from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids_xai import __version__
from ids_xai.artefacts import FICHIER_API
from ids_xai.explications import contributions, phrase_synthese, type_explainer, valeurs_shap

CHEMIN_MODELE = Path(os.getenv("MODELE_CHEMIN", FICHIER_API))
ETAT: dict[str, Any] = {}


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    ETAT["bundle"] = joblib.load(CHEMIN_MODELE) if CHEMIN_MODELE.exists() else None
    yield
    ETAT.clear()


app = FastAPI(
    title="API de détection d'intrusion explicable",
    description=(
        "Classe des flux réseau (CSE-CIC-IDS2018, format CICFlowMeter) en **normal** ou **attaque** "
        "avec le modèle retenu (XGBoost).\n\n"
        "- `GET /exemple` fournit un flux prêt à envoyer ;\n"
        "- `POST /predict` renvoie la probabilité d'attaque et la décision au seuil choisi ;\n"
        "- `POST /explain` renvoie, pour chaque flux, les contributions SHAP des variables "
        "(φ0 + Σ contributions = logit de la probabilité) et une phrase de synthèse."
    ),
    version=__version__,
    lifespan=cycle_de_vie,
)


class RequetePrediction(BaseModel):
    flux: list[dict[str, float | int | str | None]] = Field(
        ..., min_length=1, max_length=10_000,
        description="Un ou plusieurs flux : dictionnaire {nom de variable: valeur}. "
                    "La liste des variables attendues est donnée par GET /model-info.")
    seuil: float | None = Field(None, ge=0, le=1, description="Seuil de décision (défaut : seuil du modèle)")


class ResultatFlux(BaseModel):
    probabilite_attaque: float
    decision: str
    niveau: str


class ReponsePrediction(BaseModel):
    modele: str
    seuil: float
    resultats: list[ResultatFlux]
    variables_ignorees: list[str] = []


class RequeteExplication(BaseModel):
    flux: list[dict[str, float | int | str | None]] = Field(
        ..., min_length=1, max_length=100, description="Un ou plusieurs flux (100 au maximum).")
    seuil: float | None = Field(None, ge=0, le=1, description="Seuil de décision (défaut : seuil du modèle)")
    top: int = Field(5, ge=1, le=50, description="Nombre de contributions renvoyées par flux")


class Contribution(BaseModel):
    variable: str
    valeur: float | None
    valeur_lisible: str
    contribution: float


class ExplicationFlux(BaseModel):
    probabilite_attaque: float
    decision: str
    niveau: str
    valeur_de_base: float = Field(..., description="φ0 : prédiction moyenne du modèle, point de départ")
    unite: str
    contributions: list[Contribution]
    phrase: str


class ReponseExplication(BaseModel):
    modele: str
    seuil: float
    explications: list[ExplicationFlux]
    variables_ignorees: list[str] = []


def _bundle() -> dict:
    if ETAT.get("bundle") is None:
        raise HTTPException(503, f"Modèle introuvable ({CHEMIN_MODELE}). Lancez scripts/entrainer.py.")
    return ETAT["bundle"]


def _vers_dataframe(flux: list[dict], schema: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    manquantes = sorted({v for f in flux for v in schema if v not in f})
    if manquantes:
        raise HTTPException(422, {"message": "Variables manquantes", "variables": manquantes})
    ignorees = sorted({k for f in flux for k in f if k not in schema})
    df = pd.DataFrame(flux)[list(schema)]
    for col, type_ in schema.items():
        if type_ == "num":
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True)
    return df, ignorees


def _niveau(p: float, seuil: float) -> str:
    if p >= max(seuil, 0.95):
        return "critique"
    if p >= seuil:
        return "élevé"
    if p >= seuil / 2:
        return "à surveiller"
    return "faible"

@app.get("/", include_in_schema=False)
def racine():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Service"])
def health():
    return {"statut": "ok", "modele_charge": ETAT.get("bundle") is not None}


@app.get("/model-info", tags=["Service"])
def model_info():
    b = _bundle()
    return {"modele": b["nom"], "seuil_par_defaut": b["seuil"], "variables": b["schema"],
            "metriques_test": b["metriques"], "versions": b["versions"], "entraine_le": b["date"]}


@app.get("/exemple", tags=["Service"])
def exemple():
    return {"flux": [_bundle().get("exemple", {})]}


@app.post("/predict", response_model=ReponsePrediction, tags=["Prédiction"])
def predict(req: RequetePrediction):
    b = _bundle()
    seuil = b["seuil"] if req.seuil is None else req.seuil
    df, ignorees = _vers_dataframe(req.flux, b["schema"])
    probas = b["pipeline"].predict_proba(df)[:, 1]
    return ReponsePrediction(
        modele=b["nom"], seuil=seuil, variables_ignorees=ignorees,
        resultats=[ResultatFlux(probabilite_attaque=round(float(p), 6),
                                decision="attaque" if p >= seuil else "normal",
                                niveau=_niveau(float(p), seuil)) for p in probas])


@app.post("/explain", response_model=ReponseExplication, tags=["Explication"])
def explain(req: RequeteExplication):
    """Contributions SHAP (TreeExplainer) de chaque variable et phrase de synthèse.

    Les contributions sont en log-odds pour XGBoost : φ0 + Σ contributions = logit(probabilité).
    Une contribution positive pousse vers « attaque ».
    """
    b = _bundle()
    if type_explainer(b["pipeline"]) != "tree":
        raise HTTPException(501, "Explications disponibles pour les modèles à base d'arbres uniquement.")
    seuil = b["seuil"] if req.seuil is None else req.seuil
    df, ignorees = _vers_dataframe(req.flux, b["schema"])
    probas = b["pipeline"].predict_proba(df)[:, 1]
    expl, unite = valeurs_shap(b["pipeline"], df)
    medianes = b.get("medianes_normal")
    sortie = []
    for i, p in enumerate(probas):
        contrib = contributions(expl, i)
        sortie.append(ExplicationFlux(
            probabilite_attaque=round(float(p), 6), decision="attaque" if p >= seuil else "normal",
            niveau=_niveau(float(p), seuil), valeur_de_base=round(float(expl.base_values[i]), 6), unite=unite,
            contributions=[Contribution(variable=r.variable,
                                        valeur=None if pd.isna(r.valeur) else float(r.valeur),
                                        valeur_lisible=r.valeur_lisible,
                                        contribution=round(float(r.contribution), 6))
                           for r in contrib.head(req.top).itertuples()],
            phrase=phrase_synthese(contrib, float(p), seuil, medianes)))
    return ReponseExplication(modele=b["nom"], seuil=seuil, explications=sortie, variables_ignorees=ignorees)
