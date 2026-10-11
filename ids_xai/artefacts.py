from __future__ import annotations

import datetime as dt
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost

from . import config as C

FICHIER_API = C.DOSSIER_MODELES / "modele_api.joblib"


def versions() -> dict:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "pandas": pd.__version__}


def schema_variables(X: pd.DataFrame) -> dict[str, str]:
    return {c: ("num" if pd.api.types.is_numeric_dtype(X[c]) else "cat") for c in X.columns}


def sauvegarder(artefacts: dict, chemin: Path = C.FICHIER_ARTEFACTS) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    artefacts = {**artefacts, "versions": versions(),
                 "date": dt.datetime.now().isoformat(timespec="seconds")}
    joblib.dump(artefacts, chemin, compress=3)
    return chemin


def flux_exemple(X: pd.DataFrame) -> dict:
    ex = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            ex[c] = float(X[c].median())
        else:
            ex[c] = str(X[c].mode().iloc[0])
    return ex


def sauvegarder_modele_api(pipeline, schema: dict, metriques: dict, nom: str, exemple: dict | None = None,
                           seuil: float = C.SEUIL_RECOMMANDE, chemin: Path = FICHIER_API) -> Path:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "schema": schema, "metriques": metriques, "nom": nom,
                 "exemple": exemple or {},
                 "seuil": seuil, "versions": versions(),
                 "date": dt.datetime.now().isoformat(timespec="seconds")}, chemin, compress=3)
    return chemin


def charger(chemin: Path = C.FICHIER_ARTEFACTS) -> dict | None:
    chemin = Path(chemin)
    return joblib.load(chemin) if chemin.exists() else None
