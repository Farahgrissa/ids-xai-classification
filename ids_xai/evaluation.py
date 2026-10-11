from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

NAIF = "Naïf (toujours normal)"


def probabilites(modeles: dict, X: pd.DataFrame) -> dict[str, np.ndarray]:
    return {nom: pipe.predict_proba(X)[:, 1] for nom, pipe in modeles.items()}


def metriques(y, p, seuil: float = 0.5, gini_ks: bool = False) -> dict:
    y = np.asarray(y)
    pred = (p >= seuil).astype(int)
    res = {"AUC-ROC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
           "Précision": precision_score(y, pred, zero_division=0), "Rappel": recall_score(y, pred),
           "F1": f1_score(y, pred), "Accuracy": accuracy_score(y, pred)}
    if gini_ks:  # demandés par l'énoncé pour le scoring de crédit uniquement
        res["Gini"] = 2 * res["AUC-ROC"] - 1
        res["KS"] = ks_2samp(p[y == 1], p[y == 0]).statistic if len(set(p)) > 1 else 0.0
    return res


def tableau_metriques(probas: dict, y, seuil: float = 0.5, avec_naif: bool = True,
                      gini_ks: bool = False, reference_cv: pd.Series | None = None) -> pd.DataFrame:
    probas = dict(probas)
    if avec_naif:
        probas[NAIF] = np.zeros(len(y))
    tab = pd.DataFrame({nom: metriques(y, p, seuil, gini_ks) for nom, p in probas.items()}).T
    tab.index.name = "Modèle"
    if reference_cv is not None:
        tab["PR-AUC en CV"] = reference_cv
        tab["Écart test - CV"] = tab["PR-AUC"] - tab["PR-AUC en CV"]
    return tab.sort_values("PR-AUC", ascending=False)

def fig_roc(probas: dict, y, zoom_log: bool = False) -> go.Figure:
    fig = go.Figure()
    for nom, p in probas.items():
        if nom == NAIF:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        x = np.clip(fpr, 1e-4, 1) if zoom_log else fpr
        fig.add_trace(go.Scatter(x=x, y=tpr, mode="lines",
                                 name=f"{nom} (AUC {roc_auc_score(y, p):.4f})"))
    if zoom_log:
        fig.update_xaxes(type="log", range=[-4, 0])
        titre = "Zoom ROC – zone à faible taux de fausses alertes (échelle log)"
    else:
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Hasard",
                                 line=dict(dash="dash", color="gray")))
        titre = "Courbes ROC superposées – jeu de test"
    fig.update_layout(title=titre, height=500, xaxis_title="Taux de faux positifs",
                      yaxis_title="Taux de vrais positifs (rappel)")
    return fig


def fig_pr(probas: dict, y) -> go.Figure:
    fig = go.Figure()
    for nom, p in probas.items():
        if nom == NAIF:
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                 name=f"{nom} (PR-AUC {average_precision_score(y, p):.4f})"))
    taux = float(np.mean(y))
    fig.add_hline(y=taux, line_dash="dash", line_color="gray", annotation_text=f"Hasard = {taux:.3f}")
    fig.update_layout(title="Courbes précision-rappel – jeu de test", height=500,
                      xaxis_title="Rappel (attaques détectées)", yaxis_title="Précision (alertes justes)")
    return fig


def points_fonctionnement(probas: dict, y, taux_fa=(0.001, 0.005, 0.01, 0.05)) -> pd.DataFrame:
    """Rappel atteignable pour un taux maximal de fausses alertes (lecture du zoom ROC)."""
    lignes = {}
    for nom, p in probas.items():
        if nom == NAIF:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        lignes[nom] = {f"FA ≤ {f:.1%}": tpr[fpr <= f].max() for f in taux_fa}
    return pd.DataFrame(lignes).T

def confusion(y, p, seuil: float = 0.5) -> dict:
    vn, fp, fn, vp = confusion_matrix(y, (p >= seuil).astype(int), labels=[0, 1]).ravel()
    return {"VN": int(vn), "FP": int(fp), "FN": int(fn), "VP": int(vp)}


def fig_confusion(cm: dict, titre: str) -> go.Figure:
    return px.imshow([[cm["VN"], cm["FP"]], [cm["FN"], cm["VP"]]], text_auto=True,
                     color_continuous_scale="Blues", x=["Prédit normal", "Prédit attaque"],
                     y=["Réel normal", "Réel attaque"], title=titre)


def erreurs_par_type(types, y, p, seuil: float = 0.5) -> pd.DataFrame:
    d = pd.DataFrame({"type": np.asarray(types), "reel": np.asarray(y), "pred": (p >= seuil).astype(int)})
    d["erreur"] = d["reel"] != d["pred"]
    t = d.groupby("type").agg(effectif=("erreur", "size"), erreurs=("erreur", "sum"),
                              taux_correct=("erreur", lambda s: 1 - s.mean()))
    t["taux_correct"] = (t["taux_correct"] * 100).round(1)
    return t.sort_values("taux_correct")


def stats_seuil(y, p, seuil: float) -> dict:
    cm = confusion(y, p, seuil)
    rappel = cm["VP"] / max(cm["VP"] + cm["FN"], 1)
    taux_fa = cm["FP"] / max(cm["FP"] + cm["VN"], 1)
    precision = cm["VP"] / (cm["VP"] + cm["FP"]) if (cm["VP"] + cm["FP"]) else 0.0
    return {**cm, "rappel": rappel, "taux_fa": taux_fa, "precision": precision}


def precision_a_prevalence(rappel: float, taux_fa: float, taux_attaque: float) -> float:
    denom = rappel * taux_attaque + taux_fa * (1 - taux_attaque)
    return rappel * taux_attaque / denom if denom > 0 else 0.0


def projection(rappel: float, taux_fa: float, taux_attaque: float, n: int = 100_000) -> dict:
    return {"detectees": n * taux_attaque * rappel, "manquees": n * taux_attaque * (1 - rappel),
            "fausses_alertes": n * (1 - taux_attaque) * taux_fa}


def courbe_seuils(y, p, pas: float = 0.01) -> pd.DataFrame:
    seuils = np.round(np.arange(pas, 1.0, pas), 2)
    return pd.DataFrame([{"seuil": s, **stats_seuil(y, p, s)} for s in seuils])


def fig_seuil(courbe: pd.DataFrame, seuil: float) -> go.Figure:
    fig = go.Figure()
    for col, nom in [("rappel", "Rappel"), ("precision", "Précision"), ("taux_fa", "Taux de fausses alertes")]:
        fig.add_trace(go.Scatter(x=courbe["seuil"], y=courbe[col], name=nom))
    fig.add_vline(x=seuil, line_dash="dash", annotation_text=f"seuil {seuil:.2f}")
    fig.update_layout(title="Effet du seuil de décision", xaxis_title="Seuil", height=420)
    return fig
