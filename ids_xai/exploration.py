from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score

from . import config as C
from .preparation import log_signe


def libelle_classe(y: pd.Series) -> pd.Series:
    return y.map({0: "Normal", 1: "Attaque"})


def statistiques(df: pd.DataFrame, num: list[str]) -> pd.DataFrame:
    stats = df[num].describe().T
    stats["asymetrie"] = df[num].skew()
    return stats.sort_values("asymetrie", ascending=False)


def valeurs_manquantes(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isna().sum().to_frame("nb_manquants")
    m["pct"] = (m["nb_manquants"] / len(df) * 100).round(3)
    return m[m["nb_manquants"] > 0]


def medianes_par_type(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    variables = [v for v in variables if v in df.columns]
    return df.groupby(C.COLONNE_TYPE)[variables].median().T


def fig_proportion_classes(df: pd.DataFrame) -> go.Figure:
    c = libelle_classe(df[C.CIBLE]).value_counts().reset_index()
    c.columns = ["classe", "nb"]
    fig = px.pie(c, names="classe", values="nb", hole=0.4, color="classe",
                 color_discrete_map=C.COULEURS, title="Répartition Normal / Attaque")
    fig.update_traces(textinfo="percent+value")
    return fig


def fig_types(df: pd.DataFrame) -> go.Figure:
    t = df[C.COLONNE_TYPE].value_counts().reset_index()
    t.columns = ["type", "nb"]
    return px.bar(t.sort_values("nb"), x="nb", y="type", orientation="h", text="nb",
                  log_x=True, title="Nombre de flux par type (échelle log)")


def fig_distribution(df: pd.DataFrame, var: str, log: bool = True) -> go.Figure:
    x = log_signe(df[var]) if log and pd.api.types.is_numeric_dtype(df[var]) else df[var]
    nom = f"log({var})" if log else var
    tmp = pd.DataFrame({nom: x, "classe": libelle_classe(df[C.CIBLE])}).dropna()
    return px.histogram(tmp, x=nom, color="classe", barmode="overlay", histnorm="percent",
                        nbins=60, marginal="box", opacity=0.6, color_discrete_map=C.COULEURS,
                        title=f"Distribution de {var}" + (" (échelle log)" if log else ""))

def auc_univariee(df: pd.DataFrame, num: list[str], cible: str = C.CIBLE) -> pd.DataFrame:
    res = []
    for v in num:
        d = df[[v, cible]].dropna()
        if d[v].nunique() < 2 or d[cible].nunique() < 2:
            continue
        auc = roc_auc_score(d[cible], d[v])
        res.append({"variable": v, "AUC": round(max(auc, 1 - auc), 3),
                    "sens": "plus élevée chez les attaques" if auc >= 0.5
                    else "plus faible chez les attaques"})
    return pd.DataFrame(res).sort_values("AUC", ascending=False).reset_index(drop=True)


def fig_auc(pouvoir: pd.DataFrame, n: int = 20) -> go.Figure:
    fig = px.bar(pouvoir.head(n)[::-1], x="AUC", y="variable", orientation="h", color="sens",
                 title="Pouvoir séparateur de chaque variable seule (AUC univariée)")
    fig.add_vline(x=0.5, line_dash="dash", annotation_text="hasard")
    return fig


def taux_attaque(df: pd.DataFrame, var: str, n_bins: int = 5, max_valeurs: int = 15) -> pd.DataFrame:
    d = df[[var, C.CIBLE]].dropna().copy()
    discrete = d[var].nunique() <= max_valeurs or not pd.api.types.is_numeric_dtype(d[var])
    d["groupe"] = d[var] if discrete else pd.qcut(d[var], q=n_bins, duplicates="drop")
    t = (d.groupby("groupe", observed=True)[C.CIBLE]
         .agg(taux="mean", effectif="size").reset_index())
    t["taux"] = (t["taux"] * 100).round(1)
    if discrete:
        t = t.sort_values("effectif", ascending=False)
    t["groupe"] = t["groupe"].astype(str)
    return t


def fig_taux(df: pd.DataFrame, var: str) -> go.Figure:
    t = taux_attaque(df, var)
    moyenne = df[C.CIBLE].mean() * 100
    fig = px.bar(t, x="groupe", y="taux", text="taux", hover_data=["effectif"],
                 title=f"Taux d'attaque selon {var}")
    fig.add_hline(y=moyenne, line_dash="dash", annotation_text=f"Moyenne {moyenne:.1f} %")
    fig.update_xaxes(type="category")
    return fig


def fig_correlations(df: pd.DataFrame, variables: list[str]) -> go.Figure:
    corr = df[variables].corr(method="spearman").round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Corrélations (Spearman)")
    fig.update_layout(height=max(450, 40 * len(variables)))
    return fig


def paires_redondantes(df: pd.DataFrame, num: list[str], seuil: float = C.SEUIL_CORR) -> pd.Series:
    c = df[num].corr(method="spearman").abs()
    paires = c.where(np.triu(np.ones(c.shape), 1).astype(bool)).stack()
    return paires[paires > seuil].sort_values(ascending=False)
