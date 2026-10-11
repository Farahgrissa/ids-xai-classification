from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C


def log_signe(x):
    return np.sign(x) * np.log1p(np.abs(x))


IMPUTATIONS = {"median": "Médiane", "mean": "Moyenne", "constant": "Valeur 0"}


def construire_preprocesseur(X: pd.DataFrame, log: bool = True, normaliser: bool = True,
                             imputation: str = "median") -> ColumnTransformer:
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    etapes = [("imputer", SimpleImputer(strategy=imputation, fill_value=0))]
    if log:
        etapes.append(("log", FunctionTransformer(log_signe, feature_names_out="one-to-one")))
    if normaliser:
        etapes.append(("scaler", StandardScaler()))
    transformateurs = [("num", Pipeline(etapes), num)]
    if cat:
        transformateurs.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    return ColumnTransformer(transformateurs)


def choisir_variables(df: pd.DataFrame, avec_empreinte: bool = False,
                      a_exclure: list[str] | None = None) -> list[str]:
    a_exclure = set(a_exclure or [])
    return [c for c in df.columns
            if c not in C.EXCLUES and c not in a_exclure
            and (avec_empreinte or c not in C.EMPREINTE)]


def colonnes_redondantes(X: pd.DataFrame, y: pd.Series, seuil: float = C.SEUIL_CORR,
                         prioritaires: list[str] | tuple = C.PRIORITAIRES) -> list[str]:
    num = X.select_dtypes(include="number").columns
    auc = {}
    for v in num:
        masque = X[v].notna()
        if X.loc[masque, v].nunique() < 2:
            auc[v] = 0.5
            continue
        a = roc_auc_score(y[masque], X.loc[masque, v])
        auc[v] = max(a, 1 - a)
    tete = [v for v in prioritaires if v in num]
    reste = sorted([v for v in num if v not in tete], key=lambda v: auc[v], reverse=True)
    ordre = tete + reste
    c = X[ordre].corr(method="spearman").abs()
    gardees, retirees = [], []
    for v in ordre:
        if any(c.loc[v, g] > seuil for g in gardees):
            retirees.append(v)
        else:
            gardees.append(v)
    return retirees


@dataclass
class JeuDonnees:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    t_train: pd.Series
    t_test: pd.Series
    variables: list[str]
    redondantes: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)


def preparer(df: pd.DataFrame, avec_empreinte: bool = False, test_size: float = C.TEST_SIZE,
             seuil_corr: float | None = C.SEUIL_CORR, prioritaires=C.PRIORITAIRES,
             a_exclure: list[str] | None = None, random_state: int = C.RANDOM_STATE) -> JeuDonnees:
    variables = choisir_variables(df, avec_empreinte, a_exclure)
    types = df[C.COLONNE_TYPE]
    strat = types if types.value_counts().min() >= 2 else df[C.CIBLE]
    X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
        df[variables], df[C.CIBLE], types, test_size=test_size, stratify=strat,
        random_state=random_state)
    redondantes = colonnes_redondantes(X_tr, y_tr, seuil_corr, prioritaires) if seuil_corr else []
    return JeuDonnees(
        X_train=X_tr.drop(columns=redondantes), X_test=X_te.drop(columns=redondantes),
        y_train=y_tr, y_test=y_te, t_train=t_tr, t_test=t_te,
        variables=[v for v in variables if v not in redondantes], redondantes=redondantes,
        params={"avec_empreinte": avec_empreinte, "test_size": test_size,
                "seuil_corr": seuil_corr, "random_state": random_state})
