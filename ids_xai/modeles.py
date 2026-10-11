from __future__ import annotations

import time
from typing import Callable

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config as C
from .preparation import construire_preprocesseur

RS = C.RANDOM_STATE

SCORING = {"AUC-ROC": "roc_auc", "PR-AUC": "average_precision",
           "F1": "f1", "Précision": "precision", "Rappel": "recall"}

CATALOGUE: dict[str, dict] = {
    "XGBoost": {
        "classe": XGBClassifier,
        "fixes": {"eval_metric": "logloss", "random_state": RS, "n_jobs": -1},
        "optimises": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1,
                      "subsample": 1.0, "colsample_bytree": 1.0},
        "methode": "random",
        "espace": {"n_estimators": [100, 200, 400], "max_depth": [3, 5, 7, 9],
                   "learning_rate": [0.03, 0.1, 0.3], "subsample": [0.7, 0.85, 1.0],
                   "colsample_bytree": [0.6, 0.8, 1.0]},
    },
    "Random Forest": {
        "classe": RandomForestClassifier,
        "fixes": {"random_state": RS, "n_jobs": -1},
        "optimises": {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 2,
                      "max_features": "sqrt"},
        "methode": "random",
        "espace": {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20, 30],
                   "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", "log2", 0.5]},
    },
    "Arbre de décision": {
        "classe": DecisionTreeClassifier,
        "fixes": {"random_state": RS},
        "optimises": {"max_depth": 15, "min_samples_leaf": 50},
        "methode": "grid",
        "espace": {"max_depth": [5, 10, 15, 20, None], "min_samples_leaf": [1, 5, 20, 50]},
    },
    "KNN": {
        "classe": KNeighborsClassifier,
        "fixes": {},
        "optimises": {"n_neighbors": 15, "weights": "uniform"},
        "methode": "grid",
        "espace": {"n_neighbors": [3, 7, 15], "weights": ["uniform", "distance"]},
    },
    "Régression logistique": {
        "classe": LogisticRegression,
        "fixes": {"max_iter": 2000},
        "optimises": {"C": 100, "class_weight": None},
        "methode": "grid",
        "espace": {"C": [0.01, 0.1, 1, 10, 100], "class_weight": [None, "balanced"]},
    },
}
NOMS_MODELES = list(CATALOGUE)


def creer_modele(nom: str, hyperparametres: dict | None = None, optimise: bool = True):
    spec = CATALOGUE[nom]
    params = dict(spec["fixes"])
    if optimise:
        params.update(spec["optimises"])
    params.update(hyperparametres or {})
    return spec["classe"](**params)


def creer_pipeline(nom: str, X: pd.DataFrame, hyperparametres: dict | None = None,
                   optimise: bool = True, log: bool = True, normaliser: bool = True,
                   imputation: str = "median") -> Pipeline:
    return Pipeline([("prep", construire_preprocesseur(X, log, normaliser, imputation)),
                     ("modele", creer_modele(nom, hyperparametres, optimise))])


def creer_folds(X: pd.DataFrame, strat: pd.Series, n_splits: int = C.N_FOLDS):
    if strat.value_counts().min() < n_splits:
        strat = strat.where(strat.map(strat.value_counts()) >= n_splits, "autre")
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RS).split(X, strat))


def evaluer_cv(pipe: Pipeline, X, y, folds) -> dict:
    cv = cross_validate(pipe, X, y, cv=folds, scoring=SCORING, return_train_score=True)
    res = {k: cv[f"test_{k}"].mean() for k in SCORING}
    res["± AUC"] = cv["test_AUC-ROC"].std()
    res["Écart train-val"] = cv["train_AUC-ROC"].mean() - cv["test_AUC-ROC"].mean()
    return res


def optimiser(nom: str, X, y, folds, n_iter: int = 10, log: bool = True, normaliser: bool = True,
              imputation: str = "median"):
    spec = CATALOGUE[nom]
    pipe = creer_pipeline(nom, X, optimise=False, log=log, normaliser=normaliser, imputation=imputation)
    grille = {f"modele__{k}": v for k, v in spec["espace"].items()}
    opts = dict(cv=folds, scoring=SCORING, refit="PR-AUC", n_jobs=-1, return_train_score=True)
    rech = (GridSearchCV(pipe, grille, **opts) if spec["methode"] == "grid"
            else RandomizedSearchCV(pipe, grille, n_iter=n_iter, random_state=RS, **opts))
    rech.fit(X, y)
    meilleurs = {k.replace("modele__", ""): v for k, v in rech.best_params_.items()}
    return rech.best_estimator_, meilleurs, rech


def temps_prediction(pipe: Pipeline, X: pd.DataFrame, n: int = 5000) -> float:
    ech = X.sample(min(n, len(X)), random_state=RS)
    t0 = time.time()
    pipe.predict_proba(ech)
    return (time.time() - t0) / len(ech) * 1e6


def entrainer_et_comparer(noms: list[str], X, y, strat, hyperparametres: dict | None = None,
                          optimise: bool = True, log: bool = True, normaliser: bool = True,
                          rechercher: bool = False, n_iter: int = 10, imputation: str = "median",
                          progression: Callable[[str], None] | None = None):
    hyperparametres = hyperparametres or {}
    folds = creer_folds(X, strat)
    lignes, modeles, retenus = [], {}, {}
    for nom in noms:
        if progression:
            progression(nom)
        if rechercher:
            _, params, _ = optimiser(nom, X, y, folds, n_iter, log, normaliser, imputation)
        else:
            params = hyperparametres.get(nom, {})
        pipe = creer_pipeline(nom, X, params, optimise or rechercher, log, normaliser, imputation)
        res = evaluer_cv(pipe, X, y, folds)
        t0 = time.time()
        pipe.fit(X, y)
        res["Entraînement (s)"] = time.time() - t0
        res["Prédiction (ms/1000 flux)"] = temps_prediction(pipe, X)
        lignes.append({"Modèle": nom, **res})
        modeles[nom] = pipe
        retenus[nom] = pipe.named_steps["modele"].get_params()
    comparatif = pd.DataFrame(lignes).set_index("Modèle").sort_values("PR-AUC", ascending=False)
    return comparatif, modeles, retenus
