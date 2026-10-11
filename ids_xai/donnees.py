from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from . import config as C


def charger_jour(nom_fichier: str, frac: float = C.FRAC, random_state: int = C.RANDOM_STATE,
                 chunksize: int = 200_000, base_url: str = C.BASE_URL,
                 progression: Callable[[int], None] | None = None) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    morceaux = []
    for i, chunk in enumerate(pd.read_csv(base_url + nom_fichier, chunksize=chunksize, dtype=str)):
        chunk = chunk[chunk[C.COLONNE_LABEL] != C.COLONNE_LABEL]
        morceaux.append(chunk.sample(frac=frac, random_state=rng))
        if progression:
            progression(i + 1)
    df = pd.concat(morceaux, ignore_index=True)
    df[C.COLONNE_JOUR] = nom_fichier.split("_")[0]
    return df


def telecharger(jours: Iterable[str] = tuple(C.JOURS), frac: float = C.FRAC,
                random_state: int = C.RANDOM_STATE, base_url: str = C.BASE_URL,
                progression: Callable[[str, int], None] | None = None) -> pd.DataFrame:

    tables = []
    for f in jours:
        cb = (lambda i, f=f: progression(f, i)) if progression else None
        tables.append(charger_jour(f, frac, random_state, base_url=base_url, progression=cb))
    return pd.concat(tables, ignore_index=True)


def nettoyer(brut: pd.DataFrame, colonne_label: str = C.COLONNE_LABEL,
             classe_negative: str = C.CLASSE_NEGATIVE,
             colonnes_a_retirer: Iterable[str] = C.COLONNES_A_RETIRER) -> tuple[pd.DataFrame, dict]:

    rapport: dict = {"lignes_brutes": len(brut)}
    df = brut[brut[colonne_label].astype(str) != colonne_label].copy().reset_index(drop=True)
    texte = [colonne_label, C.COLONNE_JOUR]
    retirees = [c for c in colonnes_a_retirer if c in df.columns]
    df = df.drop(columns=retirees)
    rapport["colonnes_retirees"] = retirees
    for c in df.columns:
        if c not in texte:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # infinis (flux de durée 0) → NaN
    num = df.select_dtypes(include="number").columns
    inf = np.isinf(df[num]).sum()
    rapport["infinis"] = {k: int(v) for k, v in inf[inf > 0].items()}
    df = df.replace([np.inf, -np.inf], np.nan)

    # colonnes constantes
    constantes = [c for c in num if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=constantes)
    rapport["constantes"] = constantes

    # doublons et conflits d'étiquettes (mêmes variables, étiquettes différentes)
    feat = [c for c in df.columns if c not in texte]
    cle = pd.util.hash_pandas_object(df[feat], index=False).values
    est_doublon = pd.Series(cle).duplicated().values
    rapport["doublons_par_label"] = (pd.Series(est_doublon).groupby(df[colonne_label].values)
                                     .sum().astype(int).to_dict())
    nb_labels = df.groupby(cle)[colonne_label].transform("nunique")
    conflits = df[nb_labels.values > 1]
    rapport["lignes_en_conflit"] = int(len(conflits))
    if len(conflits):
        rapport["conflits"] = (conflits.groupby(cle[nb_labels.values > 1])[colonne_label]
                               .agg(lambda s: " + ".join(sorted(s.unique())))
                               .value_counts().to_dict())

    #suppression des doublons
    df = df.loc[~est_doublon].reset_index(drop=True)
    rapport["doublons_supprimes"] = int(est_doublon.sum())
    df[C.COLONNE_TYPE] = df[colonne_label].astype(str)
    df[C.CIBLE] = (df[colonne_label] != classe_negative).astype(int)
    if "Protocol" in df.columns:
        df["Protocol"] = df["Protocol"].astype("Int64").astype(str)
    df = df.drop(columns=[colonne_label])

    rapport["lignes_finales"] = len(df)
    rapport["taux_positif"] = float(df[C.CIBLE].mean())
    return df, rapport


def preparer_csv_utilisateur(brut: pd.DataFrame, cible: str, valeur_positive) -> tuple[pd.DataFrame, dict]:
    df = brut.copy().replace(r"^\s*$", np.nan, regex=True)
    for col in df.columns:
        if col == cible or pd.api.types.is_numeric_dtype(df[col]):
            continue
        conv = pd.to_numeric(df[col], errors="coerce")
        if conv.notna().sum() >= 0.95 * df[col].notna().sum():
            df[col] = conv
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    avant = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df[C.COLONNE_TYPE] = df[cible].astype(str)
    df[C.CIBLE] = (df[cible].astype(str) == str(valeur_positive)).astype(int)
    df = df.drop(columns=[cible])
    return df, {"lignes_brutes": avant, "doublons_supprimes": avant - len(df),
                "lignes_finales": len(df), "taux_positif": float(df[C.CIBLE].mean())}


def charger_donnees_propres(chemin=C.FICHIER_DONNEES) -> pd.DataFrame:
    return pd.read_parquet(chemin)


def colonnes_variables(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = [c for c in df.select_dtypes(include="number").columns if c not in C.EXCLUES]
    cat = [c for c in df.select_dtypes(exclude="number").columns if c not in C.EXCLUES]
    return num, cat
