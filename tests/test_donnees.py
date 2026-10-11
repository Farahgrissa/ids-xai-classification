import numpy as np

from ids_xai import config as C
from ids_xai.donnees import nettoyer, preparer_csv_utilisateur


def test_entetes_repetes_retires(brut):
    assert (brut[C.COLONNE_LABEL] != C.COLONNE_LABEL).all()
    assert set(brut[C.COLONNE_JOUR]) == {f.split("_")[0] for f in C.JOURS}


def test_nettoyage(brut):
    df, rapport = nettoyer(brut)
    num = df.select_dtypes("number")
    assert not np.isinf(num.to_numpy(dtype=float, na_value=0)).any()
    assert "Timestamp" not in df and "Dst Port" not in df and C.COLONNE_LABEL not in df
    assert len(rapport["constantes"]) == 10
    feat = [c for c in df.columns if c not in C.EXCLUES]
    assert not df[feat].duplicated().any()
    assert set(df[C.CIBLE]) == {0, 1}
    assert df["Protocol"].dtype == object
    assert df.shape[1] == 70


def test_csv_utilisateur():
    import pandas as pd
    brut = pd.DataFrame({"age": ["20", "30", " ", "40"], "ville": list("abca"), "churn": ["Yes", "No", "No", "Yes"]})
    df, rapport = preparer_csv_utilisateur(brut, "churn", "Yes")
    assert df[C.CIBLE].tolist() == [1, 0, 0, 1]
    assert df["age"].isna().sum() == 1 and df["age"].dtype.kind == "f"
