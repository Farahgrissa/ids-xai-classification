import numpy as np
import pandas as pd

from ids_xai import config as C
from ids_xai.preparation import colonnes_redondantes, construire_preprocesseur, log_signe


def test_log_signe():
    x = np.array([-1.0, 0.0, 1e7])
    r = log_signe(x)
    assert r[0] < 0 and r[1] == 0 and r[2] < 17


def test_redondances_garde_prioritaires():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 500))
    base = rng.exponential(1, 500) + y
    X = pd.DataFrame({"Flow Duration": base, "copie": base * 3, "autre": rng.normal(size=500)})
    assert colonnes_redondantes(X, y, 0.95) == ["copie"]


def test_split_sans_fuite(jeu):
    assert set(jeu.X_train.index).isdisjoint(jeu.X_test.index)
    assert abs(jeu.y_train.mean() - jeu.y_test.mean()) < 0.02
    assert not set(C.EMPREINTE) & set(jeu.variables)
    assert not set(C.EXCLUES) & set(jeu.variables)


def test_preprocesseur(jeu):
    prep = construire_preprocesseur(jeu.X_train)
    Xtr = prep.fit_transform(jeu.X_train)
    assert not np.isnan(Xtr).any()
    assert np.allclose(Xtr[:, :5].mean(axis=0), 0, atol=1e-6)
