import numpy as np
import pytest
from scipy.special import logit

from ids_xai import explications as E
from ids_xai.modeles import creer_pipeline


@pytest.fixture(scope="module")
def xgb(jeu):
    return creer_pipeline("XGBoost", jeu.X_train).fit(jeu.X_train, jeu.y_train)


@pytest.fixture(scope="module")
def expl(xgb, jeu):
    return E.valeurs_shap(xgb, jeu.X_test.iloc[:200])


def test_regroupement_one_hot(xgb, jeu):
    noms = xgb.named_steps["prep"].get_feature_names_out()
    m = E.matrice_regroupement(noms, list(jeu.X_train.columns))
    assert m.shape == (len(noms), jeu.X_train.shape[1])
    assert (m.sum(axis=1) == 1).all()              # chaque colonne transformée → une seule variable


def test_shap_additivite_log_odds(xgb, jeu, expl):
    e, unite = expl
    assert unite == E.UNITE_LOG_ODDS
    assert e.values.shape == (200, jeu.X_train.shape[1])
    p = np.clip(xgb.predict_proba(jeu.X_test.iloc[:200])[:, 1], 1e-6, 1 - 1e-6)
    assert np.abs(e.values.sum(1) + e.base_values - logit(p)).max() < 1e-2


def test_importances(xgb, jeu):
    nat = E.importance_native(xgb, list(jeu.X_train.columns))
    assert abs(nat.sum() - 1) < 1e-9
    perm = E.importance_permutation(xgb, jeu.X_test, jeu.y_test, n_repetitions=2, n_max=500)
    assert list(perm.columns) == ["baisse_moyenne", "ecart_type"] and len(perm) == jeu.X_test.shape[1]


def test_lime_et_comparaison(xgb, jeu, expl):
    lime = E.explication_lime(xgb, jeu.X_train, jeu.X_test.iloc[[0]], n_echantillons=500)
    assert {"variable", "regle", "poids"} <= set(lime["poids"].columns) and len(lime["poids"]) == 10
    comp = E.comparer_shap_lime(E.contributions(expl[0], 0), lime["poids"])
    assert set(comp["meme_sens"]) <= set(comp["communes"])


def test_phrase_synthese(expl, jeu):
    contrib = E.contributions(expl[0], 0)
    attaque = E.phrase_synthese(contrib.assign(contribution=contrib["contribution"].abs()), 0.95, 0.8,
                                {v: 1.0 for v in jeu.X_train.columns})
    assert attaque.startswith("Le risque d'attaque est élevé (95,0 %, seuil 0,80) principalement à cause de")
    assert "médiane du trafic normal" in attaque
    normal = E.phrase_synthese(contrib.assign(contribution=-contrib["contribution"].abs()), 0.1, 0.8)
    assert normal.startswith("Le flux est jugé normal")


def test_formater_valeur():
    assert E.formater_valeur("Flow Duration", 8400) == "8,4 ms"
    assert E.formater_valeur("Fwd IAT Mean", 2_500_000) == "2,5 s"
    assert E.formater_valeur("Fwd Pkt Len Max", 0) == "0 octets"
    assert E.formater_valeur("Protocol", 6) == "TCP"
