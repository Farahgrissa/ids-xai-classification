import pytest

from ids_xai.evaluation import precision_a_prevalence, probabilites, tableau_metriques
from ids_xai.modeles import entrainer_et_comparer


def test_precision_bayes():
    # Chiffres du notebook (XGBoost, seuil 0,5) : 1 % d'attaques → ≈ 52 % de précision
    assert precision_a_prevalence(0.981, 0.0092, 0.01) == pytest.approx(0.519, abs=0.005)


def test_entrainement_et_evaluation(jeu):
    comp, modeles, _ = entrainer_et_comparer(["XGBoost", "Régression logistique"],
                                             jeu.X_train, jeu.y_train, jeu.t_train)
    assert comp.loc["XGBoost", "AUC-ROC"] > 0.9
    tab = tableau_metriques(probabilites(modeles, jeu.X_test), jeu.y_test)
    assert tab.loc["Naïf (toujours normal)", "Rappel"] == 0
