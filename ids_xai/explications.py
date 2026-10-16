"""Module 5 – Explainable AI.

Explications globales : importance propre au modèle, importance par permutation, SHAP (barres,
beeswarm, dépendance). Explications locales : SHAP (waterfall, decision plot), LIME, comparaison
SHAP / LIME et phrase de synthèse automatique.

Le modèle est un pipeline scikit-learn (prétraitement + classifieur). SHAP explique le classifieur
sur les variables transformées ; les contributions sont ensuite regroupées par variable d'origine
(les colonnes one-hot de Protocol sont additionnées) et affichées avec les valeurs d'origine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

UNITE_LOG_ODDS = "log-odds"
UNITE_PROBA = "probabilité"


# --------------------------------------------------------------------------------------------
# Outils
# --------------------------------------------------------------------------------------------
def etapes(pipeline):
    return pipeline.named_steps["prep"], pipeline.named_steps["modele"]


def variable_origine(nom_transforme: str, variables: list[str]) -> str:
    """'num__Flow Duration' → 'Flow Duration' ; 'cat__Protocol_6' → 'Protocol'."""
    nom = nom_transforme.split("__", 1)[-1]
    if nom in variables:
        return nom
    for v in variables:
        if nom.startswith(f"{v}_"):
            return v
    return nom


def matrice_regroupement(noms_transformes, variables: list[str]) -> np.ndarray:
    """Matrice 0/1 (colonnes transformées × variables d'origine) qui additionne les colonnes one-hot."""
    m = np.zeros((len(noms_transformes), len(variables)))
    for i, nom in enumerate(noms_transformes):
        m[i, variables.index(variable_origine(nom, variables))] = 1
    return m


def en_numerique(X: pd.DataFrame) -> pd.DataFrame:
    """Copie entièrement numérique (Protocol '6' → 6.0) pour les graphiques et LIME."""
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.astype(float)


def unite_de(var: str) -> str:
    if "IAT" in var or var in ("Flow Duration", "Active Std", "Active Max", "Idle Std", "Idle Max"):
        return "µs"
    if var.endswith("/s"):
        return "/s"
    if "Len" in var or "Size" in var or var == "TotLen Fwd Pkts":
        return "octets"
    return ""


def formater_valeur(var: str, v) -> str:
    """Valeur lisible : 12000 µs → '12 ms', 1500000 octets/s → '1,5 M/s'."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "manquante"
    v = float(v)
    unite = unite_de(var)
    if var == "Protocol":
        return {6: "TCP", 17: "UDP", 0: "0"}.get(int(v), str(int(v)))
    if unite == "µs":
        a = abs(v)
        if a >= 1e6:
            return f"{v / 1e6:.3g} s".replace(".", ",")
        if a >= 1e3:
            return f"{v / 1e3:.3g} ms".replace(".", ",")
        return f"{v:.3g} µs".replace(".", ",")
    if unite == "/s":
        mot = " octets/s" if "Byts" in var else " paquets/s" if "Pkts" in var else "/s"
        a = abs(v)
        if a >= 1e6:
            return f"{v / 1e6:.3g} M{mot.strip()}".replace(".", ",")
        if a >= 1e3:
            return f"{v / 1e3:.3g} k{mot.strip()}".replace(".", ",")
        return f"{v:.3g}{mot}".replace(".", ",")
    texte = f"{v:.3g}".replace(".", ",")
    return f"{texte} {unite}".strip()


# --------------------------------------------------------------------------------------------
# Explications globales
# --------------------------------------------------------------------------------------------
def importance_native(pipeline, variables: list[str]) -> pd.Series:
    """Importance propre au modèle, regroupée par variable et normalisée (somme = 1).

    Arbres / forêts / XGBoost : importance par impureté (gain) ;
    régression logistique : |coefficient| (les variables sont standardisées, donc comparables).
    """
    prep, modele = etapes(pipeline)
    noms = prep.get_feature_names_out()
    if hasattr(modele, "feature_importances_"):
        brute = np.asarray(modele.feature_importances_, dtype=float)
    elif hasattr(modele, "coef_"):
        brute = np.abs(np.ravel(modele.coef_))
    else:
        raise ValueError(f"{type(modele).__name__} n'a pas d'importance native (KNN par exemple).")
    imp = pd.Series(brute @ matrice_regroupement(noms, variables), index=variables)
    return (imp / imp.sum()).sort_values(ascending=False)


def importance_permutation(pipeline, X: pd.DataFrame, y, n_repetitions: int = 5,
                           scoring: str = "average_precision", n_max: int | None = 3000,
                           random_state: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Perte de PR-AUC quand on mélange une variable : comparable d'un modèle à l'autre."""
    from sklearn.inspection import permutation_importance

    if n_max and len(X) > n_max:
        idx = pd.Series(np.asarray(y)).groupby(np.asarray(y)).sample(
            frac=n_max / len(X), random_state=random_state).index
        X, y = X.iloc[idx], np.asarray(y)[idx]
    r = permutation_importance(pipeline, X, y, scoring=scoring, n_repeats=n_repetitions,
                               random_state=random_state, n_jobs=1)
    return (pd.DataFrame({"baisse_moyenne": r.importances_mean, "ecart_type": r.importances_std},
                         index=X.columns).sort_values("baisse_moyenne", ascending=False))


# --------------------------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------------------------
def type_explainer(pipeline) -> str | None:
    modele = etapes(pipeline)[1]
    nom = type(modele).__name__
    if nom in ("XGBClassifier", "RandomForestClassifier", "DecisionTreeClassifier"):
        return "tree"
    if nom == "LogisticRegression":
        return "linear"
    return None


def valeurs_shap(pipeline, X: pd.DataFrame, X_fond: pd.DataFrame | None = None, n_fond: int = 200):
    """Valeurs SHAP regroupées par variable d'origine.

    Renvoie (Explanation, unité). XGBoost et la régression logistique sont expliqués en log-odds
    (additivité exacte sur le logit), les arbres et forêts scikit-learn en probabilité.
    """
    import shap

    prep, modele = etapes(pipeline)
    variables = list(X.columns)
    Xt = prep.transform(X)
    genre = type_explainer(pipeline)
    if genre == "tree":
        expl = shap.TreeExplainer(modele)(Xt)
        valeurs, base = expl.values, expl.base_values
        if valeurs.ndim == 3:  # classifieurs scikit-learn : une sortie par classe
            valeurs, base = valeurs[:, :, 1], base[:, 1]
        unite = UNITE_LOG_ODDS if type(modele).__name__ == "XGBClassifier" else UNITE_PROBA
    elif genre == "linear":
        fond = X if X_fond is None else X_fond
        fond = prep.transform(fond.sample(min(n_fond, len(fond)), random_state=C.RANDOM_STATE))
        expl = shap.LinearExplainer(modele, fond)(Xt)
        valeurs, base, unite = expl.values, expl.base_values, UNITE_LOG_ODDS
    else:
        raise ValueError(f"Pas d'explainer SHAP rapide pour {type(modele).__name__}.")

    groupees = np.asarray(valeurs) @ matrice_regroupement(prep.get_feature_names_out(), variables)
    return shap.Explanation(values=groupees, base_values=np.ravel(base).astype(float),
                            data=en_numerique(X).to_numpy(), feature_names=variables), unite


def _couleurs_en_rangs(expl):
    """Copie dont les données sont remplacées par leur rang (0 = plus faible, 1 = plus forte).

    Les variables réseau ont des queues très lourdes : colorées par leur valeur brute, presque tous
    les points seraient « faibles ». Le rang garde l'information « valeur faible / élevée ».
    """
    import shap

    rangs = pd.DataFrame(expl.data).rank(pct=True).to_numpy()
    return shap.Explanation(values=expl.values, base_values=expl.base_values, data=rangs,
                            feature_names=expl.feature_names)


def _lisible(expl_i):
    """Flux unique avec ses valeurs formatées (12 ms, 0 octets…) pour le waterfall."""
    import shap

    vars_ = expl_i.feature_names
    texte = np.array([formater_valeur(v, x) for v, x in zip(vars_, expl_i.data)], dtype=object)
    return shap.Explanation(values=expl_i.values, base_values=expl_i.base_values, data=texte,
                            feature_names=vars_)


def importance_shap(expl) -> pd.Series:
    """Importance globale = moyenne des |valeurs SHAP| (graphique en barres du cours)."""
    return pd.Series(np.abs(expl.values).mean(0), index=expl.feature_names).sort_values(ascending=False)


def contributions(expl, i: int) -> pd.DataFrame:
    """Contributions SHAP d'un flux, triées par importance absolue."""
    vars_ = expl.feature_names
    df = pd.DataFrame({"variable": vars_, "valeur": expl.data[i],
                       "valeur_lisible": [formater_valeur(v, x) for v, x in zip(vars_, expl.data[i])],
                       "contribution": expl.values[i]})
    return df.reindex(df["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------------------------
def explication_lime(pipeline, X_reference: pd.DataFrame, flux: pd.DataFrame, n_variables: int = 10,
                     n_echantillons: int = 5000, largeur_noyau: float | None = 1.0,
                     discretiseur: str = "decile", random_state: int = C.RANDOM_STATE) -> dict:
    """LIME : modèle linéaire pondéré ajusté autour du flux (étapes 1 à 4 du cours).

    1. perturbation : LIME tire des flux voisins à partir de la distribution du jeu de référence ;
    2. le pipeline prédit la probabilité d'attaque de chaque voisin ;
    3. chaque voisin est pondéré par sa proximité avec le flux (noyau exponentiel π_x) ;
    4. une régression linéaire pondérée g donne les poids (en points de probabilité).

    Réglages : avec le noyau par défaut de LIME (0,75·√d ≈ 4,4 pour 34 variables) et des quartiles,
    le voisinage couvre presque tout l'espace et l'explication devient la même pour tous les flux
    (R² ≈ 0,13). Un noyau étroit (1) et des déciles rendent l'explication vraiment locale (R² ≈ 0,7-0,8).
    `largeur_noyau=None` reprend le réglage par défaut de LIME.
    """
    from lime.lime_tabular import LimeTabularExplainer

    variables = list(X_reference.columns)
    cat = [j for j, v in enumerate(variables) if not pd.api.types.is_numeric_dtype(X_reference[v])]
    ref = en_numerique(X_reference).fillna(en_numerique(X_reference).median())
    x = en_numerique(flux[variables]).fillna(ref.median()).to_numpy()[0]

    def predire(tableau: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(tableau, columns=variables)
        for j in cat:
            df[variables[j]] = df[variables[j]].round().astype(int).astype(str)
        return pipeline.predict_proba(df)

    explainer = LimeTabularExplainer(
        ref.to_numpy(), feature_names=variables, class_names=["normal", "attaque"],
        categorical_features=cat, discretize_continuous=True, discretizer=discretiseur,
        kernel_width=largeur_noyau, mode="classification", random_state=random_state)
    exp = explainer.explain_instance(x, predire, num_features=n_variables, num_samples=n_echantillons,
                                     labels=(1,))
    regles = dict(exp.as_list(label=1))
    poids = pd.DataFrame([{"variable": variables[j], "regle": r, "poids": w}
                          for (j, w), r in zip(exp.as_map()[1], regles)])
    return {"poids": poids, "intercept": float(exp.intercept[1]), "r2": float(exp.score),
            "prediction_locale": float(np.ravel(exp.local_pred)[0])}


def comparer_shap_lime(contrib_shap: pd.DataFrame, poids_lime: pd.DataFrame, k: int = 5) -> dict:
    """Accord entre SHAP et LIME : variables communes dans le top-k et accord sur le sens."""
    s = contrib_shap.set_index("variable")["contribution"]
    lime = poids_lime.set_index("variable")["poids"]
    top_s, top_l = list(s.abs().sort_values(ascending=False).index[:k]), list(lime.index[:k])
    toutes = list(dict.fromkeys(top_s + top_l))
    rang_s = {v: r + 1 for r, v in enumerate(s.abs().sort_values(ascending=False).index)}
    rang_l = {v: r + 1 for r, v in enumerate(lime.index)}
    tableau = pd.DataFrame({
        "rang SHAP": [rang_s.get(v) for v in toutes],
        "contribution SHAP": [s.get(v) for v in toutes],
        "rang LIME": [rang_l.get(v) for v in toutes],
        "poids LIME": [lime.get(v) for v in toutes],
    }, index=pd.Index(toutes, name="variable"))
    communes = [v for v in top_s if v in top_l]
    meme_sens = [v for v in communes if np.sign(s[v]) == np.sign(lime[v])]
    return {"tableau": tableau, "communes": communes, "meme_sens": meme_sens, "k": k}


# --------------------------------------------------------------------------------------------
# Phrase de synthèse
# --------------------------------------------------------------------------------------------
def phrase_synthese(contrib: pd.DataFrame, probabilite: float, seuil: float,
                    medianes_normales: pd.Series | dict | None = None, k: int = 2) -> str:
    """Phrase automatique à partir des contributions SHAP d'un flux.

    On garde les k variables qui poussent le plus dans le sens de la décision, et on les compare
    à la médiane du trafic normal quand elle est fournie.
    """
    attaque = probabilite >= seuil
    sens = contrib[contrib["contribution"] > 0] if attaque else contrib[contrib["contribution"] < 0]
    raisons = []
    for _, r in sens.head(k).iterrows():
        desc = C.DESCRIPTIONS.get(r["variable"], r["variable"])
        desc = desc.replace(" (µs)", "")
        texte = f"{desc} = {r['valeur_lisible']}"
        if medianes_normales is not None and r["variable"] in medianes_normales:
            texte += f" (médiane du trafic normal : {formater_valeur(r['variable'], medianes_normales[r['variable']])})"
        raisons.append(texte)
    proba = "moins de 0,1 %" if probabilite < 0.001 else f"{probabilite * 100:.1f} %".replace(".", ",")
    s = f"{seuil:.2f}".replace(".", ",")
    if attaque:
        debut = f"Le risque d'attaque est élevé ({proba}, seuil {s})"
        lien = "principalement à cause de"
    else:
        debut = f"Le flux est jugé normal ({proba} de risque, seuil {s})"
        lien = "surtout grâce à"
    if not raisons:
        return f"{debut}."
    return f"{debut} {lien} : " + " ; ".join(raisons) + "."


# --------------------------------------------------------------------------------------------
# Graphiques (matplotlib, comme dans le cours : shap.plots.*)
# --------------------------------------------------------------------------------------------
def _figure(trace, largeur=8, hauteur=5):
    import matplotlib.pyplot as plt

    plt.close("all")
    plt.figure(figsize=(largeur, hauteur))
    trace()
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def fig_barres(expl, max_display: int = 12):
    import shap
    return _figure(lambda: shap.plots.bar(expl, max_display=max_display, show=False))


def fig_beeswarm(expl, max_display: int = 12):
    import shap
    return _figure(lambda: shap.plots.beeswarm(_couleurs_en_rangs(expl), max_display=max_display,
                                               show=False), 8, 6)


def fig_dependance(expl, variable: str, couleur: str | None = None, echelle_log: bool = True):
    import matplotlib.pyplot as plt
    import shap

    def trace():
        ax = plt.gca()
        col = expl[:, couleur] if couleur else None
        shap.plots.scatter(expl[:, variable], color=col, hist=False, show=False, ax=ax)
        if echelle_log:
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xlim(left=-0.5)
        ax.axhline(0, color="grey", lw=0.8)
        unite = unite_de(variable)
        ax.set_xlabel(f"{variable} ({unite}, échelle log)" if unite else variable)
        ax.set_ylabel(f"valeur SHAP de {variable}")
    return _figure(trace, 8, 4.5)


def fig_waterfall(expl, i: int, max_display: int = 10):
    import shap
    return _figure(lambda: shap.plots.waterfall(_lisible(expl[i]), max_display=max_display, show=False), 8, 5)


def fig_decision(expl, indices, seuil: float | None = None, surligner: int | None = None,
                 unite: str = UNITE_LOG_ODDS, max_display: int = 15):
    """Graphique de décision : trajet cumulé de plusieurs flux, du φ0 à la prédiction."""
    import matplotlib.pyplot as plt
    import shap

    indices = list(indices)

    def trace():
        base = float(np.mean(expl.base_values[indices]))
        hl = [indices.index(surligner)] if surligner in indices else None
        shap.decision_plot(base, expl.values[indices], feature_names=list(expl.feature_names),
                           feature_display_range=slice(None, -max_display - 1, -1),
                           highlight=hl, show=False, auto_size_plot=False)
        if seuil is not None:
            x = np.log(seuil / (1 - seuil)) if unite == UNITE_LOG_ODDS else seuil
            plt.axvline(x, color="black", lw=1.2)
    return _figure(trace, 8, 6)
