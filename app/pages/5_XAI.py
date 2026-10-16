import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import commun  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import shap  # noqa: E402
import streamlit as st  # noqa: E402

from ids_xai import config as C  # noqa: E402
from ids_xai import explications as E  # noqa: E402

commun.configurer_page("Module 5 – Explainable AI")
st.title("Module 5 – Explainable AI")
commun.exiger("modeles", "Chargez ou entraînez d'abord des modèles (module 3).", "pages/3_Modelisation.py")

s = st.session_state
X, y, t = s["test"]["X"], np.asarray(s["test"]["y"]), np.asarray(s["test"]["t"])
variables = list(X.columns)

# Jeu de référence (fond de LIME, médianes du trafic normal) : le train s'il correspond, sinon le test
if "jeu" in s and list(s["jeu"].X_train.columns) == variables:
    X_ref, y_ref = s["jeu"].X_train, np.asarray(s["jeu"].y_train)
else:
    X_ref, y_ref = X, y
medianes_normal = E.en_numerique(X_ref[y_ref == 0]).median()

explicables = [n for n, p in s["modeles"].items() if E.type_explainer(p)]
if not explicables:
    st.error("Aucun modèle compatible avec un explainer SHAP rapide (XGBoost, forêts, arbre, régression).")
    st.stop()
c1, c2 = st.columns([2, 3])
nom = c1.selectbox("Modèle à expliquer", explicables,
                   index=explicables.index("XGBoost") if "XGBoost" in explicables else 0)
pipe = s["modeles"][nom]
c2.caption("XGBoost et la régression logistique sont expliqués en **log-odds** (additivité exacte sur le "
           "logit), les arbres et forêts scikit-learn en **probabilité**. KNN n'a pas d'explainer rapide.")

cle = f"{s.get('id_modeles', '')}-{nom}"
N_GLOBAL = 3000


@st.cache_data(show_spinner="Calcul des valeurs SHAP…", max_entries=6)
def shap_global(_pipe, _X, _X_ref, cle: str, n: int):
    ech = _X.sample(min(n, len(_X)), random_state=C.RANDOM_STATE)
    expl, unite = E.valeurs_shap(_pipe, ech, X_fond=_X_ref)
    return expl.values, expl.base_values, expl.data, unite, ech.index


@st.cache_data(show_spinner="Importance par permutation…", max_entries=6)
def permutation(_pipe, _X, _y, cle: str):
    return E.importance_permutation(_pipe, _X, _y, n_repetitions=3, n_max=2000)


def montrer(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


valeurs, bases, donnees, unite, index_ech = shap_global(pipe, X, X_ref, cle, N_GLOBAL)
expl = shap.Explanation(values=valeurs, base_values=bases, data=donnees, feature_names=variables)
imp_shap = E.importance_shap(expl)

onglet_global, onglet_local = st.tabs(["Explications globales", "Expliquer un flux"])

# ------------------------------------------------------------------------------------------------
with onglet_global:
    st.subheader("1. Importance propre au modèle et importance par permutation")
    g1, g2 = st.columns(2)
    try:
        nat = E.importance_native(pipe, variables).head(12)
        mesure = "|coefficient|" if hasattr(pipe.named_steps["modele"], "coef_") else "gain (impureté)"
        fig = px.bar(nat[::-1], orientation="h", title=f"Importance propre au modèle ({mesure}, somme = 1)",
                     labels={"value": "part de l'importance", "index": ""})
        fig.update_layout(showlegend=False, height=420)
        g1.plotly_chart(fig, use_container_width=True)
    except ValueError as err:
        g1.info(str(err))
    perm = permutation(pipe, X, y, cle).head(12)
    fig = px.bar(perm["baisse_moyenne"][::-1], orientation="h", error_x=perm["ecart_type"][::-1],
                 title="Importance par permutation (baisse de PR-AUC)",
                 labels={"value": "baisse de PR-AUC", "index": ""})
    fig.update_layout(showlegend=False, height=420)
    g2.plotly_chart(fig, use_container_width=True)
    st.caption("Le gain d'un modèle d'arbres mesure les coupures faites à l'entraînement : il peut concentrer "
               "l'importance sur une seule variable. La permutation est comparable d'un modèle à l'autre, "
               "mais sous-estime les variables corrélées (le modèle se rabat sur leurs jumelles).")

    st.subheader("2. SHAP : importance globale et sens de l'effet")
    st.caption(f"Calculé sur {len(index_ech):,} flux du jeu de test · unité : {unite} · "
               f"φ0 = {bases[0]:.3f}")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Barres** – moyenne des |valeurs SHAP|")
        montrer(E.fig_barres(expl))
    with b2:
        st.markdown("**Beeswarm** – un point par flux, couleur = rang de la valeur")
        montrer(E.fig_beeswarm(expl))
    st.caption("À droite de 0 : la variable pousse vers « attaque ». La couleur est le rang de la valeur "
               "(bleu : parmi les plus faibles, rouge : parmi les plus fortes), car les variables réseau ont "
               "des queues très lourdes.")

    st.subheader("3. SHAP : graphique de dépendance")
    d1, d2, d3 = st.columns([2, 2, 1])
    var = d1.selectbox("Variable", list(imp_shap.index), index=0, format_func=commun.libelle)
    autres = [v for v in imp_shap.index if v != var]
    couleur = d2.selectbox("Couleur (interaction)", autres, index=0, format_func=commun.libelle)
    log = d3.checkbox("Échelle log", value=E.unite_de(var) != "")
    montrer(E.fig_dependance(expl, var, couleur, echelle_log=log))
    st.caption("Chaque point est un flux : valeur de la variable en abscisse, contribution SHAP en ordonnée. "
               "Une courbe en cloche ou en paliers est un effet non linéaire, qu'une régression logistique "
               "ne peut pas représenter.")

# ------------------------------------------------------------------------------------------------
with onglet_local:
    st.subheader("1. Choisir un flux")
    source = st.radio("Source", ["Flux du jeu de test", "Flux fictif (modifier un flux)"], horizontal=True)
    p_test = pipe.predict_proba(X)[:, 1]
    f1, f2, f3, f4 = st.columns([2, 2, 1, 2])
    seuil = f4.slider("Seuil de décision", 0.01, 0.99, C.SEUIL_RECOMMANDE, 0.01)
    filtres = {
        "Tous": np.ones(len(y), bool),
        "Alertes justes": (y == 1) & (p_test >= seuil),
        "Fausses alertes": (y == 0) & (p_test >= seuil),
        "Attaques manquées": (y == 1) & (p_test < seuil),
        "Normaux bien classés": (y == 0) & (p_test < seuil),
    }
    filtre = f1.selectbox("Catégorie", list(filtres), index=1)
    types = ["Tous"] + sorted(set(t[filtres[filtre]]))
    type_choisi = f2.selectbox("Type de flux", types)
    masque = filtres[filtre] & ((t == type_choisi) if type_choisi != "Tous" else True)
    candidats = np.where(masque)[0]
    if len(candidats) == 0:
        st.info("Aucun flux dans cette catégorie pour ce seuil.")
        st.stop()
    pos = f3.number_input(f"N° (0–{len(candidats) - 1})", 0, len(candidats) - 1, 0)
    i = int(candidats[pos])
    flux = X.iloc[[i]].copy()
    verite = f"{t[i]} ({'attaque' if y[i] else 'normal'})"

    if source.startswith("Flux fictif"):
        st.caption("Modifiez les valeurs (les variables sont triées par importance SHAP), puis observez "
                   "l'effet sur la probabilité et sur l'explication.")
        edition = pd.DataFrame({"variable": imp_shap.index,
                                "description": [C.DESCRIPTIONS.get(v, "") for v in imp_shap.index],
                                "valeur": [E.en_numerique(flux)[v].iloc[0] for v in imp_shap.index]})
        edition = st.data_editor(edition, disabled=["variable", "description"], hide_index=True,
                                 use_container_width=True, height=320, key=f"edit-{i}")
        for v, val in zip(edition["variable"], edition["valeur"]):
            flux[v] = str(int(val)) if v == "Protocol" and pd.notna(val) else val
        verite = "flux fictif"

    proba = float(pipe.predict_proba(flux)[:, 1][0])
    expl_flux, _ = E.valeurs_shap(pipe, flux, X_fond=X_ref)
    contrib = E.contributions(expl_flux, 0)

    st.subheader("2. Prédiction et phrase de synthèse")
    m = st.columns(3)
    m[0].metric("Probabilité d'attaque", f"{proba * 100:.1f} %".replace(".", ","))
    m[1].metric("Décision", "attaque" if proba >= seuil else "normal", f"seuil {seuil:.2f}", delta_color="off")
    m[2].metric("Vérité terrain", verite)
    phrase = E.phrase_synthese(contrib, proba, seuil, medianes_normal)
    (st.error if proba >= seuil else st.success)(phrase)

    st.subheader("3. SHAP : cascade et graphique de décision")
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Cascade (waterfall)** – de φ0 à la prédiction")
        montrer(E.fig_waterfall(expl_flux, 0))
    with w2:
        st.markdown("**Décision** – ce flux (trait épais) parmi 20 flux de probabilité proche")
        p_ech = p_test[X.index.get_indexer(index_ech)]
        proches = np.argsort(np.abs(p_ech - proba))
        voisins = [k for k in proches if index_ech[k] != X.index[i]][:20]
        tous = shap.Explanation(
            values=np.vstack([expl.values[voisins], expl_flux.values]),
            base_values=np.concatenate([expl.base_values[voisins], expl_flux.base_values]),
            data=np.vstack([expl.data[voisins], expl_flux.data]), feature_names=variables)
        montrer(E.fig_decision(tous, list(range(len(voisins) + 1)), seuil=seuil, surligner=len(voisins),
                               unite=unite))
    with st.expander("Contributions détaillées"):
        st.dataframe(contrib[["variable", "valeur_lisible", "contribution"]].style.format({"contribution": "{:+.3f}"}),
                     use_container_width=True, hide_index=True)

    st.subheader("4. LIME et comparaison avec SHAP")
    l1, l2 = st.columns([1, 3])
    noyau = l1.select_slider("Largeur du noyau", [0.5, 1.0, 2.0, "défaut"], value=1.0,
                             help="Un noyau étroit rend l'explication vraiment locale. Le réglage par défaut "
                                  "de LIME (≈ 4,4 ici) donne presque la même explication pour tous les flux.")
    with st.spinner("LIME : perturbation, prédiction, pondération, régression…"):
        lime = E.explication_lime(pipe, X_ref, flux,
                                  largeur_noyau=None if noyau == "défaut" else float(noyau),
                                  discretiseur="quartile" if noyau == "défaut" else "decile")
    l1.metric("R² du modèle local", f"{lime['r2']:.2f}")
    l1.metric("Prédiction de g", f"{lime['prediction_locale']:.2f}", f"XGBoost : {proba:.2f}", delta_color="off")
    poids = lime["poids"].iloc[::-1]
    fig = px.bar(poids, x="poids", y="regle", orientation="h", color=poids["poids"] > 0,
                 color_discrete_map={True: "#E45756", False: "#4C78A8"},
                 title="Poids LIME (points de probabilité ; > 0 : vers « attaque »)",
                 labels={"regle": "", "poids": "poids"})
    fig.update_layout(showlegend=False, height=400)
    l2.plotly_chart(fig, use_container_width=True)

    comp = E.comparer_shap_lime(contrib, lime["poids"])
    st.markdown(f"**Accord SHAP / LIME** : {len(comp['communes'])} variable(s) commune(s) dans le top "
                f"{comp['k']}, dont {len(comp['meme_sens'])} avec le même sens.")
    st.dataframe(comp["tableau"].style.format("{:+.3f}", subset=["contribution SHAP", "poids LIME"], na_rep="–")
                 .format("{:.0f}", subset=["rang SHAP", "rang LIME"], na_rep="–"), use_container_width=True)
    if lime["r2"] < 0.3:
        st.warning("R² faible : le modèle linéaire de LIME approxime mal XGBoost autour de ce flux. "
                   "Préférez l'explication SHAP, exacte et additive.")
    st.caption("SHAP (TreeExplainer) est exact, additif et déterministe ; LIME est aléatoire et dépend du "
               "voisinage choisi. Quand les deux méthodes citent les mêmes variables dans le même sens, "
               "l'explication est plus fiable.")
