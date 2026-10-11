import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import commun  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from ids_xai import config as C  # noqa: E402
from ids_xai.modeles import CATALOGUE, NOMS_MODELES, entrainer_et_comparer  # noqa: E402

commun.configurer_page("Module 3 – Modélisation")
st.title("Module 3 – Modélisation et comparaison")

MODES = ["Utiliser les modèles pré-entraînés (instantané)", "Entraîner dans l'application"]
artefacts = commun.charger_artefacts(str(C.FICHIER_ARTEFACTS)) if C.FICHIER_ARTEFACTS.exists() else None
mode = st.radio("Mode", MODES if artefacts else MODES[1:], horizontal=True)

if mode == MODES[0]:
    st.caption(f"Modèles entraînés par `scripts/entrainer.py` le {artefacts['date']} "
               f"(scikit-learn {artefacts['versions']['scikit-learn']}, "
               f"xgboost {artefacts['versions']['xgboost']}), sur l'échantillon nettoyé, "
               "sans empreinte système, hyperparamètres optimisés à l'étape 12 du notebook.")
    if st.button("Charger les modèles", type="primary"):
        commun.definir_modeles(artefacts["modeles"], artefacts["comparatif"],
                               {"X": artefacts["X_test"], "y": artefacts["y_test"], "t": artefacts["t_test"]},
                               "pré-entraînés")
else:
    commun.exiger("jeu", "Préparez d'abord les données (module 2).", "pages/2_Preparation.py")
    jeu = st.session_state["jeu"]
    p = jeu.params

    choix = st.multiselect("Modèles à entraîner", NOMS_MODELES,
                           default=[m for m in NOMS_MODELES if m != "KNN"])
    if "KNN" in choix:
        st.warning("KNN est lent : il compare chaque flux à tout le jeu d'entraînement.")

    rechercher = st.checkbox("Optimiser automatiquement les hyperparamètres "
                             "(GridSearch / RandomizedSearch, critère PR-AUC)")
    n_iter = st.slider("Combinaisons testées (RandomizedSearch)", 5, 30, 10, 5, disabled=not rechercher)

    hyper = {}
    if not rechercher:
        st.markdown("**Hyperparamètres** (valeurs par défaut = meilleurs réglages trouvés à l'étape 12)")
        for nom in choix:
            spec = CATALOGUE[nom]
            with st.expander(nom):
                cols = st.columns(len(spec["espace"]))
                hyper[nom] = {}
                for col, (param, valeurs) in zip(cols, spec["espace"].items()):
                    defaut = spec["optimises"].get(param, valeurs[0])
                    hyper[nom][param] = col.selectbox(param, valeurs, index=valeurs.index(defaut),
                                                      key=f"{nom}_{param}")

    st.caption(f"Validation croisée stratifiée à {C.N_FOLDS} folds sur le train "
               f"({len(jeu.X_train):,} flux), puis entraînement final sur tout le train.")
    if st.button("Entraîner et comparer", type="primary", disabled=not choix):
        barre = st.progress(0.0, "Démarrage…")
        etat = {"i": 0}

        def progression(nom):
            barre.progress(etat["i"] / len(choix), f"{nom}…")
            etat["i"] += 1

        comparatif, modeles, _ = entrainer_et_comparer(
            choix, jeu.X_train, jeu.y_train, jeu.t_train, hyperparametres=hyper,
            log=p.get("log", True), normaliser=p.get("normaliser", True),
            imputation=p.get("imputation", "median"), rechercher=rechercher, n_iter=n_iter,
            progression=progression)
        barre.progress(1.0, "Terminé")
        commun.definir_modeles(modeles, comparatif, {"X": jeu.X_test, "y": jeu.y_test, "t": jeu.t_test},
                               "entraînés dans l'application")

commun.exiger("comparatif", "Chargez ou entraînez des modèles.")
comparatif = st.session_state["comparatif"]

st.subheader("Tableau comparatif (validation croisée)")
st.dataframe(comparatif.style.format("{:.4f}").highlight_max(
    subset=["AUC-ROC", "PR-AUC", "F1"], color="#d4edda"), use_container_width=True)
fig = px.bar(comparatif.reset_index(), x="Modèle", y=["AUC-ROC", "PR-AUC", "F1"], barmode="group",
             title="Comparaison des modèles")
fig.update_yaxes(range=[max(0.0, comparatif[["AUC-ROC", "PR-AUC", "F1"]].min().min() - 0.05), 1.0])
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    "- **Écart train-val** proche de 0 : le modèle généralise (élevé = surapprentissage).\n"
    "- **Prédiction (ms/1000 flux)** : critère décisif pour une API temps réel.\n"
    "- Modèle retenu dans le projet : **XGBoost** (meilleure PR-AUC, rapide, stable)."
)
commun.lien("pages/4_Evaluation.py", "Étape suivante : module 4 – Évaluation")
