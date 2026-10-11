import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import commun  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ids_xai import config as C  # noqa: E402
from ids_xai.preparation import IMPUTATIONS, construire_preprocesseur, preparer  # noqa: E402

commun.configurer_page("Module 2 – Préparation")
st.title("Module 2 – Préparation")
commun.exiger("df", "Chargez d'abord des données (module 1).", "pages/1_Donnees.py")
df = st.session_state["df"]

st.subheader("1. Variable cible")
st.markdown(f"Cible binaire **`{C.CIBLE}`** : 1 = classe d'intérêt (attaque), 0 = normal. "
            f"Taux de la classe positive : **{df[C.CIBLE].mean():.1%}**.  \n"
            f"Le détail (`{C.COLONNE_TYPE}`) sert uniquement à stratifier et à analyser les erreurs : "
            "il n'est jamais donné au modèle.")

st.subheader("2. Variables explicatives")
candidates = [c for c in df.columns if c not in C.EXCLUES]
avec_empreinte = st.checkbox(
    "Inclure les variables « empreinte système » (Init Fwd/Bwd Win Byts, Fwd Seg Size Min)", value=False,
    help="Elles identifient la machine (système d'exploitation) plutôt que le comportement : raccourci.")
defaut = [c for c in candidates if avec_empreinte or c not in C.EMPREINTE]
choisies = st.multiselect("Variables utilisées", candidates, default=defaut)
c1, c2 = st.columns(2)
retirer_redondantes = c1.checkbox("Retirer les variables redondantes", value=True)
seuil_corr = c2.slider("Seuil de corrélation (Spearman)", 0.80, 1.0, C.SEUIL_CORR, 0.01,
                       disabled=not retirer_redondantes)
st.caption("Dans chaque groupe de jumelles, on garde la plus interprétable "
           f"({', '.join(C.PRIORITAIRES)}), sinon la plus séparatrice (AUC calculée sur le train).")

st.subheader("3. Prétraitement")
c1, c2, c3 = st.columns(3)
imputation = c1.selectbox("Valeurs manquantes (numériques)", list(IMPUTATIONS),
                          format_func=IMPUTATIONS.get)
log = c2.checkbox("Transformation log signée", value=True, help="Compresse les queues lourdes")
normaliser = c3.checkbox("Standardisation", value=True, help="Indispensable pour KNN et la régression logistique")
st.caption("Variables catégorielles : modalité la plus fréquente pour les manquants, puis encodage one-hot.")


st.subheader("4. Séparation entraînement / test")
test_size = st.slider("Part du jeu de test", 0.10, 0.40, C.TEST_SIZE, 0.05)
st.caption("Hold-out stratifié sur le type d'attaque : chaque attaque garde la même proportion "
           "dans le train et dans le test.")

if st.button("Préparer les données", type="primary", disabled=not choisies):
    with st.spinner("Préparation…"):
        jeu = preparer(df, avec_empreinte=True, test_size=test_size,
                       seuil_corr=seuil_corr if retirer_redondantes else None,
                       a_exclure=[c for c in candidates if c not in choisies])
        jeu.params.update(avec_empreinte=avec_empreinte, imputation=imputation, log=log,
                          normaliser=normaliser)
    st.session_state["jeu"] = jeu
    for cle in ("modeles", "comparatif", "test", "probas", "origine_modeles"):
        st.session_state.pop(cle, None)

commun.exiger("jeu", "Réglez les options puis cliquez sur « Préparer les données ».")
jeu = st.session_state["jeu"]

st.subheader("Résultat")
c1, c2, c3 = st.columns(3)
c1.metric("Variables finales", len(jeu.variables), f"-{len(jeu.redondantes)} redondantes")
c2.metric("Train", f"{len(jeu.X_train):,}", f"{jeu.y_train.mean():.1%} d'attaques", delta_color="off")
c3.metric("Test", f"{len(jeu.X_test):,}", f"{jeu.y_test.mean():.1%} d'attaques", delta_color="off")

with st.expander("Variables retirées car redondantes"):
    st.write(jeu.redondantes or "Aucune")
with st.expander("Répartition des types dans le test"):
    st.dataframe(pd.crosstab(jeu.t_test, jeu.y_test), use_container_width=True)

st.markdown("**Vérification du prétraitement** (paramètres appris sur le train uniquement)")
p = jeu.params
prep = construire_preprocesseur(jeu.X_train, p.get("log", True), p.get("normaliser", True),
                                p.get("imputation", "median"))
Xtr = prep.fit_transform(jeu.X_train)
Xte = prep.transform(jeu.X_test)
noms = prep.get_feature_names_out()
c1, c2 = st.columns(2)
c1.write(f"Train transformé : {Xtr.shape} · NaN restants : {int(np.isnan(Xtr).sum())}")
c1.dataframe(pd.DataFrame(Xtr, columns=noms).describe().T[["mean", "std", "min", "max"]].round(2),
             use_container_width=True)
c2.write(f"Test transformé : {Xte.shape} · NaN restants : {int(np.isnan(Xte).sum())}")
c2.dataframe(pd.DataFrame(Xte, columns=noms).describe().T[["mean", "std"]].round(2),
             use_container_width=True)
st.caption("Train : moyenne 0 et écart-type 1 exactement. Test : proches mais pas exacts → pas de fuite.")

commun.lien("pages/3_Modelisation.py", "Étape suivante : module 3 – Modélisation")
