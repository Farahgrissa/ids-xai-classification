import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import commun  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ids_xai import config as C  # noqa: E402
from ids_xai import exploration as E  # noqa: E402
from ids_xai.donnees import colonnes_variables, nettoyer, preparer_csv_utilisateur, telecharger  # noqa: E402

commun.configurer_page("Module 1 – Données")
st.title("Module 1 – Données")


SOURCES = ["Échantillon nettoyé (rapide)", "Téléchargement automatique depuis AWS S3", "Importer un CSV"]
source = st.radio("Source des données", SOURCES, horizontal=True)

if source == SOURCES[0]:
    if C.FICHIER_DONNEES.exists():
        st.caption("Échantillon de 1,5 % des journées 14/02, 16/02 et 02/03, nettoyé "
                   "(mêmes étapes que le notebook, cellules 3 à 4f).")
        if st.button("Charger l'échantillon", type="primary"):
            commun.definir_donnees(commun.charger_echantillon(str(C.FICHIER_DONNEES)), source)
    else:
        st.error(f"Fichier introuvable : `{C.FICHIER_DONNEES.relative_to(C.RACINE)}`. "
                 "Lancez `python scripts/preparer_donnees.py` ou utilisez une autre source.")

elif source == SOURCES[1]:
    st.caption("Lecture en streaming du bucket public `cse-cic-ids2018` (aucun compte AWS requis), "
               "par morceaux de 200 000 lignes, avec échantillonnage aléatoire reproductible.")
    jours = st.multiselect("Journées", list(C.JOURS), default=list(C.JOURS),
                           format_func=lambda f: f"{f.split('_')[0]} – {C.JOURS[f]}")
    frac = st.select_slider("Fraction gardée de chaque journée", [0.002, 0.005, 0.01, 0.015], value=0.005)
    st.warning("Chaque journée pèse ≈ 350 Mo : comptez plusieurs minutes.")
    if st.button("Télécharger et nettoyer", type="primary", disabled=not jours):
        statut = st.status("Téléchargement en cours…", expanded=True)
        brut = telecharger(jours, frac=frac,
                           progression=lambda f, i: statut.write(f"{f.split('_')[0]} : morceau {i} lu"))
        statut.write("Nettoyage…")
        df, rapport = nettoyer(brut)
        statut.update(label="Terminé", state="complete")
        commun.definir_donnees(df, source, rapport)

else:
    fichier = st.file_uploader("Fichier CSV", type="csv")
    if fichier:
        brut = pd.read_csv(fichier, low_memory=False)
        st.dataframe(brut.head(), use_container_width=True)
        format_cic = C.COLONNE_LABEL in brut.columns and "Flow Duration" in brut.columns
        if format_cic and st.checkbox("Format CSE-CIC-IDS2018 détecté : appliquer le nettoyage du projet",
                                      value=True):
            if st.button("Nettoyer et charger", type="primary"):
                df, rapport = nettoyer(brut)
                commun.definir_donnees(df, "CSV (format CIC)", rapport)
        else:
            cible = st.selectbox("Variable cible (obligatoire)", brut.columns)
            valeurs = brut[cible].dropna().astype(str).unique().tolist()
            positive = st.selectbox("Classe d'intérêt (positive)", valeurs)
            if len(valeurs) > 2:
                st.info(f"{len(valeurs)} classes : ramenées à « {positive} » contre « autres ».")
            if st.button("Charger", type="primary"):
                df, rapport = preparer_csv_utilisateur(brut, cible, positive)
                commun.definir_donnees(df, f"CSV ({fichier.name})", rapport)

commun.exiger("df", "Chargez des données pour continuer.")
df = st.session_state["df"]
num, cat = colonnes_variables(df)
id_donnees = st.session_state["id_donnees"]

st.success(f"**{len(df):,} flux** · {len(num)} variables numériques · "
           f"{len(cat)} catégorielle(s) · source : {st.session_state['source']}")
if st.session_state.get("rapport"):
    with st.expander("Rapport de nettoyage"):
        st.json(st.session_state["rapport"])

onglets = st.tabs(["Aperçu", "Statistiques", "Valeurs manquantes", "Classes", "Distributions",
                   "Pouvoir séparateur", "Taux d'attaque par variable", "Corrélations"])

with onglets[0]:
    st.dataframe(df.head(50), use_container_width=True)
    st.write("Types :", df.dtypes.astype(str).value_counts().to_dict())

with onglets[1]:
    st.markdown("Triées par **asymétrie** : les queues lourdes justifient la transformation log (module 2).")
    st.dataframe(E.statistiques(df, num).round(2), use_container_width=True)
    if df[C.COLONNE_TYPE].nunique() > 2:
        st.markdown("**Médianes par type de flux**")
        defaut = [v for v in ["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts",
                              "Flow Pkts/s", "Flow IAT Mean", "Init Fwd Win Byts"] if v in num]
        choix = st.multiselect("Variables", num, default=defaut or num[:6])
        if choix:
            st.dataframe(E.medianes_par_type(df, choix).round(1), use_container_width=True)

with onglets[2]:
    manq = E.valeurs_manquantes(df)
    if manq.empty:
        st.success("Aucune valeur manquante.")
    else:
        st.dataframe(manq, use_container_width=True)
        st.caption("Pour CIC-IDS2018 : flux de durée 0 (débit = quantité ÷ 0). Imputés au module 2.")

with onglets[3]:
    c1, c2 = st.columns(2)
    c1.plotly_chart(E.fig_proportion_classes(df), use_container_width=True)
    c2.plotly_chart(E.fig_types(df), use_container_width=True)

with onglets[4]:
    var = st.selectbox("Variable", num + cat, format_func=commun.libelle, key="dist_var")
    log = st.checkbox("Échelle log (recommandé pour le trafic réseau)", value=var in num)
    st.plotly_chart(E.fig_distribution(df, var, log and var in num), use_container_width=True)

with onglets[5]:
    pouvoir = commun.auc_univariee(df, id_donnees, tuple(num))
    st.markdown("AUC de chaque variable utilisée seule comme score (0,5 = hasard). "
                "Ne détecte que les séparations « dans un seul sens ».")
    c1, c2 = st.columns([1, 1])
    c1.dataframe(pouvoir, use_container_width=True, height=500)
    c2.plotly_chart(E.fig_auc(pouvoir), use_container_width=True)

with onglets[6]:
    defaut = "Fwd Seg Size Min" if "Fwd Seg Size Min" in df.columns else (num + cat)[0]
    var = st.selectbox("Variable", num + cat, index=(num + cat).index(defaut),
                       format_func=commun.libelle, key="taux_var")
    st.plotly_chart(E.fig_taux(df, var), use_container_width=True)
    st.dataframe(E.taux_attaque(df, var), use_container_width=True)

with onglets[7]:
    pouvoir = commun.auc_univariee(df, id_donnees, tuple(num))
    n = st.slider("Nombre de variables (les plus séparatrices)", 5, 30, 15)
    st.plotly_chart(E.fig_correlations(df, pouvoir["variable"].head(n).tolist()), use_container_width=True)
    seuil = st.slider("Seuil de redondance", 0.80, 1.0, C.SEUIL_CORR, 0.01)
    paires = E.paires_redondantes(df, num, seuil)
    st.write(f"**{len(paires)} paires** avec |corrélation| > {seuil:.2f}")
    st.dataframe(paires.rename("corrélation").round(3).reset_index()
                 .rename(columns={"level_0": "variable 1", "level_1": "variable 2"}),
                 use_container_width=True)

commun.lien("pages/2_Preparation.py", "Étape suivante : module 2 – Préparation")
