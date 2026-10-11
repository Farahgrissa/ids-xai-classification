
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import commun
import streamlit as st

from ids_xai import config as C

commun.configurer_page("Détection d'intrusion explicable")

st.title("Détection d'intrusion réseau explicable")
st.markdown(
    "**Cas métier :** un SOC doit repérer les flux réseau malveillants (force brute, déni de service, "
    "botnet) et **comprendre pourquoi** une alerte est levée.  \n"
    "**Données :** CSE-CIC-IDS2018, trafic capturé sur une infrastructure **AWS** "
    "(Registry of Open Data), au format CICFlowMeter.  \n"
    "**Cible :** flux **normal (0)** ou **attaque (1)**."
)

col1, col2, col3 = st.columns(3)
col1.metric("Flux après nettoyage", "37 395", help="Échantillon de 1,5 % de 3 journées, doublons retirés")
col2.metric("Taux d'attaque", "27,3 %")
col3.metric("PR-AUC test (XGBoost)", "0,994")

st.subheader("Parcours")
st.markdown(
    "| Module | Contenu |\n|---|---|\n"
    "| 1 – Données | Téléchargement automatique (AWS S3) ou import CSV, statistiques, visualisations |\n"
    "| 2 – Préparation | Cible et variables, valeurs manquantes, encodage, normalisation, split |\n"
    "| 3 – Modélisation | 5 modèles, hyperparamètres, validation croisée, tableau comparatif |\n"
    "| 4 – Évaluation | Métriques, ROC et précision-rappel superposées, confusion, seuil en direct |\n"
    "| 5 – Explainable AI | SHAP, LIME, phrase d'explication *(en cours)* |\n"
    "| 6 – API | Appel du service FastAPI documenté |"
)

st.subheader("Disponibilité des ressources")
st.write(("✅" if C.FICHIER_DONNEES.exists() else "⚠️")
         + f" Échantillon nettoyé : `{C.FICHIER_DONNEES.relative_to(C.RACINE)}`")
st.write(("✅" if C.FICHIER_ARTEFACTS.exists() else "⚠️")
         + f" Modèles pré-entraînés : `{C.FICHIER_ARTEFACTS.relative_to(C.RACINE)}`")
if not C.FICHIER_ARTEFACTS.exists():
    st.info("Générez les modèles avec `python scripts/entrainer.py` (voir le README), "
            "ou entraînez-les directement dans le module 3.")

commun.lien("pages/1_Donnees.py", "Commencer par le module 1 – Données")
