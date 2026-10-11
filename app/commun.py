from __future__ import annotations

import sys
import uuid
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from ids_xai import artefacts, exploration  # noqa: E402
from ids_xai import config as C  # noqa: E402
from ids_xai.donnees import charger_donnees_propres  # noqa: E402


def configurer_page(titre: str, icone: str = "🛡️"):
    st.set_page_config(page_title=titre, page_icon=icone, layout="wide")
    barre_laterale()


@st.cache_data(show_spinner="Chargement de l'échantillon nettoyé…")
def charger_echantillon(chemin: str) -> pd.DataFrame:
    return charger_donnees_propres(chemin)


@st.cache_resource(show_spinner="Chargement des modèles pré-entraînés…")
def charger_artefacts(chemin: str):
    return artefacts.charger(Path(chemin))


@st.cache_data(show_spinner="Calcul du pouvoir séparateur des variables…")
def auc_univariee(_df: pd.DataFrame, id_donnees: str, num: tuple) -> pd.DataFrame:
    return exploration.auc_univariee(_df, list(num))


def definir_donnees(df: pd.DataFrame, source: str, rapport: dict | None = None):
    st.session_state.update(df=df, source=source, rapport=rapport or {},
                            id_donnees=uuid.uuid4().hex[:8])
    for cle in ("jeu", "modeles", "comparatif", "test", "probas", "origine_modeles"):
        st.session_state.pop(cle, None)


def definir_modeles(modeles: dict, comparatif: pd.DataFrame, test: dict, origine: str):
    """test = {"X": X_test, "y": y_test, "t": t_test}."""
    st.session_state.update(modeles=modeles, comparatif=comparatif, test=test,
                            origine_modeles=origine, id_modeles=uuid.uuid4().hex[:8])
    st.session_state.pop("probas", None)


def lien(page: str, label: str):

    try:
        st.page_link(page, label=label, icon="➡️")
    except Exception:
        pass


def exiger(cle: str, message: str, page: str | None = None):
    if cle not in st.session_state:
        st.warning(message)
        if page:
            lien(page, "Aller à cette étape")
        st.stop()


def barre_laterale():
    s = st.session_state
    with st.sidebar:
        st.markdown("### État du projet")
        st.write(("✅ " if "df" in s else "⬜ ") + "Données" + (f" ({len(s['df']):,} flux)" if "df" in s else ""))
        st.write(("✅ " if "jeu" in s else "⬜ ") + "Préparation")
        st.write(("✅ " if "modeles" in s else "⬜ ") + "Modèles"
                 + (f" ({s.get('origine_modeles')})" if "modeles" in s else ""))
        st.caption("Les étapes se suivent : chaque page utilise le résultat de la précédente.")


def libelle(var: str) -> str:
    return f"{var} – {C.DESCRIPTIONS[var]}" if var in C.DESCRIPTIONS else var
