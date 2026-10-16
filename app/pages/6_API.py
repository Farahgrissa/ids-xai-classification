import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import commun  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from ids_xai import config as C  # noqa: E402

commun.configurer_page("Module 6 – API")
st.title("Module 6 – API d'interprétation (FastAPI)")


def url_par_defaut() -> str:

    try:
        from streamlit.runtime.secrets import secrets_singleton
        if secrets_singleton.load_if_toml_exists() and "API_URL" in st.secrets:
            return st.secrets["API_URL"]
    except Exception:
        pass
    return os.getenv("API_URL", "http://localhost:8000")


url = st.text_input("Adresse de l'API", url_par_defaut()).rstrip("/")
st.caption(f"Documentation interactive : [{url}/docs]({url}/docs) · En production, l'adresse est "
           "définie dans les secrets Streamlit (`API_URL`).")


def appeler(methode: str, route: str, **kwargs):
    try:
        r = requests.request(methode, f"{url}{route}", timeout=30, **kwargs)
        return r.status_code, r.json()
    except requests.RequestException as e:
        return None, str(e)


c1, c2 = st.columns(2)
if c1.button("GET /health"):
    code, rep = appeler("GET", "/health")
    (c1.success if code == 200 else c1.error)(f"Code : {code}")
    c1.json(rep)
if c2.button("GET /model-info"):
    code, rep = appeler("GET", "/model-info")
    c2.write(f"Code : {code}")
    c2.json(rep)

st.subheader("Prédire un flux via l'API")
artefacts = commun.charger_artefacts(str(C.FICHIER_ARTEFACTS)) if C.FICHIER_ARTEFACTS.exists() else None
if artefacts is None:
    st.info("Les flux de démonstration viennent du jeu de test pré-entraîné : lancez `scripts/entrainer.py`.")
    st.stop()

X, y, t = artefacts["X_test"], artefacts["y_test"].values, artefacts["t_test"].values
c1, c2, c3 = st.columns(3)
type_choisi = c1.selectbox("Type de flux", sorted(set(t)))
indices = np.where(t == type_choisi)[0]
pos = c2.number_input(f"Flux n° (0 à {len(indices) - 1})", 0, len(indices) - 1, 0)
seuil = c3.slider("Seuil", 0.01, 0.99, C.SEUIL_RECOMMANDE, 0.01)
i = int(indices[pos])
flux = json.loads(X.iloc[[i]].to_json(orient="records"))[0]  # types natifs, NaN → null
with st.expander("Flux envoyé (JSON)"):
    st.json(flux)

if st.button("POST /predict", type="primary"):
    code, rep = appeler("POST", "/predict", json={"flux": [flux], "seuil": seuil})
    retenu = artefacts["modele_retenu"]
    p_local = float(artefacts["modeles"][retenu].predict_proba(X.iloc[[i]])[:, 1][0])
    if code == 200:
        res = rep["resultats"][0]
        m = st.columns(3)
        m[0].metric("Probabilité d'attaque (API)", f"{res['probabilite_attaque']:.4f}")
        m[1].metric("Décision", res["decision"], res["niveau"], delta_color="off")
        m[2].metric("Vérité terrain", "attaque" if y[i] else "normal", type_choisi, delta_color="off")
        st.json(rep)
    else:
        st.error(f"API injoignable ou erreur ({code}) : {rep}")
        st.info(f"Calcul local de secours ({retenu}) : probabilité d'attaque = {p_local:.4f} → "
                f"{'attaque' if p_local >= seuil else 'normal'}")

st.subheader("Expliquer un flux via l'API")
st.caption("`POST /explain` renvoie les contributions SHAP des variables (en log-odds : φ0 + Σ contributions = "
           "logit de la probabilité) et une phrase de synthèse.")
if st.button("POST /explain"):
    code, rep = appeler("POST", "/explain", json={"flux": [flux], "seuil": seuil, "top": 8})
    if code == 200:
        e = rep["explications"][0]
        (st.error if e["decision"] == "attaque" else st.success)(e["phrase"])
        contrib = pd.DataFrame(e["contributions"]).iloc[::-1]
        fig = px.bar(contrib, x="contribution", y="variable", orientation="h",
                     color=contrib["contribution"] > 0, color_discrete_map={True: "#E45756", False: "#4C78A8"},
                     hover_data=["valeur_lisible"],
                     title=f"Contributions SHAP renvoyées par l'API (φ0 = {e['valeur_de_base']:.3f})")
        fig.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Réponse JSON"):
            st.json(rep)
    elif code == 501:
        st.warning("L'API déployée ne fournit pas encore /explain : redéployez-la avec la dernière version.")
    else:
        st.error(f"API injoignable ou erreur ({code}) : {rep}")
        st.info("L'explication locale complète (SHAP, LIME) est disponible au module 5.")
