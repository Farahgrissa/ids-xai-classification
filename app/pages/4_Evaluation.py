import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import commun  # noqa: E402
import streamlit as st  # noqa: E402

from ids_xai import config as C  # noqa: E402
from ids_xai import evaluation as V  # noqa: E402

commun.configurer_page("Module 4 – Évaluation")
st.title("Module 4 – Évaluation sur le jeu de test")
commun.exiger("modeles", "Chargez ou entraînez d'abord des modèles (module 3).", "pages/3_Modelisation.py")

s = st.session_state
test = s["test"]
X, y, t = test["X"], test["y"].values, test["t"].values
if "probas" not in s:
    with st.spinner("Calcul des probabilités sur le jeu de test…"):
        s["probas"] = V.probabilites(s["modeles"], X)
probas = s["probas"]
st.caption(f"{len(y):,} flux jamais vus pendant l'entraînement · {y.mean():.1%} d'attaques · "
           f"modèles {s['origine_modeles']}")

st.subheader("1. Métriques sur la classe d'intérêt (attaque)")
c1, c2 = st.columns([1, 3])
seuil_tab = c1.number_input("Seuil de décision", 0.01, 0.99, C.SEUIL_DECISION, 0.05)
gini_ks = c1.checkbox("Ajouter Gini et Kolmogorov-Smirnov", value=False,
                      help="Demandés par l'énoncé pour les cas de scoring de crédit.")
tableau = V.tableau_metriques(probas, y, seuil_tab, gini_ks=gini_ks,
                              reference_cv=s["comparatif"]["PR-AUC"])
c2.dataframe(tableau.style.format("{:.4f}", na_rep="–"), use_container_width=True)
st.caption("La PR-AUC est privilégiée (classes déséquilibrées) ; son niveau « hasard » vaut la proportion "
           "d'attaques. L'accuracy est trompeuse : le modèle naïf atteint ≈ 73 % sans détecter aucune attaque.")

st.subheader("2. Courbes ROC et précision-rappel superposées")
o1, o2, o3 = st.tabs(["ROC", "Zoom ROC (faibles fausses alertes)", "Précision-rappel"])
o1.plotly_chart(V.fig_roc(probas, y), use_container_width=True)
with o2:
    st.plotly_chart(V.fig_roc(probas, y, zoom_log=True), use_container_width=True)
    st.markdown("**Rappel atteignable pour un taux maximal de fausses alertes**")
    st.dataframe(V.points_fonctionnement(probas, y).style.format("{:.1%}"), use_container_width=True)
o3.plotly_chart(V.fig_pr(probas, y), use_container_width=True)

st.subheader("3. Seuil de décision et matrice de confusion")
noms = list(probas)
c1, c2, c3 = st.columns([2, 2, 3])
modele = c1.selectbox("Modèle", noms, index=noms.index("XGBoost") if "XGBoost" in noms else 0)
seuil = c2.slider("Seuil de décision", 0.01, 0.99, C.SEUIL_DECISION, 0.01)
taux_reel = c3.select_slider("Taux d'attaques du trafic réel (simulation)",
                             [0.273, 0.1, 0.05, 0.01, 0.001], value=0.01,
                             format_func=lambda v: f"{v:.1%}")
p = probas[modele]
st_ = V.stats_seuil(y, p, seuil)
prec_reelle = V.precision_a_prevalence(st_["rappel"], st_["taux_fa"], taux_reel)
proj = V.projection(st_["rappel"], st_["taux_fa"], taux_reel)

m = st.columns(5)
m[0].metric("Rappel", f"{st_['rappel']:.1%}", help="Part des attaques détectées")
m[1].metric("Précision (test)", f"{st_['precision']:.1%}", help="Part des alertes justes sur le test")
m[2].metric("Taux de fausses alertes", f"{st_['taux_fa']:.2%}")
m[3].metric(f"Précision à {taux_reel:.1%} d'attaques", f"{prec_reelle:.1%}",
            delta=f"{(prec_reelle - st_['precision']) * 100:+.1f} pts", delta_color="normal")
m[4].metric("Attaques manquées (test)", st_["FN"])
st.info(f"Sur **100 000 flux** avec {taux_reel:.1%} d'attaques : **{proj['detectees']:,.0f}** attaques "
        f"détectées, **{proj['manquees']:,.0f}** manquées, **{proj['fausses_alertes']:,.0f}** fausses alertes.")

c1, c2 = st.columns(2)
c1.plotly_chart(V.fig_confusion(st_, f"Matrice de confusion – {modele} (seuil {seuil:.2f})"),
                use_container_width=True)
c2.plotly_chart(V.fig_seuil(V.courbe_seuils(y, p), seuil), use_container_width=True)

st.markdown("**Erreurs par type de flux** (attaques : manquées · Benign : fausses alertes)")
st.dataframe(V.erreurs_par_type(t, y, p, seuil), use_container_width=True)
st.caption("La précision dépend du taux d'attaques (formule de Bayes), pas le rappel ni le taux de "
           "fausses alertes : un excellent détecteur appliqué à un événement rare produit beaucoup de "
           "fausses alertes. Seuil recommandé ≈ 0,8, avec agrégation des alertes par source.")
