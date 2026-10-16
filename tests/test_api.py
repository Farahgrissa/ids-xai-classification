import importlib

import pytest
from fastapi.testclient import TestClient

from ids_xai.artefacts import flux_exemple, medianes_trafic_normal, sauvegarder_modele_api, schema_variables
from ids_xai.modeles import creer_pipeline


@pytest.fixture(scope="module")
def client(jeu, tmp_path_factory, monkeypatch_module):
    pipe = creer_pipeline("XGBoost", jeu.X_train).fit(jeu.X_train, jeu.y_train)
    chemin = tmp_path_factory.mktemp("m") / "modele_api.joblib"
    sauvegarder_modele_api(pipe, schema_variables(jeu.X_train), {"PR-AUC": 0.99}, "XGBoost",
                           exemple=flux_exemple(jeu.X_train), chemin=chemin,
                           medianes_normal=medianes_trafic_normal(jeu.X_train, jeu.y_train))
    monkeypatch_module.setenv("MODELE_CHEMIN", str(chemin))
    import api.main
    importlib.reload(api.main)
    with TestClient(api.main.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    assert client.get("/health").json() == {"statut": "ok", "modele_charge": True}


def test_predict(client):
    flux = client.get("/exemple").json()["flux"]
    r = client.post("/predict", json={"flux": flux * 3, "seuil": 0.5})
    assert r.status_code == 200
    res = r.json()["resultats"]
    assert len(res) == 3 and 0 <= res[0]["probabilite_attaque"] <= 1


def test_variables_manquantes(client):
    flux = client.get("/exemple").json()["flux"][0]
    flux.pop(next(iter(flux)))
    assert client.post("/predict", json={"flux": [flux]}).status_code == 422


def test_explain(client):
    flux = client.get("/exemple").json()["flux"]
    r = client.post("/explain", json={"flux": flux * 2, "seuil": 0.5, "top": 3})
    assert r.status_code == 200
    rep = r.json()
    assert rep["modele"] == "XGBoost" and len(rep["explications"]) == 2
    e = rep["explications"][0]
    assert len(e["contributions"]) == 3 and e["unite"] == "log-odds"
    assert e["phrase"].startswith(("Le risque d'attaque", "Le flux est jugé normal"))
