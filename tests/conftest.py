import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids_xai.donnees import nettoyer, telecharger  # noqa: E402
from ids_xai.preparation import preparer  # noqa: E402
from tests.donnees_synthetiques import generer  # noqa: E402


@pytest.fixture(scope="session")
def dossier_csv(tmp_path_factory):
    return generer(tmp_path_factory.mktemp("cic"), n_par_jour=3000)


@pytest.fixture(scope="session")
def brut(dossier_csv):
    return telecharger(frac=0.5, base_url=f"{dossier_csv}/")


@pytest.fixture(scope="session")
def propre(brut):
    df, _ = nettoyer(brut)
    return df


@pytest.fixture(scope="session")
def jeu(propre):
    return preparer(propre)
