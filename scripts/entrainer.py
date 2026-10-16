import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids_xai import config as C  # noqa: E402
from ids_xai.artefacts import (  # noqa: E402
    FICHIER_API,
    flux_exemple,
    medianes_trafic_normal,
    sauvegarder,
    sauvegarder_modele_api,
    schema_variables,
)
from ids_xai.donnees import charger_donnees_propres  # noqa: E402
from ids_xai.evaluation import probabilites, tableau_metriques  # noqa: E402
from ids_xai.modeles import NOMS_MODELES, entrainer_et_comparer  # noqa: E402
from ids_xai.preparation import preparer  # noqa: E402

MODELE_RETENU = "XGBoost"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donnees", default=str(C.FICHIER_DONNEES))
    ap.add_argument("--avec-empreinte", action="store_true")
    ap.add_argument("--rechercher", action="store_true", help="relancer Grid/RandomizedSearch")
    ap.add_argument("--modeles", nargs="+", default=NOMS_MODELES)
    args = ap.parse_args()

    df = charger_donnees_propres(args.donnees)
    print(f"Données : {df.shape} | taux d'attaque {df[C.CIBLE].mean():.3f}")

    jeu = preparer(df, avec_empreinte=args.avec_empreinte)
    print(f"Variables : {len(jeu.variables)} ({len(jeu.redondantes)} redondantes retirées)")
    print(f"Train {jeu.X_train.shape} | Test {jeu.X_test.shape}")

    comparatif, modeles, params = entrainer_et_comparer(
        args.modeles, jeu.X_train, jeu.y_train, jeu.t_train, rechercher=args.rechercher,
        progression=lambda nom: print(f"  → {nom}"))
    print("\nValidation croisée (5 folds) :\n" + comparatif.round(4).to_string())

    probas = probabilites(modeles, jeu.X_test)
    eval_test = tableau_metriques(probas, jeu.y_test, reference_cv=comparatif["PR-AUC"])
    print("\nJeu de test (seuil 0,5) :\n" + eval_test.round(4).to_string())

    suffixe = "_avec_empreinte" if args.avec_empreinte else ""
    C.DOSSIER_RAPPORTS.mkdir(exist_ok=True)
    comparatif.round(4).to_csv(C.DOSSIER_RAPPORTS / f"comparatif_cv{suffixe}.csv")
    eval_test.round(4).to_csv(C.DOSSIER_RAPPORTS / f"evaluation_test{suffixe}.csv")

    if args.avec_empreinte:
        print("\nModèle de comparaison : seuls les rapports CSV sont enregistrés.")
        return

    retenu = MODELE_RETENU if MODELE_RETENU in modeles else comparatif.index[0]
    chemin = sauvegarder({
        "modeles": modeles, "modele_retenu": retenu, "hyperparametres": params,
        "comparatif": comparatif, "eval_test": eval_test,
        "X_test": jeu.X_test, "y_test": jeu.y_test, "t_test": jeu.t_test,
        "variables": jeu.variables, "redondantes": jeu.redondantes, "params_preparation": jeu.params,
        "schema": schema_variables(jeu.X_train),
    })
    sauvegarder_modele_api(modeles[retenu], schema_variables(jeu.X_train),
                           eval_test.loc[retenu].round(4).to_dict(), retenu,
                           exemple=flux_exemple(jeu.X_train),
                           medianes_normal=medianes_trafic_normal(jeu.X_train, jeu.y_train))
    print(f"\nEnregistré : {chemin}\nEnregistré : {FICHIER_API}")


if __name__ == "__main__":
    main()
