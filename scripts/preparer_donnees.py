import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids_xai import config as C  # noqa: E402
from ids_xai.donnees import nettoyer, telecharger  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=C.FRAC)
    ap.add_argument("--base-url", default=C.BASE_URL, help="URL ou dossier local contenant les CSV")
    ap.add_argument("--sortie", default=str(C.FICHIER_DONNEES))
    args = ap.parse_args()

    brut = telecharger(frac=args.frac, base_url=args.base_url,
                       progression=lambda f, i: print(f"  {f[:20]} : morceau {i}", end="\r"))
    print(f"\nDonnées brutes : {brut.shape}")
    df, rapport = nettoyer(brut)
    Path(args.sortie).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.sortie, index=False)
    C.DOSSIER_RAPPORTS.mkdir(exist_ok=True)
    (C.DOSSIER_RAPPORTS / "rapport_nettoyage.json").write_text(
        json.dumps(rapport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Données propres : {df.shape} → {args.sortie}")
    print(json.dumps(rapport, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
