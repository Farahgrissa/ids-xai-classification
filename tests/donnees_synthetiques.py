"""Génère de faux fichiers au format CSE-CIC-IDS2018 (80 colonnes) pour les tests et la CI.

Reproduit les défauts réels : en-têtes répétés, valeurs « Infinity », doublons, colonnes constantes.
Les valeurs sont plausibles mais ne remplacent pas les vraies données.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ids_xai import config as C

TYPES_PAR_JOUR = {
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv": {"FTP-BruteForce": 0.18, "SSH-Bruteforce": 0.18},
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv": {"DoS attacks-Hulk": 0.45, "DoS attacks-SlowHTTPTest": 0.12},
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv": {"Bot": 0.27},
}


def _flux(rng, n, type_):
    if type_ == "Benign":
        duree = rng.lognormal(13, 2.5, n)
        fwd, bwd = rng.integers(1, 30, n), rng.integers(0, 30, n)
        taille = rng.lognormal(5, 1.2, n)
        win = rng.choice([8192, -1, 218, 236, 278, 62317], n)
        proto = rng.choice([6, 17, 0], n, p=[0.82, 0.17, 0.01])
    elif type_ == "DoS attacks-Hulk":
        duree = rng.lognormal(9.8, 1.0, n)
        fwd, bwd = rng.integers(1, 4, n), rng.integers(0, 2, n)
        taille = np.where(rng.random(n) < 0.65, 0, rng.lognormal(5.5, 0.5, n))
        win, proto = np.full(n, 225), np.full(n, 6)
    elif type_ == "Bot":
        duree = rng.lognormal(9.3, 0.6, n)
        fwd, bwd = rng.integers(2, 5, n), rng.integers(3, 6, n)
        taille = rng.normal(80, 10, n).clip(1)
        win, proto = np.full(n, 8192), np.full(n, 6)
    elif type_ == "SSH-Bruteforce":
        duree = rng.lognormal(12.8, 0.3, n)
        fwd, bwd = rng.integers(18, 26, n), rng.integers(18, 26, n)
        taille = rng.normal(90, 8, n).clip(1)
        win, proto = np.full(n, 26883), np.full(n, 6)
    else:  # FTP-BruteForce, SlowHTTPTest : connexions vides quasi identiques
        duree = rng.choice([1, 2, 3, 5, 17, 23], n).astype(float)
        fwd, bwd = np.ones(n, int), np.ones(n, int)
        taille = np.zeros(n)
        win, proto = np.full(n, 26883), np.full(n, 6)

    fwd_len = (fwd * taille).round()
    bwd_len = (bwd * taille * rng.uniform(0.5, 3, n)).round()
    iat = duree / np.maximum(fwd + bwd - 1, 1)
    d = {c: np.zeros(n) for c in C.COLONNES_CIC}
    d.update({
        "Dst Port": rng.choice([80, 443, 21, 22, 53, 8080], n), "Protocol": proto,
        "Timestamp": "14/02/2018 10:00:00", "Flow Duration": duree.round(),
        "Tot Fwd Pkts": fwd, "Tot Bwd Pkts": bwd, "TotLen Fwd Pkts": fwd_len, "TotLen Bwd Pkts": bwd_len,
        "Fwd Pkt Len Max": taille * 1.5, "Fwd Pkt Len Min": taille * 0.2, "Fwd Pkt Len Mean": taille,
        "Fwd Pkt Len Std": taille * 0.4, "Bwd Pkt Len Max": taille * 2, "Bwd Pkt Len Min": taille * 0.1,
        "Bwd Pkt Len Mean": taille * 1.2, "Bwd Pkt Len Std": taille * 0.5,
        "Flow IAT Mean": iat, "Flow IAT Std": iat * rng.uniform(0, 1.5, n), "Flow IAT Max": iat * 2,
        "Flow IAT Min": iat * rng.uniform(0, 0.5, n), "Fwd IAT Tot": duree * 0.9,
        "Fwd IAT Mean": iat * 1.1, "Fwd IAT Std": iat * 0.6, "Fwd IAT Max": iat * 2.1,
        "Fwd IAT Min": iat * rng.uniform(0, 0.4, n), "Bwd IAT Tot": duree * 0.8, "Bwd IAT Mean": iat,
        "Bwd IAT Std": iat * 0.5, "Bwd IAT Max": iat * 1.9, "Bwd IAT Min": iat * 0.3,
        "Fwd PSH Flags": rng.integers(0, 2, n), "Fwd Header Len": fwd * 32, "Bwd Header Len": bwd * 32,
        "Bwd Pkts/s": bwd / np.maximum(duree, 1) * 1e6,
        "Pkt Len Min": taille * 0.1, "Pkt Len Max": taille * 2, "Pkt Len Mean": taille * 1.1,
        "Pkt Len Std": taille * 0.45, "FIN Flag Cnt": rng.integers(0, 2, n),
        "RST Flag Cnt": rng.integers(0, 2, n), "PSH Flag Cnt": rng.integers(0, 2, n),
        "ACK Flag Cnt": rng.integers(0, 2, n), "URG Flag Cnt": rng.integers(0, 2, n),
        "Down/Up Ratio": rng.integers(0, 3, n), "Pkt Size Avg": taille * 1.15,
        "Init Fwd Win Byts": win, "Init Bwd Win Byts": rng.choice([-1, 0, 211, 219, 230], n),
        "Fwd Act Data Pkts": (fwd * (taille > 0)), "Fwd Seg Size Min": np.where(proto == 17, 8, 32),
        "Active Std": np.where(rng.random(n) < 0.1, rng.exponential(1e4, n), 0),
        "Active Max": np.where(rng.random(n) < 0.1, rng.exponential(1e5, n), 0),
        "Idle Std": np.where(rng.random(n) < 0.1, rng.exponential(1e5, n), 0),
        "Idle Max": np.where(rng.random(n) < 0.1, rng.exponential(1e7, n), 0),
        "Label": type_,
    })
    d["Flow Byts/s"] = (fwd_len + bwd_len) / duree * 1e6
    d["Flow Pkts/s"] = (fwd + bwd) / duree * 1e6
    d["Fwd Pkts/s"] = fwd / duree * 1e6
    d["Pkt Len Var"] = d["Pkt Len Std"] ** 2
    d["Fwd Seg Size Avg"], d["Bwd Seg Size Avg"] = d["Fwd Pkt Len Mean"], d["Bwd Pkt Len Mean"]
    d["Subflow Fwd Pkts"], d["Subflow Fwd Byts"] = fwd, fwd_len
    d["Subflow Bwd Pkts"], d["Subflow Bwd Byts"] = bwd, bwd_len
    d["SYN Flag Cnt"], d["ECE Flag Cnt"] = d["Fwd PSH Flags"], d["RST Flag Cnt"]
    for a, b in [("Active Mean", "Active Max"), ("Active Min", "Active Max"),
                 ("Idle Mean", "Idle Max"), ("Idle Min", "Idle Max")]:
        d[a] = d[b] * 0.8
    return pd.DataFrame(d)[C.COLONNES_CIC]


def generer(dossier: Path, n_par_jour: int = 4000, seed: int = 0) -> Path:
    """Écrit 3 faux CSV journaliers dans `dossier` et renvoie le dossier."""
    rng = np.random.default_rng(seed)
    dossier.mkdir(parents=True, exist_ok=True)
    for fichier, attaques in TYPES_PAR_JOUR.items():
        parts = []
        reste = n_par_jour
        for t, p in attaques.items():
            k = int(n_par_jour * p)
            parts.append(_flux(rng, k, t))
            reste -= k
        parts.append(_flux(rng, reste, "Benign"))
        df = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        df.loc[rng.choice(len(df), 15, replace=False), "Flow Byts/s"] = np.inf
        df = pd.concat([df, df.iloc[:200]])  # doublons
        lignes = df.to_csv(index=False).replace("inf", "Infinity").splitlines()
        lignes.insert(len(lignes) // 2, lignes[0])  # en-tête répété
        (dossier / fichier).write_text("\n".join(lignes) + "\n")
    return dossier


if __name__ == "__main__":
    import sys
    print(generer(Path(sys.argv[1] if len(sys.argv) > 1 else "data/synthetique")))
