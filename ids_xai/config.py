
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
DOSSIER_DATA = RACINE / "data"
DOSSIER_MODELES = RACINE / "models"
DOSSIER_RAPPORTS = RACINE / "reports"

FICHIER_DONNEES = DOSSIER_DATA / "ids2018_propre.parquet"
FICHIER_ARTEFACTS = DOSSIER_MODELES / "artefacts.joblib"

NOM_PROJET = "CSE-CIC-IDS2018 – Détection d'intrusion réseau explicable (infrastructure AWS)"

BASE_URL = ("https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
            "Processed%20Traffic%20Data%20for%20ML%20Algorithms/")
JOURS = {
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv": "Force brute FTP/SSH",
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv": "DoS Hulk / SlowHTTPTest",
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv": "Botnet",
}
FRAC = 0.015
RANDOM_STATE = 42

COLONNE_LABEL = "Label"
CLASSE_NEGATIVE = "Benign"
CIBLE = "_y"                  # 1 = attaque, 0 = normal
COLONNE_TYPE = "type_attaque"
COLONNE_JOUR = "jour"
EXCLUES = [CIBLE, COLONNE_TYPE, COLONNE_JOUR]

COLONNES_A_RETIRER = ["Timestamp", "Dst Port", "Flow ID", "Src IP", "Src Port", "Dst IP"]

EMPREINTE = ["Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Seg Size Min"]

PRIORITAIRES = ["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "Pkt Size Avg"]

SEUIL_CORR = 0.95
TEST_SIZE = 0.2
N_FOLDS = 5
SEUIL_DECISION = 0.5
SEUIL_RECOMMANDE = 0.8

COULEURS = {"Normal": "#4C78A8", "Attaque": "#E45756"}

COLONNES_CIC = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Tot", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
    "Bwd URG Flags", "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "FIN Flag Cnt",
    "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "CWE Flag Count", "ECE Flag Cnt", "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg",
    "Bwd Seg Size Avg", "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg", "Subflow Fwd Pkts",
    "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts", "Init Fwd Win Byts",
    "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label",
]

DESCRIPTIONS = {
    "Flow Duration": "durée du flux (µs)",
    "Tot Fwd Pkts": "paquets envoyés par le client",
    "Tot Bwd Pkts": "paquets renvoyés par le serveur",
    "TotLen Fwd Pkts": "octets envoyés par le client",
    "Pkt Size Avg": "taille moyenne des paquets",
    "Flow Byts/s": "débit en octets par seconde",
    "Bwd Pkts/s": "paquets retour par seconde",
    "Flow IAT Std": "irrégularité des intervalles entre paquets",
    "Flow IAT Min": "intervalle minimal entre deux paquets",
    "Fwd IAT Min": "intervalle minimal entre deux paquets envoyés",
    "Fwd IAT Mean": "intervalle moyen entre deux paquets envoyés",
    "Init Fwd Win Byts": "fenêtre TCP initiale du client (empreinte système)",
    "Init Bwd Win Byts": "fenêtre TCP initiale du serveur (empreinte système)",
    "Fwd Seg Size Min": "taille minimale de segment (≈ en-tête, empreinte système)",
    "Protocol": "protocole (6 = TCP, 17 = UDP)",
}
