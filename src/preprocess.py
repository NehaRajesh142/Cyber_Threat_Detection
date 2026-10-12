"""
preprocess.py
-------------
Loads traffic flow data (real CICIDS2017 CSVs if present in data/raw/,
otherwise the synthetic fallback) and prepares two feature sets:

  1. UNIDIRECTIONAL feature set — only fields derivable from traffic
     observed in a single direction (source -> destination), i.e. what
     a data-diode / one-way tap / unidirectional gateway can see.
  2. FULL (bidirectional) feature set — adds the "Bwd" (return-path)
     statistics, which require seeing both directions of the flow.

Comparing model performance on (1) vs (2) is the core experiment of
this project: can AI still detect threats reliably when only one-way
traffic is visible?
"""

import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Columns that describe the return path (destination -> source). These
# are EXCLUDED from the unidirectional feature set because a one-way
# gateway never observes them.
BWD_COLUMNS = [
    "Total Backward Packets", "Total Length of Bwd Packets",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd Header Length", "Bwd Packets/s",
]

LABEL_COLUMN = "Label"


def _find_real_dataset_files():
    """Look for real CICIDS2017 CSVs the user has downloaded into data/raw/."""
    candidates = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    real_files = [f for f in candidates if "synthetic" not in os.path.basename(f).lower()]
    return real_files


def load_raw_dataframe():
    real_files = _find_real_dataset_files()
    if real_files:
        print(f"Found {len(real_files)} real dataset file(s) in data/raw/ — using them.")
        frames = [pd.read_csv(f, low_memory=False) for f in real_files]
        df = pd.concat(frames, ignore_index=True)
        df.columns = [c.strip() for c in df.columns]
    else:
        synthetic_path = os.path.join(RAW_DIR, "synthetic_traffic.csv")
        if not os.path.exists(synthetic_path):
            raise FileNotFoundError(
                "No dataset found. Either place real CICIDS2017 CSVs in data/raw/, "
                "or run `python data/generate_synthetic_traffic.py` first."
            )
        print("No real dataset found in data/raw/ — using synthetic fallback data. "
              "See README.md to plug in the real CICIDS2017 dataset.")
        df = pd.read_csv(synthetic_path)

    return df


def clean_dataframe(df):
    # Replace inf values (common in Flow Bytes/s and Flow Packets/s columns
    # of the real CICIDS2017 CSVs when Flow Duration is 0), drop NaNs.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()

    # Normalize label text (real CICIDS2017 has variants like
    # "Web Attack – Brute Force", "DoS Hulk", etc.) into the broad
    # attack families used in this project.
    def normalize_label(label):
        l = str(label).strip().lower()
        if l == "benign":
            return "BENIGN"
        if "dos" in l or "ddos" in l:
            return "DoS"
        if "portscan" in l or "port scan" in l:
            return "PortScan"
        if "brute" in l or "ftp-patator" in l or "ssh-patator" in l:
            return "BruteForce"
        if "bot" in l:
            return "Botnet"
        if "web attack" in l or "webattack" in l:
            return "WebAttack"
        if "infiltration" in l:
            return "Infiltration"
        return "Other"

    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(normalize_label)
    df = df[df[LABEL_COLUMN] != "Other"]
    return df


def get_feature_sets(df):
    """Returns (unidirectional_columns, full_columns) present in df."""
    all_cols = [c for c in df.columns if c != LABEL_COLUMN]
    full_cols = all_cols
    uni_cols = [c for c in all_cols if c not in BWD_COLUMNS]
    return uni_cols, full_cols


def prepare_splits(df, feature_cols, test_size=0.25, random_state=42):
    X = df[feature_cols].copy()
    y = df[LABEL_COLUMN].copy()

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state, stratify=y_enc
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "label_encoder": label_encoder, "scaler": scaler,
        "feature_names": feature_cols,
    }


def load_and_prepare(feature_mode="unidirectional"):
    """
    feature_mode: "unidirectional" or "full"
    Returns the prepared train/test split dict for the requested feature set.
    """
    df = load_raw_dataframe()
    df = clean_dataframe(df)
    uni_cols, full_cols = get_feature_sets(df)
    cols = uni_cols if feature_mode == "unidirectional" else full_cols
    return prepare_splits(df, cols), df


if __name__ == "__main__":
    data, df = load_and_prepare("unidirectional")
    print("Class distribution:\n", df[LABEL_COLUMN].value_counts())
    print("\nUnidirectional feature count:", len(data["feature_names"]))
    print("Train shape:", data["X_train"].shape, " Test shape:", data["X_test"].shape)
