"""
generate_synthetic_traffic.py
------------------------------
Generates a CICIDS2017-schema-compatible labeled traffic dataset so the
rest of the pipeline can be built, run, and demoed end to end without
needing the ~3 GB real dataset on disk.

IMPORTANT: This is a STAND-IN, not the real dataset. For your actual
submission, download the real CICIDS2017 CSVs (see README.md) and place
them in data/raw/ — preprocess.py will automatically use the real files
instead of calling this script if they are present.

The feature set mirrors the CICFlowMeter output used by CICIDS2017,
which cleanly separates "Fwd" (source -> destination, i.e. the direction
visible to a one-way / unidirectional tap or data-diode) from "Bwd"
(destination -> source, the return path) statistics. That separation is
exactly what this project needs to simulate a unidirectional-visibility
scenario.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# Attack classes and how many flows to simulate for each (imbalanced,
# similar in spirit to the real CICIDS2017 class distribution).
CLASS_COUNTS = {
    "BENIGN": 12000,
    "DoS": 2500,
    "PortScan": 2000,
    "BruteForce": 1200,
    "Botnet": 800,
    "WebAttack": 600,
    "Infiltration": 150,
}

COLUMNS = [
    "Destination Port", "Protocol", "Flow Duration",
    # Forward-direction (unidirectional-visible) features
    "Total Fwd Packets", "Total Length of Fwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Fwd Header Length", "Fwd Packets/s",
    "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std",
    "SYN Flag Count", "ACK Flag Count", "PSH Flag Count",
    # Backward-direction (return-path) features — only visible with
    # bidirectional capture. Kept so the pipeline can compare
    # "unidirectional-only" vs "full bidirectional" feature sets.
    "Total Backward Packets", "Total Length of Bwd Packets",
    "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Bwd IAT Mean", "Bwd IAT Std", "Bwd Header Length", "Bwd Packets/s",
    "Label",
]

# Per-class (mean, std) profiles for a handful of the most
# discriminative raw features; the rest are derived/correlated from
# these so the synthetic data has realistic-looking structure rather
# than being pure noise.
PROFILES = {
    "BENIGN":        dict(fwd_pkts=(12, 4),  fwd_len=(300, 80),  duration=(2e5, 6e4), iat=(1.5e4, 5e3), bwd_ratio=(0.9, 0.15), flags=(0.3, 0.3, 0.2)),
    "DoS":           dict(fwd_pkts=(400, 120), fwd_len=(60, 20),  duration=(5e4, 2e4), iat=(200, 80),   bwd_ratio=(0.05, 0.05), flags=(0.9, 0.1, 0.05)),
    "PortScan":      dict(fwd_pkts=(3, 1.5),  fwd_len=(40, 10),  duration=(500, 200), iat=(50, 20),    bwd_ratio=(0.02, 0.03), flags=(0.95, 0.02, 0.01)),
    "BruteForce":    dict(fwd_pkts=(25, 8),   fwd_len=(120, 30), duration=(3e4, 1e4), iat=(800, 300),  bwd_ratio=(0.6, 0.2),  flags=(0.4, 0.5, 0.1)),
    "Botnet":        dict(fwd_pkts=(18, 6),   fwd_len=(200, 60), duration=(1.2e5, 4e4), iat=(5e3, 2e3), bwd_ratio=(0.5, 0.2),  flags=(0.3, 0.4, 0.2)),
    "WebAttack":     dict(fwd_pkts=(35, 10),  fwd_len=(500, 150),duration=(4e4, 1.5e4), iat=(900, 400), bwd_ratio=(0.7, 0.2),  flags=(0.2, 0.6, 0.2)),
    "Infiltration":  dict(fwd_pkts=(60, 20),  fwd_len=(700, 200),duration=(3e5, 1e5), iat=(2e4, 8e3),  bwd_ratio=(0.8, 0.15), flags=(0.2, 0.5, 0.3)),
}

PROTOCOLS = [6, 17, 1]  # TCP, UDP, ICMP
PROTOCOL_WEIGHTS = [0.75, 0.20, 0.05]


def _clip_pos(arr):
    return np.clip(arr, 0.001, None)


def generate_class(label, n):
    p = PROFILES[label]
    fwd_pkts = _clip_pos(RNG.normal(p["fwd_pkts"][0], p["fwd_pkts"][1], n)).round()
    fwd_len_mean = _clip_pos(RNG.normal(p["fwd_len"][0], p["fwd_len"][1], n))
    duration = _clip_pos(RNG.normal(p["duration"][0], p["duration"][1], n))
    iat_mean = _clip_pos(RNG.normal(p["iat"][0], p["iat"][1], n))
    bwd_ratio = np.clip(RNG.normal(p["bwd_ratio"][0], p["bwd_ratio"][1], n), 0, 1.5)

    fwd_len_std = fwd_len_mean * RNG.uniform(0.1, 0.4, n)
    fwd_len_max = fwd_len_mean + fwd_len_std * RNG.uniform(1, 3, n)
    fwd_len_min = np.clip(fwd_len_mean - fwd_len_std * RNG.uniform(1, 3, n), 0, None)
    total_fwd_len = fwd_len_mean * fwd_pkts

    fwd_iat_std = iat_mean * RNG.uniform(0.1, 0.5, n)
    fwd_iat_max = iat_mean + fwd_iat_std * RNG.uniform(1, 4, n)
    fwd_iat_min = np.clip(iat_mean - fwd_iat_std * RNG.uniform(1, 4, n), 0, None)

    fwd_header_len = fwd_pkts * RNG.uniform(20, 32, n)
    fwd_pps = fwd_pkts / (duration / 1e6 + 1e-6)

    pkt_len_mean = fwd_len_mean * RNG.uniform(0.8, 1.1, n)
    pkt_len_std = fwd_len_std * RNG.uniform(0.9, 1.2, n)
    min_pkt_len = np.clip(pkt_len_mean - pkt_len_std, 0, None)
    max_pkt_len = pkt_len_mean + pkt_len_std * 2

    syn_w, ack_w, psh_w = p["flags"]
    syn = RNG.binomial(1, np.clip(syn_w, 0, 1), n)
    ack = RNG.binomial(1, np.clip(ack_w, 0, 1), n)
    psh = RNG.binomial(1, np.clip(psh_w, 0, 1), n)

    bwd_pkts = _clip_pos(fwd_pkts * bwd_ratio)
    bwd_len_mean = _clip_pos(fwd_len_mean * RNG.uniform(0.7, 1.3, n))
    bwd_len_std = bwd_len_mean * RNG.uniform(0.1, 0.4, n)
    bwd_len_max = bwd_len_mean + bwd_len_std * 2
    bwd_len_min = np.clip(bwd_len_mean - bwd_len_std * 2, 0, None)
    total_bwd_len = bwd_len_mean * bwd_pkts
    bwd_iat_mean = iat_mean * RNG.uniform(0.8, 1.5, n)
    bwd_iat_std = bwd_iat_mean * RNG.uniform(0.1, 0.5, n)
    bwd_header_len = bwd_pkts * RNG.uniform(20, 32, n)
    bwd_pps = bwd_pkts / (duration / 1e6 + 1e-6)

    dest_port = RNG.choice([80, 443, 22, 21, 3389, 8080, 53, 3306, 25],
                            size=n, p=[0.28, 0.25, 0.12, 0.08, 0.07, 0.08, 0.06, 0.04, 0.02])
    protocol = RNG.choice(PROTOCOLS, size=n, p=PROTOCOL_WEIGHTS)

    df = pd.DataFrame({
        "Destination Port": dest_port,
        "Protocol": protocol,
        "Flow Duration": duration.round(),
        "Total Fwd Packets": fwd_pkts,
        "Total Length of Fwd Packets": total_fwd_len.round(),
        "Fwd Packet Length Max": fwd_len_max.round(2),
        "Fwd Packet Length Min": fwd_len_min.round(2),
        "Fwd Packet Length Mean": fwd_len_mean.round(2),
        "Fwd Packet Length Std": fwd_len_std.round(2),
        "Fwd IAT Mean": iat_mean.round(2),
        "Fwd IAT Std": fwd_iat_std.round(2),
        "Fwd IAT Max": fwd_iat_max.round(2),
        "Fwd IAT Min": fwd_iat_min.round(2),
        "Fwd Header Length": fwd_header_len.round(),
        "Fwd Packets/s": fwd_pps.round(3),
        "Min Packet Length": min_pkt_len.round(2),
        "Max Packet Length": max_pkt_len.round(2),
        "Packet Length Mean": pkt_len_mean.round(2),
        "Packet Length Std": pkt_len_std.round(2),
        "SYN Flag Count": syn,
        "ACK Flag Count": ack,
        "PSH Flag Count": psh,
        "Total Backward Packets": bwd_pkts.round(),
        "Total Length of Bwd Packets": total_bwd_len.round(),
        "Bwd Packet Length Max": bwd_len_max.round(2),
        "Bwd Packet Length Min": bwd_len_min.round(2),
        "Bwd Packet Length Mean": bwd_len_mean.round(2),
        "Bwd Packet Length Std": bwd_len_std.round(2),
        "Bwd IAT Mean": bwd_iat_mean.round(2),
        "Bwd IAT Std": bwd_iat_std.round(2),
        "Bwd Header Length": bwd_header_len.round(),
        "Bwd Packets/s": bwd_pps.round(3),
        "Label": label,
    })
    return df[COLUMNS]


def generate_dataset():
    frames = [generate_class(label, n) for label, n in CLASS_COUNTS.items()]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/raw/synthetic_traffic.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} flows -> {out_path}")
    print(df["Label"].value_counts())
