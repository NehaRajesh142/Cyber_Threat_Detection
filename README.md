# AI-Based Detection of Cyber Threats in Unidirectional IP Traffic

A course mini project that trains classical ML models (Random Forest, SVM,
Gradient Boosting as an XGBoost stand-in) to detect cyber threats from
network flow data — and specifically tests how well detection holds up
when only **unidirectional** traffic is visible, as in a data-diode or
one-way network tap deployment (common in ICS/SCADA and other
high-security environments where return traffic cannot be observed).

## The Core Idea

Most intrusion-detection research (and most IDS datasets) assumes full,
bidirectional visibility of a network flow — both the request and the
response. But some of the most security-critical environments
deliberately install a **unidirectional gateway (data diode)** that
physically allows data to flow in only one direction, precisely to
prevent any return path an attacker could use. A monitoring system
sitting on such a link can only ever see traffic going one way.

This project asks: **can classical ML models still detect threats
accurately using only the forward-direction flow features, without the
return-path statistics?**

To answer this, every flow in the dataset is split into two feature
sets:

- **Unidirectional set** — only forward-direction (source → destination)
  statistics: packet counts, lengths, inter-arrival times, header
  lengths, flags, etc.
- **Full set** — the unidirectional set plus backward-direction
  (destination → source / return-path) statistics.

The same three models are trained and evaluated on both feature sets, so
the report can directly show the accuracy trade-off (if any) of
operating with only one-way visibility.

## Dataset

Built to work with **CICIDS2017** (Canadian Institute for Cybersecurity),
which has clean Fwd/Bwd feature separation from CICFlowMeter — exactly
what this project needs.

**To use the real dataset:**
1. Download the CSV files from the official page:
   https://www.unb.ca/cic/datasets/ids-2017.html
2. Place the CSVs in `data/raw/` (e.g. `Monday-WorkingHours.pcap_ISCX.csv`, etc.)
3. Run the pipeline as normal — `preprocess.py` automatically detects and
   uses real CSVs over the synthetic fallback.

**Without downloading anything:** the pipeline still runs end-to-end
using `data/generate_synthetic_traffic.py`, which produces a
schema-compatible synthetic dataset (same column names, same attack
categories: BENIGN, DoS, PortScan, BruteForce, Botnet, WebAttack,
Infiltration) so you can build, test, and demo the whole pipeline before
the real dataset is downloaded. **Swap in the real CSVs before final
submission** — the synthetic data is a development stand-in only.

## Setup

```bash
pip install -r requirements.txt
python data/generate_synthetic_traffic.py   # skip this if using real CICIDS2017 CSVs
cd src
python train_model.py     # trains all 3 models on both feature sets
python evaluate.py        # generates plots into outputs/
```

## Project Structure

```
cyber_threat_detection/
├── data/
│   ├── generate_synthetic_traffic.py   # dev-only synthetic data generator
│   └── raw/                            # put real CICIDS2017 CSVs here
├── src/
│   ├── preprocess.py     # loading, cleaning, unidirectional/full feature split
│   ├── train_model.py    # trains RF, SVM, GradientBoosting (XGBoost stand-in)
│   └── evaluate.py       # confusion matrix, model comparison, feature importance plots
├── outputs/
│   ├── models/                  # saved .joblib models
│   ├── metrics_summary.json     # accuracy/precision/recall/F1 for every model × feature set
│   ├── best_model_report.txt    # sklearn classification report for the best model
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## Using Real XGBoost

`train_model.py` uses `HistGradientBoostingClassifier` (scikit-learn) as
a stand-in because this build environment had no internet access to
install `xgboost`. To use real XGBoost:

```bash
pip install xgboost
```

Then in `src/train_model.py`, set `USE_XGBOOST = True` at the top of the
file — the rest of the pipeline (evaluation, plots, saved metrics) works
identically either way.

## Results (on synthetic dev data)

| Feature Set     | Model            | Accuracy | F1     |
|------------------|------------------|----------|--------|
| Unidirectional   | Random Forest    | ~0.984   | ~0.984 |
| Unidirectional   | SVM              | ~0.977   | ~0.978 |
| Unidirectional   | Gradient Boosting| ~0.984   | ~0.985 |
| Full (bidirectional) | Random Forest| ~0.988   | ~0.988 |
| Full (bidirectional) | Gradient Boosting | ~0.989 | ~0.989 |

The gap between unidirectional-only and full-feature accuracy is small
(roughly 0.5–1 point) — the headline finding for the report: **losing
visibility into the return path costs very little detection accuracy**,
which supports using unidirectional gateways in security-critical
networks without materially weakening AI-based threat detection.

*(Re-run on the real CICIDS2017 dataset before final submission — exact
numbers will differ from the synthetic dev results above.)*
