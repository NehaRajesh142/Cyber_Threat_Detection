"""
train_model.py
--------------
Trains and compares three classical ML models on the traffic dataset:
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting (used as a drop-in for XGBoost; see note below)

NOTE ON XGBOOST: the real `xgboost` package could not be installed in
this build environment (no internet access). sklearn's
HistGradientBoostingClassifier is used instead — it is the same family
of algorithm (gradient-boosted trees) and the code is written so that
swapping in real XGBoost is a one-line change (see USE_XGBOOST below)
once you run this on a machine with internet access / pip installed.
"""

import json
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from preprocess import load_and_prepare, LABEL_COLUMN

USE_XGBOOST = False  # flip to True and `pip install xgboost` to use real XGBoost

if USE_XGBOOST:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        USE_XGBOOST = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "models")


def build_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1, random_state=42, class_weight="balanced"
        ),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
    }
    if USE_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            eval_metric="mlogloss", random_state=42
        )
    else:
        models["GradientBoosting (XGBoost stand-in)"] = HistGradientBoostingClassifier(
            max_iter=200, max_depth=8, learning_rate=0.1, random_state=42
        )
    return models


def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, zero_division=0
    )
    return {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": cm, "report": report, "y_pred": y_pred,
    }


def run_experiment(feature_mode):
    print(f"\n{'='*70}\nFEATURE MODE: {feature_mode.upper()}\n{'='*70}")
    data, df = load_and_prepare(feature_mode)
    label_encoder = data["label_encoder"]

    results = {}
    models = build_models()
    for name, model in models.items():
        t0 = time.time()
        model.fit(data["X_train"], data["y_train"])
        train_time = time.time() - t0

        metrics = evaluate_model(model, data["X_test"], data["y_test"], label_encoder)
        metrics["train_time_sec"] = round(train_time, 2)
        results[name] = metrics

        print(f"\n--- {name} ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}  "
              f"Precision: {metrics['precision']:.4f}  "
              f"Recall: {metrics['recall']:.4f}  "
              f"F1: {metrics['f1']:.4f}  "
              f"(train time: {train_time:.2f}s)")

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name.split(' ')[0]}_{feature_mode}.joblib"))

    return results, data, df


def summarize_to_json(all_results):
    summary = {}
    for feature_mode, results in all_results.items():
        summary[feature_mode] = {
            name: {
                "accuracy": round(m["accuracy"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "train_time_sec": m["train_time_sec"],
            }
            for name, m in results.items()
        }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    all_results = {}
    all_results["unidirectional"], data_uni, df = run_experiment("unidirectional")
    all_results["full"], data_full, _ = run_experiment("full")

    summary = summarize_to_json(all_results)

    print(f"\n{'='*70}\nSUMMARY: unidirectional-only vs full bidirectional features\n{'='*70}")
    for feature_mode, models in summary.items():
        print(f"\n{feature_mode.upper()}:")
        for name, m in models.items():
            print(f"  {name:35s} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")

    # Save the best model's classification report + confusion matrix for the report
    best_name = max(all_results["unidirectional"], key=lambda n: all_results["unidirectional"][n]["f1"])
    best = all_results["unidirectional"][best_name]
    with open(os.path.join(OUTPUT_DIR, "best_model_report.txt"), "w") as f:
        f.write(f"Best model (unidirectional features): {best_name}\n\n")
        f.write(best["report"])
    np.save(os.path.join(OUTPUT_DIR, "best_confusion_matrix.npy"), best["confusion_matrix"])
    print(f"\nBest unidirectional model: {best_name} (F1={best['f1']:.4f})")
    print(f"Metrics summary written to outputs/metrics_summary.json")
