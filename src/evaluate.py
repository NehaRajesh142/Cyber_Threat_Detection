"""
evaluate.py
-----------
Produces the plots used in the project report:
  1. Confusion matrix heatmap for the best unidirectional model
  2. Bar chart comparing accuracy/F1 across models and feature modes
  3. Feature importance chart (Random Forest, unidirectional features)

Run this AFTER train_model.py.
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from preprocess import load_and_prepare

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")


def plot_confusion_matrix():
    cm = np.load(os.path.join(OUTPUT_DIR, "best_confusion_matrix.npy"))
    data, df = load_and_prepare("unidirectional")
    labels = data["label_encoder"].classes_

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix — Best Model (Unidirectional Features)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved outputs/confusion_matrix.png")


def plot_model_comparison():
    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json")) as f:
        summary = json.load(f)

    modes = list(summary.keys())
    model_names = list(summary[modes[0]].keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(model_names))
    width = 0.35

    for ax, metric in zip(axes, ["accuracy", "f1"]):
        for i, mode in enumerate(modes):
            values = [summary[mode][m][metric] for m in model_names]
            ax.bar(x + i * width, values, width, label=mode)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([m.replace(" (XGBoost stand-in)", "") for m in model_names], rotation=15, ha="right")
        ax.set_ylim(0.9, 1.0)
        ax.set_title(metric.upper())
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Unidirectional vs Full Feature Set — Model Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print("Saved outputs/model_comparison.png")


def plot_feature_importance():
    rf_path = os.path.join(MODEL_DIR, "RandomForest_unidirectional.joblib")
    if not os.path.exists(rf_path):
        print("RandomForest model not found — run train_model.py first.")
        return
    model = joblib.load(rf_path)
    data, df = load_and_prepare("unidirectional")
    feature_names = data["feature_names"]

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(9, 6))
    plt.barh([feature_names[i] for i in order][::-1], importances[order][::-1], color="#0b5fa5")
    plt.title("Top 15 Feature Importances — Random Forest (Unidirectional Features)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("Saved outputs/feature_importance.png")


if __name__ == "__main__":
    plot_confusion_matrix()
    plot_model_comparison()
    plot_feature_importance()
