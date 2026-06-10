"""Stage 05 — Evaluation: metrics, confusion matrices, master comparison table.

Reads prediction CSVs from results/predictions/ (any file with columns
label + pred_label), computes Accuracy, Macro-F1, Weighted-F1, per-class F1,
binary (bully/not) metrics, parse/refusal rate, saves a confusion-matrix figure
per experiment, and writes results/metrics/summary_metrics.csv (the headline
table for the paper) plus results/metrics/report_<exp>.txt per experiment.

Scores experiments 1-7 (rule baseline, the five prompt styles, and the hybrid).

Usage:  python 05_evaluate_metrics.py
"""
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_score, recall_score,
)

from config import PRED_DIR, METRICS_DIR, FIGURES_DIR, LABELS

# Files that are predictions we want scored. rule predictions use a different col.
PRED_FILES = {
    "exp2_zero_shot": "pred_label",
    "exp3_few_shot": "pred_label",
    "exp4_cot": "pred_label",
    "exp5_persona": "pred_label",
    "exp6_few_shot_cot": "pred_label",
    "exp1_rule_eval": "rule_pred",
    "exp7_hybrid": "pred_label",
}


def binary(series):
    """6-class label -> bully(1)/not(0)."""
    return (series != "not_cyberbullying").astype(int)


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, cbar=False)
    plt.ylabel("Gold")
    plt.xlabel("Predicted")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def evaluate_file(name: str, pred_col: str) -> dict | None:
    path = PRED_DIR / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "label" not in df.columns or pred_col not in df.columns:
        return None
    y_true = df["label"].astype(str)
    y_pred = df[pred_col].astype(str)

    row = {
        "experiment": name,
        "n": len(df),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0), 4),
        "binary_acc": round(accuracy_score(binary(y_true), binary(y_pred)), 4),
        "binary_precision": round(precision_score(binary(y_true), binary(y_pred), zero_division=0), 4),
        "binary_recall": round(recall_score(binary(y_true), binary(y_pred), zero_division=0), 4),
    }
    if "parse_ok" in df.columns:
        row["parse_ok_rate"] = round(df["parse_ok"].mean(), 4)
    # per-class F1
    per_class = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    for lab, f in zip(LABELS, per_class):
        row[f"f1_{lab}"] = round(f, 4)

    # save confusion matrix + full report
    plot_confusion(y_true, y_pred, name, FIGURES_DIR / f"cm_{name}.png")
    rep = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    (METRICS_DIR / f"report_{name}.txt").write_text(rep, encoding="utf-8")
    return row


def main() -> None:
    rows = []
    for name, col in PRED_FILES.items():
        r = evaluate_file(name, col)
        if r:
            rows.append(r)
            print(f"Scored {name}: acc={r['accuracy']} macro_f1={r['macro_f1']}")
    if not rows:
        print("No prediction files found yet.")
        return
    summary = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    out = METRICS_DIR / "summary_metrics.csv"
    summary.to_csv(out, index=False)
    print(f"\nMaster comparison table -> {out}")
    cols = ["experiment", "accuracy", "macro_f1", "binary_recall"]
    print(summary[cols].to_string(index=False))

    # Bar chart of macro-F1 across experiments
    plt.figure(figsize=(9, 5))
    s = summary.sort_values("macro_f1")
    sns.barplot(data=s, y="experiment", x="macro_f1", color="#4C72B0")
    plt.title("Macro-F1 by experiment")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "macro_f1_comparison.png", dpi=130)
    plt.close()
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
