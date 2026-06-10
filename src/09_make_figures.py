"""Stage 09 — Thesis figure factory (Chapter 5: Results & Discussion).

Reads everything already produced by the pipeline (results/predictions/ and
results/metrics/) and renders a
complete, publication-quality battery of figures into figures/. It is OFFLINE
(no API calls) and idempotent: re-run any time after 05_evaluate_metrics.py.

Every figure is written with a `ch5_` prefix so they are easy to drop into the
thesis. Each section is guarded — if an input CSV is missing, that figure is
skipped (with a note) instead of crashing, so partial pipelines still produce
all the figures the available data supports.

Figures produced (when data is present):
  Overall model comparison
    ch5_accuracy_by_experiment      ch5_macro_f1_by_experiment
    ch5_core_metrics_grouped        ch5_binary_metrics_grouped
    ch5_parse_ok_rate
  Per-class behaviour
    ch5_per_class_f1_heatmap        ch5_per_class_f1_grouped
    ch5_class_distribution
  Prompt engineering
    ch5_prompt_style_ablation
  Confusion matrices (row-normalised %)
    ch5_cm_norm_<experiment>        (one per experiment)
  Rule-based lexicon analysis (Exp 1)
    ch5_tox_score_distribution      ch5_sentiment_by_class
    ch5_threshold_sweep
  Robustness & hybrid
    ch5_hard_vs_easy                ch5_hybrid_vs_best_llm
  Explanation quality (Exp 9, if available)
    ch5_explanation_quality_rubric  ch5_explanation_quality_by_class

Usage:  python 09_make_figures.py
"""
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from config import PRED_DIR, METRICS_DIR, FIGURES_DIR, LABELS

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "axes.titleweight": "bold"})

# Pretty display names + canonical ordering of experiments.
EXP_DISPLAY = {
    "exp1_rule_eval": "Exp1\nRule-based",
    "exp2_zero_shot": "Exp2\nZero-shot",
    "exp3_few_shot": "Exp3\nFew-shot",
    "exp4_cot": "Exp4\nCoT",
    "exp5_persona": "Exp5\nPersona",
    "exp6_few_shot_cot": "Exp6\nFew-shot+CoT",
    "exp7_hybrid": "Exp7\nHybrid",
}
EXP_ORDER = list(EXP_DISPLAY.keys())
# Prediction column for confusion matrices differs for the rule baseline.
PRED_COL = {e: ("rule_pred" if e == "exp1_rule_eval" else "pred_label") for e in EXP_ORDER}
# Short class labels for tick marks.
SHORT = {
    "religion": "religion", "age": "age", "gender": "gender", "ethnicity": "ethnicity",
    "other_cyberbullying": "other_cb", "not_cyberbullying": "not_cb",
}
SHORT_LABELS = [SHORT[l] for l in LABELS]
PALETTE = sns.color_palette("viridis", n_colors=len(EXP_ORDER))

_made = []


def save(fig, name):
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    _made.append(name)
    print(f"  + {name}.png")


def _disp(exp):
    return EXP_DISPLAY.get(exp, exp).replace("\n", " ")


# --------------------------------------------------------------------------- #
# Section 1 — overall comparison from summary_metrics.csv
# --------------------------------------------------------------------------- #
def section_summary():
    p = METRICS_DIR / "summary_metrics.csv"
    if not p.exists():
        print("[skip] summary_metrics.csv not found — run 05_evaluate_metrics.py first")
        return None
    s = pd.read_csv(p)
    s["order"] = s["experiment"].map(lambda e: EXP_ORDER.index(e) if e in EXP_ORDER else 99)
    s = s.sort_values("order").reset_index(drop=True)
    s["disp"] = s["experiment"].map(_disp)
    colors = [PALETTE[EXP_ORDER.index(e)] if e in EXP_ORDER else "#888" for e in s["experiment"]]

    # 1a. Accuracy by experiment (sorted bar, value labels)
    sa = s.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sa["disp"], sa["accuracy"], color=[PALETTE[EXP_ORDER.index(e)] if e in EXP_ORDER else "#888" for e in sa["experiment"]])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Accuracy"); ax.set_xlim(0, max(sa["accuracy"]) * 1.18)
    ax.set_title("Overall accuracy by experiment")
    save(fig, "ch5_accuracy_by_experiment")

    # 1b. Macro-F1 by experiment
    sm = s.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sm["disp"], sm["macro_f1"], color=[PALETTE[EXP_ORDER.index(e)] if e in EXP_ORDER else "#888" for e in sm["experiment"]])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("Macro-F1"); ax.set_xlim(0, max(sm["macro_f1"]) * 1.18)
    ax.set_title("Macro-F1 by experiment")
    save(fig, "ch5_macro_f1_by_experiment")

    # 1c. Core metrics grouped (accuracy / macro-F1 / weighted-F1)
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    x = np.arange(len(s)); w = 0.26
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(metrics):
        b = ax.bar(x + (i - 1) * w, s[m], w, label=m.replace("_", "-"))
        ax.bar_label(b, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(s["disp"], fontsize=10)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.0)
    ax.set_title("Core 6-class metrics by experiment")
    ax.legend(title="Metric", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "ch5_core_metrics_grouped")

    # 1d. Binary (bully vs not) metrics grouped
    bmetrics = ["binary_acc", "binary_precision", "binary_recall"]
    if all(m in s.columns for m in bmetrics):
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, m in enumerate(bmetrics):
            b = ax.bar(x + (i - 1) * w, s[m], w, label=m.replace("binary_", "").replace("_", "-"))
            ax.bar_label(b, fmt="%.2f", padding=2, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(s["disp"], fontsize=10)
        ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
        ax.set_title("Binary detection (bully vs. not) — accuracy / precision / recall")
        ax.legend(title="Metric", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        save(fig, "ch5_binary_metrics_grouped")

    # 1e. Parse-ok rate (valid-JSON compliance) for LLM experiments
    if "parse_ok_rate" in s.columns and s["parse_ok_rate"].notna().any():
        sp = s[s["parse_ok_rate"].notna()]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(sp["disp"], sp["parse_ok_rate"], color="#2a9d8f")
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_ylabel("Valid-JSON rate"); ax.set_ylim(0, 1.05)
        ax.set_title("Output format compliance (parse-ok rate)")
        ax.tick_params(axis="x", labelsize=10)
        save(fig, "ch5_parse_ok_rate")

    # 1f. Per-class F1 heatmap
    f1cols = [f"f1_{l}" for l in LABELS]
    if all(c in s.columns for c in f1cols):
        mat = s.set_index("disp")[f1cols]
        mat.columns = SHORT_LABELS
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1,
                    cbar_kws={"label": "F1"}, ax=ax, linewidths=.5)
        ax.set_title("Per-class F1 by experiment")
        ax.set_ylabel(""); ax.set_xlabel("Class")
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save(fig, "ch5_per_class_f1_heatmap")

        # 1g. Per-class F1 grouped bars (one cluster per class)
        fig, ax = plt.subplots(figsize=(13, 6))
        xc = np.arange(len(LABELS)); n = len(s); bw = 0.8 / max(n, 1)
        for i, (_, r) in enumerate(s.iterrows()):
            ax.bar(xc + i * bw - 0.4 + bw / 2, [r[c] for c in f1cols], bw,
                   label=r["disp"].replace("\n", " "),
                   color=PALETTE[EXP_ORDER.index(r["experiment"])] if r["experiment"] in EXP_ORDER else "#888")
        ax.set_xticks(xc); ax.set_xticklabels(SHORT_LABELS, rotation=30, ha="right")
        ax.set_ylabel("F1"); ax.set_ylim(0, 1.0)
        ax.set_title("Per-class F1 across experiments")
        ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        save(fig, "ch5_per_class_f1_grouped")
    return s


# --------------------------------------------------------------------------- #
# Section 2 — prompt-style ablation
# --------------------------------------------------------------------------- #
def section_ablation(s):
    if s is None:
        return
    order = ["exp2_zero_shot", "exp3_few_shot", "exp4_cot", "exp5_persona", "exp6_few_shot_cot"]
    abl = s[s["experiment"].isin(order)].copy()
    if abl.empty:
        return
    abl["order"] = abl["experiment"].map(order.index)
    abl = abl.sort_values("order")
    labels = ["Zero-shot", "Few-shot", "CoT", "Persona", "Few-shot\n+CoT"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(abl)), abl["macro_f1"], "-o", color="#e76f51", lw=2.5, ms=10, label="Macro-F1")
    ax.plot(range(len(abl)), abl["accuracy"], "-s", color="#264653", lw=2, ms=8, label="Accuracy")
    base = abl.iloc[0]["macro_f1"]
    for i, (_, r) in enumerate(abl.iterrows()):
        d = r["macro_f1"] - base
        ax.annotate(f"{r['macro_f1']:.3f}\n({d:+.3f})", (i, r["macro_f1"]),
                    textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9)
    ax.set_xticks(range(len(abl))); ax.set_xticklabels(labels[:len(abl)])
    ax.set_ylabel("Score"); ax.set_title("Prompt-style ablation (Δ macro-F1 vs. zero-shot)")
    ax.legend(loc="best")
    save(fig, "ch5_prompt_style_ablation")


# --------------------------------------------------------------------------- #
# Section 3 — normalised confusion matrices + class distribution
# --------------------------------------------------------------------------- #
def section_confusion():
    eval_gold_done = False
    for exp in EXP_ORDER:
        p = PRED_DIR / f"{exp}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        col = PRED_COL[exp]
        if "label" not in df.columns or col not in df.columns:
            continue
        y_true = df["label"].astype(str)
        y_pred = df[col].astype(str)

        cm = confusion_matrix(y_true, y_pred, labels=LABELS).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cmn = np.divide(cm, cm.sum(axis=1, keepdims=True),
                            out=np.zeros_like(cm), where=cm.sum(axis=1, keepdims=True) != 0)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        sns.heatmap(cmn * 100, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                    xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS,
                    cbar_kws={"label": "% of gold row"}, ax=ax, linewidths=.5)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Gold")
        ax.set_title(f"Normalised confusion matrix — {_disp(exp)}")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        save(fig, f"ch5_cm_norm_{exp}")

        # class distribution from the first eval file we see
        if not eval_gold_done:
            vc = y_true.value_counts().reindex(LABELS).fillna(0).astype(int)
            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.bar(SHORT_LABELS, vc.values, color=sns.color_palette("crest", len(LABELS)))
            ax.bar_label(bars, padding=3)
            ax.set_ylabel("Tweets"); ax.set_title("Gold class distribution (evaluation sample)")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            save(fig, "ch5_class_distribution")
            eval_gold_done = True


# --------------------------------------------------------------------------- #
# Section 4 — rule-based lexicon analysis (Exp 1)
# --------------------------------------------------------------------------- #
def section_lexicon():
    p = PRED_DIR / "exp1_rule_eval.csv"
    if not p.exists():
        print("[skip] exp1_rule_eval.csv not found")
        return
    df = pd.read_csv(p)
    df["is_bully"] = (df["label"] != "not_cyberbullying")

    # 4a. Toxicity score distribution by gold class (bully vs not)
    if "tox_score" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        bins = np.linspace(0, 1, 26)
        ax.hist(df.loc[df["is_bully"], "tox_score"], bins=bins, alpha=0.6, label="bully (gold)", color="#e63946")
        ax.hist(df.loc[~df["is_bully"], "tox_score"], bins=bins, alpha=0.6, label="not_cyberbullying", color="#457b9d")
        ax.axvline(0.35, color="k", ls="--", lw=1.5, label="decision threshold (0.35)")
        ax.set_xlabel("Rule toxicity score"); ax.set_ylabel("Tweets")
        ax.set_title("Rule toxicity score distribution by gold label")
        ax.legend()
        save(fig, "ch5_tox_score_distribution")

    # 4b. Sentiment by class (box)
    if "sentiment" in df.columns:
        fig, ax = plt.subplots(figsize=(11, 6))
        data = [df.loc[df["label"] == l, "sentiment"].dropna().values for l in LABELS]
        sns.boxplot(data=data, ax=ax, palette="crest")
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(SHORT_LABELS, rotation=30, ha="right")
        ax.set_ylabel("VADER compound sentiment")
        ax.set_title("Sentiment distribution by gold class")
        save(fig, "ch5_sentiment_by_class")

    # 4c. Threshold sweep for the binary rule detector
    if "tox_score" in df.columns:
        gold = df["is_bully"].astype(int).values
        thr = np.linspace(0, 1, 101)
        acc, prec, rec, f1 = [], [], [], []
        for t in thr:
            pred = (df["tox_score"].values >= t).astype(int)
            acc.append(accuracy_score(gold, pred))
            prec.append(precision_score(gold, pred, zero_division=0))
            rec.append(recall_score(gold, pred, zero_division=0))
            f1.append(f1_score(gold, pred, zero_division=0))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thr, acc, label="accuracy", lw=2)
        ax.plot(thr, prec, label="precision", lw=2)
        ax.plot(thr, rec, label="recall", lw=2)
        ax.plot(thr, f1, label="F1", lw=2)
        best = int(np.argmax(f1))
        ax.axvline(thr[best], color="k", ls="--", lw=1.2, label=f"best-F1 thr={thr[best]:.2f}")
        ax.axvline(0.35, color="grey", ls=":", lw=1.2, label="used thr=0.35")
        ax.set_xlabel("Toxicity threshold"); ax.set_ylabel("Binary score")
        ax.set_title("Rule detector — binary metrics vs. threshold")
        ax.legend(fontsize=10)
        save(fig, "ch5_threshold_sweep")


# --------------------------------------------------------------------------- #
# Section 5 — robustness (Exp 8) + hybrid (Exp 7)
# --------------------------------------------------------------------------- #
def section_hard_cases():
    p = METRICS_DIR / "exp8_hard_summary.csv"
    if not p.exists():
        print("[skip] exp8_hard_summary.csv not found")
        return
    df = pd.read_csv(p)
    x = np.arange(len(df)); w = 0.26
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (m, lab) in enumerate([("acc_hard", "hard"), ("acc_easy", "easy"), ("acc_overall", "overall")]):
        b = ax.bar(x + (i - 1) * w, df[m], w, label=lab)
        ax.bar_label(b, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(df["style"], rotation=15)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, max(df[["acc_hard", "acc_easy", "acc_overall"]].max()) * 1.2)
    ax.set_title("Robustness: accuracy on hard vs. easy tweets (Exp 8)")
    ax.legend(title="Subset")
    save(fig, "ch5_hard_vs_easy")


def section_hybrid(s):
    if s is None:
        return
    order = ["exp2_zero_shot", "exp3_few_shot", "exp4_cot", "exp5_persona", "exp6_few_shot_cot"]
    llm = s[s["experiment"].isin(order)]
    hyb = s[s["experiment"] == "exp7_hybrid"]
    if llm.empty or hyb.empty:
        return
    best = llm.sort_values("macro_f1", ascending=False).iloc[0]
    metrics = ["macro_f1", "accuracy", "binary_recall"]
    labels = ["Macro-F1", "Accuracy", "Binary recall"]
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w / 2, [best[m] for m in metrics], w, label=f"Best LLM ({_disp(best['experiment'])})", color="#8d99ae")
    b2 = ax.bar(x + w / 2, [hyb.iloc[0][m] for m in metrics], w, label="Hybrid (Exp7)", color="#ef233c")
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=9); ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Hybrid (reasoning + rules) vs. best pure LLM")
    ax.legend(loc="upper right")
    save(fig, "ch5_hybrid_vs_best_llm")


# --------------------------------------------------------------------------- #
# Section 6 — explanation quality (Exp 9, optional)
# --------------------------------------------------------------------------- #
def section_explanations():
    files = glob.glob(str(METRICS_DIR / "exp9_explanation_quality_*.csv"))
    if not files:
        print("[skip] no exp9_explanation_quality_*.csv — run 07_judge_explanations.py to enable")
        return
    q = pd.read_csv(files[0])
    rubric = ["names_target", "cites_reason", "intent_coherent", "consistent_with_gold"]
    present = [r for r in rubric if r in q.columns]
    if present:
        means = [q[r].mean() for r in present] + ([q["expl_quality"].mean()] if "expl_quality" in q else [])
        names = [r.replace("_", " ") for r in present] + (["composite"] if "expl_quality" in q else [])
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names, means, color=sns.color_palette("flare", len(names)))
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_ylabel("Mean score"); ax.set_ylim(0, 1.05)
        ax.set_title("Explanation-quality rubric (LLM-as-judge, Exp 9)")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        save(fig, "ch5_explanation_quality_rubric")
    if "expl_quality" in q.columns and "label" in q.columns:
        m = q.groupby("label")["expl_quality"].mean().reindex(LABELS).fillna(0)
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(SHORT_LABELS, m.values, color=sns.color_palette("flare", len(LABELS)))
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_ylabel("Mean composite quality"); ax.set_ylim(0, 1.05)
        ax.set_title("Explanation quality by class (Exp 9)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save(fig, "ch5_explanation_quality_by_class")


def main():
    print(f"Rendering thesis figures -> {FIGURES_DIR}")
    s = section_summary()
    section_ablation(s)
    section_confusion()
    section_lexicon()
    section_hard_cases()
    section_hybrid(s)
    section_explanations()
    print(f"\nDone. {len(_made)} figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
