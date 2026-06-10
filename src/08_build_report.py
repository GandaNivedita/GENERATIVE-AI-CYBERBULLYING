"""Stage 08 (Experiment 10) — Prompt benchmark synthesis & final report.

Rolls everything up into the paper's headline artifacts:
  - prompt ablation table (zero -> few -> cot -> persona -> few+cot deltas)
  - binary vs 6-class comparison
  - hybrid improvement over the best pure LLM
  - a markdown report (results/reports/REPORT.md) summarizing all 10 experiments

Run AFTER stages 04 (hybrid), 05 (metrics), 06 (hard cases), 07 (explanations).

Usage:  python 08_build_report.py
"""
import pandas as pd

from config import METRICS_DIR, REPORTS_DIR

EXP_LABELS = {
    "exp1_rule_eval": "Exp1 Rule-based (no LLM)",
    "exp2_zero_shot": "Exp2 Zero-shot",
    "exp3_few_shot": "Exp3 Few-shot",
    "exp4_cot": "Exp4 Chain-of-Thought",
    "exp5_persona": "Exp5 Persona",
    "exp6_few_shot_cot": "Exp6 Few-shot + CoT",
    "exp7_hybrid": "Exp7 Hybrid (Reasoning+Rules)",
}


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join("---" for _ in cols) + " |\n"
    body = ""
    for _, r in df.iterrows():
        body += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return head + sep + body


def main() -> None:
    summ_path = METRICS_DIR / "summary_metrics.csv"
    if not summ_path.exists():
        print("Run 05_evaluate_metrics.py first to produce summary_metrics.csv")
        return
    summ = pd.read_csv(summ_path)
    summ["name"] = summ["experiment"].map(lambda e: EXP_LABELS.get(e, e))

    lines = ["# Cyberbullying Detection — Experimental Results\n",
             "Training-free Gen-AI prompting + rule-based hybrid scoring on Twitter data.\n",
             f"Model: Groq llama-3.1-8b-instant | Eval sample: balanced, 6 classes.\n",
             "\n## 1. Master comparison (6-class)\n"]
    main_cols = ["name", "accuracy", "macro_f1", "weighted_f1", "binary_recall"]
    show = summ[["name", "accuracy", "macro_f1", "weighted_f1", "binary_recall"]].copy()
    lines.append(md_table(show, main_cols))

    # Prompt ablation (LLM-only styles)
    order = ["exp2_zero_shot", "exp3_few_shot", "exp4_cot", "exp5_persona", "exp6_few_shot_cot"]
    abl = summ[summ["experiment"].isin(order)].set_index("experiment").reindex(order).reset_index()
    if not abl["macro_f1"].isna().all():
        base = abl.loc[abl["experiment"] == "exp2_zero_shot", "macro_f1"].values
        base = base[0] if len(base) else 0
        abl["delta_vs_zero_shot"] = (abl["macro_f1"] - base).round(4)
        lines.append("\n## 2. Prompt-style ablation (macro-F1 vs zero-shot)\n")
        lines.append(md_table(abl, ["name", "macro_f1", "delta_vs_zero_shot"]))

    # Hybrid improvement
    llm_best = summ[summ["experiment"].isin(order)].sort_values("macro_f1", ascending=False)
    hyb = summ[summ["experiment"] == "exp7_hybrid"]
    if not llm_best.empty and not hyb.empty:
        best_f1 = llm_best.iloc[0]["macro_f1"]
        best_recall = llm_best.iloc[0]["binary_recall"]
        lines.append("\n## 3. Hybrid (Reasoning+Rules) vs best pure LLM\n")
        lines.append(f"- Best pure LLM: **{llm_best.iloc[0]['name']}** — macro-F1 {best_f1}, binary recall {best_recall}\n")
        lines.append(f"- Hybrid: macro-F1 {hyb.iloc[0]['macro_f1']}, binary recall {hyb.iloc[0]['binary_recall']}\n")
        lines.append(f"- Macro-F1 delta: {round(hyb.iloc[0]['macro_f1'] - best_f1, 4)}; "
                     f"recall delta: {round(hyb.iloc[0]['binary_recall'] - best_recall, 4)}\n")

    # Hard cases (Exp8)
    hp = METRICS_DIR / "exp8_hard_summary.csv"
    if hp.exists():
        hard = pd.read_csv(hp)
        lines.append("\n## 4. Robustness on hard cases (Exp8: sarcasm/coded/quoted)\n")
        lines.append(md_table(hard, list(hard.columns)))

    # Explanation quality (Exp9)
    import glob
    eq = glob.glob(str(METRICS_DIR / "exp9_explanation_quality_*.csv"))
    if eq:
        q = pd.read_csv(eq[0])
        lines.append("\n## 5. Explanation quality (Exp9, LLM-as-judge)\n")
        lines.append(f"- Mean composite explanation quality: **{q['expl_quality'].mean():.3f}**\n")
        for k in ["names_target", "cites_reason", "intent_coherent", "consistent_with_gold"]:
            if k in q:
                lines.append(f"  - {k}: {q[k].mean():.3f}\n")

    report = "".join(lines)
    (REPORTS_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nFull report saved -> {REPORTS_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
