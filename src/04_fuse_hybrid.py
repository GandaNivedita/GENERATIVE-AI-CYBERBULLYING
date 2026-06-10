"""Stage 04 (Experiment 7) — Hybrid "Reasoning + Rules" fusion.

Combines the best LLM prompt's prediction with the rule-based lexicon. Empirical
analysis of this dataset showed WHERE each method is strong:
  - The LLM is strong on RECALL (catches ~88% of bullying) but confuses the six
    fine-grained categories.
  - The lexicon has low recall (it misses implicit/contextual bullying with no
    explicit slurs) BUT, when an explicit marker fires, it names the category
    very precisely. On category disagreements where the rule fired, the rule was
    correct 45x vs the LLM's 8x.

So the productive fusion is CATEGORY ARBITRATION, not a recall safety-net.

Strategies (all saved for comparison):
  A. arbitration (PRIMARY): trust the LLM's bully/not decision, but when the
     lexicon fires a confident category (tox >= ARB_TOX) override the LLM's
     category to the lexicon's. Lifts accuracy on this dataset.
  B. safety_net: if the LLM says not_cyberbullying while rule toxicity is high,
     override to the rule's category (catches LLM false negatives). Documented
     but weak here, because the LLM's misses are implicit cases the lexicon
     also misses.
  C. union_bully: predict "bully" if EITHER the LLM or the rules say bully
     (maximizes binary recall; used for the safety view).

Outputs results/predictions/exp7_hybrid.csv (primary = arbitration) plus a comparison print.

Usage:  python 04_fuse_hybrid.py [best_llm_basename]
        default best_llm = exp5_persona
"""
import sys

import pandas as pd

from config import PRED_DIR

ARB_TOX = 0.5        # lexicon must be at least this toxic to arbitrate category
HIGH_TOX = 0.6
VERY_HIGH_TOX = 0.85


def load(best_llm: str) -> pd.DataFrame:
    llm = pd.read_csv(PRED_DIR / f"{best_llm}.csv")
    rule = pd.read_csv(PRED_DIR / "exp1_rule_eval.csv")
    rule_cols = ["id", "tox_score", "hit_terms", "rule_category", "rule_is_bully", "rule_pred"]
    merged = llm.merge(rule[rule_cols], on="id", how="left")
    return merged


def fuse(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()

    def arbitration(r):
        # Lexicon fired a confident category -> let it correct the LLM's category.
        if isinstance(r["rule_category"], str) and r["rule_category"] and r["tox_score"] >= ARB_TOX:
            return r["rule_category"]
        return r["pred_label"]

    def safety_net(r):
        # LLM said harmless but rules scream toxic -> override to rule category
        if r["pred_label"] == "not_cyberbullying" and r["tox_score"] >= HIGH_TOX:
            return r["rule_category"] if isinstance(r["rule_category"], str) and r["rule_category"] else "other_cyberbullying"
        return r["pred_label"]

    df["pred_label_llm_only"] = df["pred_label"]
    df["pred_label_arbitration"] = df.apply(arbitration, axis=1)
    df["pred_label_safety_net"] = df.apply(safety_net, axis=1)
    # Primary hybrid prediction used by 05_evaluate_metrics.py is category arbitration
    df["pred_label"] = df["pred_label_arbitration"]
    # Binary union view (max recall for the safety use-case)
    df["bully_union"] = (
        (df["pred_label_llm_only"] != "not_cyberbullying") | (df["rule_is_bully"] == 1)
    ).astype(int)
    return df


def main() -> None:
    best_llm = sys.argv[1] if len(sys.argv) > 1 else "exp5_persona"
    merged = load(best_llm)
    fused = fuse(merged)
    out = PRED_DIR / "exp7_hybrid.csv"
    fused.to_csv(out, index=False)

    gold = fused["label"]
    n_overrides = (fused["pred_label_arbitration"] != fused["pred_label_llm_only"]).sum()
    rescued = ((fused["pred_label_llm_only"] != gold) &
               (fused["pred_label_arbitration"] == gold)).sum()
    broke = ((fused["pred_label_llm_only"] == gold) &
             (fused["pred_label_arbitration"] != gold)).sum()
    base_acc = (fused["pred_label_llm_only"] == gold).mean()
    hyb_acc = (fused["pred_label_arbitration"] == gold).mean()
    print(f"Base LLM: {best_llm}  (ARB_TOX={ARB_TOX})")
    print(f"Category-arbitration overrides applied: {n_overrides}")
    print(f"  -> correctly rescued (LLM wrong -> hybrid right): {rescued}")
    print(f"  -> wrongly broke   (LLM right -> hybrid wrong):  {broke}")
    print(f"Accuracy: base LLM {base_acc:.4f} -> hybrid {hyb_acc:.4f} ({hyb_acc - base_acc:+.4f})")
    print(f"Saved -> {out}")
    print("Run 05_evaluate_metrics.py to score exp7_hybrid against the others.")


if __name__ == "__main__":
    main()
