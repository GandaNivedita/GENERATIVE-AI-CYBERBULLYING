"""Stage 07 (Experiment 9) — Explanation-quality evaluation (LLM-as-judge).

Explainability is a first-class objective of this research, so we score the
QUALITY of the generated justifications, not just the label. A judge model
rates each explanation on a 4-point rubric:
  R1 names the correct target group (0/1)
  R2 cites the offending words/phrase or specific reason (0/1)
  R3 states intent/intensity coherently (0/1)
  R4 the explanation is consistent with the gold label (0/1)
Composite score = mean of the four (0..1).

We judge the best reasoning style (default few_shot_cot) on a stratified subset
to keep cost low. Uses the same Groq client (cached + key-rotated).

Usage:  python 07_judge_explanations.py [style] [n_per_class]
        default style=few_shot_cot, n_per_class=20
"""
import sys
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from config import PRED_DIR, METRICS_DIR, LABELS, RANDOM_SEED
from prompts import STYLE_TO_EXP
from llm_client import GroqRotator

JUDGE_SYSTEM = "You are a strict evaluator of moderation explanations. Output only JSON."

JUDGE_TEMPLATE = """A classifier judged a tweet. Evaluate the QUALITY of its explanation.

Tweet: "{tweet}"
Gold category: {gold}
Predicted category: {pred}
Model explanation: "{explanation}"

Score each criterion as 0 or 1:
- names_target: does the explanation correctly identify who/what group is targeted (or correctly say none)?
- cites_reason: does it cite the specific offending words or a concrete reason?
- intent_coherent: does it describe intent/tone coherently?
- consistent_with_gold: is the explanation consistent with the GOLD category?

Respond ONLY with JSON:
{{"names_target":0/1,"cites_reason":0/1,"intent_coherent":0/1,"consistent_with_gold":0/1}}"""

_BOT = None


def _judge_one(row):
    prompt = JUDGE_TEMPLATE.format(
        tweet=row["tweet_text"], gold=row["label"],
        pred=row["pred_label"], explanation=row["explanation"],
    )
    res = _BOT.complete_json(prompt, system=JUDGE_SYSTEM)
    keys = ["names_target", "cites_reason", "intent_coherent", "consistent_with_gold"]
    scores = {k: int(res.get(k, 0)) if str(res.get(k, 0)) in ("0", "1") else 0 for k in keys}
    composite = sum(scores.values()) / 4.0
    return {**{"id": row["id"], "label": row["label"], "pred_label": row["pred_label"]},
            **scores, "expl_quality": round(composite, 3)}


def main() -> None:
    global _BOT
    style = sys.argv[1] if len(sys.argv) > 1 else "few_shot_cot"
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    exp = STYLE_TO_EXP[style]
    df = pd.read_csv(PRED_DIR / f"{exp}_{style}.csv")

    # stratified subset
    parts = [df[df["label"] == lab].sample(n=min(n_per, (df["label"] == lab).sum()),
                                           random_state=RANDOM_SEED) for lab in LABELS]
    sub = pd.concat(parts).reset_index(drop=True)

    _BOT = GroqRotator()
    records = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for rec in tqdm(ex.map(_judge_one, [r for _, r in sub.iterrows()]),
                        total=len(sub), desc="judge"):
            records.append(rec)
    out = pd.DataFrame(records)
    path = METRICS_DIR / f"exp9_explanation_quality_{style}.csv"
    out.to_csv(path, index=False)

    print(f"\n[Exp9] Explanation quality for {style} (n={len(out)}):")
    print(f"  mean composite quality : {out['expl_quality'].mean():.3f}")
    for k in ["names_target", "cites_reason", "intent_coherent", "consistent_with_gold"]:
        print(f"  {k:24s}: {out[k].mean():.3f}")
    print("\nPer-class mean explanation quality:")
    print(out.groupby("label")["expl_quality"].mean().round(3).to_string())
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
