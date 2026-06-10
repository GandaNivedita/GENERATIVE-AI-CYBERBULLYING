"""Stage 06 (Experiment 8) — Robustness / hard-case stress test.

Documents WHERE the model struggles: sarcasm, coded language, quoted irony,
reclaimed words, and tweets the rule layer flags but the LLM may miss. We reuse
the predictions already produced for Exp 2-6 (no new API calls) and slice them
by a "hard-case" heuristic, then report per-style accuracy on hard vs easy and
dump a failure gallery for qualitative analysis in the paper.

Usage:  python 06_analyze_hard_cases.py
"""
import re

import pandas as pd

from config import PRED_DIR, METRICS_DIR
from prompts import STYLE_TO_EXP

# Heuristic markers of "hard" tweets.
SARCASM_MARKERS = [":p", ":)", "lol", "lmao", "/s", "sure...", "yeah right",
                   "great job", "oh wow", "so brave", "totally"]
EMOJI_RE = re.compile(r"[\U0001F600-\U0001FAFF☀-➿]")
QUOTE_RE = re.compile(r'[\"“”].+[\"“”]')
# Reclaimed / in-group terms that flip meaning by context.
RECLAIMED = ["queer", "dyke", "nigga", "bitch", "slut", "crazy"]


def is_hard(text: str) -> bool:
    t = str(text).lower()
    if any(m in t for m in SARCASM_MARKERS):
        return True
    if EMOJI_RE.search(text):
        return True
    if QUOTE_RE.search(text):
        return True
    if any(w in t.split() for w in RECLAIMED):
        return True
    return False


def main() -> None:
    styles = list(STYLE_TO_EXP.keys())
    rows = []
    gallery = []
    for style in styles:
        exp = STYLE_TO_EXP[style]
        path = PRED_DIR / f"{exp}_{style}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["hard"] = df["tweet_text"].map(is_hard)
        hard = df[df["hard"]]
        easy = df[~df["hard"]]
        rows.append({
            "style": style,
            "n_hard": len(hard),
            "acc_hard": round((hard["pred_label"] == hard["label"]).mean(), 4) if len(hard) else None,
            "acc_easy": round((easy["pred_label"] == easy["label"]).mean(), 4) if len(easy) else None,
            "acc_overall": round((df["pred_label"] == df["label"]).mean(), 4),
        })
        # collect failures on hard cases for the gallery (from CoT style)
        if style == "cot":
            fails = hard[hard["pred_label"] != hard["label"]]
            for _, r in fails.iterrows():
                gallery.append({
                    "tweet_text": r["tweet_text"],
                    "gold": r["label"],
                    "pred": r["pred_label"],
                    "explanation": r["explanation"],
                })

    summary = pd.DataFrame(rows)
    summary.to_csv(METRICS_DIR / "exp8_hard_summary.csv", index=False)
    pd.DataFrame(gallery).to_csv(METRICS_DIR / "exp8_failure_gallery.csv", index=False)
    print("Hard-case accuracy by prompt style (hard vs easy vs overall):")
    print(summary.to_string(index=False))
    print(f"\nFailure gallery ({len(gallery)} hard CoT errors) -> exp8_failure_gallery.csv")


if __name__ == "__main__":
    main()
