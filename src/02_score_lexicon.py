"""Stage 02 (Experiment 1) — Rule-based toxicity + sentiment scorer.

This is the "safety net" layer of the hybrid system and also the non-LLM
baseline (Experiment 1). It is fully offline and free, so it runs on the
WHOLE dataset as well as the eval sample.

It produces, per tweet:
  - tox_score        : 0..1 toxicity score from keyword/slur hits + sentiment
  - sentiment        : VADER compound (-1..1)
  - hit_terms        : which lexicon terms fired (for explainability)
  - rule_is_bully    : binary prediction (bully vs not) via a threshold
  - rule_category    : a weak guess of the category from which lexicon fired

The lexicons here are intentionally compact and transparent (the point of the
research is an explainable, low-compute net) and grouped by target category so
the rule layer can offer a weak category hint, not just a binary flag.

Outputs:
  results/predictions/exp1_rule_eval.csv  -> scored eval sample (consumed by stage 04)
  results/predictions/exp1_rule_full.csv  -> scored full dataset (free, all-data baseline)

Usage:  python 02_score_lexicon.py
"""
import sys

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import CLEAN_CSV, EVAL_CSV, PRED_DIR, LABELS

# --- Category-grouped lexicons (lowercase, matched as whole words/substrings) ---
# Compact and auditable on purpose. Extend freely; every term is human-readable.
LEXICONS = {
    "ethnicity": [
        "nigger", "nigga", "chink", "spic", "wetback", "gook", "paki",
        "raghead", "coon", "negro", "white trash", "go back to your country",
        "savage", "monkey", "your race", "ethnic",
    ],
    "religion": [
        "terrorist", "isis", "jihad", "muzzie", "raghead", "kike", "infidel",
        "islam is", "muslims are", "christians are", "jews are", "your religion",
        "allah", "burn the quran", "crusade",
    ],
    "gender": [
        "bitch", "slut", "whore", "cunt", "skank", "dyke", "faggot", "fag",
        "tranny", "women are", "men are", "feminazi", "rape", "sexist",
        "get back to the kitchen", "thot",
    ],
    "age": [
        "old hag", "boomer", "geezer", "senile", "too old", "too young",
        "immature kid", "grow up", "act your age", "old fart", "wrinkly",
    ],
    "other_cyberbullying": [
        "kill yourself", "kys", "loser", "ugly", "stupid", "idiot", "pathetic",
        "nobody likes you", "worthless", "freak", "go die", "shut up", "trash",
        "moron", "dumb",
    ],
}

# Strong terms that almost always indicate bullying regardless of category.
HARD_FLAGS = {"kill yourself", "kys", "go die", "nigger", "faggot", "cunt", "rape"}

analyzer = SentimentIntensityAnalyzer()


def score_text(text: str):
    """Return (tox_score, sentiment, hit_terms, rule_category)."""
    t = (text or "").lower()
    cat_hits = {cat: [] for cat in LEXICONS}
    all_hits = []
    for cat, terms in LEXICONS.items():
        for term in terms:
            if term in t:
                cat_hits[cat].append(term)
                all_hits.append(term)

    sentiment = analyzer.polarity_scores(t)["compound"]  # -1..1

    # Toxicity score: lexicon density + negative sentiment, plus hard-flag boost
    n_hits = len(all_hits)
    lex_component = min(n_hits / 3.0, 1.0)               # saturates at 3 hits
    neg_component = max(0.0, -sentiment)                  # only negative counts
    tox = 0.7 * lex_component + 0.3 * neg_component
    if any(h in HARD_FLAGS for h in all_hits):
        tox = max(tox, 0.9)
    tox = round(min(tox, 1.0), 3)

    # Weak category guess = category with most hits (ties -> first); else None
    rule_category = None
    best = 0
    for cat, hits in cat_hits.items():
        if len(hits) > best:
            best = len(hits)
            rule_category = cat

    return tox, round(sentiment, 3), "|".join(sorted(set(all_hits))), rule_category


def annotate(df: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    rows = df["clean_text"].fillna("").map(score_text)
    df = df.copy()
    df["tox_score"] = [r[0] for r in rows]
    df["sentiment"] = [r[1] for r in rows]
    df["hit_terms"] = [r[2] for r in rows]
    df["rule_category"] = [r[3] for r in rows]
    df["rule_is_bully"] = (df["tox_score"] >= threshold).astype(int)
    # Rule prediction in the 6-class space: category guess if bully, else not_cb
    df["rule_pred"] = df.apply(
        lambda r: (r["rule_category"] or "other_cyberbullying")
        if r["rule_is_bully"] else "not_cyberbullying",
        axis=1,
    )
    return df


def main() -> None:
    # Eval sample (used by hybrid fusion, Exp 7)
    eval_df = pd.read_csv(EVAL_CSV)
    eval_scored = annotate(eval_df)
    out_eval = PRED_DIR / "exp1_rule_eval.csv"
    eval_scored.to_csv(out_eval, index=False)
    print(f"Scored eval sample -> {out_eval}")

    # Full dataset (the free, all-data Experiment 1 baseline)
    full = pd.read_csv(CLEAN_CSV)
    full_scored = annotate(full)
    out_full = PRED_DIR / "exp1_rule_full.csv"
    full_scored[["tweet_text", "label", "tox_score", "sentiment",
                 "hit_terms", "rule_is_bully", "rule_pred"]].to_csv(out_full, index=False)
    print(f"Scored full dataset -> {out_full}")

    # Quick sanity: binary accuracy (bully vs not) on full data
    full_scored["gold_is_bully"] = (full_scored["label"] != "not_cyberbullying").astype(int)
    acc = (full_scored["rule_is_bully"] == full_scored["gold_is_bully"]).mean()
    print(f"\n[Exp1] Rule-based BINARY accuracy on full data: {acc:.3f}")


if __name__ == "__main__":
    sys.exit(main())
