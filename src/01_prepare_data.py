"""Stage 01 — Clean the raw tweets and build the stratified evaluation sample.

We deliberately do only LIGHT cleaning: the LLM benefits from seeing the tweet
nearly as a human would (hashtags, mentions, casing all carry signal). We keep
two text columns:
  - tweet_text : original, fed to the LLM
  - clean_text : lightly normalized, used by the rule-based lexicon scorer

Inputs:
  data/cyberbullying_tweets.csv  -> raw dataset
Outputs:
  data/clean_full.csv   -> full deduplicated dataset
  data/eval_sample.csv  -> stratified sample (EVAL_PER_CLASS per class) for LLM runs

Usage:  python 01_prepare_data.py
"""
import re
import sys
import html

import pandas as pd

from config import RAW_CSV, CLEAN_CSV, EVAL_CSV, LABELS, RANDOM_SEED, EVAL_PER_CLASS

URL_RE = re.compile(r"http\S+|www\.\S+")
WS_RE = re.compile(r"\s+")


def light_clean(text: str) -> str:
    """Normalization for the lexicon scorer (NOT used for the LLM input)."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)            # &#128049; -> emoji, &amp; -> &
    text = URL_RE.sub(" ", text)          # drop URLs
    text = re.sub(r"@\w+", " ", text)     # drop @mentions
    text = text.replace("#", " ")         # keep hashtag words, drop the '#'
    text = WS_RE.sub(" ", text).strip()
    return text


def main() -> None:
    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df):,} rows, columns = {list(df.columns)}")

    # Standardize column names
    df = df.rename(columns={"tweet_text": "tweet_text", "cyberbullying_type": "label"})
    df["label"] = df["label"].str.strip().str.lower()

    # Drop rows with unknown labels (defensive) and empty text
    before = len(df)
    df = df[df["label"].isin(LABELS)].copy()
    df["tweet_text"] = df["tweet_text"].astype(str).map(lambda t: html.unescape(t).strip())
    df = df[df["tweet_text"].str.len() > 0]
    print(f"Dropped {before - len(df):,} rows with bad label / empty text")

    # Remove exact-duplicate tweets (common in this dataset)
    before = len(df)
    df = df.drop_duplicates(subset=["tweet_text"]).reset_index(drop=True)
    print(f"Dropped {before - len(df):,} duplicate tweets")

    # Build the lexicon-friendly text
    df["clean_text"] = df["tweet_text"].map(light_clean)

    df.to_csv(CLEAN_CSV, index=False)
    print(f"\nSaved cleaned full dataset -> {CLEAN_CSV}  ({len(df):,} rows)")
    print("\nClass distribution (clean):")
    print(df["label"].value_counts())

    # --- Stratified evaluation sample: EVAL_PER_CLASS per class ---
    parts = []
    for lab in LABELS:
        pool = df[df["label"] == lab]
        n = min(EVAL_PER_CLASS, len(pool))
        parts.append(pool.sample(n=n, random_state=RANDOM_SEED))
    eval_df = pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    eval_df.insert(0, "id", range(len(eval_df)))
    eval_df.to_csv(EVAL_CSV, index=False)
    print(f"\nSaved evaluation sample -> {EVAL_CSV}  ({len(eval_df):,} rows)")
    print(eval_df["label"].value_counts())


if __name__ == "__main__":
    sys.exit(main())
