"""Stage 03 (Experiments 2-6) — Run the LLM prompt-style benchmark.

For each tweet it builds the prompt, calls Groq (cached + key-rotated), parses
the JSON, and saves per-tweet predictions to results/predictions/<expN>_<style>.csv with:
  id, tweet_text, label (gold), pred_label, target_group, intent, intensity,
  explanation, parse_ok

The five prompt styles map to experiments 2-6 (see prompts.STYLE_TO_EXP):
  zero_shot -> exp2, few_shot -> exp3, cot -> exp4, persona -> exp5,
  few_shot_cot -> exp6.

Usage:
    python 03_run_prompts.py zero_shot
    python 03_run_prompts.py all          # runs all five styles
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from config import EVAL_CSV, PRED_DIR, LABELS, GROQ_MODEL

# Concurrent workers: kept below the live-key count so each key stays under its
# per-minute limit (9 validated keys -> 8 workers).
N_WORKERS = 8
_BOT = None  # shared client, set by run_style (thread-safe key rotation inside)
from prompts import build_prompt, STYLES, STYLE_TO_EXP
from llm_client import GroqRotator


def _normalize_label(raw) -> str:
    """Map a model label string onto one of the 6 canonical labels."""
    if not isinstance(raw, str):
        return "parse_error"
    r = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if r in LABELS:
        return r
    # common variants
    aliases = {
        "not_cyberbullying": ["none", "no", "not_bullying", "neutral", "non_cyberbullying", "not_cyber_bullying"],
        "other_cyberbullying": ["other", "general", "other_cyber_bullying", "miscellaneous"],
        "gender": ["sex", "sexism", "sexual_orientation"],
        "ethnicity": ["race", "racism", "racial", "nationality"],
        "religion": ["religious", "faith"],
        "age": ["ageism"],
    }
    for canon, alts in aliases.items():
        if r == canon or r in alts:
            return canon
    return "parse_error"


def _classify_one(args):
    style, row = args
    system, user = build_prompt(style, row["tweet_text"])
    result = _BOT.complete_json(user, system=system)
    return {
        "id": row["id"],
        "tweet_text": row["tweet_text"],
        "label": row["label"],
        "pred_label": _normalize_label(result.get("label")),
        "target_group": result.get("target_group", ""),
        "intent": result.get("intent", ""),
        "intensity": result.get("intensity", ""),
        "explanation": result.get("explanation", ""),
        "parse_ok": not result.get("_parse_error", False),
    }


def run_style(style: str, bot: GroqRotator) -> pd.DataFrame:
    global _BOT
    _BOT = bot
    df = pd.read_csv(EVAL_CSV)
    tasks = [(style, row) for _, row in df.iterrows()]
    records = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        for rec in tqdm(ex.map(_classify_one, tasks), total=len(tasks), desc=style):
            records.append(rec)
    out = pd.DataFrame(records).sort_values("id").reset_index(drop=True)
    exp = STYLE_TO_EXP[style]
    path = PRED_DIR / f"{exp}_{style}.csv"
    out.to_csv(path, index=False)
    acc = (out["pred_label"] == out["label"]).mean()
    parse_rate = out["parse_ok"].mean()
    print(f"\n[{exp}/{style}] acc={acc:.3f}  parse_ok={parse_rate:.3f}  -> {path}")
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python 03_run_prompts.py <style|all>")
        print("styles:", STYLES)
        return
    target = sys.argv[1]
    print(f"Model: {GROQ_MODEL}")
    bot = GroqRotator()
    styles = STYLES if target == "all" else [target]
    for style in styles:
        run_style(style, bot)


if __name__ == "__main__":
    main()
