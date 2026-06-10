"""Library module (imported, not run directly) — shared configuration.

Holds paths, the canonical label set, and run constants. Every stage script
imports from here so paths and the label list never drift between experiments.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of this src/ folder
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Directories ---
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"             # original, immutable input data
PROCESSED_DIR = DATA_DIR / "processed"  # generated/derived datasets
RESULTS_DIR = ROOT / "results"
PRED_DIR = RESULTS_DIR / "predictions"   # per-tweet model prediction CSVs (exp*_*.csv)
METRICS_DIR = RESULTS_DIR / "metrics"    # scores, sklearn reports, analysis tables
REPORTS_DIR = RESULTS_DIR / "reports"    # human-facing narrative reports (REPORT.md)
FIGURES_DIR = ROOT / "figures"
CACHE_DIR = ROOT / "cache"
for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, RESULTS_DIR, PRED_DIR, METRICS_DIR,
           REPORTS_DIR, FIGURES_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Files ---
RAW_CSV = RAW_DIR / "cyberbullying_tweets.csv"
CLEAN_CSV = PROCESSED_DIR / "clean_full.csv"
EVAL_CSV = PROCESSED_DIR / "eval_sample.csv"

# --- The 6 canonical labels (exactly as they appear in the dataset) ---
LABELS = [
    "religion",
    "age",
    "gender",
    "ethnicity",
    "other_cyberbullying",
    "not_cyberbullying",
]
# Human-readable category descriptions injected into prompts.
LABEL_DESCRIPTIONS = {
    "religion": "attacks or slurs targeting a person's religion or faith",
    "age": "bullying based on age (e.g. ageism, school/teen bullying about age)",
    "gender": "attacks based on gender, sexism, misogyny, or sexual orientation",
    "ethnicity": "racist or ethnic slurs and attacks on race/nationality",
    "other_cyberbullying": "harassment/bullying that is NOT specifically about religion, age, gender, or ethnicity",
    "not_cyberbullying": "no bullying or harassment; neutral, friendly, or merely negative-but-not-abusive",
}

# --- Run config (overridable via .env) ---
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
EVAL_PER_CLASS = int(os.getenv("EVAL_PER_CLASS", "100"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def get_api_keys() -> list[str]:
    """Return the Groq API keys for rotation.

    If cache/good_keys.txt exists (written by 00_check_api_keys.py), use only those
    validated keys so restricted/banned keys are never tried.
    """
    good = CACHE_DIR / "good_keys.txt"
    if good.exists():
        keys = [k.strip() for k in good.read_text(encoding="utf-8").split(",") if k.strip()]
        if keys:
            return keys
    raw = os.getenv("GROQ_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]
