# Cyberbullying Detection — Prompt Engineering + Rule-Based Hybrid model


## 📖 Overview
This repository contains the research framework, datasets, and experimental configurations for **Generative AI-driven cyberbullying detection**.  
The study benchmarks multiple approaches — including zero-shot, few-shot, persona prompting, chain-of-thought reasoning, and hybrid rule+reasoning systems — to evaluate their effectiveness in detecting harmful online behavior. Study of cyberbullying detection on Twitter data. It compares prompt-engineering styles on a Groq-hosted LLM against a transparent rule-based lexicon baseline, then fuses the two into a hybrid "reasoning + rules" classifier.

---
## Project layout
├── data/
│   ├── raw/                       # original, immutable input
│   │   └── cyberbullying_tweets.csv
│   └── processed/                 # generated (git-ignored)
│       ├── clean_full.csv             # deduplicated, cleaned full set
│       └── eval_sample.csv            # stratified eval sample (EVAL_PER_CLASS/class)
├── src/                       # all source code (run scripts from here)
│   │  # Library modules (imported, never run directly) — no number prefix
│   ├── config.py                  # shared paths, label set, run constants, key loader
│   ├── llm_client.py              # Groq client: key rotation + disk cache + JSON parsing
│   ├── prompts.py                 # the 5 prompt styles (zero/few-shot, CoT, persona, few+CoT)
│   │  # Pipeline stages (run in order) — NN_verb_noun.py
│   ├── 00_check_api_keys.py       # (optional) probe each key -> cache/good_keys.txt
│   ├── 01_prepare_data.py         # Stage 01: clean raw tweets -> processed/
│   ├── 02_score_lexicon.py        # Stage 02 / Exp 1: rule-based toxicity/sentiment scorer
│   ├── 03_run_prompts.py          # Stage 03 / Exp 2-6: run a prompt style over the eval sample
│   ├── 04_fuse_hybrid.py          # Stage 04 / Exp 7: fuse best LLM prediction with the rule layer
│   ├── 05_evaluate_metrics.py     # Stage 05: metrics, confusion matrices, summary_metrics.csv
│   ├── 06_analyze_hard_cases.py   # Stage 06 / Exp 8: robustness on hard cases (sarcasm/coded)
│   ├── 07_judge_explanations.py   # Stage 07 / Exp 9: LLM-as-judge explanation quality
│   ├── 08_build_report.py         # Stage 08 / Exp 10: roll-up tables + results/REPORT.md
│   └── 09_make_figures.py         # Stage 09: thesis figure factory (Chapter 5) -> figures/
├── tests/
│   ├── conftest.py            # puts src/ on the import path
│   └── test_smoke.py          # fast offline sanity tests (pytest -q)
├── results/                   # generated outputs, split by artifact type
│   ├── predictions/               # per-tweet model prediction CSVs (exp*_*.csv)
│   ├── metrics/                   # summary_metrics.csv, report_*.txt, exp8/exp9 tables
│   └── reports/                   # REPORT.md (human-facing synthesis)
├── figures/                   # generated confusion matrices + comparison charts
├── cache/                     # generated: on-disk LLM response cache (+ good_keys.txt)
├── docs/                      # project docs (extend as needed)
├── requirements.txt
├── .env.example               # template — copy to .env and fill in
└── .gitignore

## 🧩 Research Objectives
- Develop a **hybrid architecture** combining rule-based systems with LLM reasoning.
- Benchmark **accuracy, F1 scores, recall, and parse validity** across experimental setups.
- Provide **transparent, reproducible methodology** for academic and stakeholder review.
- Contribute to **explainable AI** in online harm detection.

---

## ⚙️ Experimental Configurations
Seven experiments were conducted:

1. **Rule-based (no LLM)**  
2. **Zero-shot prompting**  
3. **Few-shot prompting**  
4. **Chain-of-Thought reasoning**  
5. **Persona prompting**  
6. **Few-shot + Chain-of-Thought**  
7. **Hybrid (Reasoning + Rules)**  

Each configuration is documented in **Appendix C** with prompt templates and screenshots.

---

## 📊 Results Summary
| Experiment | Accuracy | Macro-F1 | Weighted-F1 | Binary Recall | Parse-OK |
|------------|----------|----------|-------------|---------------|----------|
| Hybrid (Reasoning+Rules) | 0.443 | 0.404 | 0.404 | 0.878 | ✅ |
| Few-shot | 0.365 | 0.378 | 0.378 | 0.838 | ✅ |
| Rule-based | 0.417 | 0.371 | 0.371 | 0.396 | ❌ |
| Persona | 0.400 | 0.364 | 0.364 | 0.878 | ✅ |
| Zero-shot | 0.383 | 0.363 | 0.363 | 0.838 | ✅ |
| Chain-of-Thought | 0.353 | 0.333 | 0.333 | 0.868 | ✅ |
| Few-shot + CoT | 0.322 | 0.327 | 0.327 | 0.872 | ✅ |

---
├── data/                  # Datasets used in experiments
├── prompts/               # Prompt templates (Exp 1–7)
├── configs/               # Configuration files and screenshots
├── results/               # Evaluation metrics and tables
├── appendix/              # Appendix C materials
└── README.md              # Project overview


## Setup

```powershell
# 1. Create / activate a virtual environment (a .venv is already present here)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets + run options
copy .env.example .env              # then edit .env and add your GROQ_API_KEYS
```

Required `.env` variables (see `.env.example`): `GROQ_API_KEYS` (comma-separated),
`GROQ_MODEL`, `EVAL_PER_CLASS`, `RANDOM_SEED`.

> **Security:** `.env` is git-ignored — never commit real keys. If a key is ever
> exposed, revoke and regenerate it in the Groq console immediately.

## Run the pipeline

> Run scripts **from inside `src/`** — they use flat imports (`from config import ...`).
> Numbered prefixes indicate run order; the three unnumbered files are imported libraries.

```powershell
cd src

py -3 00_check_api_keys.py     # (optional) validate keys -> cache/good_keys.txt
py -3 01_prepare_data.py       # Stage 01 — clean data + build the stratified eval sample
py -3 02_score_lexicon.py      # Stage 02 / Exp 1 — rule-based baseline (offline, free)
py -3 03_run_prompts.py all    # Stage 03 / Exp 2-6 — prompt benchmark (API; cached)
py -3 04_fuse_hybrid.py        # Stage 04 / Exp 7 — hybrid fusion (base = exp6_few_shot_cot)
py -3 05_evaluate_metrics.py   # Stage 05 — score everything -> summary_metrics.csv + figures/
py -3 06_analyze_hard_cases.py # Stage 06 / Exp 8 — hard-case robustness (reuses Exp 2-6, no API)
py -3 07_judge_explanations.py # Stage 07 / Exp 9 — explanation-quality judging (API)
py -3 08_build_report.py       # Stage 08 / Exp 10 — final synthesis -> results/REPORT.md
py -3 09_make_figures.py       # Stage 09 — render all Chapter-5 figures (offline) -> figures/
```

### Chapter-5 figures

`09_make_figures.py` writes a full battery of publication-quality figures (all
prefixed `ch5_`) into `figures/` — overall metric comparisons, per-class F1
heatmap, prompt-style ablation, normalised confusion matrices, lexicon/threshold
analysis, hard-case robustness, and the hybrid-vs-best-LLM comparison. It is
offline and idempotent: re-run any time after `05_evaluate_metrics.py`.

### Dependency order (what reads what)

```
01_prepare_data ──> data/processed/eval_sample.csv ─┬─> 02_score_lexicon ─> results/predictions/exp1_rule_eval.csv
                                                    │
                                                    └─> 03_run_prompts ──> results/predictions/exp{2..6}_*.csv
                                                                              │            │
exp1_rule_eval + exp5_persona ──> 04_fuse_hybrid ──> results/predictions/exp7_hybrid.csv
                                                                              │
   results/predictions/exp{1..7}_*.csv ──> 05_evaluate_metrics ──> results/metrics/summary_metrics.csv, figures/
                                                                              │
       metrics/ (summary + 06_hard_cases + 07_explanations) ──> 08_build_report ──> results/reports/REPORT.md
```

Full reproduction: **00 → 01 → 02 → 03 (all) → 04 → 05 → 06 → 07 → 08**.

## Tests

```powershell
pip install pytest
pytest -q
```

## Notes & reproducibility

- **Caching:** every `(model, temperature, prompt)` response is cached under `cache/`
  as JSON, so re-running an experiment is free and deterministic. Delete a cache
  file to force a fresh call.
- **Key rotation:** `llm_client.py` round-robins across all keys and drops
  restricted/invalid ones, so a 600+ call batch survives free-tier limits.
- **Determinism:** all sampling uses `RANDOM_SEED`; LLM calls use `temperature=0.0`.
- **Console encoding (Windows):** if `08_build_report.py` prints a `�` for the em
  dash, that's just the cp1252 terminal — `results/REPORT.md` is written as UTF-8
  and is correct. To make the console match: `chcp 65001`.


---

## 🚀 Getting Started
### Prerequisites
- Python 3.9+
- Required libraries: `transformers`, `scikit-learn`, `pandas`, `matplotlib`
- Access to LLM API (e.g., OpenAI, Azure OpenAI)

### Installation
```bash
git clone https://github.com/GandaNivedita/cyberbullying-genai.git
cd cyberbullying-genai
pip install -r requirements.txt


python run_experiment.py --config configs/exp7_hybrid.json


📌 Key Contributions
First comparative study of prompt engineering strategies for cyberbullying detection.

Demonstrates that Hybrid (Reasoning+Rules) achieves the best balance of accuracy and recall.

Provides reproducible templates and configurations for academic validation.




---

This README is publication-ready** and aligns with academic + GitHub standards.  

Would you like me to also prepare a **shorter “executive summary” version** of the README (1–2 pages) that you can share with stakeholders or interview panels, so they don’t have to read the full technical details?




## 📂 Repository Structure
