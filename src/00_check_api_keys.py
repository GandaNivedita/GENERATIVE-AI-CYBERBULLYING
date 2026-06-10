"""Stage 00 (preflight) — Health-check every Groq API key.

Probes every key in GROQ_API_KEYS and reports which are usable vs
restricted/invalid, then writes the working subset to cache/good_keys.txt so the
LLM stages (03, 07) automatically skip dead keys.

Usage:  python 00_check_api_keys.py
"""
from groq import Groq
from config import get_api_keys, GROQ_MODEL

good, bad = [], []
for i, k in enumerate(get_api_keys()):
    try:
        c = Groq(api_key=k)
        c.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1, temperature=0,
        )
        good.append(k)
        print(f"[{i:2d}] OK    ...{k[-6:]}")
    except Exception as e:
        msg = str(e)
        short = msg[:90]
        bad.append((k, short))
        print(f"[{i:2d}] DEAD  ...{k[-6:]}  -> {short}")

print(f"\nUsable keys: {len(good)} / {len(get_api_keys())}")
# Write the working keys out so the client can prefer them.
from config import ROOT
(ROOT / "cache" / "good_keys.txt").write_text(",".join(good), encoding="utf-8")
print("Saved working keys -> cache/good_keys.txt")
