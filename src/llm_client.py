"""Library module (imported, not run directly) — Groq LLM client.

Provides API-key rotation, on-disk caching, and JSON parsing.

- Key rotation: cycles through all keys in GROQ_API_KEYS. On a rate-limit (429)
  or transient error it advances to the next key and retries, so a batch of
  600+ calls survives per-key free-tier limits.
- Caching: every (model, prompt) result is cached to cache/ as JSON keyed by a
  hash, so re-running an experiment costs nothing and is fully reproducible.
- JSON parsing: extracts the first {...} block from the model's reply and
  validates it; returns a structured dict the experiments can trust.
"""
import json
import time
import hashlib
import re
import threading
from pathlib import Path

from groq import Groq

from config import CACHE_DIR, GROQ_MODEL, get_api_keys

_KEYS = get_api_keys()
if not _KEYS:
    raise RuntimeError("No GROQ_API_KEYS found in .env")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class GroqRotator:
    def __init__(self, model: str = GROQ_MODEL, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.keys = _KEYS
        self.idx = 0
        self._lock = threading.Lock()
        self._clients = {k: Groq(api_key=k) for k in self.keys}

    def _take_key(self):
        """Atomically grab the next live key (round-robin) to spread load evenly
        across all keys, keeping each under its per-minute limit."""
        with self._lock:
            if not self.keys:
                raise RuntimeError("No live Groq keys remaining.")
            self.idx %= len(self.keys)
            key = self.keys[self.idx]
            self.idx = (self.idx + 1) % len(self.keys)
        return key

    def _drop_key(self, key: str):
        """Permanently remove a restricted/invalid key from rotation."""
        with self._lock:
            if key in self.keys:
                self.keys.remove(key)
                print(f"[client] dropped dead key ...{key[-6:]} ({len(self.keys)} left)")

    def _cache_path(self, prompt: str) -> Path:
        h = hashlib.sha256(f"{self.model}|{self.temperature}|{prompt}".encode()).hexdigest()
        return CACHE_DIR / f"{h}.json"

    def complete(self, prompt: str, system: str | None = None,
                 max_retries: int = 6) -> str:
        """Return the raw text completion, using cache + key rotation."""
        cache_file = self._cache_path((system or "") + "\n####\n" + prompt)
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["text"]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err = None
        for attempt in range(max_retries):
            try:
                key = self._take_key()  # spread load across all keys
                resp = self._clients[key].chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=700,
                )
                text = resp.choices[0].message.content
                cache_file.write_text(json.dumps({"text": text}), encoding="utf-8")
                return text
            except Exception as e:  # rate limit / transient -> next iter takes a new key
                last_err = e
                msg = str(e).lower()
                # Permanent per-key failures: drop the key and retry immediately.
                if "organization_restricted" in msg or "restricted" in msg or \
                   "invalid_api_key" in msg or "invalid api key" in msg:
                    self._drop_key(key)
                    continue
                wait = min(2 ** attempt, 20) if ("429" in msg or "rate" in msg) else 1.5
                time.sleep(wait)
        raise RuntimeError(f"All retries failed. Last error: {last_err}")

    def complete_json(self, prompt: str, system: str | None = None) -> dict:
        """Return a parsed JSON dict from the model reply (robust to extra prose)."""
        raw = self.complete(prompt, system=system)
        return parse_json(raw)


def parse_json(raw: str) -> dict:
    """Extract and parse the first JSON object from model text. Never raises."""
    if not raw:
        return {"_parse_error": True, "_raw": raw}
    m = _JSON_RE.search(raw)
    candidate = m.group(0) if m else raw
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try to repair common issues: trailing commas, single quotes
        fixed = re.sub(r",\s*}", "}", candidate)
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return {"_parse_error": True, "_raw": raw}


if __name__ == "__main__":
    # Smoke test: one call, confirm key rotation + JSON parsing work.
    bot = GroqRotator()
    out = bot.complete_json(
        'Reply ONLY with JSON: {"ok": true, "model": "<your model>"}'
    )
    print("Smoke test result:", out)
    print("Keys available for rotation:", len(_KEYS))
