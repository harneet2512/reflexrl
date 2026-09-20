"""TypeSafe Jev 1.13: the decision half of the teacher.

Live calls go to OpenRouter's decisions endpoint (typed Choice question ->
choice + probabilities). Answers are cached per typed state, so a repeated
state costs nothing, and a cached table fitted earlier serves as the offline
fallback when no API key is available.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"


class JevClient:
    def __init__(self, table_path: str | Path, criteria: dict[str, str] | None = None,
                 api_key: str | None = None, timeout: float = 30.0):
        data = json.loads(Path(table_path).read_text())
        self.actions: list[str] = data["actions"]
        self.table = {k: np.asarray(v, np.float64) for k, v in data["table"].items()}
        self.criteria = criteria or {a: a.replace("_", " ").lower() for a in self.actions}
        self.key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.timeout = timeout
        self.cache: dict[str, np.ndarray] = {}
        self.calls = 0
        self.seconds = 0.0
        self.failures = 0

    @property
    def live(self) -> bool:
        return bool(self.key)

    def decide(self, state: str, cache_key: str | None = None) -> np.ndarray:
        """Typed state -> action distribution. Falls back to the cached table on failure."""
        key = cache_key or state
        if key in self.cache:
            return self.cache[key]
        if not self.live:
            raise RuntimeError("no OPENROUTER_API_KEY: use table_probs() instead")
        body = {"model": MODEL, "state": state, "questions": {"action": {
            "type": "choice",
            "instructions": ("Which action should the player take right now to kill monsters "
                             "before they reach the player, without wasting limited ammo?"),
            "criteria": self.criteria}}}
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.loads(r.read())
            probs = resp["answers"]["action"]["probabilities"]
            p = np.array([float(probs[a]) for a in self.actions], np.float64)
        except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as e:
            self.failures += 1
            print(f"[jev] call failed ({e}); using the cached table for this state", flush=True)
            raise
        finally:
            self.seconds += time.perf_counter() - t0
            self.calls += 1
        p = p / max(p.sum(), 1e-8)
        self.cache[key] = p
        return p

    def table_probs(self, classes: tuple[str, ...]) -> np.ndarray:
        """(C, A) matrix of the offline table, rows in `classes` order."""
        M = np.stack([self.table[c] for c in classes])
        return M / M.sum(1, keepdims=True)
