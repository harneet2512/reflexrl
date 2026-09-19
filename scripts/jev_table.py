"""Query Jev 1.13 once per perception class; cache the action distributions.

The OpenRouter key is read from the environment (OPENROUTER_API_KEY), never
written to disk.

    OPENROUTER_API_KEY=... python scripts/jev_table.py --out experiments/configs/jev_table_dtc.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reflexrl.env.scenarios import get_scenario  # noqa: E402

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
PERCEPTION = {
    "left": "The nearest monster is on the left side of the screen.",
    "center": "The nearest monster is in the center of the screen, straight ahead.",
    "right": "The nearest monster is on the right side of the screen.",
    "none": "No monster is visible on the screen.",
}
CRITERIA = {
    "TURN_LEFT": "turn left without shooting",
    "TURN_RIGHT": "turn right without shooting",
    "FIRE": "shoot straight ahead without turning",
    "TURN_LEFT_FIRE": "turn left while shooting",
    "TURN_RIGHT_FIRE": "turn right while shooting",
}


def ask(key: str, state: str) -> dict:
    body = {"model": MODEL, "state": state, "questions": {"action": {
        "type": "choice",
        "instructions": ("Which action should the player take right now to kill monsters "
                         "before they reach the player, without wasting limited ammo?"),
        "criteria": CRITERIA}}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    out["latency_s"] = time.perf_counter() - t0
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiments/configs/jev_table_dtc.json")
    args = p.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    spec = get_scenario("dtc")
    assert list(CRITERIA) == spec.action_names
    table, raw = {}, {}
    for cls, perception in PERCEPTION.items():
        state = (f"Doom, scenario '{spec.name}'. {spec.briefing} "
                 f"Perception of the current frame: {perception}")
        resp = ask(key, state)
        probs = resp["answers"]["action"]["probabilities"]
        table[cls] = [float(probs[a]) for a in spec.action_names]
        raw[cls] = {"state": state, "response": resp}
        print(cls, dict(zip(spec.action_names, table[cls])), f"{resp['latency_s']:.2f}s", flush=True)
    Path(args.out).write_text(json.dumps({"model": MODEL, "actions": spec.action_names,
                                          "table": table, "raw": raw}, indent=2))


if __name__ == "__main__":
    main()
