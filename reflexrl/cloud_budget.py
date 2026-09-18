"""Hard $20 spend guard for Modal runs.

The ledger lives on the reflexrl-runs volume at /runs/budget_ledger.json.
Every paid function calls ``spend_gate(est_usd, label)`` before doing
work: if recorded + estimated spend would exceed the cap, the run refuses
to start. On completion it calls ``record_spend(seconds, rate, label)``
so the ledger converges to *actual* wall-time cost, not estimates.

Defense-in-depth: the real hard stop is the workspace spend limit in the
Modal dashboard (Settings -> Billing). This ledger additionally makes
every run's cost auditable next to its artifacts, and prevents a single
run from silently burning the whole budget.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

CAP_USD = float(os.environ.get("REFLEXRL_CAP_USD", "10"))
LEDGER_PATH = "/runs/budget_ledger.json"

# USD per hour, rounded up from Modal list prices (mid-2025).
# "cpu" is per-vCPU-hour incl. memory overhead; container mem scales it.
RATES_PER_HR = {
    "L4": 0.81,
    "A10G": 1.11,
    "L40S": 1.96,
    "T4": 0.60,
    "cpu": 0.15,
}


def _read(path: str = LEDGER_PATH) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {"cap_usd": CAP_USD, "spent_usd": 0.0, "events": []}


def spent_usd(path: str = LEDGER_PATH) -> float:
    return float(_read(path)["spent_usd"])


def remaining_usd(path: str = LEDGER_PATH) -> float:
    return CAP_USD - spent_usd(path)


def spend_gate(est_usd: float, label: str, path: str = LEDGER_PATH) -> float:
    """Refuse to start if est + recorded spend would exceed the cap."""
    led = _read(path)
    proj = led["spent_usd"] + est_usd
    if proj > CAP_USD:
        raise RuntimeError(
            f"budget guard: '{label}' est ${est_usd:.2f} + spent "
            f"${led['spent_usd']:.2f} would exceed cap ${CAP_USD:.2f}. "
            f"Refusing to start."
        )
    return led["spent_usd"]


def record_spend(seconds: float, rate_per_hr: float, label: str,
                 path: str = LEDGER_PATH) -> float:
    """Append actual wall-time cost to the ledger."""
    led = _read(path)
    usd = seconds / 3600.0 * rate_per_hr
    led["spent_usd"] = round(led["spent_usd"] + usd, 4)
    led["events"].append({
        "t": time.time(), "label": label,
        "seconds": round(seconds, 1), "usd": round(usd, 4),
    })
    Path(path).write_text(json.dumps(led, indent=2))
    return usd


def estimate_train_usd(n_decisions: int, ms_per_decision: float = 90.0,
                       gpu: str = "T4", overhead: float = 1.4) -> float:
    """Pre-launch estimate: decisions x latency x rate + overhead margin."""
    return n_decisions * ms_per_decision / 1000 / 3600 * RATES_PER_HR[gpu] * overhead


class BudgetExhausted(RuntimeError):
    """Raised inside a running job when cumulative spend reaches the cap."""


class BudgetWatchdog:
    """In-run self-termination: call ``check()`` once per loop iteration.

    Tracks this run's own accrued cost (elapsed x rate) on top of the
    recorded ledger spend; when the sum reaches the cap it raises
    ``BudgetExhausted`` so the caller can save artifacts and wind down
    instead of burning past $20 on a bad estimate.
    """

    def __init__(self, rate_per_hr: float, label: str, path: str = LEDGER_PATH,
                 cap_usd: float = CAP_USD, reload=None):
        self.rate, self.label, self.path, self.cap = rate_per_hr, label, path, cap_usd
        self.reload = reload  # e.g. volume.reload — defeat mount caching
        self.t0 = time.time()
        self.base_spent = spent_usd(path)

    def accrued(self) -> float:
        """Current ledger spend + this run's wall-time cost so far.

        Re-reads the ledger each call so concurrent runs' spend counts too.
        """
        if self.reload is not None:
            try:
                self.reload()
            except Exception:
                pass
        return spent_usd(self.path) + (time.time() - self.t0) / 3600.0 * self.rate

    def check(self) -> float:
        usd = self.accrued()
        if usd >= self.cap:
            raise BudgetExhausted(
                f"budget watchdog: '{self.label}' accrued ${usd:.2f} >= cap "
                f"${self.cap:.2f} — winding down"
            )
        return usd
