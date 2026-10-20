"""Spending guard: every model call is priced and counted, and calls stop at a daily cap.

Azure's own cost reports can lag by hours, so this guard is the first line of defence.
It works before money is spent: ``check()`` refuses a call once today's estimated spend
reaches the cap, and only a person raising ``MATCH_DAILY_CAP_USD`` lets calls continue.

Usage is stored in ``~/.match-intelligence/usage.json`` (one file per machine), so the
cap holds across scripts and restarts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Prices in US$ per 1M tokens (input, output). Deliberately rounded UP from current
# Azure listings, so the guard over-estimates spend rather than under-estimates it.
# Check https://azure.microsoft.com/pricing/details/azure-openai/ and update if they change.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-6-luna": (0.15, 0.75),
    "gpt-6-sol": (2.50, 12.00),
    "gpt-6-astra": (12.00, 60.00),
    "gpt-5-mini": (0.30, 2.50),
    "gpt-5-nano": (0.08, 0.50),
    "gpt-4.1-mini": (0.50, 2.00),
    "gpt-oss-120b": (0.20, 0.80),
}
UNKNOWN_MODEL_PRICE = (15.00, 75.00)  # an unknown model is priced as the most expensive, to stay safe
DEFAULT_DAILY_CAP_USD = 5.0


class BudgetExceeded(RuntimeError):
    """Raised before a call that would run past today's cap."""


def price_for(model: str) -> tuple[float, float]:
    """Match a deployment name like 'gpt-6-sol-1' or 'analyst-sol' to its model's price."""
    name = model.lower()
    for key in sorted(PRICES, key=len, reverse=True):
        if key in name:
            return PRICES[key]
    for alias, key in (("luna", "gpt-6-luna"), ("sol", "gpt-6-sol"), ("astra", "gpt-6-astra")):
        if alias in name:
            return PRICES[key]
    return UNKNOWN_MODEL_PRICE


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0


class BudgetGuard:
    def __init__(self, daily_cap_usd: float | None = None, path: str | Path | None = None):
        env_cap = os.environ.get("MATCH_DAILY_CAP_USD")
        self.cap = float(daily_cap_usd if daily_cap_usd is not None else env_cap or DEFAULT_DAILY_CAP_USD)
        self.path = Path(path or Path.home() / ".match-intelligence" / "usage.json")

    # -- storage ---------------------------------------------------------------------------
    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # -- the guard -------------------------------------------------------------------------
    def spent_today(self) -> float:
        day = self._load().get(date.today().isoformat(), {})
        return round(sum(m.get("usd", 0.0) for m in day.values()), 6)

    def remaining(self) -> float:
        return round(max(0.0, self.cap - self.spent_today()), 6)

    def check(self, model: str, expected_input_tokens: int = 0, expected_output_tokens: int = 0) -> None:
        """Call before every model request. Raises BudgetExceeded instead of spending past the cap."""
        p_in, p_out = price_for(model)
        expected = (expected_input_tokens * p_in + expected_output_tokens * p_out) / 1e6
        if self.spent_today() + expected >= self.cap:
            raise BudgetExceeded(
                f"Daily model budget reached: ${self.spent_today():.2f} of ${self.cap:.2f} spent today. "
                "No more calls will be made. To allow more, set MATCH_DAILY_CAP_USD higher (your decision)."
            )

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Call after every model response with its token counts. Returns the estimated cost."""
        p_in, p_out = price_for(model)
        usd = (input_tokens * p_in + output_tokens * p_out) / 1e6
        data = self._load()
        day = data.setdefault(date.today().isoformat(), {})
        m = day.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        m["calls"] += 1
        m["input_tokens"] += int(input_tokens)
        m["output_tokens"] += int(output_tokens)
        m["usd"] = round(m["usd"] + usd, 6)
        self._save(data)
        return usd

    def report(self) -> str:
        day = self._load().get(date.today().isoformat(), {})
        lines = [f"Today: ${self.spent_today():.4f} of ${self.cap:.2f} daily cap"]
        for model, m in sorted(day.items()):
            lines.append(
                f"  {model}: {m['calls']} calls, {m['input_tokens']:,} in / {m['output_tokens']:,} out, ${m['usd']:.4f}"
            )
        return "\n".join(lines)
