"""Momentum: who is generating danger right now.

Each team accumulates *threat*: the expected-threat gained by completed passes
and carries, plus the xG of every shot and a little for corners. Threat decays
with a half-life of a few minutes, so the index reflects the current spell of
play. ``momentum`` runs from -1 (away on top) to +1 (home on top).
"""

from __future__ import annotations

import math

from ..pitch import threat
from .base import Metric, register

HALF_LIFE_MIN = 4.0
SHIFT_THRESHOLD = 0.35


@register
class Momentum(Metric):
    name = "momentum"
    order = 50

    def __init__(self, analyzer):
        super().__init__(analyzer)
        self.level = {self.ctx.home_id: 0.0, self.ctx.away_id: 0.0}
        self._at = 0.0
        self.series: list[dict] = []
        self._last_minute = -1
        self._shift_at = -99.0
        self._extreme: tuple[float, float] = (0.0, 0.0)  # (value, minute) of the last strong spell

    def _decay_to(self, minute: float) -> None:
        if minute > self._at:
            k = math.pow(0.5, (minute - self._at) / HALF_LIFE_MIN)
            for tid in self.level:
                self.level[tid] *= k
            self._at = minute

    def index(self) -> float:
        h, a = self.level[self.ctx.home_id], self.level[self.ctx.away_id]
        return round((h - a) / (h + a + 0.03), 3)

    def on_event(self, e: dict) -> None:
        ctx = self.ctx
        minute = (e["period"] - 1) * 45.0 + e["t"] / 60.0
        self._decay_to(minute)
        tid = e["team_id"]
        gain = 0.0
        if e["type"] in ("pass", "carry") and tid and e["location"]:
            d = e["details"]
            if e["type"] == "carry" or d.get("outcome") == "complete":
                x0, y0 = ctx.tf(e["location"], tid, e["period"])
                x1, y1 = ctx.tf(d["end_location"], tid, e["period"])
                gain = max(0.0, threat(x1, y1) - threat(x0, y0))
            if e["type"] == "pass" and d.get("restart_type") == "corner":
                gain += 0.02
        elif e["type"] == "shot" and tid:
            gain = ctx.ann.get(e["event_id"], {}).get("xg", 0.05)
        if gain > 0 and tid:
            self.level[tid] += gain
        idx = self.index()
        whole = int(minute)
        if whole != self._last_minute:
            self._last_minute = whole
            self.series.append(
                {"minute": whole, "momentum": idx, **{k: round(v, 4) for k, v in self.level.items()}}
            )
        self._check_shift(idx, minute, e)

    def _check_shift(self, idx: float, minute: float, e: dict) -> None:
        prev_val, prev_min = self._extreme
        if abs(idx) >= SHIFT_THRESHOLD:
            flipped = prev_val * idx < 0 and minute - prev_min <= 10
            if flipped and minute - self._shift_at >= 8:
                self._shift_at = minute
                team = self.ctx.home_id if idx > 0 else self.ctx.away_id
                self.signal(
                    "momentum_shift",
                    team_id=team,
                    momentum_before=prev_val,
                    momentum_after=idx,
                    period=e["period"],
                    t=e["t"],
                )
            if abs(idx) >= abs(prev_val) or prev_val * idx < 0 or minute - prev_min > 10:
                self._extreme = (idx, minute)

    def snapshot(self) -> dict:
        return {
            "momentum": self.index(),
            "threat_level": {k: round(v, 4) for k, v in self.level.items()},
            "series": self.series,
            "notes": "+1 = home team fully on top, -1 = away team fully on top; threat half-life 4 minutes",
        }
