"""Control vs. chaos, and game rhythm.

Over a rolling 5-minute window of ball-in-play events:

* **chaos index (0-100)**: turnovers, duels and loose-ball scrambles per minute,
  scaled against reference levels learned from simulated matches.
* **team control (0-100)**: how calmly a team keeps the ball: completed passes
  in longer sequences, minus losses.
* **rhythm**: tempo (passes per minute) and stoppages per 10 minutes.
"""

from __future__ import annotations

import json
from collections import deque

from ..models import WEIGHTS_DIR
from .base import Metric, register

WINDOW_MIN = 5.0
CHAOTIC = {
    "possession_change",
    "tackle",
    "aerial_duel",
    "dribble",
    "ball_recovery",
    "interception",
    "block",
    "miscontrol",
    "clearance",
}
STOPPAGES = {"ball_out", "foul", "offside", "substitution", "goal", "card"}

DEFAULT_REFERENCE = {"chaos_rate_p10": 3.0, "chaos_rate_p90": 7.5}


def load_reference() -> dict:
    path = WEIGHTS_DIR / "control_reference.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(DEFAULT_REFERENCE)


@register
class Control(Metric):
    name = "control"
    order = 60

    def __init__(self, analyzer):
        super().__init__(analyzer)
        self.ref = load_reference()
        self.window: deque[tuple[float, str, str | None, bool]] = deque()
        self.series: list[dict] = []
        self._last_minute = -1
        self._spike_at = -99.0
        self.stoppages = 0
        self.passes_total = 0

    def _now(self, e: dict) -> float:
        return (e["period"] - 1) * 45.0 + e["t"] / 60.0

    def on_event(self, e: dict) -> None:
        now = self._now(e)
        t = e["type"]
        completed = t == "pass" and e["details"].get("outcome") == "complete"
        self.window.append((now, t, e["team_id"], completed))
        while self.window and self.window[0][0] < now - WINDOW_MIN:
            self.window.popleft()
        if t in STOPPAGES:
            self.stoppages += 1
        if t == "pass":
            self.passes_total += 1
        whole = int(now)
        if whole != self._last_minute:
            self._last_minute = whole
            snap = self.current()
            self.series.append({"minute": whole, **snap})
            if snap["chaos_index"] >= 88 and now - self._spike_at >= 10 and now > 6:
                self._spike_at = now
                self.signal("chaos_spike", chaos_index=snap["chaos_index"], period=e["period"], t=e["t"])

    def chaos_rate(self) -> float:
        span = (
            min(WINDOW_MIN, max(self.window[-1][0] - self.window[0][0], 1.0)) if self.window else WINDOW_MIN
        )
        return sum(1 for w in self.window if w[1] in CHAOTIC) / span

    def current(self) -> dict:
        lo, hi = self.ref["chaos_rate_p10"], self.ref["chaos_rate_p90"]
        rate = self.chaos_rate()
        chaos = max(0.0, min(100.0, 10 + 80 * (rate - lo) / max(hi - lo, 1e-6)))
        span = (
            min(WINDOW_MIN, max(self.window[-1][0] - self.window[0][0], 1.0)) if self.window else WINDOW_MIN
        )
        passes = [w for w in self.window if w[1] == "pass"]
        tempo = len(passes) / span
        control = {}
        for tid in (self.ctx.home_id, self.ctx.away_id):
            comp = sum(1 for w in passes if w[2] == tid and w[3])
            lost = sum(1 for w in self.window if w[1] == "possession_change" and w[2] != tid)
            total_comp = sum(1 for w in passes if w[3]) or 1
            control[tid] = round(max(0.0, min(100.0, 100 * comp / total_comp - 4 * lost)), 1)
        return {"chaos_index": round(chaos, 1), "tempo_passes_per_min": round(tempo, 1), "control": control}

    def snapshot(self) -> dict:
        cur = self.current()
        minutes = max(self.ctx.match_minute_float(), 1.0)
        label = (
            "chaotic"
            if cur["chaos_index"] >= 70
            else ("controlled" if cur["chaos_index"] <= 35 else "balanced")
        )
        return {
            **cur,
            "state": label,
            "stoppages_per_10_min": round(self.stoppages / minutes * 10, 1),
            "series": self.series,
        }
