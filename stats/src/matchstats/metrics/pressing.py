"""Pressing intensity: pressures by zone, PPDA, high turnovers and the minute-by-minute Pressure Pulse."""

from __future__ import annotations

from collections import defaultdict

from ..pitch import third
from .base import Metric, ratio, register

DEFENSIVE_ACTIONS = {"tackle", "interception", "foul", "block"}
SURGE_WINDOW_MIN = 5
SURGE_MIN_PRESSURES = 14
SURGE_FACTOR = 1.8


@register
class Pressing(Metric):
    name = "pressing"
    order = 40

    def __init__(self, analyzer):
        super().__init__(analyzer)
        ids = (self.ctx.home_id, self.ctx.away_id)
        self.pressures = dict.fromkeys(ids, 0)
        self.by_third = {tid: {"defensive": 0, "middle": 0, "attacking": 0} for tid in ids}
        self.per_minute = defaultdict(lambda: dict.fromkeys(ids, 0))
        self.opp_passes_in_build_up = dict.fromkeys(ids, 0)  # opponent passes in their own 60% (for our PPDA)
        self.def_actions_high = dict.fromkeys(ids, 0)
        self.high_turnovers = dict.fromkeys(ids, 0)
        self._surge_at = dict.fromkeys(ids, -99)

    def _minute(self, e: dict) -> int:
        return int((e["period"] - 1) * 45 + e["t"] / 60)

    def on_event(self, e: dict) -> None:
        ctx = self.ctx
        t, tid = e["type"], e["team_id"]
        if not tid or not e["location"]:
            return
        x, _ = ctx.tf(e["location"], tid, e["period"])
        if t == "pressure":
            self.pressures[tid] += 1
            self.by_third[tid][third(x)] += 1
            m = self._minute(e)
            self.per_minute[m][tid] += 1
            self._check_surge(tid, m, e)
        elif t == "pass" and x < 63:
            self.opp_passes_in_build_up[ctx.opponent(tid)] += 1
        if t in DEFENSIVE_ACTIONS and x > 42:
            self.def_actions_high[tid] += 1
        won_ball = t in ("interception", "ball_recovery") or (
            t == "tackle" and e["details"].get("outcome") == "won"
        )
        if won_ball and x >= 70:
            self.high_turnovers[tid] += 1
            self.signal(
                "high_turnover",
                team_id=tid,
                player_id=e["player_id"],
                event_id=e["event_id"],
                period=e["period"],
                t=e["t"],
            )

    def _check_surge(self, tid: str, minute: int, e: dict) -> None:
        if minute < 10 or minute - self._surge_at[tid] < 12:
            return
        recent = sum(self.per_minute[m][tid] for m in range(minute - SURGE_WINDOW_MIN + 1, minute + 1))
        played = max(minute + 1, 1)
        avg_window = self.pressures[tid] / played * SURGE_WINDOW_MIN
        if recent >= SURGE_MIN_PRESSURES and recent >= SURGE_FACTOR * avg_window:
            self._surge_at[tid] = minute
            self.signal(
                "pressing_surge",
                team_id=tid,
                pressures=recent,
                window_min=SURGE_WINDOW_MIN,
                usual=round(avg_window, 1),
                period=e["period"],
                t=e["t"],
            )

    def snapshot(self) -> dict:
        teams = {}
        for tid in self.pressures:
            teams[tid] = {
                "pressures": self.pressures[tid],
                "pressures_by_third": self.by_third[tid],
                "high_press_share": ratio(self.by_third[tid]["attacking"], self.pressures[tid]),
                "ppda": round(self.opp_passes_in_build_up[tid] / self.def_actions_high[tid], 2)
                if self.def_actions_high[tid]
                else None,
                "high_turnovers": self.high_turnovers[tid],
            }
        pulse = [{"minute": m, **v} for m, v in sorted(self.per_minute.items())]
        return {
            "teams": teams,
            "pressure_pulse": pulse,
            "notes": {
                "ppda": "opponent passes allowed per defensive action in their build-up zone; lower = more intense press"
            },
        }
