"""Win probability and Moment Weight.

Win probability comes from a model trained on simulated matches (score, time,
chance creation, red cards). **Moment Weight** is how many percentage points a
moment moved the acting team's chance of winning:

* goals and red cards: the realised swing;
* shots that did not go in: the *potential* swing had it gone in, times its xG
  (the "expected swing"), so a big miss carries weight even though the score
  did not change.
"""

from __future__ import annotations

from .base import Metric, register


@register
class WinProbability(Metric):
    name = "win_probability"
    order = 80

    def __init__(self, analyzer):
        super().__init__(analyzer)
        self.current = self._predict(0.0, 0, 0.0, 0)
        self.series: list[dict] = [{"minute": 0, **self._as_dict(self.current)}]
        self._last_minute = 0

    # (home, draw, away)
    def _predict(
        self, minute: float, goal_diff: int, xg_diff: float, red_diff: int
    ) -> tuple[float, float, float]:
        return self.models.win_probabilities(minute, goal_diff, xg_diff, red_diff)

    def _as_dict(self, p: tuple[float, float, float]) -> dict:
        return {"home": round(p[0], 4), "draw": round(p[1], 4), "away": round(p[2], 4)}

    def _state(self, minute: float, extra_home_goals: int = 0, extra_away_goals: int = 0) -> tuple:
        ctx = self.ctx
        shooting = self.analyzer.metric("shooting")
        xgd = shooting.xg_total(ctx.home_id) - shooting.xg_total(ctx.away_id)
        gd = ctx.score["home"] - ctx.score["away"] + extra_home_goals - extra_away_goals
        red = ctx.red_cards[ctx.home_id] - ctx.red_cards[ctx.away_id]
        return minute, gd, xgd, red

    def _team_prob(self, p: tuple[float, float, float], team_id: str) -> float:
        return p[0] if team_id == self.ctx.home_id else p[2]

    def on_event(self, e: dict) -> None:
        ctx = self.ctx
        minute = (e["period"] - 1) * 45.0 + min(e["t"], 2700.0) / 60.0
        if e["type"] == "period_end" and e["period"] == 2:
            minute = 95.0
        before = self.current
        after = self._predict(*self._state(minute))
        ann = ctx.ann.setdefault(e["event_id"], {})
        tid = e["team_id"]
        if e["type"] == "shot" and tid:
            d = e["details"]
            xg = ann.get("xg", 0.0)
            is_home = tid == ctx.home_id
            if_goal = self._predict(*self._state(minute, int(is_home), int(not is_home)))
            potential = self._team_prob(if_goal, tid) - self._team_prob(after, tid)
            ann["potential_swing_pts"] = round(100 * potential, 1)
            ann["expected_swing_pts"] = round(100 * potential * xg, 1)
            if d.get("outcome") != "goal":
                ann["moment_weight"] = ann["expected_swing_pts"]
        if e["type"] in ("goal", "card") and tid:
            # Goal: the score in the event already includes it, so compare with the pre-goal state.
            ann["win_prob_before"] = self._as_dict(before)
            ann["win_prob_after"] = self._as_dict(after)
            swing = self._team_prob(after, tid) - self._team_prob(before, tid)
            ann["moment_weight"] = round(100 * abs(swing), 1)
        if e["type"] == "match_end":
            h, a = ctx.score["home"], ctx.score["away"]
            after = (1.0, 0.0, 0.0) if h > a else ((0.0, 0.0, 1.0) if a > h else (0.0, 1.0, 0.0))
        self.current = after
        whole = int(minute)
        if whole != self._last_minute or e["type"] in ("goal", "match_end"):
            self._last_minute = whole
            self.series.append(
                {
                    "minute": whole,
                    **self._as_dict(after),
                    "event_id": e["event_id"] if e["type"] in ("goal", "card") else None,
                }
            )

    def snapshot(self) -> dict:
        return {
            "current": self._as_dict(self.current),
            "series": self.series,
            "model": "trained" if self.models.win_probability.trained else "fallback",
        }
