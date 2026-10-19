"""Physical metrics from tracking frames: distance, high-speed running, sprints, top speed.

Also drives the rules' "player identification" overlays: speed and distance
indicators raised when predefined thresholds are reached.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .base import Metric, register

HSR_MS = 5.5  # high-speed running threshold, 19.8 km/h
SPRINT_MS = 7.0  # sprint threshold, 25.2 km/h
SPRINT_MIN_S = 1.0
SPEED_ALERT_KMH = 33.0  # overlay: "player hits 33+ km/h"
DISTANCE_MILESTONES_KM = (5, 8, 10, 11, 12)


@register
class Physical(Metric):
    name = "physical"
    order = 5

    def __init__(self, analyzer):
        super().__init__(analyzer)
        self.dist = defaultdict(float)
        self.hsr = defaultdict(float)
        self.sprints = defaultdict(int)
        self.top = defaultdict(float)
        self.now = {}
        self._last: dict[str, tuple[int, float, float, float]] = {}
        self._above = defaultdict(float)
        self._counted: set[str] = set()
        self.match_top = (0.0, None, None)  # (kmh, player_id, period/t)
        self.ball_top = (0.0, None)
        self._last_alert: dict[str, float] = {}
        self._milestones: set[tuple[str, int]] = set()
        self._next_leader_check = 60.0

    def on_frame(self, f: dict) -> None:
        period, t = f["period"], f["t"]
        clock_s = (period - 1) * 2700 + t
        for pid, x, y, v in f["players"]:
            last = self._last.get(pid)
            if last and last[0] == period:
                dt = t - last[1]
                if 0 < dt <= 1.5:
                    step = math.hypot(x - last[2], y - last[3])
                    if step / dt < 12.5:  # ignore teleports (restarts, substitutions)
                        self.dist[pid] += step
                        if v > HSR_MS:
                            self.hsr[pid] += step
                    if v > SPRINT_MS:
                        self._above[pid] += dt
                        if self._above[pid] >= SPRINT_MIN_S and pid not in self._counted:
                            self.sprints[pid] += 1
                            self._counted.add(pid)
                    else:
                        self._above[pid] = 0.0
                        self._counted.discard(pid)
            self._last[pid] = (period, t, x, y)
            kmh = v * 3.6
            self.now[pid] = kmh
            if kmh > self.top[pid]:
                self.top[pid] = kmh
            if kmh >= SPEED_ALERT_KMH and clock_s - self._last_alert.get(pid, -1e9) > 1200:
                self._last_alert[pid] = clock_s
                record = kmh > self.match_top[0] + 0.2
                self.signal(
                    "speed_alert", player_id=pid, speed_kmh=round(kmh, 1), record=record, period=period, t=t
                )
            if kmh > self.match_top[0]:
                self.match_top = (kmh, pid, (period, t))
        ball = f.get("ball", {})
        bkmh = ball.get("speed_ms", 0.0) * 3.6
        if ball.get("state") == "flight" and bkmh > self.ball_top[0]:
            self.ball_top = (bkmh, (period, t))
        if clock_s >= self._next_leader_check:
            self._next_leader_check = clock_s + 60.0
            self._check_milestones(period, t)

    def _check_milestones(self, period: int, t: float) -> None:
        if not self.dist:
            return
        leader = max(self.dist, key=self.dist.get)
        km = self.dist[leader] / 1000
        for m in DISTANCE_MILESTONES_KM:
            if km >= m and (leader, m) not in self._milestones and all(k[1] != m for k in self._milestones):
                self._milestones.add((leader, m))
                self.signal(
                    "distance_milestone", player_id=leader, km=m, distance_km=round(km, 2), period=period, t=t
                )

    def player(self, pid: str) -> dict:
        return {
            "distance_km": round(self.dist[pid] / 1000, 2),
            "high_speed_m": round(self.hsr[pid]),
            "sprints": self.sprints[pid],
            "top_speed_kmh": round(self.top[pid], 1),
            "speed_kmh_now": round(self.now.get(pid, 0.0), 1),
        }

    def snapshot(self) -> dict:
        if not self.ctx.has_frames:
            return {"available": False}
        teams = {}
        for tid in (self.ctx.home_id, self.ctx.away_id):
            pids = [p for p in self.dist if self.ctx.team_of(p) == tid]
            teams[tid] = {
                "distance_km": round(sum(self.dist[p] for p in pids) / 1000, 2),
                "high_speed_km": round(sum(self.hsr[p] for p in pids) / 1000, 2),
                "sprints": sum(self.sprints[p] for p in pids),
            }
        top_dist = sorted(self.dist, key=self.dist.get, reverse=True)[:5]
        top_speed = sorted(self.top, key=self.top.get, reverse=True)[:5]
        return {
            "available": True,
            "teams": teams,
            "players": {p: self.player(p) for p in self.dist},
            "leaders": {
                "distance": [
                    {"player_id": p, "distance_km": round(self.dist[p] / 1000, 2)} for p in top_dist
                ],
                "top_speed": [{"player_id": p, "top_speed_kmh": round(self.top[p], 1)} for p in top_speed],
            },
            "records": {
                "fastest_player_kmh": round(self.match_top[0], 1),
                "fastest_player_id": self.match_top[1],
                "fastest_ball_kmh": round(self.ball_top[0], 1),
            },
        }
