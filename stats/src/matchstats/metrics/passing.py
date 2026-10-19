"""Passing: volume, accuracy, difficulty (trained model), progression, lines broken, networks.

Difficulty is 0-10, defined as ``10 x (1 - predicted completion probability)``
from our pass-completion model, so a 7.5 means "this pass is completed only
about 25% of the time".
"""

from __future__ import annotations

from collections import defaultdict

from ..pitch import in_box
from .base import Metric, ratio, register, team_dict

HARD_PASS = 7.5  # difficulty at which a completed pass is noteworthy


def _counts():
    return {
        "passes": 0,
        "completed": 0,
        "progressive": 0,
        "passes_into_final_third": 0,
        "passes_into_box": 0,
        "crosses": 0,
        "crosses_completed": 0,
        "long_balls": 0,
        "long_balls_completed": 0,
        "through_balls": 0,
        "under_pressure": 0,
        "under_pressure_completed": 0,
        "key_passes": 0,
        "assists": 0,
        "lines_broken": 0,
        "length_sum": 0.0,
        "difficulty_sum": 0.0,
        "completed_difficulty_sum": 0.0,
        "fastest_kmh": 0.0,
    }


def _finish(c: dict) -> dict:
    out = {k: v for k, v in c.items() if not k.endswith("_sum")}
    out["accuracy"] = ratio(c["completed"], c["passes"])
    out["under_pressure_accuracy"] = ratio(c["under_pressure_completed"], c["under_pressure"])
    out["avg_length_m"] = round(c["length_sum"] / c["passes"], 1) if c["passes"] else 0.0
    out["avg_difficulty"] = round(c["difficulty_sum"] / c["passes"], 2) if c["passes"] else 0.0
    out["avg_completed_difficulty"] = (
        round(c["completed_difficulty_sum"] / c["completed"], 2) if c["completed"] else 0.0
    )
    out["fastest_kmh"] = round(c["fastest_kmh"], 1)
    return out


@register
class Passing(Metric):
    name = "passing"
    order = 20

    def __init__(self, analyzer):
        super().__init__(analyzer)
        self.teams = team_dict(self.ctx, _counts)
        self.players = defaultdict(_counts)
        self.edges = defaultdict(int)  # (passer, recipient) -> completed passes
        self.touch_pos = defaultdict(lambda: [0.0, 0.0, 0])  # player -> sum x, sum y, n (attacking frame)
        self.hardest: list[tuple[float, str]] = []
        self._last_completed: dict[
            str, tuple[str, str, float, int]
        ] = {}  # team -> (event_id, recipient, t, period)

    def on_event(self, e: dict) -> None:
        t = e["type"]
        if t == "pass":
            self._pass(e)
        elif t == "shot":
            lc = self._last_completed.get(e["team_id"])
            if lc and lc[1] == e["player_id"] and lc[3] == e["period"] and e["t"] - lc[2] < 12:
                passer = self.ctx.by_id[lc[0]]["player_id"]
                self.teams[e["team_id"]]["key_passes"] += 1
                self.players[passer]["key_passes"] += 1
                self.ctx.ann.setdefault(e["event_id"], {})["key_pass_event_id"] = lc[0]
                self.ctx.ann.setdefault(lc[0], {})["key_pass"] = True
        elif t == "goal":
            aid = e["details"].get("assist_id")
            if aid:
                self.teams[e["team_id"]]["assists"] += 1
                self.players[aid]["assists"] += 1
        elif t == "possession_change":
            self._last_completed.pop(e["details"].get("from_team_id"), None)

    def _pass(self, e: dict) -> None:
        ctx = self.ctx
        d = e["details"]
        tid = e["team_id"]
        pid = e["player_id"]
        x, y = ctx.tf(e["location"], tid, e["period"])
        ex, ey = ctx.tf(d["end_location"], tid, e["period"])
        p_complete = self.models.pass_completion(d, x, y, ex, ey)
        difficulty = round(10.0 * (1.0 - p_complete), 2)
        complete = d["outcome"] == "complete"
        progression = float(d.get("progression_m", ex - x))
        progressive = progression >= 15 or (progression >= 10 and ex >= 52.5)
        lines = self._lines_broken(e, x, ex) if complete and progression > 3 else 0
        ann = ctx.ann.setdefault(e["event_id"], {})
        ann.update(
            {
                "difficulty": difficulty,
                "p_complete": round(p_complete, 3),
                "progressive": progressive,
                "lines_broken": lines,
            }
        )
        for c in (self.teams[tid], self.players[pid]):
            c["passes"] += 1
            c["length_sum"] += float(d.get("length_m", 0.0))
            c["difficulty_sum"] += difficulty
            c["fastest_kmh"] = max(c["fastest_kmh"], float(d.get("release_speed_kmh", 0.0)))
            kind = d.get("pass_kind")
            if kind in ("cross", "corner", "free_kick_cross") and d.get("height") != "ground":
                c["crosses"] += 1
                c["crosses_completed"] += complete
            if kind in ("long", "switch") or float(d.get("length_m", 0)) >= 32:
                c["long_balls"] += 1
                c["long_balls_completed"] += complete
            if kind == "through":
                c["through_balls"] += 1
            if d.get("under_pressure"):
                c["under_pressure"] += 1
                c["under_pressure_completed"] += complete
            if complete:
                c["completed"] += 1
                c["completed_difficulty_sum"] += difficulty
                c["progressive"] += progressive
                c["passes_into_final_third"] += x < 70 <= ex
                c["passes_into_box"] += (not in_box(x, y)) and in_box(ex, ey)
                c["lines_broken"] += lines
        pos = self.touch_pos[pid]
        lx, ly = x, y  # each team's network is drawn in its own attacking direction
        pos[0] += lx
        pos[1] += ly
        pos[2] += 1
        if complete:
            rid = d.get("recipient_id") or d.get("completed_by")
            if rid:
                self.edges[(pid, rid)] += 1
                self._last_completed[tid] = (e["event_id"], rid, e["t"], e["period"])
            self.hardest.append((difficulty, e["event_id"]))
            self.hardest = sorted(self.hardest, reverse=True)[:8]
            if difficulty >= HARD_PASS and (progressive or lines >= 2 or in_box(ex, ey)):
                self.signal(
                    "hard_pass",
                    event_id=e["event_id"],
                    player_id=pid,
                    team_id=tid,
                    difficulty=difficulty,
                    lines_broken=lines,
                    period=e["period"],
                    t=e["t"],
                )

    def _lines_broken(self, e: dict, x0: float, x1: float) -> int:
        """Opponent outfielders the pass bypassed (between passer and receiver, in depth), from tracking."""
        frame = self.ctx.frame_at(e["period"], e["t"])
        if frame is None:
            return 0
        tid = e["team_id"]
        n = 0
        for pid, px, py, _v in frame["players"]:
            info = self.ctx.players.get(pid)
            if info is None or info.team_id == tid or info.role == "GK":
                continue
            ox, _ = self.ctx.tf({"x": px, "y": py}, tid, e["period"])
            if x0 < ox < x1:
                n += 1
        return n

    def network(self, team_id: str, min_passes: int = 3) -> dict:
        nodes = []
        for pid, (sx, sy, n) in self.touch_pos.items():
            if self.ctx.team_of(pid) == team_id and n:
                nodes.append({"player_id": pid, "x": round(sx / n, 1), "y": round(sy / n, 1), "passes": n})
        edges = [
            {"from": a, "to": b, "count": c}
            for (a, b), c in self.edges.items()
            if c >= min_passes and self.ctx.team_of(a) == team_id
        ]
        edges.sort(key=lambda x: -x["count"])
        return {
            "nodes": nodes,
            "edges": edges,
            "orientation": "drawn in the team's own attacking direction (towards x = 105)",
        }

    def snapshot(self) -> dict:
        return {
            "teams": {tid: _finish(c) for tid, c in self.teams.items()},
            "players": {pid: _finish(c) for pid, c in self.players.items()},
            "networks": {tid: self.network(tid) for tid in self.teams},
            "records": {
                "hardest_completed": [{"event_id": eid, "difficulty": dif} for dif, eid in self.hardest[:5]]
            },
        }
