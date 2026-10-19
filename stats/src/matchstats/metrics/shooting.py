"""Shots and expected goals (our trained xG model), shot speed, chance quality."""

from __future__ import annotations

from collections import defaultdict

from .base import Metric, ratio, register, team_dict

BIG_CHANCE_XG = 0.30


def _team():
    return {
        "shots": 0,
        "on_target": 0,
        "goals": 0,
        "xg": 0.0,
        "big_chances": 0,
        "blocked": 0,
        "off_target": 0,
        "headers": 0,
        "fastest_shot_kmh": 0.0,
        "speed_sum": 0.0,
    }


@register
class Shooting(Metric):
    name = "shooting"
    order = 10

    def __init__(self, analyzer):
        super().__init__(analyzer)
        self.teams = team_dict(self.ctx, _team)
        self.players = defaultdict(lambda: {"shots": 0, "on_target": 0, "goals": 0, "xg": 0.0})
        self.keepers = defaultdict(lambda: {"saves": 0, "xg_on_target_faced": 0.0, "goals_conceded": 0})
        self.timeline: list[dict] = []
        self.best_chance: tuple[float, str | None] = (0.0, None)
        self.fastest: tuple[float, str | None] = (0.0, None)

    def on_event(self, e: dict) -> None:
        if e["type"] != "shot":
            return
        ctx = self.ctx
        d = e["details"]
        tid = e["team_id"]
        x, y = ctx.tf(e["location"], tid, e["period"])
        xg = self.models.shot_xg(d, x, y)
        ctx.ann.setdefault(e["event_id"], {})["xg"] = round(xg, 4)
        t = self.teams[tid]
        p = self.players[e["player_id"]]
        t["shots"] += 1
        p["shots"] += 1
        t["xg"] += xg
        p["xg"] += xg
        speed = float(d.get("speed_kmh", 0.0))
        t["speed_sum"] += speed
        t["fastest_shot_kmh"] = max(t["fastest_shot_kmh"], speed)
        if d.get("body_part") == "head":
            t["headers"] += 1
        if xg >= BIG_CHANCE_XG:
            t["big_chances"] += 1
        outcome = d.get("outcome")
        if d.get("on_target"):
            t["on_target"] += 1
            p["on_target"] += 1
            gk = d.get("goalkeeper_id")
            if gk:
                self.keepers[gk]["xg_on_target_faced"] += xg
                if outcome == "saved":
                    self.keepers[gk]["saves"] += 1
                elif outcome == "goal":
                    self.keepers[gk]["goals_conceded"] += 1
        if outcome == "goal":
            t["goals"] += 1
            p["goals"] += 1
        elif outcome == "blocked":
            t["blocked"] += 1
        elif outcome in ("off_target", "post"):
            t["off_target"] += 1
        if xg > self.best_chance[0]:
            self.best_chance = (xg, e["event_id"])
        if speed > self.fastest[0]:
            self.fastest = (speed, e["event_id"])
        self.timeline.append(
            {
                "minute": e["minute"],
                "team_id": tid,
                "xg": round(xg, 3),
                "cumulative": round(t["xg"], 3),
                "outcome": outcome,
                "event_id": e["event_id"],
            }
        )

    def xg_total(self, team_id: str) -> float:
        return self.teams[team_id]["xg"]

    def snapshot(self) -> dict:
        teams = {}
        for tid, t in self.teams.items():
            teams[tid] = {
                k: (round(v, 3) if isinstance(v, float) else v) for k, v in t.items() if k != "speed_sum"
            }
            teams[tid]["xg_per_shot"] = ratio(t["xg"], t["shots"])
            teams[tid]["avg_shot_speed_kmh"] = round(t["speed_sum"] / t["shots"], 1) if t["shots"] else 0.0
        players = {pid: {**v, "xg": round(v["xg"], 3)} for pid, v in self.players.items()}
        keepers = {
            pid: {
                **v,
                "xg_on_target_faced": round(v["xg_on_target_faced"], 3),
                "goals_prevented": round(v["xg_on_target_faced"] - v["goals_conceded"], 3),
            }
            for pid, v in self.keepers.items()
        }
        return {
            "teams": teams,
            "players": players,
            "goalkeepers": keepers,
            "xg_timeline": self.timeline,
            "records": {
                "best_chance_xg": round(self.best_chance[0], 3),
                "best_chance_event_id": self.best_chance[1],
                "fastest_shot_kmh": round(self.fastest[0], 1),
                "fastest_shot_event_id": self.fastest[1],
            },
        }
