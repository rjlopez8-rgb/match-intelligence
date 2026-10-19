"""Shared match context: who is playing, the clock, score, possession, and lookups.

Metrics read from the context; only the analyzer writes to it. Per-event
annotations computed by one metric (e.g. a shot's xG) are stored in
``ctx.ann[event_id]`` so later metrics and the moment detector can reuse them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .pitch import attacking_frame


@dataclass(frozen=True)
class PlayerInfo:
    player_id: str
    name: str
    short_name: str
    team_id: str
    shirt: int
    position: str
    role: str
    bio: str = ""


class MatchContext:
    FRAME_BUFFER = 120

    def __init__(self, meta: dict):
        self.meta = meta
        self.match_id: str = meta["match_id"]
        self.home_id: str = meta["home"]["team_id"]
        self.away_id: str = meta["away"]["team_id"]
        self.teams = {
            side["team_id"]: {
                "name": side["name"],
                "short_name": side["short_name"],
                "code": side["code"],
                "colors": side.get("colors", {}),
                "formation": side.get("formation"),
                "manager": side.get("manager"),
                "side": key,
            }
            for key, side in (("home", meta["home"]), ("away", meta["away"]))
        }
        self.players: dict[str, PlayerInfo] = {}
        for side in (meta["home"], meta["away"]):
            positions = {p["player_id"]: p["position"] for p in side.get("lineup", [])}
            for p in side.get("squad", []):
                self.players[p["player_id"]] = PlayerInfo(
                    p["player_id"],
                    p["name"],
                    p["short_name"],
                    side["team_id"],
                    p["shirt"],
                    positions.get(p["player_id"], p.get("preferred_position", "")),
                    p["role"],
                    p.get("bio", ""),
                )
        self.on_pitch: set[str] = {
            p["player_id"] for s in (meta["home"], meta["away"]) for p in s.get("lineup", [])
        }
        self.period = 1
        self.t = 0.0
        self.clock = "00:00"
        self.minute = 0
        self.minute_display = "1'"
        self.score = {"home": 0, "away": 0}
        self.red_cards = {self.home_id: 0, self.away_id: 0}
        self.possession_team: str | None = None
        self.possession_id: int | None = None
        self.events: list[dict] = []
        self.by_id: dict[str, dict] = {}
        self.ann: dict[str, dict] = {}
        self.frames: deque[dict] = deque(maxlen=self.FRAME_BUFFER)
        self.has_frames = False
        self.finished = False

    # -- identity helpers -------------------------------------------------------------
    def side(self, team_id: str | None) -> str | None:
        return None if team_id is None else self.teams[team_id]["side"]

    def opponent(self, team_id: str) -> str:
        return self.away_id if team_id == self.home_id else self.home_id

    def team_name(self, team_id: str | None, short: bool = True) -> str:
        if team_id is None:
            return ""
        return self.teams[team_id]["short_name" if short else "name"]

    def player_name(self, player_id: str | None, short: bool = True) -> str:
        if not player_id or player_id not in self.players:
            return ""
        p = self.players[player_id]
        return p.short_name if short else p.name

    def team_of(self, player_id: str) -> str | None:
        p = self.players.get(player_id)
        return p.team_id if p else None

    # -- geometry ---------------------------------------------------------------------
    def attacks_right(self, team_id: str, period: int | None = None) -> bool:
        period = self.period if period is None else period
        home_right = period == 1
        return home_right if team_id == self.home_id else not home_right

    def tf(self, loc: dict | None, team_id: str, period: int | None = None) -> tuple[float, float]:
        if not loc:
            return (52.5, 34.0)
        return attacking_frame(loc["x"], loc["y"], self.attacks_right(team_id, period))

    def frame_at(self, period: int, t: float, tolerance: float = 0.6) -> dict | None:
        best, best_dt = None, tolerance
        for f in reversed(self.frames):
            if f["period"] != period:
                continue
            dt = abs(f["t"] - t)
            if dt <= best_dt:
                best, best_dt = f, dt
            elif f["t"] < t - tolerance:
                break
        return best

    def match_minute_float(self) -> float:
        return (self.period - 1) * 45.0 + self.t / 60.0

    # -- updates (analyzer only) ---------------------------------------------------------
    def update_from_event(self, e: dict) -> None:
        self.events.append(e)
        self.by_id[e["event_id"]] = e
        self.period = e["period"]
        self.t = max(self.t, e["t"]) if e["period"] == self.period else e["t"]
        self.clock = e["clock"]
        self.minute = e["minute"]
        self.minute_display = e["minute_display"]
        self.score = dict(e["score"])
        poss = e.get("possession") or {}
        self.possession_team = poss.get("team_id")
        self.possession_id = poss.get("id")
        t = e["type"]
        d = e["details"]
        if t == "card" and d.get("card") in ("red", "second_yellow") and e["team_id"]:
            self.red_cards[e["team_id"]] += 1
            self.on_pitch.discard(e["player_id"])
        elif t == "substitution":
            self.on_pitch.discard(d["player_out_id"])
            self.on_pitch.add(d["player_in_id"])
        elif t == "match_end":
            self.finished = True
        elif t == "period_start":
            self.t = 0.0

    def update_from_frame(self, f: dict) -> None:
        self.has_frames = True
        self.frames.append(f)
        if f["period"] != self.period:
            self.period = f["period"]
            self.t = f["t"]
        else:
            self.t = max(self.t, f["t"])
