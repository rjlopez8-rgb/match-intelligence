"""Scripted scenarios: inject incidents (red cards, injuries, tactical switches,
momentum swings, penalties) at chosen minutes, and search for seeds whose natural
flow matches a storyline (e.g. "home trail at 60' and win").

Scenarios keep the match football-realistic: incidents happen through the
normal engine mechanics (a red card is still a foul on a real ball carrier), and
storyline constraints are met by *selecting* seeds, not by rigging outcomes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .models import MatchConfig
from .params import DEFAULT_PARAMS, EngineParams

IncidentKind = Literal["red_card", "yellow_card", "injury", "tactics", "momentum", "penalty"]


@dataclass(frozen=True)
class Incident:
    minute: float
    kind: IncidentKind
    team: Literal["home", "away"]
    player_id: str | None = None
    changes: dict = field(default_factory=dict)  # for kind == "tactics"
    value: float = 0.0  # for kind == "momentum"
    reason: str | None = None
    description: str | None = None
    condition: Literal["leading", "trailing", "level"] | None = None  # only fire if the team is in this state


@dataclass(frozen=True)
class Storyline:
    """Conditions a seed must satisfy. All fields optional."""

    result: Literal["home_win", "draw", "away_win"] | None = None
    trailing_at: tuple[str, float] | None = None  # (team, minute): team must be behind at that minute
    min_goals: int | None = None
    max_goals: int | None = None
    scorer: str | None = None  # player_id who must score

    def check_final(self, summary: dict, trailing_ok: bool) -> bool:
        h, a = summary["final_score"]["home"], summary["final_score"]["away"]
        if self.result == "home_win" and not h > a:
            return False
        if self.result == "away_win" and not a > h:
            return False
        if self.result == "draw" and h != a:
            return False
        if self.min_goals is not None and h + a < self.min_goals:
            return False
        if self.max_goals is not None and h + a > self.max_goals:
            return False
        if self.scorer and not any(g["scorer_id"] == self.scorer for g in summary["goals"]):
            return False
        return trailing_ok


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str = ""
    home: str = "RIV"
    away: str = "KIN"
    incidents: tuple[Incident, ...] = ()
    storyline: Storyline | None = None
    seed: int | None = None  # a known seed that satisfies the storyline

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        incidents = tuple(Incident(**i) for i in d.get("incidents", []))
        sl = d.get("storyline")
        storyline = None
        if sl:
            if sl.get("trailing_at"):
                sl = {**sl, "trailing_at": tuple(sl["trailing_at"])}
            storyline = Storyline(**sl)
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            home=d.get("home", "RIV"),
            away=d.get("away", "KIN"),
            incidents=incidents,
            storyline=storyline,
            seed=d.get("seed"),
        )

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        return cls.from_dict(json.loads(Path(path).read_text()))


def find_seed(
    make_config: Callable[[int], MatchConfig],
    scenario: Scenario,
    start: int = 1,
    tries: int = 200,
    params: EngineParams = DEFAULT_PARAMS,
    progress: Callable[[int, str], None] | None = None,
) -> int | None:
    """Search seeds until one produces a match that satisfies the scenario's storyline."""
    from .engine import MatchEngine
    from .summary import build_summary

    sl = scenario.storyline
    for seed in range(start, start + tries):
        eng = MatchEngine(make_config(seed), params=params, scenario=scenario, emit_frames=False)
        ok: bool | None = None
        trailing_ok = sl is None or sl.trailing_at is None
        for kind, _ in eng.run():
            if kind != "event" or sl is None:
                continue
            minute = eng.match_minute() if eng.period == 1 else 45 + eng.t / 60.0
            score = (eng.teams[0].score, eng.teams[1].score)
            if sl.trailing_at and not trailing_ok and minute >= sl.trailing_at[1]:
                team = sl.trailing_at[0]
                trailing_ok = score[0] < score[1] if team == "home" else score[1] < score[0]
                if not trailing_ok:
                    ok = False
                    break
            if sl.max_goals is not None and sum(score) > sl.max_goals:
                ok = False
                break
        if ok is False:
            if progress:
                progress(seed, "rejected early")
            continue
        summary = build_summary(eng, eng.meta())
        if sl is None or sl.check_final(summary, trailing_ok):
            if progress:
                progress(seed, "match found")
            return seed
        if progress:
            progress(seed, "rejected")
    return None
