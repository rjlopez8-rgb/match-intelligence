"""Typed contracts for what the stats layer emits.

Agents and the frontend depend on these shapes, so they are validated in tests
and exported as JSON Schema (``matchstats schema``) next to the event schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MomentType = Literal[
    "goal",
    "big_chance",
    "great_save",
    "woodwork",
    "penalty_awarded",
    "red_card",
    "yellow_card",
    "tactical_change",
    "substitution",
    "period_summary",
    "speed_alert",
    "top_speed",
    "distance_milestone",
    "hard_pass",
    "long_sequence",
    "dominance",
    "pressing_surge",
    "momentum_shift",
    "chaos_spike",
]


class Probabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    home: float = Field(ge=0, le=1)
    draw: float = Field(ge=0, le=1)
    away: float = Field(ge=0, le=1)


class Score(BaseModel):
    model_config = ConfigDict(extra="forbid")
    home: int = Field(ge=0)
    away: int = Field(ge=0)


class Moment(BaseModel):
    """Something worth talking about, with the numbers and event ids that back it up."""

    model_config = ConfigDict(extra="forbid")
    moment_id: str
    type: MomentType
    period: int = Field(ge=1, le=2)
    t: float = Field(ge=0, description="seconds since the start of the period")
    clock: str
    minute_display: str = Field(description="broadcast-style minute, e.g. 45+2'")
    team_id: str | None
    team_name: str | None
    player_ids: list[str]
    player_names: list[str]
    title: str = Field(description="factual template; narration is the agents' job")
    facts: dict = Field(description="numbers behind the moment, all computed by the stats layer")
    evidence: list[str] = Field(description="event ids that prove the moment")
    moment_weight: float | None = Field(
        description="percentage points of win probability moved (realised for goals/red cards, expected for misses)"
    )
    importance: float = Field(ge=0, le=1)
    score: Score
    win_probability: Probabilities
    momentum: float = Field(ge=-1, le=1, description="+1 home fully on top, -1 away fully on top")


class TeamHeader(BaseModel):
    team_id: str
    name: str
    short_name: str
    code: str
    formation: str
    manager: str
    colors: dict


class MatchHeader(BaseModel):
    match_id: str
    competition: str | None
    venue: dict | None = Field(description="stadium, city, capacity")
    home: TeamHeader
    away: TeamHeader
    finished: bool


class Clock(BaseModel):
    period: int
    clock: str
    minute_display: str


class TopMoments(BaseModel):
    count: int = Field(ge=0)
    top: list[Moment]


class Snapshot(BaseModel):
    """Complete state of the stats layer at one instant (``MatchAnalyzer.snapshot()``)."""

    snapshot_version: str
    match: MatchHeader
    clock: Clock
    score: Score
    models: dict[str, Literal["trained", "fallback"]]
    metrics: dict[str, dict]
    moments: TopMoments


def export_schemas(directory: str | Path) -> list[Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for model in (Moment, Snapshot):
        path = out / f"stats_{model.__name__.lower()}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written
