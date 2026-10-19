"""Public data contract (v1) for events, tracking frames and match metadata.

These Pydantic models are the single source of truth for what the feed emits.
Tests validate every event the engine produces against them, and
``matchengine schema --out DIR`` exports JSON Schema files that other services
(and other languages) can validate against.

Coordinates: metres on a 105 x 68 pitch, origin at the bottom-left corner flag.
Each event's ``t`` is when the action *started*; events are published once their
outcome is known (so a pass appears when it is received or lost), exactly like a
live data feed. ``seq`` is the publication order.

Fields inside ``details.truth`` are simulator ground truth (e.g. the engine's own
xG). They are for evaluating analytics models and must not be shown to fans or
used as model inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .events import SCHEMA_VERSION, EventType


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Location(_Strict):
    x: float = Field(ge=-5, le=110, description="metres along the pitch length")
    y: float = Field(ge=-5, le=73, description="metres along the pitch width")


class Location3D(Location):
    z: float = Field(ge=0, description="height in metres")


class Possession(_Strict):
    id: int
    team_id: str | None
    play_pattern: str = Field(description="how the possession started, e.g. regular_play, from_corner")


class Score(_Strict):
    home: int
    away: int


class Truth(BaseModel):
    """Simulator ground truth. Never display or use as a model feature."""

    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------------------
# Per-type details
# --------------------------------------------------------------------------------------
class PassDetails(_Strict):
    recipient_id: str | None
    recipient_name: str | None
    end_location: Location
    length_m: float
    angle_deg: float = Field(description="direction of travel in pitch coordinates, degrees")
    progression_m: float = Field(description="metres gained towards the opponent's goal")
    height: Literal["ground", "low", "high"]
    pass_kind: str = Field(
        description="ground | back | through | long | switch | cross | cutback | corner | "
        "free_kick | free_kick_cross | throw_in | goal_kick | kick_off | "
        "keeper_distribution"
    )
    restart_type: str | None
    release_speed_kmh: float
    under_pressure: bool
    nearest_opponent_m: float
    outcome: Literal["complete", "incomplete", "intercepted", "out", "offside"]
    intercepted_by: str | None = None
    completed_by: str | None = None
    truth: Truth


class CarryDetails(_Strict):
    end_location: Location
    distance_m: float
    duration_s: float
    progression_m: float
    max_speed_kmh: float
    under_pressure: bool


class ShotDetails(_Strict):
    shot_kind: Literal["open_play", "free_kick", "penalty"]
    body_part: Literal["head", "left_foot", "right_foot"]
    outcome: Literal["goal", "saved", "blocked", "off_target", "post"]
    on_target: bool
    end_location: Location3D
    goal_mouth: dict = Field(description="y_offset_m from goal centre (+ = shooter's left) and z_m height")
    distance_m: float
    angle_deg: float = Field(description="angle subtended by the goal mouth")
    speed_kmh: float
    under_pressure: bool
    defenders_in_lane: int
    goalkeeper_id: str | None
    truth: Truth


class GoalDetails(_Strict):
    scorer_id: str
    scorer_name: str
    assist_id: str | None
    assist_name: str | None
    shot_kind: str
    body_part: Literal["head", "foot"]
    xg: float
    big_chance: bool
    shot_event_id: str | None
    new_score: Score
    minute_display: str


class PressureDetails(_Strict):
    target_player_id: str
    distance_m: float
    closing_speed_kmh: float


class TackleDetails(_Strict):
    outcome: Literal["won", "lost"]
    opponent_id: str
    truth: Truth
    ball_out: bool | None = None


class DribbleDetails(_Strict):
    outcome: Literal["success", "fail"]
    opponent_id: str
    truth: Truth


class AerialDetails(_Strict):
    opponent_id: str
    outcome: Literal["won"]
    truth: Truth


class SaveDetails(_Strict):
    kind: Literal["catch", "parry"]
    shot_speed_kmh: float
    shooter_id: str


class BlockDetails(_Strict):
    blocked: Literal["shot", "pass"]
    shooter_id: str | None = None
    passer_id: str | None = None


class ClearanceDetails(_Strict):
    end_location: Location
    body_part: Literal["foot", "head"]
    height: Literal["ground", "low", "high"]


class FoulDetails(_Strict):
    victim_id: str
    victim_name: str
    kind: str
    penalty: bool


class CardDetails(_Strict):
    card: Literal["yellow", "second_yellow", "red"]
    reason: str


class BallOutDetails(_Strict):
    line: str
    last_touch_team_id: str | None
    restart: Literal["throw_in", "goal_kick", "corner"]
    restart_team_id: str | None


class PossessionChangeDetails(_Strict):
    from_team_id: str | None
    how: str
    previous_duration_s: float
    previous_completed_passes: int


class SubstitutionDetails(_Strict):
    player_out_id: str
    player_out_name: str
    player_in_id: str
    player_in_name: str
    position: str
    reason: Literal["tactical", "fatigue", "injury"]
    energy_out: float


class TacticalChangeDetails(_Strict):
    reason: str
    description: str
    manager: str
    before: dict[str, float]
    after: dict[str, float]


DETAILS_BY_TYPE: dict[str, type[BaseModel]] = {
    "pass": PassDetails,
    "carry": CarryDetails,
    "shot": ShotDetails,
    "goal": GoalDetails,
    "pressure": PressureDetails,
    "tackle": TackleDetails,
    "dribble": DribbleDetails,
    "aerial_duel": AerialDetails,
    "save": SaveDetails,
    "block": BlockDetails,
    "clearance": ClearanceDetails,
    "foul": FoulDetails,
    "card": CardDetails,
    "ball_out": BallOutDetails,
    "possession_change": PossessionChangeDetails,
    "substitution": SubstitutionDetails,
    "tactical_change": TacticalChangeDetails,
}


class MatchEvent(_Strict):
    schema_version: str = SCHEMA_VERSION
    event_id: str = Field(description="unique id: <match_id>-<seq>")
    match_id: str
    seq: int = Field(ge=1, description="publication order")
    type: EventType
    period: Literal[1, 2]
    t: float = Field(ge=0, description="seconds since the start of the period when the action began")
    minute: int
    second: int
    clock: str = Field(description="broadcast clock, e.g. '67:12' or '90:00+2:31'")
    minute_display: str = Field(description='e.g. "68\'" or "90+3\'"')
    elapsed_s: float = Field(description="seconds since kick-off including half-time, for real-time pacing")
    team_id: str | None
    player_id: str | None
    player_name: str | None
    location: Location | None
    possession: Possession
    score: Score
    details: dict

    def typed_details(self) -> BaseModel | dict:
        model = DETAILS_BY_TYPE.get(self.type.value)
        return model.model_validate(self.details) if model else self.details


class FrameBall(_Strict):
    x: float
    y: float
    z: float
    speed_ms: float
    state: Literal["held", "flight", "loose", "dead"]
    owner_id: str | None


class TrackingFrame(_Strict):
    frame: int
    period: Literal[1, 2]
    t: float
    clock: str
    ball: FrameBall
    possession_team_id: str | None
    players: list[tuple[str, float, float, float]] = Field(
        description="[player_id, x, y, speed_m_per_s] for every player on the pitch"
    )


def validate_event(e: dict) -> MatchEvent:
    ev = MatchEvent.model_validate(e)
    ev.typed_details()
    return ev


def export_json_schemas(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, model in {
        "event": MatchEvent,
        "frame": TrackingFrame,
        **{f"details.{k}": v for k, v in DETAILS_BY_TYPE.items()},
    }.items():
        path = out / f"{name}.schema.json"
        path.write_text(json.dumps(model.model_json_schema(), indent=2), encoding="utf-8")
        written.append(path)
    return written
