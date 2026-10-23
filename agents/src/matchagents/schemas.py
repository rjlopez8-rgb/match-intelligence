"""What each agent must hand back: strict, typed outputs instead of free text.

Model outputs are validated against these Pydantic models (OpenAI strict structured
output), so a malformed answer is rejected by the API rather than breaking the broadcast.
Strict mode requires every field, so optional values are nullable rather than defaulted.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------ viewer
Style = Literal["fan", "tactician", "analyst", "academy"]


class ViewerProfile(BaseModel):
    """Who is watching (the Personalize stage). Not a model output, so defaults are fine."""

    style: Style = "fan"
    favourite_team: str | None = None
    language: str = "English"


LANGUAGES = {"en": "English", "es": "Spanish", "fr": "French"}
STYLES: tuple[str, ...] = ("fan", "tactician", "analyst", "academy")


# ----------------------------------------------------------------------------- producer
class ProducerPlan(_Strict):
    cover: bool = Field(description="false if this moment is not worth airtime right now")
    segment: Literal["live_call", "replay_breakdown", "what_if", "stat_drop"]
    angle: str = Field(description="one sentence: the story this segment tells")
    priority: Literal["low", "normal", "high", "breaking"]
    telestrate: bool = Field(description="draw on the pitch during the replay")
    what_if: str | None = Field(description="a counterfactual worth simulating, or null")


# ------------------------------------------------------------------------------ analyst
class Fact(_Strict):
    statement: str = Field(description="one checkable claim in plain words")
    value: str = Field(description="the number exactly as a tool returned it, e.g. '0.41', '74', '0-1'")
    source_tool: str
    evidence: list[str] = Field(description="event ids that prove it (from tool results)")


class AnalystBrief(_Strict):
    headline: str
    facts: list[Fact]
    insight: str = Field(description="the tactical why, two sentences at most")
    decision: str = Field(
        description="the key decision at the action, from the options tool: what he chose, the best alternative "
        "with its numbers, and whether the choice was right. Empty string when there was no decision to judge."
    )


# --------------------------------------------------------------------------- telestrator
class Shape(_Strict):
    kind: Literal["ring", "arrow", "run", "zone", "label"]
    player: str | None = Field(description="player name or id the shape is attached to")
    event_id: str | None = Field(description="event whose location anchors the shape")
    to_player: str | None = Field(description="arrow/run target player")
    to_event_id: str | None = Field(description="arrow target event (or use the event's end location)")
    label: str | None = Field(description="2-4 words drawn next to the shape")
    color: Literal["home", "away", "highlight", "warning"]


class Telestration(_Strict):
    freeze_event_id: str | None = Field(description="event to freeze the replay on")
    shapes: list[Shape]
    caption: str


# ----------------------------------------------------------------------------- director
Camera = Literal["broadcast", "behind", "tactical", "skycam", "player", "sweep"]


class Shot(_Strict):
    camera: Camera
    focus_player: str | None
    seconds: float
    slow_motion: bool
    purpose: str


class DirectorCue(_Strict):
    live_camera: Camera
    replay: bool
    replay_from_event_id: str | None
    shots: list[Shot]
    show_telestration: bool
    graphic: Literal["none", "win_probability", "xg", "momentum", "pass_network", "player_card", "scoreboard"]


# --------------------------------------------------------------------------- commentator
class Line(_Strict):
    text: str
    facts_used: list[int] = Field(description="indices into the analyst's facts")


class Commentary(_Strict):
    lines: list[Line]
    headline: str = Field(description="the on-screen title for this moment, under 10 words, in the viewer's language")
    caption: str = Field(description="a 2-8 word caption for the drawing on the replay, in the viewer's language")
    question: str | None = Field(description="the what-if question in the viewer's language, or null if none")


# -------------------------------------------------------------------------- fact-checker
class Issue(_Strict):
    line: int
    claim: str
    problem: str


class Verdict(_Strict):
    approved: bool
    issues: list[Issue]


# ----------------------------------------------------------------------------- scenario
class ScenarioSpec(_Strict):
    question: str = Field(description="the what-if in one sentence")
    intervention: Literal["shot_scores", "shot_misses", "goal_disallowed", "no_red_card", "player_swap"]
    event_id: str = Field(description="the shot the intervention changes (its event id)")
    simulations: int = Field(description="how many futures to simulate (10-40)")
    hypothesis: str = Field(description="what you expect and why, to compare with the result")


# -------------------------------------------------------------------------- scout (Club mode)
class ScoutReport(_Strict):
    headline: str
    report: str
    verdict: Literal["sign", "shortlist", "pass"]
