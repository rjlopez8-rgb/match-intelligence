"""matchengine: synthetic football match engine for explainable match intelligence."""

from .engine import ENGINE_VERSION, MatchEngine, MatchResult
from .events import SCHEMA_VERSION, EventType
from .league import get_club, load_league
from .models import MatchConfig
from .params import DEFAULT_PARAMS, EngineParams
from .scenario import Incident, Scenario, Storyline, find_seed
from .stream import MatchFeed, SyntheticFeed

__all__ = [
    "ENGINE_VERSION",
    "SCHEMA_VERSION",
    "DEFAULT_PARAMS",
    "EngineParams",
    "EventType",
    "Incident",
    "MatchConfig",
    "MatchEngine",
    "MatchFeed",
    "MatchResult",
    "Scenario",
    "Storyline",
    "SyntheticFeed",
    "find_seed",
    "get_club",
    "load_league",
    "simulate_match",
]


def simulate_match(
    home: str = "RIV",
    away: str = "KIN",
    seed: int = 7,
    scenario: Scenario | None = None,
    frame_hz: int = 10,
    keep_frames: bool = True,
    **config,
) -> MatchResult:
    """One-call helper: simulate a full match between two league clubs."""
    cfg = MatchConfig(home=get_club(home), away=get_club(away), seed=seed, **config)
    return MatchEngine(cfg, scenario=scenario, frame_hz=frame_hz, emit_frames=keep_frames).simulate(
        keep_frames
    )
