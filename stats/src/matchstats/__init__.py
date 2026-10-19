"""matchstats: real-time football match intelligence."""

from .analyzer import SNAPSHOT_VERSION, MatchAnalyzer, analyze
from .feed import live_feed, replay_bundle
from .models import Models, default_models

__all__ = [
    "SNAPSHOT_VERSION",
    "MatchAnalyzer",
    "Models",
    "analyze",
    "default_models",
    "live_feed",
    "replay_bundle",
]
