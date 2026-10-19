"""MatchAnalyzer: the stats layer's single entry point.

Feed it ``("meta" | "event" | "frame", payload)`` items, one at a time or from any
iterable source. It keeps an incremental match state (O(1) work per item), runs
every registered metric, detects key moments, and can produce a complete
JSON-serialisable snapshot at any instant. One analyzer per match, so a service
can follow many matches in parallel.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .context import MatchContext
from .metrics import REGISTRY, Metric
from .models import Models, default_models
from .moments import MomentDetector

SNAPSHOT_VERSION = "1.0.0"


class MatchAnalyzer:
    def __init__(self, models: Models | None = None, frames: bool = True):
        self.models = models or default_models()
        self.use_frames = frames
        self.ctx: MatchContext | None = None
        self.metrics: list[Metric] = []
        self._by_name: dict[str, Metric] = {}
        self.detector: MomentDetector | None = None
        self.signals: list[dict] = []
        self.new_moments: list[dict] = []
        self.items = 0

    # -- feeding ------------------------------------------------------------------------
    def process(self, kind: str, item: dict) -> list[dict]:
        """Process one feed item; returns any moments it produced."""
        self.new_moments = []
        self.items += 1
        if kind == "meta":
            self._start(item)
            return []
        if self.ctx is None:
            raise RuntimeError("feed must start with a 'meta' item")
        if kind == "event":
            self.ctx.update_from_event(item)
            for m in self.metrics:
                m.on_event(item)
            self.detector.on_event(item)
        elif kind == "frame" and self.use_frames:
            self.ctx.update_from_frame(item)
            for m in self.metrics:
                m.on_frame(item)
        self._flush_signals()
        return self.new_moments

    def _start(self, meta: dict) -> None:
        self.ctx = MatchContext(meta)
        self.metrics = [cls(self) for cls in REGISTRY]
        self._by_name = {m.name: m for m in self.metrics}
        self.detector = MomentDetector(self)

    def _flush_signals(self) -> None:
        while self.signals:
            self.detector.on_signal(self.signals.pop(0))

    def run(self, source: Iterable[tuple[str, dict]]) -> MatchAnalyzer:
        for kind, item in source:
            self.process(kind, item)
        return self

    def stream(self, source: Iterable[tuple[str, dict]]) -> Iterator[dict]:
        """Yield moments as soon as they are detected (for live use)."""
        for kind, item in source:
            yield from self.process(kind, item)

    # -- access ---------------------------------------------------------------------------
    def metric(self, name: str) -> Metric:
        return self._by_name[name]

    @property
    def moments(self) -> list[dict]:
        return self.detector.moments if self.detector else []

    def snapshot(self, include_series: bool = True) -> dict:
        ctx = self.ctx
        if ctx is None:
            return {}
        metrics = {m.name: m.snapshot() for m in self.metrics}
        if not include_series:
            for snap in metrics.values():
                for key in [
                    k for k in snap if k in ("series", "per_minute", "pressure_pulse", "xg_timeline")
                ]:
                    snap.pop(key)
        return {
            "snapshot_version": SNAPSHOT_VERSION,
            "match": {
                "match_id": ctx.match_id,
                "competition": ctx.meta.get("competition"),
                "venue": ctx.meta.get("venue"),
                "home": self._team_header(ctx.home_id),
                "away": self._team_header(ctx.away_id),
                "finished": ctx.finished,
            },
            "clock": {"period": ctx.period, "clock": ctx.clock, "minute_display": ctx.minute_display},
            "score": dict(ctx.score),
            "models": self.models.status(),
            "metrics": metrics,
            "moments": {
                "count": len(self.moments),
                "top": sorted(self.moments, key=lambda m: -m["importance"])[:10],
            },
        }

    def _team_header(self, team_id: str) -> dict:
        t = self.ctx.teams[team_id]
        return {
            "team_id": team_id,
            "name": t["name"],
            "short_name": t["short_name"],
            "code": t["code"],
            "formation": t["formation"],
            "manager": t["manager"],
            "colors": t["colors"],
        }


def analyze(
    source: Iterable[tuple[str, dict]], frames: bool = True, models: Models | None = None
) -> MatchAnalyzer:
    """Convenience: run a whole source through a fresh analyzer."""
    return MatchAnalyzer(models=models, frames=frames).run(source)
