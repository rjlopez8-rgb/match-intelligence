"""Metric plug-in interface.

A metric is a small class that updates itself incrementally from events and/or
tracking frames and can produce a JSON-serialisable snapshot at any time. Adding
a new statistic means adding one module with one ``@register``-ed class.

Metrics may raise *signals* (e.g. "new top speed") that the moment detector turns
into key moments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:  # pragma: no cover
    from ..analyzer import MatchAnalyzer
    from ..context import MatchContext

REGISTRY: list[type[Metric]] = []


def register(cls: type[Metric]) -> type[Metric]:
    REGISTRY.append(cls)
    REGISTRY.sort(key=lambda c: c.order)
    return cls


class Metric:
    name: ClassVar[str] = "metric"
    order: ClassVar[int] = 100  # lower runs first; metrics may depend on earlier ones' annotations

    def __init__(self, analyzer: MatchAnalyzer):
        self.analyzer = analyzer
        self.ctx: MatchContext = analyzer.ctx
        self.models = analyzer.models

    def on_event(self, e: dict) -> None:  # pragma: no cover - default no-op
        pass

    def on_frame(self, f: dict) -> None:  # pragma: no cover - default no-op
        pass

    def snapshot(self) -> dict:  # pragma: no cover - default
        return {}

    def signal(self, kind: str, **data) -> None:
        self.analyzer.signals.append({"kind": kind, **data})


def team_dict(ctx: MatchContext, factory) -> dict:
    return {ctx.home_id: factory(), ctx.away_id: factory()}


def ratio(a: float, b: float, digits: int = 3) -> float:
    return round(a / b, digits) if b else 0.0
