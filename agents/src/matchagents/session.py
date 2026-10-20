"""A match the agents can work on, replayed from a bundle as if it were live.

The analyzer is only ever advanced to the moment being covered, so agents cannot see the
future: when the desk covers the 74th-minute penalty, the stats tools answer as of the
74th minute. Tracking frames are kept so the Telestrator's drawings can be pinned to where
players really were.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Iterator
from pathlib import Path

from matchstats import MatchAnalyzer, replay_bundle
from matchstats.queries import MatchQueries


class MatchSession:
    def __init__(self, bundle: str | Path):
        self.bundle = Path(bundle)
        self.meta = json.loads((self.bundle / "meta.json").read_text(encoding="utf-8"))
        self.analyzer = MatchAnalyzer()
        self.queries: MatchQueries | None = None
        self._feed = replay_bundle(self.bundle)
        self._frame_keys: list[tuple[int, float]] = []
        self._frames: list[dict] = []
        self.done = False

    # -- feeding ----------------------------------------------------------------------------
    def _feed_one(self) -> list[dict]:
        try:
            kind, item = next(self._feed)
        except StopIteration:
            self.done = True
            return []
        if kind == "frame":
            self._frame_keys.append((item["period"], item["t"]))
            self._frames.append(item)
        new = self.analyzer.process(kind, item)
        if self.queries is None and getattr(self.analyzer, "ctx", None) is not None:
            self.queries = MatchQueries(self.analyzer)
        return new

    def moments(self, min_importance: float = 0.0) -> Iterator[dict]:
        """Advance the match, pausing at each new moment (the stats tools answer as of that instant)."""
        while not self.done:
            for m in self._feed_one():
                if m["importance"] >= min_importance:
                    yield m

    def advance_to(self, moment_id: str) -> dict:
        """Play the match forward until a given moment has just happened."""
        for m in self.moments():
            if m["moment_id"] == moment_id:
                return m
        raise KeyError(f"moment '{moment_id}' never happened in this match")

    @property
    def ctx(self):
        return self.analyzer.ctx

    # -- geometry for the telestrator ---------------------------------------------------------
    def frame_at(self, period: int, t: float) -> dict | None:
        """The tracking frame closest to a match time (only frames already seen)."""
        if not self._frames:
            return None
        i = bisect.bisect_left(self._frame_keys, (period, t))
        best = None
        for j in (i - 1, i):
            if not (0 <= j < len(self._frames)) or self._frame_keys[j][0] != period:
                continue
            if best is None or abs(self._frame_keys[j][1] - t) < abs(self._frame_keys[best][1] - t):
                best = j
        return self._frames[best] if best is not None else None

    def player_xy(self, player_id: str, period: int, t: float) -> tuple[float, float] | None:
        f = self.frame_at(period, t)
        if not f:
            return None
        for row in f["players"]:
            if row[0] == player_id:
                return float(row[1]), float(row[2])
        return None
