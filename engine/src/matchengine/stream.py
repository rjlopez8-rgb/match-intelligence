"""Real-time streaming.

``MatchFeed`` is the contract the rest of the system ingests from. Today it is
backed by the simulator (``SyntheticFeed``); in production it would be backed by
an official event and tracking provider, and nothing downstream would change.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterator
from typing import Protocol

from .engine import MatchEngine
from .models import MatchConfig
from .params import DEFAULT_PARAMS, EngineParams

Item = tuple[str, dict]


class MatchFeed(Protocol):
    """Anything that yields ("meta" | "event" | "frame", payload) in match order."""

    def __iter__(self) -> Iterator[Item]: ...


class SyntheticFeed:
    """Streams a simulated match, optionally paced to the match clock.

    ``speed=1`` plays in real time, ``speed=10`` ten times faster, ``speed=0`` as fast as possible.
    ``frame_hz`` controls how many tracking frames per second are published (0 disables frames).
    """

    def __init__(
        self,
        config: MatchConfig,
        speed: float = 1.0,
        frame_hz: int = 10,
        scenario=None,
        params: EngineParams = DEFAULT_PARAMS,
        skip_halftime: bool = True,
    ):
        self.engine = MatchEngine(
            config, params=params, scenario=scenario, frame_hz=max(frame_hz, 1), emit_frames=frame_hz > 0
        )
        self.speed = speed
        self.skip_halftime = skip_halftime

    def __iter__(self) -> Iterator[Item]:
        eng = self.engine
        yield "meta", eng.meta()
        start_wall = time.monotonic()
        offset = 0.0
        last_elapsed = 0.0
        for kind, item in eng.run():
            if self.speed > 0:
                elapsed = item.get("elapsed_s") if kind == "event" else eng._elapsed(item["t"])
                if elapsed is not None:
                    if self.skip_halftime and elapsed - last_elapsed > 600:
                        offset += elapsed - last_elapsed
                    last_elapsed = max(last_elapsed, elapsed)
                    due = start_wall + (elapsed - offset) / self.speed
                    wait = due - time.monotonic()
                    if wait > 0:
                        time.sleep(wait)
            yield kind, item


def stream_to(feed: MatchFeed, sink: Callable[[str, dict], None]) -> None:
    for kind, item in feed:
        sink(kind, item)


def stdout_sink(include_frames: bool = False) -> Callable[[str, dict], None]:
    def _sink(kind: str, item: dict) -> None:
        if kind == "frame" and not include_frames:
            return
        sys.stdout.write(
            json.dumps({"kind": kind, "data": item}, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        sys.stdout.flush()

    return _sink
