"""Feed sources.

The analyzer consumes ``(kind, payload)`` items where ``kind`` is ``"meta"``,
``"event"`` or ``"frame"``. Two sources are provided:

* :func:`replay_bundle` replays a recorded match bundle written by ``matchengine``
  (``meta.json``, ``events.jsonl``, optional ``tracking.jsonl.gz``), interleaving
  events and frames in match-time order.
* :func:`live_feed` wraps ``matchengine.SyntheticFeed`` for a live, paced stream.

In production either can be replaced by an adapter over an official provider
that yields the same three kinds of items.
"""

from __future__ import annotations

import gzip
import heapq
import json
from collections.abc import Iterator
from pathlib import Path

Item = tuple[str, dict]


def _events(path: Path) -> Iterator[dict]:
    if not path.exists() and path.with_suffix(".jsonl.gz").exists():
        path = path.with_suffix(".jsonl.gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _frames(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def replay_bundle(bundle: str | Path, frames: bool = True) -> Iterator[Item]:
    """Yield a recorded match in match-time order."""
    root = Path(bundle)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    yield "meta", meta
    events = sorted(_events(root / "events.jsonl"), key=lambda e: (e["period"], e["t"], e["seq"]))
    tracking = root / "tracking.jsonl.gz"
    streams: list[Iterator[tuple[tuple, str, dict]]] = [
        ((e["period"], e["t"], 1, e["seq"]), "event", e) for e in events
    ]  # type: ignore[list-item]
    if frames and tracking.exists():
        frame_iter = (((f["period"], f["t"], 0, f["frame"]), "frame", f) for f in _frames(tracking))
        merged = heapq.merge(iter(streams), frame_iter, key=lambda x: x[0])
    else:
        merged = iter(streams)
    for _key, kind, item in merged:
        yield kind, item


def live_feed(
    home: str = "RIV", away: str = "KIN", seed: int = 7, speed: float = 0.0, frame_hz: int = 5, scenario=None
) -> Iterator[Item]:
    """Live stream from the synthetic engine (``speed`` 1 = real time, 0 = as fast as possible)."""
    from matchengine import MatchConfig, SyntheticFeed, get_club

    cfg = MatchConfig(home=get_club(home), away=get_club(away), seed=seed)
    yield from SyntheticFeed(cfg, speed=speed, frame_hz=frame_hz, scenario=scenario)
