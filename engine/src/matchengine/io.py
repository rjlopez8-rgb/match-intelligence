"""Writing and reading match bundles.

A bundle is a folder with:

* ``meta.json``        match metadata, lineups, squads, tactics
* ``events.jsonl``     one event per line, in publication order
* ``tracking.jsonl.gz`` one tracking frame per line (gzip)
* ``summary.json``     ground-truth box score
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

from .engine import MatchResult


def write_bundle(result: MatchResult, out_dir: str | Path, tracking: bool = True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(result.meta, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "events.jsonl").open("w", encoding="utf-8") as fh:
        for e in result.events:
            fh.write(json.dumps(e, ensure_ascii=False, separators=(",", ":")) + "\n")
    if tracking and result.frames:
        with gzip.open(out / "tracking.jsonl.gz", "wt", encoding="utf-8") as fh:
            for f in result.frames:
                fh.write(json.dumps(f, separators=(",", ":")) + "\n")
    (out / "summary.json").write_text(json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def read_events(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_frames(path: str | Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
