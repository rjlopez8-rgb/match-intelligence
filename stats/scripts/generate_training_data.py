"""Simulate matches for model training.

Writes one gzip JSON-lines file per match: a ``{"meta": ...}`` header line, then
every event. Every ordered pair of clubs is played in turn, so home advantage
and all style match-ups are covered evenly. Re-running resumes where it stopped.

    python scripts/generate_training_data.py --matches 168 --out ../data/training
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations
from pathlib import Path

from matchengine import MatchConfig, MatchEngine
from matchengine.league import load_league


def simulate(job: tuple[str, str, int, str]) -> str:
    home, away, seed, out = job
    path = Path(out) / f"{home}-{away}-{seed}.jsonl.gz"
    if path.exists():
        return path.name
    league = load_league()
    engine = MatchEngine(MatchConfig(home=league[home], away=league[away], seed=seed), emit_frames=False)
    meta = engine.meta()
    for _ in engine.run():
        pass
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        header = {
            "match_id": meta["match_id"],
            "home": meta["home"]["team_id"],
            "away": meta["away"]["team_id"],
        }
        fh.write(json.dumps({"meta": header}) + "\n")
        for event in engine.events:
            fh.write(json.dumps(event, separators=(",", ":"), ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path.name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--matches", type=int, default=168, help="168 = every fixture three times")
    p.add_argument("--out", default="../data/training")
    p.add_argument("--seed-start", type=int, default=50000)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    fixtures = list(permutations(load_league(), 2))
    jobs = [(*fixtures[k % len(fixtures)], args.seed_start + k, args.out) for k in range(args.matches)]
    with ProcessPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(simulate, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} matches", flush=True)


if __name__ == "__main__":
    main()
