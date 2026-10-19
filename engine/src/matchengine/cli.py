"""Command-line interface.

Examples::

    matchengine league                                   # list clubs and star players
    matchengine simulate --home RIV --away KIN --seed 7 --out data/matches/riv-kin-7
    matchengine stream --home RIV --away KIN --speed 10  # live JSONL feed on stdout
    matchengine scenario scenarios/comeback.json --out data/matches/comeback
    matchengine find-seed scenarios/comeback.json --tries 150
    matchengine calibrate --matches 24
    matchengine schema --out ../docs/schema
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

from .calibration import format_table, run_calibration
from .engine import MatchEngine
from .io import write_bundle
from .league import get_club, load_league
from .models import MatchConfig
from .scenario import Scenario, find_seed
from .stream import SyntheticFeed, stdout_sink, stream_to


def _config(home: str, away: str, seed: int, match_id: str | None = None) -> MatchConfig:
    return MatchConfig(home=get_club(home), away=get_club(away), seed=seed, match_id=match_id)


def cmd_league(args: argparse.Namespace) -> None:
    for code, club in load_league().items():
        print(f"{code}  {club.name:20s} {club.formation:8s} {club.tactics.style:28s} manager: {club.manager}")
        for p in club.squad:
            if p.bio:
                print(f"       #{p.shirt:<3d}{p.name:18s} {p.preferred_position:4s} {p.bio}")


def cmd_simulate(args: argparse.Namespace) -> None:
    cfg = _config(args.home, args.away, args.seed, args.match_id)
    t0 = time.time()
    result = MatchEngine(cfg, frame_hz=args.frame_hz, emit_frames=not args.no_tracking).simulate(
        keep_frames=not args.no_tracking
    )
    out = write_bundle(result, args.out, tracking=not args.no_tracking)
    s = result.summary
    print(
        f"{cfg.home.name} {s['final_score']['home']}-{s['final_score']['away']} {cfg.away.name}  "
        f"({len(result.events)} events, {len(result.frames)} frames, {time.time() - t0:.1f}s) -> {out}"
    )


def cmd_stream(args: argparse.Namespace) -> None:
    scenario = Scenario.load(args.scenario) if args.scenario else None
    home = args.home or (scenario.home if scenario else "RIV")
    away = args.away or (scenario.away if scenario else "KIN")
    seed = args.seed if args.seed is not None else (scenario.seed if scenario and scenario.seed else 7)
    feed = SyntheticFeed(
        _config(home, away, seed), speed=args.speed, frame_hz=args.frame_hz, scenario=scenario
    )
    with contextlib.suppress(BrokenPipeError, KeyboardInterrupt):
        stream_to(feed, stdout_sink(include_frames=args.frame_hz > 0))


def cmd_scenario(args: argparse.Namespace) -> None:
    sc = Scenario.load(args.file)
    seed = args.seed if args.seed is not None else (sc.seed or 7)
    cfg = _config(sc.home, sc.away, seed, args.match_id)
    result = MatchEngine(cfg, scenario=sc, frame_hz=args.frame_hz).simulate()
    out = write_bundle(result, args.out)
    s = result.summary
    print(
        f"[{sc.name}] {cfg.home.name} {s['final_score']['home']}-{s['final_score']['away']} {cfg.away.name} -> {out}"
    )


def cmd_find_seed(args: argparse.Namespace) -> None:
    sc = Scenario.load(args.file)

    def progress(seed: int, msg: str) -> None:
        print(f"seed {seed}: {msg}", file=sys.stderr)

    seed = find_seed(
        lambda s: _config(sc.home, sc.away, s), sc, start=args.start, tries=args.tries, progress=progress
    )
    print(json.dumps({"scenario": sc.name, "seed": seed}))


def cmd_calibrate(args: argparse.Namespace) -> None:
    report = run_calibration(args.matches, workers=args.workers)
    print(format_table(report))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")


def cmd_schema(args: argparse.Namespace) -> None:
    from .schema import export_json_schemas

    paths = export_json_schemas(args.out)
    for p in paths:
        print(p)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="matchengine", description="Synthetic football match engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("league", help="list clubs and star players").set_defaults(func=cmd_league)

    s = sub.add_parser("simulate", help="simulate a match and write a bundle")
    s.add_argument("--home", default="RIV")
    s.add_argument("--away", default="KIN")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--match-id")
    s.add_argument("--frame-hz", type=int, default=10)
    s.add_argument("--no-tracking", action="store_true")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_simulate)

    st = sub.add_parser("stream", help="stream a match as JSON lines (kind=meta|event|frame)")
    st.add_argument("--home")
    st.add_argument("--away")
    st.add_argument("--seed", type=int)
    st.add_argument("--scenario")
    st.add_argument("--speed", type=float, default=1.0, help="1 = real time, 10 = 10x, 0 = no pacing")
    st.add_argument(
        "--frame-hz", type=int, default=0, help="tracking frames per second to include (0 = none)"
    )
    st.set_defaults(func=cmd_stream)

    sc = sub.add_parser("scenario", help="run a scripted scenario file")
    sc.add_argument("file")
    sc.add_argument("--seed", type=int)
    sc.add_argument("--match-id")
    sc.add_argument("--frame-hz", type=int, default=10)
    sc.add_argument("--out", required=True)
    sc.set_defaults(func=cmd_scenario)

    fs = sub.add_parser("find-seed", help="search for a seed matching a scenario storyline")
    fs.add_argument("file")
    fs.add_argument("--start", type=int, default=1)
    fs.add_argument("--tries", type=int, default=150)
    fs.set_defaults(func=cmd_find_seed)

    ca = sub.add_parser("calibrate", help="compare simulated averages with real-world targets")
    ca.add_argument("--matches", type=int, default=24)
    ca.add_argument("--workers", type=int, default=2)
    ca.add_argument("--json")
    ca.set_defaults(func=cmd_calibrate)

    sh = sub.add_parser("schema", help="export JSON Schemas for events, frames and metadata")
    sh.add_argument("--out", required=True)
    sh.set_defaults(func=cmd_schema)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
