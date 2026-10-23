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


def _slot_of(home: str, away: str, side: str, position: str) -> int:
    from .tryout import lineup

    for pos in lineup(home, away)[side]:
        if position.upper() in (pos["label"], str(pos["slot"])) or position == pos["player_id"]:
            return pos["slot"]
    raise SystemExit(f"No position {position!r}; options: {[p['label'] for p in lineup(home, away)[side]]}")


def cmd_tryout(args: argparse.Namespace) -> None:
    import json

    from .tryout import run_tryout

    slot = _slot_of(args.home, args.away, args.side, args.position)

    def show(r: dict) -> None:
        pr, d = r["progress"], r["difference"]
        print(f"  {pr['done']}/{pr['total']} matches  expected points {d.get('xpts', 0):+.2f} "
              f"(+/-{d.get('xpts_margin_95', 0):.2f})", flush=True)

    r = run_tryout(args.home, args.away, args.side, slot, args.player, n=args.n, cache_dir=args.cache, on_progress=show)
    w, i, d = r["with_candidate"], r["with_incumbent"], r["difference"]
    print(f"\n{r['candidate']['name']} at {r['position']['label']} for {r['fixture']['team']} "
          f"(instead of {r['incumbent']['name']}), {w['futures']} matches each:")
    print(f"  expected points  {w['xpts']:.2f} vs {i['xpts']:.2f}  ({d['xpts']:+.2f} +/-{d['xpts_margin_95']:.2f})")
    print(f"  team xG for      {w['xg_for']:.2f} vs {i['xg_for']:.2f};  against {w['xg_against']:.2f} vs {i['xg_against']:.2f}")
    print(f"  results          W{w['win']:.0%} D{w['draw']:.0%} L{w['loss']:.0%}  vs  W{i['win']:.0%} D{i['draw']:.0%} L{i['loss']:.0%}")
    for k in ("goals", "xg", "assists", "key_passes", "dribbles_completed", "tackles_won", "interceptions"):
        print(f"  {k:<18} {w['per90'][k]:.2f} vs {i['per90'][k]:.2f} per 90")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1)
        print(f"saved {args.out}")


def cmd_board(args: argparse.Namespace) -> None:
    import json
    from pathlib import Path

    from .tryout import board, featured_clip, run_tryout, save_clip

    sides = ["home", "away"] if args.side == "both" else [args.side]
    out: dict = {"home": args.home, "away": args.away, "n": args.n, "sides": {}}
    path = Path(args.out)
    if path.exists():
        out = json.loads(path.read_text(encoding="utf-8"))
        out.setdefault("sides", {})
    clips = Path(args.cache) / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for side in sides:
        b = board(args.home, args.away, side, ready=args.ready)
        done = {(r["position"]["slot"], r["candidate"]["player_id"]): r for r in out["sides"].get(side, {}).get("tryouts", [])}
        results = list(done.values())
        for pos in b:
            if args.positions and pos["label"] not in args.positions:
                continue
            for c in [c for c in pos["candidates"] if c["ready"]]:
                key = (pos["slot"], c["player_id"])
                if key in done and done[key]["with_candidate"]["futures"] >= args.n:
                    if not args.refresh:
                        continue
                    old = done[key]   # re-assemble from the cached runs (no new simulations), keep notes and clips
                    r = run_tryout(args.home, args.away, side, pos["slot"], c["player_id"], n=args.n, cache_dir=args.cache)
                    if old.get("featured") and r.get("featured"):
                        r["featured"]["clip"] = old["featured"].get("clip")
                    for k in ("scout",):
                        if old.get(k):
                            r[k] = old[k]
                    done[key] = r
                    continue
                print(f"{side} {pos['label']}: {c['name']} ({c['club']}) vs {pos['name']}", flush=True)
                r = run_tryout(args.home, args.away, side, pos["slot"], c["player_id"], n=args.n, cache_dir=args.cache)
                fe = r.get("featured")
                if fe:
                    clip = featured_clip(args.home, args.away, side, pos["slot"], c["player_id"], fe["seed"], fe)
                    if clip:
                        name = f"{side}-{pos['slot']}-{c['player_id']}.json.gz"
                        save_clip(clip, clips / name)
                        r["featured"]["clip"] = f"clips/{name}"
                d = r["difference"]
                print(f"   expected points {d['xpts']:+.2f} (+/-{d['xpts_margin_95']:.2f})", flush=True)
                done[key] = r
                results = list(done.values())
                out["sides"][side] = {"board": b, "tryouts": results}
                path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        out["sides"][side] = {"board": b, "tryouts": list(done.values())}
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved {path}")


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

    to = sub.add_parser("tryout", help="try any league player in a position for full matches")
    to.add_argument("--home", default="RIV")
    to.add_argument("--away", default="KIN")
    to.add_argument("--side", default="home", choices=["home", "away"])
    to.add_argument("--position", required=True, help="slot label (ST, LW, RCB...) or index")
    to.add_argument("--player", required=True, help="player id of the candidate (e.g. wes-20)")
    to.add_argument("--n", type=int, default=20, help="matches each way")
    to.add_argument("--cache", default=None, help="folder to cache runs in")
    to.add_argument("--out")
    to.set_defaults(func=cmd_tryout)

    bo = sub.add_parser("board", help="recruitment board: rank every candidate, pre-simulate the top picks")
    bo.add_argument("--home", default="RIV")
    bo.add_argument("--away", default="KIN")
    bo.add_argument("--side", default="both", choices=["home", "away", "both"])
    bo.add_argument("--ready", type=int, default=2, help="players to pre-simulate per position")
    bo.add_argument("--positions", nargs="*", help="only these slot labels")
    bo.add_argument("--n", type=int, default=12)
    bo.add_argument("--refresh", action="store_true", help="rebuild finished try-outs from the cache (no simulations)")
    bo.add_argument("--cache", required=True)
    bo.add_argument("--out", required=True)
    bo.set_defaults(func=cmd_board)

    sv = sub.add_parser("serve", help="live try-out server for the 3D page (engine only, daily cap)")
    sv.add_argument("--cache", required=True)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--origin", default="*")
    sv.set_defaults(func=lambda a: __import__("matchengine.serve", fromlist=["main"]).main(
        ["--cache", a.cache, "--host", a.host, "--port", str(a.port), "--origin", a.origin]))

    se = sub.add_parser("season", help="simulate a full league season (56 matches, no tracking)")
    se.add_argument("--out", required=True)
    se.add_argument("--demo", default=None, help="recorded match bundle to slot into the schedule")
    se.add_argument("--workers", type=int, default=None)
    se.set_defaults(func=lambda a: __import__("matchengine.season", fromlist=["main"]).main(
        ["--out", a.out] + (["--demo", a.demo] if a.demo else []) + (["--workers", str(a.workers)] if a.workers else [])))

    sh = sub.add_parser("schema", help="export JSON Schemas for events, frames and metadata")
    sh.add_argument("--out", required=True)
    sh.set_defaults(func=cmd_schema)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
