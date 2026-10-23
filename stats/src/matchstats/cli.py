"""Command line for the stats layer.

matchstats analyze BUNDLE [--out snapshot.json] [--moments moments.jsonl]
matchstats live [--home RIV --away KIN --seed 7 --speed 60]
matchstats ask BUNDLE TOOL [--args '{"player": "Duarte"}']
matchstats tools
matchstats train DIR [--report docs/MODEL_REPORT.md]
matchstats schema [--out docs/schema]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

from .analyzer import MatchAnalyzer
from .feed import live_feed, replay_bundle
from .queries import TOOLS, MatchQueries, QueryError

WIDTH = 64


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):  # non-standard streams
            stream.reconfigure(encoding="utf-8")


def _rule(char: str = "-") -> str:
    return char * WIDTH


def _row(label: str, home, away) -> str:
    return f"  {str(home):>12}   {label:^32}   {str(away):<12}"


def print_report(a: MatchAnalyzer) -> None:
    q = MatchQueries(a)
    sb = q.scoreboard()
    home, away = sb["home"], sb["away"]
    print(_rule("="))
    print(f"  {home['team']} {home['goals']}-{away['goals']} {away['team']}".center(WIDTH))
    status = "Full-time" if sb["as_of"]["finished"] else sb["as_of"]["minute"]
    print(status.center(WIDTH))
    for g in sb["goals"]:
        side = "  " if g["team"] == home["team"] else " " * 34
        print(f"{side}{g['minute']} {g['scorer']}")
    print(_rule("="))
    comparison = q.team_comparison(["attack", "possession", "passing", "defending", "physical"])
    picks = {
        "Expected goals (xG)",
        "Shots",
        "Shots on target",
        "Big chances",
        "Possession",
        "Field tilt",
        "Pass accuracy",
        "Progressive passes",
        "Opponents bypassed by passes",
        "PPDA (lower = more intense press)",
        "High turnovers",
        "Distance (km)",
        "Sprints",
    }
    for group in ("attack", "possession", "passing", "defending", "physical"):
        for r in comparison[group]:
            if r["stat"] in picks:
                h, w = r["home"], r["away"]
                if r["stat"] in ("Possession", "Field tilt", "Pass accuracy"):
                    h, w = f"{h:.0%}", f"{w:.0%}"
                elif isinstance(h, float):
                    fmt = ".2f" if "xG" in r["stat"] else ".1f"
                    h, w = format(h, fmt), format(w, fmt)
                print(_row(r["stat"].replace(" (lower = more intense press)", ""), h, w))
    wp = sb["win_probability"]
    print(_rule())
    print(
        f"  Win probability   {home['team']} {wp['home']:.0%}   draw {wp['draw']:.0%}   {away['team']} {wp['away']:.0%}"
    )
    print(f"  Models            {', '.join(f'{k} ({v})' for k, v in a.models.status().items())}")
    print(_rule())
    print("  Key moments (by importance)")
    for m in q.moments(order="importance", limit=8)["moments"]:
        weight = f"  ({m['moment_weight']:.0f}-pt win-prob swing)" if m["moment_weight"] else ""
        print(f"  {m['minute']:>6}  {m['title']}{weight}")
    print(_rule("="))


def cmd_analyze(args) -> int:
    t0 = time.perf_counter()
    a = MatchAnalyzer(frames=not args.no_frames).run(replay_bundle(args.bundle, frames=not args.no_frames))
    elapsed = time.perf_counter() - t0
    print_report(a)
    print(f"  {a.items:,} feed items analysed in {elapsed:.1f}s, {len(a.moments)} moments detected")
    if args.out:
        Path(args.out).write_text(json.dumps(a.snapshot(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  snapshot  -> {args.out}")
    if args.moments:
        with open(args.moments, "w", encoding="utf-8") as fh:
            for m in a.moments:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"  moments   -> {args.moments}")
    return 0


def cmd_live(args) -> int:
    a = MatchAnalyzer(frames=True)
    feed = live_feed(args.home, args.away, seed=args.seed, speed=args.speed, scenario=args.scenario)
    print(f"Following {args.home} v {args.away} (seed {args.seed}); moments appear as they are detected.\n")
    for m in a.stream(feed):
        if m["importance"] < args.min_importance:
            continue
        s = m["score"]
        wp = m["win_probability"]
        print(
            f"{m['minute_display']:>7}  {s['home']}-{s['away']}  {m['title']}"
            f"   (win prob {wp['home']:.0%}/{wp['draw']:.0%}/{wp['away']:.0%})",
            flush=True,
        )
    print()
    print_report(a)
    return 0


def cmd_ask(args) -> int:
    a = MatchAnalyzer(frames=not args.no_frames).run(replay_bundle(args.bundle, frames=not args.no_frames))
    try:
        result = MatchQueries(a).call(args.tool, json.loads(args.args) if args.args else {})
    except (QueryError, TypeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_replay(args) -> int:
    from .replay import build_replay, render_html

    view = "broadcast" if args.view_3d else "replay"
    out = Path(args.out or f"{Path(args.bundle).name}-{'broadcast' if args.view_3d else 'replay'}.html")
    out.write_text(render_html(build_replay(args.bundle), view=view), encoding="utf-8")
    print(f"replay -> {out} ({out.stat().st_size / 1e6:.1f} MB). Open it in any browser; no server needed.")
    return 0


def cmd_season(args) -> int:
    from .season import build_season, render_season

    links = dict(x.split("=", 1) for x in (args.link or []))
    data = build_season(args.root, links=links, board=args.board)
    out = Path(args.out)
    out.write_text(render_season(data), encoding="utf-8")
    print(f"season hub -> {out} ({out.stat().st_size / 1e6:.1f} MB, {len(data['matches'])} matches, {len(data['players'])} players)")
    return 0


def cmd_tools(_args) -> int:
    for name, spec in TOOLS.items():
        params = ", ".join(spec["parameters"]["properties"]) or "-"
        print(f"  {name:<16} {spec['description']}\n  {'':<16} params: {params}")
    return 0


def cmd_train(args) -> int:
    from .training import plot_report, report_markdown, train_all

    t0 = time.perf_counter()
    report = train_all(args.directory)
    md = report_markdown(report, Path(args.report).with_suffix(".png").name if args.report else None)
    print(md)
    if args.report:
        Path(args.report).write_text(md, encoding="utf-8")
        Path(args.report).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        plot_report(report, Path(args.report).with_suffix(".png"))
        print(f"report -> {args.report}")
    print(f"trained in {time.perf_counter() - t0:.1f}s; weights saved with the package")
    return 0


def cmd_schema(args) -> int:
    from .schema import export_schemas

    for path in export_schemas(args.out):
        print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8()
    p = argparse.ArgumentParser(prog="matchstats", description="Match statistics, models and key moments.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("analyze", help="analyse a recorded match bundle and print a report")
    s.add_argument("bundle")
    s.add_argument("--out", help="write the full snapshot JSON here")
    s.add_argument("--moments", help="write every moment as JSON lines here")
    s.add_argument("--no-frames", action="store_true", help="events only (no tracking-based metrics)")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("live", help="simulate a match and print moments as they happen")
    s.add_argument("--home", default="RIV")
    s.add_argument("--away", default="KIN")
    s.add_argument("--seed", type=int, default=7)
    s.add_argument(
        "--speed", type=float, default=0.0, help="0 = as fast as possible, 60 = one minute per second"
    )
    s.add_argument("--scenario", help="scenario JSON for the engine")
    s.add_argument("--min-importance", type=float, default=0.3)
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("ask", help="run one agent query against a recorded match")
    s.add_argument("bundle")
    s.add_argument("tool", choices=list(TOOLS))
    s.add_argument("--args", help="JSON object of tool arguments")
    s.add_argument("--no-frames", action="store_true")
    s.set_defaults(fn=cmd_ask)

    s = sub.add_parser("replay", help="build a self-contained 2D replay page for a recorded match")
    s.add_argument("bundle")
    s.add_argument("--out", help="output .html (default: <bundle>-replay.html)")
    s.add_argument(
        "--3d", dest="view_3d", action="store_true", help="3D broadcast view with the auto director"
    )
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("tools", help="list the agent queries")
    s.set_defaults(fn=cmd_tools)

    s = sub.add_parser("train", help="train and evaluate the models on simulated matches")
    s.add_argument("directory")
    s.add_argument("--report", help="write the accuracy report (markdown, plus .json) here")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("season", help="build the season hub page (table, results, players) from `matchengine season`")
    s.add_argument("root", help="season folder (with season.json)")
    s.add_argument("--out", default="season.html")
    s.add_argument("--link", action="append", help="MATCH_ID=URL of a 3D page for that match (repeatable)")
    s.add_argument("--board", default=None, help="Club mode board.json, to show try-outs on player pages")
    s.set_defaults(fn=cmd_season)

    s = sub.add_parser("mcp", help="run the Match Intelligence MCP server (stdio; --http for streamable HTTP)")
    s.add_argument("--http", action="store_true")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8766)
    s.add_argument("--data", default=None, help="the repo's data folder")
    s.set_defaults(fn=lambda a: __import__("matchstats.mcp_server", fromlist=["main"]).main(
        (["--http"] if a.http else []) + ["--host", a.host, "--port", str(a.port)] + (["--data", a.data] if a.data else [])) or 0)
    s = sub.add_parser("schema", help="export JSON Schemas for moments and snapshots")
    s.add_argument("--out", default="docs/schema")
    s.set_defaults(fn=cmd_schema)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
