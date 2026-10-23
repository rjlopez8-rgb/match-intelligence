"""``matchagents``: run the broadcast desk from the command line.

matchagents cover ../data/sample/riv-kin-7 --moment riv-kin-7-m044      # one moment, live models
matchagents match ../data/sample/riv-kin-7 --offline                     # whole match, $0
matchagents match ../data/sample/riv-kin-7 --limit 3                     # first 3 big moments, live
matchagents diagram                                                      # the workflow as Mermaid
matchagents spend                                                        # today's model spending
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from pathlib import Path

from .budget import BudgetGuard
from .config import all_settings
from .schemas import ViewerProfile

EST_USD_PER_MOMENT = 0.03  # gpt-5-mini incl. reasoning tokens, rounded up; GPT-6 Sol everywhere ~ $0.15


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]


def _print_segment(s: dict) -> None:
    plan = s["plan"] or {}
    head = f"{s['minute']:>6}  {s['type']}"
    if not s["aired"]:
        why = "; ".join(s["errors"]) or "producer: not worth airtime"
        print(f"{head}  -- skipped ({why})")
        return
    print(f"\n{head}  [{plan.get('segment')}, {plan.get('priority')}]  {plan.get('angle')}")
    for line in s["commentary"]["lines"]:
        print(f"        > {line}")
    checks = s["factcheck"]
    if s["commentary"]["fallback"]:
        print("        fact-check: drafts rejected, aired the stats layer's factual title instead")
    elif checks:
        print(f"        fact-check: approved on attempt {checks[-1]['attempt']}")
    d = s["director"]
    if d:
        cams = " -> ".join(f"{x['camera']}{' (slow-mo)' if x['slow_motion'] else ''}" for x in d["shots"])
        print(
            f"        director: live {d['live_camera']}"
            + (f"; replay {cams}" if d["replay"] else "")
            + (f"; graphic {d['graphic']}" if d["graphic"] != "none" else "")
        )
    t = s["telestration"]
    if t:
        print(f'        telestrator: {len(t["shapes"])} shapes, "{t["caption"]}"')
    sc, res = s["scenario"]["spec"], s["scenario"]["result"] or {}
    if sc:
        print(
            f"        scenario: {sc['question']} ({sc['intervention']} on {sc['event_id']}, {sc['simulations']} sims)"
        )
        _print_ghosts(res)
    c = s["cost"]
    print(f"        {c['model_calls']} model calls, {c['seconds']}s, ${c['usd']:.4f}")
    for e in s["errors"]:
        print(f"        ! {e}")


def _pct(d: dict) -> str:
    return f"home {d['home_win']:.0%} / draw {d['draw']:.0%} / away {d['away_win']:.0%}"


def _print_ghosts(res: dict) -> None:
    if not res:
        return
    if res.get("status") != "ok":
        print(f"        ghosts: {res.get('reason', res.get('status'))}")
        return
    if res["intervention"] == "player_swap":
        tk = res["taker"]
        print(
            f"        ghost scout ({res['candidates_evaluated']} players, {res['seconds']}s):"
            f" {tk['name']} {tk['goal_probability']:.0%} to score"
        )
        for r in res["best"][:3]:
            print(
                f"          {r['name']} ({r['club']}) {r['goal_probability']:.0%}"
                f" -> team win {r['team_win_probability']:.0%}"
            )
        return
    print(f"        ghost futures ({res['with_change']['futures']} each, {res['seconds']}s):")
    print(f"          with the change:  {_pct(res['with_change'])}")
    print(f"          as it happened:   {_pct(res['as_it_happened'])}")
    print(f"          margin of error:  +/-{res['with_change']['margin_95']:.0%} (more futures = smaller)")


async def _run(args, moments_filter) -> list[dict]:
    from .desk import Desk
    from .session import MatchSession

    session = MatchSession(args.bundle)
    profile = ViewerProfile(style=args.style, favourite_team=args.team, language=args.language)
    runner = None
    if not args.no_ghosts:
        from .ghost import GhostRunner

        runner = GhostRunner()
    if args.futures:
        import os

        os.environ["MATCH_GHOST_FUTURES"] = str(args.futures)
    desk = Desk(session, profile, offline_mode=args.offline, scenario_runner=runner)
    out = []
    for m in session.moments(args.min_importance):
        if not moments_filter(m):
            continue
        seg = await desk.cover(m)
        _print_segment(seg)
        out.append(seg)
        if args.limit and len(out) >= args.limit:
            break
        if desk.guard is not None and desk.guard.remaining() <= 0:
            print("\nDaily budget reached: stopping. Raise MATCH_DAILY_CAP_USD to continue (your call).")
            break
    total = sum(s["cost"]["usd"] for s in out)
    print(f"\n{len(out)} moments, {sum(s['aired'] for s in out)} aired, ${total:.4f} total")
    if desk.guard is not None:
        print(desk.guard.report())
    return out


def _confirm(args, n_estimate: int) -> bool:
    if args.offline or args.yes:
        return True
    guard = BudgetGuard()
    print("Models: " + ", ".join(f"{a}={s.deployment}" for a, s in all_settings().items()))
    print(f"About {n_estimate} moment(s), roughly ${n_estimate * EST_USD_PER_MOMENT:.2f} at most on gpt-5-mini.")
    print(f"Budget guard: ${guard.remaining():.2f} left of today's ${guard.cap:.2f} cap.")
    return input("Run with real models? [y/N] ").strip().lower() == "y"


EST_USD_PER_NOTE = 0.004
EST_USD_PER_CALL = 0.004  # gpt-5-mini with low reasoning, priced high like the budget guard
LANG_CODES = ("en", "es", "fr")


def edition_list(styles: list[str], languages: list[str], channels: bool) -> list:
    from .desk import Edition

    out = []
    for st in styles:
        for ch in (("neutral", "home", "away") if channels and st == "fan" else ("neutral",)):
            for lang in languages:
                out.append(Edition(st, ch, lang))
    return out


def _seed_cache(desk, segs: list[dict], want: int) -> None:
    """Reuse simulations from earlier runs (offline or live: the engine is the same) when they have enough futures."""
    for s in segs:
        sc = s.get("scenario") or {}
        spec, res = sc.get("spec"), sc.get("result") or {}
        if not spec or res.get("status") != "ok":
            continue
        n = (res.get("with_change") or {}).get("futures") or res.get("candidates_evaluated") or 0
        key = (spec["event_id"], spec["intervention"])
        have = (desk.scenario_cache.get(key, {}).get("with_change") or {}).get("futures", 0)
        if n >= want and n > have:
            desk.scenario_cache[key] = res


def _editions(args) -> int:
    from .desk import Desk
    from .session import MatchSession

    styles = [x.strip() for x in args.styles.split(",") if x.strip()]
    langs = [x.strip() for x in args.languages.split(",") if x.strip()]
    bad = [x for x in langs if x not in LANG_CODES]
    bad += [x for x in styles if x not in ("fan", "tactician", "analyst", "academy")]
    if bad:
        print(f"Unknown style or language: {', '.join(bad)} (languages: {', '.join(LANG_CODES)})")
        return 1
    eds = edition_list(styles, langs, not args.no_channels)
    mode = "offline" if args.offline else "live"
    saved = {}
    if args.out.exists():
        for s in json.loads(args.out.read_text(encoding="utf-8")):
            saved[s["moment_id"]] = s
    session = MatchSession(args.bundle)
    mid = session.meta["match_id"]
    picks = None if args.moments == "auto" else {m if "-m" in m else f"{mid}-{m}" for m in args.moments.split(",")}

    def chosen(m: dict) -> bool:
        return m["moment_id"] in picks if picks else (m["importance"] >= args.min_importance)

    # count the work left before spending anything
    probe = MatchSession(args.bundle)
    todo_moments, todo_eds, fresh = [], 0, 0
    for m in probe.moments(0.0):
        if not chosen(m):
            continue
        s = saved.get(m["moment_id"])
        reuse = s if s and s.get("mode") == mode else None
        missing = [e for e in eds if not reuse or e.key not in (reuse.get("editions") or {})]
        if missing:
            todo_moments.append(m["moment_id"])
            todo_eds += len(missing)
            fresh += 0 if reuse and reuse.get("brief") else 1
    if not todo_moments:
        print(f"Every edition is already in {args.out}. Nothing to do.")
        return 0
    calls = fresh * 6 + len(todo_moments) * len(styles) + round(todo_eds * 2 * 1.25)
    print(f"{len(todo_moments)} moments, {todo_eds} editions to write "
          f"({len(styles)} styles x channels x {len(langs)} languages).")
    if not args.offline and not args.yes:
        guard = BudgetGuard()
        est = calls * EST_USD_PER_CALL
        print("Models: " + ", ".join(f"{a}={s.deployment}" for a, s in all_settings().items() if a != "scout"))
        print(f"About {calls} model calls, roughly ${est:.2f} at most on gpt-5-mini.")
        print(f"Budget guard: ${guard.remaining():.2f} left of today's ${guard.cap:.2f} cap.")
        if est > guard.remaining():
            print("That is more than today's allowance: the run stops itself at the cap and keeps what it finished. "
                  "Run the same command tomorrow to finish (finished editions are never paid for twice).")
        if input("Run with real models? [y/N] ").strip().lower() != "y":
            return 1
    if args.futures:
        import os

        os.environ["MATCH_GHOST_FUTURES"] = str(args.futures)
    runner = None
    if not args.no_ghosts:
        from .ghost import GhostRunner

        runner = GhostRunner()
    desk = Desk(session, offline_mode=args.offline, scenario_runner=runner)
    seed = list(saved.values())
    for extra in args.reuse or []:
        seed += json.loads(Path(extra).read_text(encoding="utf-8"))
    _seed_cache(desk, seed, args.futures or 6)
    order = [m for m in saved]

    async def go():
        for m in session.moments(0.0):
            if m["moment_id"] not in todo_moments:
                continue
            prev = saved.get(m["moment_id"])
            seg = await desk.cover_editions(m, eds, prev if prev and prev.get("mode") == mode else None)
            seg["mode"] = mode
            saved[m["moment_id"]] = seg
            if m["moment_id"] not in order:
                order.append(m["moment_id"])
            done = sum(1 for e in eds if e.key in seg["editions"])
            fb = sum(1 for v in seg["editions"].values() if v["fallback"])
            print(f"{m['minute_display']:>6}  {m['type']:<16} {done}/{len(eds)} editions"
                  + (f", {fb} fell back to the verified headline" if fb else "")
                  + f"  ({seg['cost']['model_calls']} calls, ${seg['cost']['usd']:.4f})")
            v = seg["editions"].get(eds[0].key)
            if v:
                print(f"        {eds[0].key}: {v['lines'][0]}")
            for e in seg["errors"]:
                print(f"        ! {e}")
            out = sorted(saved.values(), key=lambda s: (s["period"], s["t"]))
            args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            if desk.guard is not None and desk.guard.remaining() <= 0:
                print("\nDaily budget reached: stopped. Finished editions are saved; run the same command tomorrow.")
                break

    asyncio.run(go())
    print(f"Saved {args.out}")
    if desk.guard is not None:
        print(desk.guard.report())
    return 0


def _scout(args) -> int:
    from matchengine.serve import compact

    from .scout import Scout

    board = json.loads(args.board.read_text(encoding="utf-8"))
    todo = [(side, r) for side, b in board.get("sides", {}).items() for r in b.get("tryouts", [])
            if args.redo or not r.get("scout")]
    if not todo:
        print("Every try-out already has a note (use --redo to rewrite).")
        return 0
    if not args.offline and not args.yes:
        guard = BudgetGuard()
        print(f"{len(todo)} notes, roughly ${len(todo) * EST_USD_PER_NOTE:.2f} at most on gpt-5-mini.")
        print(f"Budget guard: ${guard.remaining():.2f} left of today's ${guard.cap:.2f} cap.")
        if input("Run with real models? [y/N] ").strip().lower() != "y":
            return 1
    scout = Scout(offline_mode=args.offline)

    async def go():
        for side, r in todo:
            t = compact(r)
            t["team"] = r["fixture"]["team"]
            note = await scout.note(t)
            r["scout"] = note
            print(f"{side} {r['position']['label']}: {r['candidate']['name']} -> {note['verdict'].upper()}: {note['headline']}")
            print(f"    {note['report']}")
            args.board.write_text(json.dumps(board, indent=1), encoding="utf-8")
            if scout.guard is not None and scout.guard.remaining() <= 0:
                print("Daily budget reached: stopping.")
                break

    asyncio.run(go())
    if scout.guard is not None:
        print(scout.guard.report())
    return 0


EST_USD_PER_RECAP_CALL = 0.004   # gpt-5-mini, low reasoning, priced high like the budget guard


def _recaps(args) -> int:
    """Recaps for every season match: research first ($0, engine), then the Recap agent (asks before spending)."""
    from matchstats.season import build_season

    from .budget import BudgetExceeded
    from .recap import TURNING_FILE, evidence, fallback_recap, research_season, squad_names

    tp_path = args.season / TURNING_FILE
    if args.futures or not tp_path.exists():
        research_season(args.season, n=args.futures or 40, workers=args.workers, demo=args.demo)
    research = json.loads(tp_path.read_text(encoding="utf-8"))["matches"] if tp_path.exists() else {}
    if args.research_only:
        return 0
    season = build_season(args.season)
    out = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    todo = [m for m in season["matches"] if args.redo or m["id"] not in out]
    if not todo:
        print(f"Every match already has a recap in {args.out} (use --redo to rewrite).")
        return 0
    calls = 1 if args.no_review else 2
    if not args.offline and not args.yes:
        guard = BudgetGuard()
        models = ["recap=" + all_settings()["recap"].deployment] + ([] if args.no_review else ["factchecker=" + all_settings()["factchecker"].deployment])
        print("Models: " + ", ".join(models))
        print(f"{len(todo)} recaps, {calls} model calls each ({'writer only' if args.no_review else 'writer + fact-check'}): "
              f"roughly ${len(todo) * calls * EST_USD_PER_RECAP_CALL:.2f}, at most ${len(todo) * calls * 2 * EST_USD_PER_RECAP_CALL:.2f} "
              f"if every recap needs its one rewrite.")
        print(f"Budget guard: ${guard.remaining():.2f} left of today's ${guard.cap:.2f} cap. The run stops itself at the cap "
              "and keeps what it finished; the same command later does the rest.")
        if input("Run with real models? [y/N] ").strip().lower() != "y":
            return 1
    from .recap import Recapper

    agent = Recapper(offline_mode=args.offline, review=not args.no_review)
    names = squad_names(season)

    async def go():
        for k, m in enumerate(todo, 1):
            ev = evidence(m, season, research.get(m["id"]))
            try:
                rec = await agent.recap(ev, names)
            except BudgetExceeded as exc:
                print(f"\nDaily budget reached: stopped ({exc}). Finished recaps are saved; run the same command later.")
                break
            except Exception as exc:  # a model or network error: the code-written recap, never a crash
                rec = {**fallback_recap(ev), "checked": True, "attempts": 0, "fallback": True, "by": "code", "error": str(exc)[:300]}
            out[m["id"]] = rec
            args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
            tag = "fallback (code)" if rec.get("fallback") else f"approved on attempt {rec['attempts']}"
            print(f"{k:>2}/{len(todo)}  {m['id']:<14} {rec['headline']}  [{tag}]")

    asyncio.run(go())
    done = sum(1 for m in season["matches"] if m["id"] in out)
    fb = sum(1 for r in out.values() if r.get("fallback"))
    print(f"{done}/{len(season['matches'])} recaps in {args.out} ({fb} code-written fallbacks); this run ${agent.trace.usd:.4f}.")
    if agent.guard is not None:
        print(agent.guard.report())
    return 0


def main(argv: list[str] | None = None) -> int:
    _utf8()
    p = argparse.ArgumentParser(prog="matchagents", description="The AI broadcast desk")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("bundle", type=Path)
        sp.add_argument("--offline", action="store_true", help="rule-based agents, no model calls, $0")
        sp.add_argument("--style", default="fan", choices=["fan", "tactician", "analyst", "academy"])
        sp.add_argument("--team", default=None, help="the viewer's favourite team")
        sp.add_argument("--language", default="English")
        sp.add_argument("--out", type=Path, default=None, help="save segments as JSON")
        sp.add_argument("--yes", action="store_true", help="skip the cost confirmation")
        sp.add_argument("--no-ghosts", action="store_true", help="skip ghost-future simulations (faster)")
        sp.add_argument("--futures", type=int, default=0, help="exact ghost futures per what-if")

    c = sub.add_parser("cover", help="cover one moment")
    common(c)
    c.add_argument("--moment", required=True)
    m = sub.add_parser("match", help="cover every big moment of a match")
    common(m)
    m.add_argument("--min-importance", type=float, default=0.6)
    m.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("render", help="build the 3D broadcast page with the desk's segments")
    r.add_argument("bundle", type=Path)
    r.add_argument("segments", type=Path, help="JSON saved with --out")
    r.add_argument("--out", type=Path, default=None)
    sc = sub.add_parser("scout", help="the Scout agent writes a note for every pre-simulated try-out on a board")
    sc.add_argument("board", type=Path, help="board.json from `matchengine board`")
    sc.add_argument("--offline", action="store_true")
    sc.add_argument("--yes", action="store_true")
    sc.add_argument("--redo", action="store_true", help="rewrite notes that already exist")
    ed = sub.add_parser("editions", help="every big moment told for every viewer: styles x channels x languages")
    ed.add_argument("bundle", type=Path)
    ed.add_argument("--out", type=Path, required=True, help="editions JSON (resumes if it exists)")
    ed.add_argument("--moments", default="auto", help="'auto' (importance >= --min-importance) or ids like m002,m026")
    ed.add_argument("--min-importance", type=float, default=0.7)
    ed.add_argument("--styles", default="fan,tactician,analyst,academy")
    ed.add_argument("--languages", default="en,es,fr")
    ed.add_argument("--no-channels", action="store_true", help="skip the clubs' own Fan channels")
    ed.add_argument("--offline", action="store_true")
    ed.add_argument("--yes", action="store_true")
    ed.add_argument("--no-ghosts", action="store_true")
    ed.add_argument("--futures", type=int, default=0, help="ghost futures per what-if (default: the agent's choice)")
    ed.add_argument("--reuse", action="append", help="segments JSON whose simulations can be reused (e.g. demo.json)")
    rc = sub.add_parser("recaps", help="a recap for every season match, with a simulated 'what decided it?'")
    rc.add_argument("season", type=Path, help="folder from `matchengine season`")
    rc.add_argument("--out", type=Path, default=Path("recaps.json"), help="recaps JSON (resumes if it exists)")
    rc.add_argument("--demo", type=Path, default=None, help="the demo bundle (its turning point uses its own checkpoints)")
    rc.add_argument("--futures", type=int, default=0,
                    help="what-if futures each way (default 40 when there is no research yet; giving it re-runs the research: "
                         "$0, engine only, from the committed frozen turning points, so any machine can do it)")
    rc.add_argument("--research-only", action="store_true", help="only the turning points and what-ifs ($0)")
    rc.add_argument("--no-review", action="store_true", help="skip the Fact-checker model review (about half the cost)")
    rc.add_argument("--workers", type=int, default=None)
    rc.add_argument("--offline", action="store_true")
    rc.add_argument("--yes", action="store_true")
    rc.add_argument("--redo", action="store_true", help="rewrite recaps that already exist")
    st = sub.add_parser("site", help="build the website: season hub (index.html) + every match in 3D (watch.html), linked together")
    st.add_argument("--season", type=Path, required=True, help="folder from `matchengine season`")
    st.add_argument("--demo", type=Path, required=True, help="the demo match bundle (with tracking)")
    st.add_argument("--segments", type=Path, default=None, help="AI desk segments JSON for the demo match")
    st.add_argument("--out", type=Path, default=Path("site"))
    st.add_argument("--tracking", type=Path, default=None,
                    help="bundles from `matchengine season --tracking DIR`: every match gets 3D highlights, "
                         "and each matchday's match of the week the full match")
    st.add_argument("--workers", type=int, default=None)
    st.add_argument("--recaps", type=Path, default=None, help="recaps JSON from `matchagents recaps`")
    sub.add_parser("diagram", help="print the desk workflow as Mermaid")
    sub.add_parser("spend", help="today's model spending")
    args = p.parse_args(argv)

    if args.cmd == "render":
        from matchstats.replay import build_replay, render_html

        segs = json.loads(args.segments.read_text(encoding="utf-8"))
        data = build_replay(args.bundle, segs)
        out = args.out or Path(f"{data['match']['id']}-desk-broadcast.html")
        out.write_text(render_html(data, view="broadcast"), encoding="utf-8")
        print(f"{len(data['segments'])} desk segments -> {out}  (open it in a browser)")
        return 0
    if args.cmd == "site":
        from .site import build_site

        r = build_site(args.season, args.demo, args.out, args.segments, args.tracking, args.workers, args.recaps)
        print(f"site -> {args.out}: index.html (season hub), watch.html (3D), {r['matches']} matches in 3D "
              f"({len(r['full'])} full, highlights for all), {r['mb']} MB "
              f"(highlights {r['hl_mb']} MB, full matches {r['full_mb']} MB).")
        print(f"Recaps: {r['recaps_by_agent']} by the Recap agent, the rest written from the stats"
              + ("" if args.recaps else " (pass --recaps recaps.json from `matchagents recaps`)") + ".")
        if r["missing"]:
            print(f"{len(r['missing'])} matches have no tracking yet, so no 3D: run "
                  f"`matchengine season --out {args.season} --demo {args.demo} --tracking ../data/matches/season`, "
                  f"then pass --tracking ../data/matches/season here.")
        print("Serve it to view (pages load their match data): cd into the folder, then `python -m http.server 8000`.")
        return 0
    if args.cmd == "scout":
        return _scout(args)
    if args.cmd == "recaps":
        return _recaps(args)
    if args.cmd == "editions":
        return _editions(args)
    if args.cmd == "spend":
        print(BudgetGuard().report())
        return 0
    if args.cmd == "diagram":
        from .desk import Desk, workflow_mermaid
        from .session import MatchSession

        sample = Path(__file__).resolve().parents[3] / "data" / "sample" / "riv-kin-7"
        print(workflow_mermaid(Desk(MatchSession(sample), offline_mode=True)))
        return 0

    if args.cmd == "cover":
        args.min_importance, args.limit = 0.0, 1
        if not _confirm(args, 1):
            return 1
        segs = asyncio.run(_run(args, lambda mo: mo["moment_id"] == args.moment))
        if not segs:
            print(f"Moment {args.moment} not found.")
            return 1
    else:
        if not _confirm(args, args.limit or 8):
            return 1
        segs = asyncio.run(_run(args, lambda mo: True))
    if args.out:
        args.out.write_text(json.dumps(segs, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
