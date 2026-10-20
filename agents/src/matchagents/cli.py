"""``matchagents``: run the broadcast desk from the command line.

matchagents cover ../data/sample/riv-kin-7 --moment riv-kin-7-m042      # one moment, live models
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

        os.environ["MATCH_GHOST_FUTURES_MAX"] = str(args.futures)
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
        sp.add_argument("--futures", type=int, default=0, help="cap ghost futures per what-if (default 40)")

    c = sub.add_parser("cover", help="cover one moment")
    common(c)
    c.add_argument("--moment", required=True)
    m = sub.add_parser("match", help="cover every big moment of a match")
    common(m)
    m.add_argument("--min-importance", type=float, default=0.6)
    m.add_argument("--limit", type=int, default=0)
    sub.add_parser("diagram", help="print the desk workflow as Mermaid")
    sub.add_parser("spend", help="today's model spending")
    args = p.parse_args(argv)

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
