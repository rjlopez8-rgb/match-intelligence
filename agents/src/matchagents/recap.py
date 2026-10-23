"""Match recaps: what happened, and what decided it, settled by simulation.

Two steps, so the paid one never needs a special machine:

1. **Research** (engine and stats layer, $0, Linux): for every season match, the turning point is the goal or chance
   that moved win probability most (the stats layer's moment weight). The match is replayed exactly to that shot and
   frozen; then the same futures are played with its outcome flipped and as it happened (common random numbers), so
   the difference is the effect of that one moment. Exact replay only works on the platform that generated the
   season, so the results are saved to ``data/season/turning_points.json`` and committed.
2. **Writing** (the Recap agent on Microsoft Foundry, or rule-based offline): a headline, a short recap and one
   sentence on what decided it. Same safety net as the commentary: every number and scoreline must be in the
   evidence and every player named must have been involved (code checks that cannot be talked round), one rewrite,
   then a code-written recap that is always true.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

TURNING_FILE = "turning_points.json"
TURNING_TYPES = ("goal", "great_save", "big_chance", "woodwork")
DEFAULT_FUTURES = 40


# ------------------------------------------------------------------------------ research
def _bundle(folder: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    path = folder / "events.jsonl.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
    else:
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return meta, events


def turning_point(meta: dict, events: list[dict]) -> dict | None:
    """The goal or chance that moved the match most, with the shot behind it (None if there was no shot)."""
    from matchstats import MatchAnalyzer

    a = MatchAnalyzer()
    a.process("meta", meta)
    for e in events:
        a.process("event", e)
    by_id = {e["event_id"]: e for e in events}
    side_of = {meta["home"]["team_id"]: "home", meta["away"]["team_id"]: "away"}
    best = None
    for m in a.moments:
        if m["type"] not in TURNING_TYPES or m.get("moment_weight") is None:
            continue
        shot = None
        for eid in m["evidence"]:
            ev = by_id.get(eid)
            if ev and ev["type"] == "goal":
                ev = by_id.get((ev.get("details") or {}).get("shot_event_id", ""))
            if ev and ev["type"] == "shot":
                shot = ev
                break
        if shot is None:         # an own goal has no shot to replay
            continue
        if best is None or abs(m["moment_weight"]) > abs(best[0]["moment_weight"]):
            best = (m, shot)
    if best is None:             # nothing the stats layer flagged: the best chance of the match
        shots = [e for e in events if e["type"] == "shot"]
        if not shots:
            return None
        shot = max(shots, key=lambda e: a.ctx.ann.get(e["event_id"], {}).get("xg") or 0.0)
        m = {"moment_id": None, "type": "chance", "title": None, "moment_weight": None}
    else:
        m, shot = best
    d = shot.get("details") or {}
    scored = d.get("outcome") == "goal"
    xg = a.ctx.ann.get(shot["event_id"], {}).get("xg")
    return {
        "moment_id": m["moment_id"], "type": m["type"], "title": m.get("title"), "weight_pts": m.get("moment_weight"),
        "event_id": shot["event_id"], "minute": shot["minute_display"], "period": shot["period"], "t": shot["t"],
        "player_id": shot.get("player_id"), "player": shot.get("player_name") or a.ctx.player_name(shot.get("player_id")),
        "side": side_of.get(shot.get("team_id"), ""), "outcome": d.get("outcome"), "shot_kind": d.get("shot_kind"),
        "xg": round(float(xg), 2) if isinstance(xg, (int, float)) else None,
        "change": "shot_misses" if scored else "shot_scores",
    }


FORKS_FILE = "turning_forks.pkl.gz"


def _freeze_one(args: tuple) -> dict:
    """Phase 1 for one match (a worker process): find the turning point and freeze the match at that shot.
    Needs an exact replay, so it only works where the season was generated (Linux)."""
    from matchengine.ghost import fork_at_shot, load_checkpoint

    folder, checkpoints = Path(args[0]), args[1]
    meta, events = _bundle(folder)
    tp = turning_point(meta, events)
    final = next((e["details"]["final_score"] for e in reversed(events) if e["type"] == "match_end"), None)
    out = {"match_id": meta["match_id"], "final": final, "turning_point": tp, "fork": None}
    if tp is None:
        return {**out, "status": "no_shots"}
    shot = next(e for e in events if e["event_id"] == tp["event_id"])
    try:   # a match that ships frozen checkpoints (the demo) uses them; the others are replayed to the shot
        fork = (load_checkpoint(checkpoints, shot["event_id"]) if checkpoints else None) or fork_at_shot(meta, shot)
    except ValueError as exc:
        return {**out, "status": "replay_failed", "reason": str(exc)}
    fork.event = shot
    return {**out, "status": "ok", "fork": fork}


def _futures_one(args: tuple) -> dict:
    """Phase 2 for one match (a worker process): the same futures with the turning point flipped and as it happened.
    Runs from the frozen state, so any machine can do it."""
    from matchengine.ghost import what_if_shot

    item, n = args
    tp, fork = item["turning_point"], item["fork"]
    t0 = time.perf_counter()
    res = what_if_shot(None, fork.event, tp["change"], n=n, record=1, parallel=False, fork=fork)
    fe = (res.get("featured") or [{}])[0]
    return {
        "match_id": item["match_id"], "final": item["final"], "turning_point": tp, "status": "ok",
        "seconds": round(time.perf_counter() - t0, 1), "futures": n,
        "actual_outcome": res["actual_outcome"], "changed_outcome": res["changed_outcome"],
        "with_change": _short(res["with_change"]), "as_it_happened": _short(res["as_it_happened"]),
        "featured": {"final": fe.get("final"), "goals_after": fe.get("goals_after", [])} if fe else None,
        "method": res["method"],
    }


def _short(s: dict) -> dict:
    return {k: s[k] for k in ("futures", "home_win", "draw", "away_win", "most_common_scores", "margin_95")}


def _workers(n_jobs: int, workers: int | None) -> int:
    return max(1, min(n_jobs, workers or int(os.environ.get("MATCH_WORKERS") or (os.cpu_count() or 2))))


def load_forks(season_dir: Path) -> dict:
    import pickle

    path = Path(season_dir) / FORKS_FILE
    if not path.exists():
        return {"matches": {}}
    with gzip.open(path, "rb") as fh:
        return pickle.load(fh)   # the repo's own data: only load files you trust


def freeze_season(season_dir: Path, demo: Path | None = None, workers: int | None = None, progress=print) -> dict:
    """Phase 1 for every match: turning points and frozen states, saved to ``turning_forks.pkl.gz`` (resumable)."""
    import pickle

    from matchengine.engine import ENGINE_VERSION

    season_dir = Path(season_dir)
    forks = load_forks(season_dir)
    forks.setdefault("engine_version", ENGINE_VERSION)
    index = json.loads((season_dir / "season.json").read_text(encoding="utf-8"))
    todo = [m["match_id"] for m in index["matches"] if (forks["matches"].get(m["match_id"]) or {}).get("status") != "ok"]
    if not todo:
        return forks
    demo_id = json.loads((Path(demo) / "meta.json").read_text(encoding="utf-8"))["match_id"] if demo else None
    k = _workers(len(todo), workers)
    progress(f"Finding the turning point of {len(todo)} matches and freezing each at that shot (exact replay) on {k} worker(s)")
    failed = []
    with ProcessPoolExecutor(max_workers=k) as ex:
        futs = [ex.submit(_freeze_one, (str(season_dir / mid), str(demo) if mid == demo_id else None)) for mid in todo]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            forks["matches"][r["match_id"]] = r
            tp = r.get("turning_point") or {}
            if r["status"] == "ok":
                progress(f"  {i}/{len(todo)}  {r['match_id']}  {tp['minute']} {tp['player']} ({tp['type']})")
            else:
                failed.append(r["match_id"])
                progress(f"  {i}/{len(todo)}  {r['match_id']}  {r['status']} {r.get('reason', '')}")
    with gzip.open(season_dir / FORKS_FILE, "wb", compresslevel=9) as fh:
        pickle.dump(forks, fh)
    if any(forks["matches"][m]["status"] == "replay_failed" for m in failed):
        progress("Some matches could not be replayed exactly on this machine. Freeze them on Linux (WSL or GitHub "
                 "Actions), where the season was generated; the futures can then run anywhere.")
    return forks


def research_season(season_dir: Path, n: int = DEFAULT_FUTURES, workers: int | None = None, only: list[str] | None = None,
                    demo: Path | None = None, progress=print) -> dict:
    """Turning points and what-ifs for every match, saved to ``turning_points.json`` (resumable; a different
    ``n`` re-runs the futures). Freezing needs the platform that made the season; the futures run anywhere."""
    season_dir = Path(season_dir)
    forks = freeze_season(season_dir, demo, workers, progress)
    path = season_dir / TURNING_FILE
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"matches": {}}
    todo = [mid for mid, f in forks["matches"].items()
            if f["status"] == "ok" and (not only or mid in only)
            and ((data["matches"].get(mid) or {}).get("status") != "ok" or data["matches"][mid].get("futures") != n)]
    for mid, f in forks["matches"].items():          # no shot, or not replayable here: say so in the results
        if f["status"] != "ok" and (data["matches"].get(mid) or {}).get("status") != "ok":
            data["matches"][mid] = {k: v for k, v in f.items() if k != "fork"}
    if not todo:
        progress(f"Every match already has its what-if ({n} futures each way) in {path}.")
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        return data
    k = _workers(len(todo), workers)
    progress(f"What-ifs for {len(todo)} matches: {n} futures with the turning point flipped and {n} as it happened, "
             f"on {k} worker(s)")
    with ProcessPoolExecutor(max_workers=k) as ex:
        futs = [ex.submit(_futures_one, (forks["matches"][mid], n)) for mid in todo]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            data["matches"][r["match_id"]] = r
            path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")   # resumable
            tp, w, a = r["turning_point"], r["with_change"], r["as_it_happened"]
            progress(f"  {i}/{len(todo)}  {r['match_id']}  {tp['minute']} {tp['player']} "
                     f"{'scores' if tp['change'] == 'shot_misses' else 'misses'} -> flipped: home win "
                     f"{a['home_win']:.0%} -> {w['home_win']:.0%}, away win {a['away_win']:.0%} -> {w['away_win']:.0%} "
                     f"[{r['seconds']}s]")
    return data


# ------------------------------------------------------------------------------- writing
WHAT = {"goal": "goal", "great_save": "big save", "big_chance": "big chance", "woodwork": "shot off the woodwork",
        "chance": "best chance"}


def _pct(x: float) -> int:
    return int(round(float(x) * 100))


def evidence(report: dict, season: dict, research: dict | None) -> dict:
    """Everything the Recap agent may say, in reader-friendly form (rounded once, here). Nothing else is allowed:
    the code check rejects any number or scoreline that is not in here."""
    C, P = season["clubs"], season["players"]
    name = {"home": C[report["home"]]["name"], "away": C[report["away"]]["name"]}

    def pname(pid, fallback=None):
        return (P.get(pid) or {}).get("name") or fallback

    st = report["stats"]
    ev = {
        "competition": season["competition"], "matchday": report["md"], "date": report["date"], "venue": report["venue"],
        "home_team": name["home"], "away_team": name["away"],
        "final_score": {"home": report["score"]["home"], "away": report["score"]["away"]},
        "goals": [{"minute": g["min"], "team": name[g["side"]], "scorer": pname(g["scorer"], g.get("scorer_name")),
                   "assist": pname(g.get("assist"), g.get("assist_name")), "penalty": bool(g.get("pen")),
                   "own_goal": bool(g.get("og"))} for g in report["goals"]],
        "red_cards": [{"minute": c["min"], "team": name[c["side"]], "player": pname(c["player"], c.get("name"))}
                      for c in report["cards"] if c["card"] != "yellow"],
        "expected_goals": {"home_team": round(report["xg"][0], 2), "away_team": round(report["xg"][1], 2)},
        "possession_percent": {"home_team": _pct(st["possession"][0]), "away_team": _pct(st["possession"][1])},
        "shots": {"home_team": st["shots"][0], "away_team": st["shots"][1]},
        "shots_on_target": {"home_team": st["shots_on_target"][0], "away_team": st["shots_on_target"][1]},
        "result_the_chances_deserved_percent": {"home_team_win": _pct(report["xres"][0]), "draw": _pct(report["xres"][1]),
                                                "away_team_win": _pct(report["xres"][2])},
    }
    if report.get("potm"):
        line = next((r for r in (P.get(report["potm"]) or {}).get("log", []) if r["m"] == report["id"]), {})
        ev["player_of_the_match"] = {"name": pname(report["potm"]), "team": C[(P.get(report["potm"]) or {}).get("club", report["home"])]["name"],
                                     "goals": int(line.get("goals", 0)), "assists": int(line.get("assists", 0))}
    r = research or {}
    tp = r.get("turning_point")
    if tp and r.get("status") == "ok":
        flipped = "had been saved" if tp["change"] == "shot_misses" else "had gone in"
        w, a = r["with_change"], r["as_it_happened"]
        ev["what_decided_it"] = {
            "moment": f"{tp['player']}'s {WHAT.get(tp['type'], 'chance')} at {tp['minute']}",
            "minute": tp["minute"], "player": tp["player"], "team": name.get(tp["side"], ""),
            "chance_quality_xg": tp["xg"],
            "simulation": f"the match was frozen the instant the shot was struck and played to full time {r['futures']} "
                          f"times as it happened and {r['futures']} times as if the shot {flipped}, with the same "
                          f"random streams",
            "if_it": flipped,
            "simulated_futures_each_way": r["futures"],
            "as_it_happened_percent": {"home_team_win": _pct(a["home_win"]), "draw": _pct(a["draw"]), "away_team_win": _pct(a["away_win"])},
            "if_flipped_percent": {"home_team_win": _pct(w["home_win"]), "draw": _pct(w["draw"]), "away_team_win": _pct(w["away_win"])},
            "margin_of_error_percent": _pct(w["margin_95"]),
            "most_common_final_score_if_flipped": (w.get("most_common_scores") or [["", 0]])[0][0],
        }
    return ev


def _winner(ev: dict) -> tuple[str | None, str | None]:
    h, a = ev["final_score"]["home"], ev["final_score"]["away"]
    if h == a:
        return None, None
    return (ev["home_team"], ev["away_team"]) if h > a else (ev["away_team"], ev["home_team"])


def fallback_recap(ev: dict) -> dict:
    """Written by code from the evidence alone: always true, never exciting."""
    h, a = ev["final_score"]["home"], ev["final_score"]["away"]
    win, lose = _winner(ev)
    head = f"{win} beat {lose} {max(h, a)}-{min(h, a)}" if win else f"{ev['home_team']} and {ev['away_team']} draw {h}-{a}"
    if ev["goals"]:
        import re   # minutes only while the sentence stays readable (at most three numbers, 45+4' counts two)
        many = sum(len(re.findall(r"\d+", g["minute"])) for g in ev["goals"]) > 3
        parts = [f"{g['scorer']}" + ("" if many else f" ({g['minute']}{', pen' if g['penalty'] else ''})")
                 + (" (own goal)" if g["own_goal"] else "") for g in ev["goals"]]
        scored = "Goals: " + ", ".join(parts) + "."
    else:
        scored = "Neither side scored."
    xg = ev["expected_goals"]
    recap = (f"{scored} {ev['home_team']} had {ev['possession_percent']['home_team']}% of the ball, and the chances were "
             f"worth {xg['home_team']:.2f} expected goals to {xg['away_team']:.2f}.")
    d = ev.get("what_decided_it")
    if d:
        team_key = "home_team_win" if d["team"] == ev["home_team"] else "away_team_win"
        decided = (f"The turning point was {d['moment']}. In {d['simulated_futures_each_way']} simulated futures where it "
                   f"{d['if_it']}, {d['team']} won {d['if_flipped_percent'][team_key]}% of the time, against "
                   f"{d['as_it_happened_percent'][team_key]}% as it happened.")
    else:
        decided = ""      # no simulation for this match yet: say nothing rather than guess
    return {"headline": head, "recap": recap, "decided": decided}


def squad_names(season: dict, report: dict | None = None) -> set[str]:
    """Every player name in the league (full and surname), to catch a recap that brings in someone not involved."""
    out = set()
    for p in season["players"].values():
        if p.get("name") and p["name"] != p.get("id"):
            out.add(p["name"])
            out.add(p["name"].split()[-1])
    return out


def check_names(texts: list[str], ev: dict, names: set[str]) -> list[dict]:
    """Issues for any player of either squad named in the recap who is not in the evidence."""
    allowed = json.dumps(ev, ensure_ascii=False)
    issues = []
    for i, text in enumerate(texts):
        for n in sorted(names, key=len, reverse=True):
            if n in text and n not in allowed:
                issues.append({"line": i, "claim": n, "problem": "player not in the evidence: only name players involved"})
    return issues


def check_recap(rec: dict, ev: dict, names: set[str]) -> list[dict]:
    from .checks import check_numbers, check_style
    from .schemas import Commentary, Line

    import re

    texts = [rec["headline"], rec["recap"], rec["decided"]]
    sentences = [x for t in texts for x in re.split(r"(?<=[.!?])\s+", t.strip()) if x]   # style is judged per sentence
    as_lines = Commentary(lines=[Line(text=t, facts_used=[]) for t in sentences], headline="", caption="", question=None)
    return check_numbers(as_lines, [ev]) + check_style(as_lines) + check_names(texts, ev, names)


def _offline_responder(agent: str, messages, options) -> str:
    from .offline import payload

    return json.dumps(fallback_recap(payload(messages)["_ev"]))


REVIEW_NOTE = ("These lines are a written match recap for the league website, one sentence per line, not live commentary: "
               "up to three numbers in a sentence is fine. Check every claim against DATA: scorers, minutes, teams, the "
               "score, who did what, and that the simulation result is stated with the right team, direction and numbers.")


def _offline_review(agent: str, messages, options) -> str:
    return json.dumps({"approved": True, "issues": []})


class Recapper:
    """The Recap agent writes (a model on Foundry, or by rule offline); the code checks and then the Fact-checker
    model review decide. One rewrite, then the code-written recap."""

    def __init__(self, offline_mode: bool = False, guard=None, http_client=None, review: bool = True):
        from agent_framework import Agent

        from .budget import BudgetGuard
        from .config import settings_for
        from .llm import ScriptedChatClient, Trace, foundry_client
        from .prompts import INSTRUCTIONS

        self.offline = offline_mode
        self.trace = Trace()
        self.guard = None if offline_mode else (guard or BudgetGuard())
        s = settings_for("recap")
        self.model = "offline" if offline_mode else s.deployment
        client = (ScriptedChatClient("recap", _offline_responder, self.trace) if offline_mode
                  else foundry_client(s, self.trace, self.guard, http_client=http_client))
        options: dict = {"model": s.deployment, "max_tokens": s.max_tokens}
        if s.reasoning_effort and not offline_mode:
            options["reasoning_effort"] = s.reasoning_effort
        self.agent = Agent(client=client, name="recap", instructions=INSTRUCTIONS["recap"], default_options=options)
        self.reviewer = None
        if review:
            f = settings_for("factchecker")
            rclient = (ScriptedChatClient("factchecker", _offline_review, self.trace) if offline_mode
                       else foundry_client(f, self.trace, self.guard, http_client=http_client))
            ropts: dict = {"model": f.deployment, "max_tokens": f.max_tokens}
            if f.reasoning_effort and not offline_mode:
                ropts["reasoning_effort"] = f.reasoning_effort
            self.reviewer = Agent(client=rclient, name="factchecker", instructions=INSTRUCTIONS["factchecker"],
                                  default_options=ropts)

    async def _ask(self, ev: dict, issues: list[dict] | None) -> dict:
        from .schemas import MatchRecap

        task = "Write the recap of this match."
        if issues:
            task += " Your previous draft failed the fact-check; fix exactly these problems: " + json.dumps(issues)
        data = {**ev, "_ev": ev} if self.offline else ev
        r = await self.agent.run(f"{task}\n\nDATA:\n{json.dumps(data, ensure_ascii=False)}",
                                 options={"response_format": MatchRecap})
        v = r.value if isinstance(r.value, MatchRecap) else MatchRecap.model_validate_json(r.text)
        return v.model_dump()

    async def _review(self, rec: dict, ev: dict) -> list[dict]:
        import re

        from .schemas import Verdict

        lines = [x for t in (rec["headline"], rec["recap"], rec["decided"]) for x in re.split(r"(?<=[.!?])\s+", t.strip()) if x]
        r = await self.reviewer.run(f"Check the lines. {REVIEW_NOTE}\n\nLINES:\n{json.dumps(lines, ensure_ascii=False)}"
                                    f"\n\nDATA:\n{json.dumps(ev, ensure_ascii=False)}", options={"response_format": Verdict})
        v = r.value if isinstance(r.value, Verdict) else Verdict.model_validate_json(r.text)
        return [] if v.approved else ([i.model_dump() for i in v.issues] or [{"line": 0, "claim": "", "problem": "rejected"}])

    async def recap(self, ev: dict, names: set[str]) -> dict:
        issues: list[dict] | None = None
        for attempt in (1, 2):
            rec = await self._ask(ev, issues)
            issues = check_recap(rec, ev, names)          # code first: it cannot be talked round
            if not issues and self.reviewer is not None:
                issues = await self._review(rec, ev)       # then every claim, by the Fact-checker
            if not issues:
                return {**rec, "checked": True, "reviewed": self.reviewer is not None, "attempts": attempt, "by": self.model}
        return {**fallback_recap(ev), "checked": True, "attempts": 2, "fallback": True, "by": "code",
                "rejected": issues[:5]}
