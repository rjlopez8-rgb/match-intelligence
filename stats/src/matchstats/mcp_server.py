"""Kairos as an MCP server: the stats layer, the season and Club mode as tools any MCP client can call
(GitHub Copilot agent mode in VS Code, Claude, the Agent Framework...).

    matchstats mcp                       # stdio (what VS Code launches, see .vscode/mcp.json)
    matchstats mcp --http --port 8766    # streamable HTTP, for a hosted server

Tools
- Season: ``list_matches``, ``league_table``, ``season_player``, ``what_decided_it``.
- One match (any season match, the demo or the sample): the stats layer's own query tools, the same ones the AI
  desk's Analyst uses. Each takes ``match_id`` and an optional ``minute``: the match is replayed up to that minute
  and the tools answer as of then, with no knowledge of what happens later.
- Club mode: ``tryout_board`` (every candidate for every position, ranked), ``tryout_result`` (a pre-simulated
  try-out), ``run_tryout`` (play the fixture again with a candidate on the match engine: $0, engine only, capped).

Everything is read-only on the repo. New try-out runs are cached in ``~/.match-intelligence/tryouts``.
"""


import argparse
import functools
import json
import os
import shutil
import threading
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from .analyzer import MatchAnalyzer
from .feed import replay_bundle
from .queries import PLAYER_STATS, SERIES, TEAM_STAT_GROUPS, TOOLS, MatchQueries, QueryError

DATA = Path(os.environ.get("MATCH_DATA") or Path(__file__).resolve().parents[3] / "data")
TRYOUT_CACHE = Path(os.environ.get("MATCH_TRYOUT_CACHE") or Path.home() / ".match-intelligence" / "tryouts")
MAX_TRYOUT_MATCHES = int(os.environ.get("MATCH_MCP_TRYOUT_MATCHES", "120"))   # per server process
_tryout_used = 0
_lock = threading.Lock()

MatchId = Annotated[str, Field(description="match id from list_matches, e.g. 'md04-elm-kin' or the demo 'riv-kin-1119'")]
Minute = Annotated[float | None, Field(description="answer as of this match minute (46-90+ is the second half); "
                                                   "omit for full time")]
Team = Annotated[str | None, Field(description="team name, code (RIV) or 'home'/'away'")]


# ------------------------------------------------------------------------------ data access
def _bundle(match_id: str) -> Path:
    """The richest bundle for a match: with tracking if it was re-played, else the committed events."""
    if not match_id or any(c in match_id for c in "/\\.") or len(match_id) > 64:
        raise QueryError("match_id must be an id from list_matches")
    for p in (DATA / "demo" / match_id, DATA / "sample" / match_id, DATA / "matches" / "season" / match_id,
              DATA / "season" / match_id):
        if (p / "meta.json").exists():
            return p
    raise QueryError(f"unknown match '{match_id}': call list_matches for the ids")


def _until(minute: float | None) -> tuple[int, float] | None:
    if minute is None:
        return None
    m = float(minute)
    return (1, m * 60) if m <= 45 else (2, (m - 45) * 60)


@functools.lru_cache(maxsize=24)
def _queries(match_id: str, minute: float | None, stop_at: tuple[int, float] | None = None) -> MatchQueries:
    """Replay the match to ``minute`` (or an exact (period, t), or the end) through the stats layer."""
    stop = stop_at or _until(minute)
    a = MatchAnalyzer()
    for kind, item in replay_bundle(_bundle(match_id)):
        if stop and kind != "meta" and (item["period"], item["t"]) > stop:
            break
        a.process(kind, item)
    return MatchQueries(a)


def _ask(match_id: str, minute: float | None, tool: str, **args: Any) -> dict:
    try:
        return _queries(match_id, None if minute is None else round(float(minute), 1)).call(
            tool, {k: v for k, v in args.items() if v is not None})
    except QueryError as e:
        return {"error": str(e)}


_season_lock = threading.Lock()
_season_data: dict | None = None


def _season() -> dict:
    """The season hub's data, built once (a few seconds; the server warms it up at start, callers wait for it)."""
    global _season_data
    with _season_lock:
        if _season_data is None:
            from .season import build_season

            board = DATA / "demo" / "riv-kin-1119" / "tryouts" / "board.json"
            _season_data = build_season(DATA / "season", board=board if board.exists() else None)
        return _season_data


def _turning() -> dict:
    p = DATA / "season" / "turning_points.json"
    return json.loads(p.read_text(encoding="utf-8"))["matches"] if p.exists() else {}


def _board() -> dict:
    p = DATA / "demo" / "riv-kin-1119" / "tryouts" / "board.json"
    if not p.exists():
        raise QueryError("no recruitment board in this repo (data/demo/riv-kin-1119/tryouts/board.json)")
    return json.loads(p.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ the server
def build_server():
    from mcp.server.mcpserver import Context, MCPServer

    mcp = MCPServer(
        name="kairos",
        title="Kairos",
        version="1.0.0",
        instructions=(
            "Kairos: football match intelligence for the simulated Meridian League 2026/27 (8 fictional clubs, 56 matches). "
            "Start with list_matches or league_table to find ids. Match tools replay a match through the stats layer "
            "and can answer as of any minute (no future knowledge). Club mode tools try any player in any position by "
            "simulating the fixture (Riverside v Kingsport) many times. Every number comes from the stats layer or the "
            "match engine: quote them, do not invent them. All clubs, players and matches are fictional."),
    )

    # ---- season
    @mcp.tool(title="List matches")
    def list_matches(matchday: Annotated[int | None, Field(description="1-14", ge=1, le=14)] = None, club: Team = None) -> dict:
        """The season's matches with ids, dates, scores and expected goals, optionally for one matchday or club."""
        S = _season()
        code = None
        if club:
            c = club.strip().lower()
            code = next((k for k, v in S["clubs"].items() if c in (k.lower(), v["name"].lower(), v["short"].lower())), None)
            if not code:
                return {"error": f"unknown club '{club}'", "clubs": {k: v["name"] for k, v in S["clubs"].items()}}
        rows = [{"match_id": m["id"], "matchday": m["md"], "date": m["date"], "home": S["clubs"][m["home"]]["name"],
                 "away": S["clubs"][m["away"]]["name"], "score": f"{m['score']['home']}-{m['score']['away']}",
                 "xg": f"{m['xg'][0]:.2f}-{m['xg'][1]:.2f}", "demo_with_ai_desk": m.get("demo", False)}
                for m in S["matches"] if (matchday is None or m["md"] == matchday) and (code is None or code in (m["home"], m["away"]))]
        return {"competition": S["competition"], "season": S["season"], "matches": rows}

    @mcp.tool(title="League table")
    def league_table(expected: Annotated[bool, Field(description="rank by expected points (from every shot's xG) instead")] = False) -> dict:
        """The league table after 14 matchdays, actual or expected (what the chances deserved)."""
        S = _season()
        rows = sorted(S["table"], key=(lambda r: -r["xpts"]) if expected else (lambda r: r["pos"]))
        return {"order": "expected points" if expected else "points", "table": [
            {"pos": i + 1, "club": S["clubs"][r["club"]]["name"], "played": r["p"], "won": r["w"], "drawn": r["d"], "lost": r["l"],
             "goals_for": r["gf"], "goals_against": r["ga"], "points": r["pts"], "expected_points": r["xpts"],
             "xg_for": r["xgf"], "xg_against": r["xga"], "form": "".join(f["r"] for f in r["form"])} for i, r in enumerate(rows)]}

    @mcp.tool(title="Season player")
    def season_player(player: Annotated[str, Field(description="player name (or surname) or id, e.g. 'Mensah' or 'riv-10'")]) -> dict:
        """A player's season: totals, per-90 numbers, attributes and match log."""
        S, q = _season(), player.strip().lower()
        hits = [p for p in S["players"].values() if q in (p["id"].lower(), (p.get("name") or "").lower())
                or (p.get("name") or "").lower().endswith(" " + q)]
        if len(hits) != 1:
            return {"error": "no player found" if not hits else "several players match", "matches": [h["name"] for h in hits[:10]]}
        p = hits[0]
        s = p.get("season") or {}
        mins = s.get("min") or 0
        per90 = {k: round(s[k] * 90 / mins, 2) for k in ("goals", "assists", "xg", "xa", "shots", "key_passes", "tackles_won")
                 if k in s and mins >= 90}
        return {"id": p["id"], "name": p["name"], "club": S["clubs"][p["club"]]["name"], "role": p.get("role"), "age": p.get("age"),
                "season": s, "per90": per90, "attributes": p.get("attr"), "player_of_the_match": p.get("potm"),
                "log": [{k: r.get(k) for k in ("m", "min", "goals", "assists", "xg", "key_passes")} for r in p.get("log", [])]}

    @mcp.tool(title="What decided it?")
    def what_decided_it(match_id: MatchId) -> dict:
        """The match's turning point (the goal or chance that moved win probability most) and what the match engine
        says would have happened had it gone the other way: the match frozen at that shot and played to full time
        many times each way, with the same random streams."""
        r = _turning().get(match_id)
        if not r:
            return {"error": f"no turning-point research for '{match_id}' (run matchagents recaps --research-only)"}
        if r.get("status") != "ok":
            return {"match_id": match_id, "status": r.get("status"), "reason": r.get("reason")}
        tp, w, a = r["turning_point"], r["with_change"], r["as_it_happened"]
        return {"match_id": match_id, "final_score": r["final"], "turning_point": {k: tp[k] for k in ("minute", "player", "type", "side", "xg", "weight_pts")},
                "if_it": "had been saved" if tp["change"] == "shot_misses" else "had gone in", "futures_each_way": r["futures"],
                "as_it_happened": a, "if_flipped": w, "method": r["method"]}

    # ---- one match: the stats layer's own tools
    @mcp.tool(title="Scoreboard", description=TOOLS["scoreboard"]["description"])
    def scoreboard(match_id: MatchId, minute: Minute = None) -> dict:
        return _ask(match_id, minute, "scoreboard")

    @mcp.tool(title="Team comparison", description=TOOLS["team_comparison"]["description"])
    def team_comparison(match_id: MatchId, groups: list[Literal[tuple(TEAM_STAT_GROUPS)]] | None = None, minute: Minute = None) -> dict:  # type: ignore[valid-type]
        return _ask(match_id, minute, "team_comparison", groups=groups)

    @mcp.tool(title="Player stats", description=TOOLS["player_stats"]["description"])
    def player_stats(match_id: MatchId, player: Annotated[str, Field(description="player name or id")], minute: Minute = None) -> dict:
        return _ask(match_id, minute, "player_stats", player=player)

    @mcp.tool(title="Top players", description=TOOLS["top_players"]["description"])
    def top_players(match_id: MatchId, stat: Literal[tuple(PLAYER_STATS)], team: Team = None,  # type: ignore[valid-type]
                    limit: Annotated[int, Field(ge=1, le=20)] = 5, minute: Minute = None) -> dict:
        return _ask(match_id, minute, "top_players", stat=stat, team=team, limit=limit)

    @mcp.tool(title="Momentum", description=TOOLS["momentum"]["description"])
    def momentum(match_id: MatchId, window_min: Annotated[int, Field(ge=3, le=45)] = 10, minute: Minute = None) -> dict:
        return _ask(match_id, minute, "momentum", window_min=window_min)

    @mcp.tool(title="Key moments", description=TOOLS["moments"]["description"])
    def moments(match_id: MatchId, types: list[str] | None = None, team: Team = None, player: str | None = None,
                since_minute: float | None = None, min_importance: Annotated[float, Field(ge=0, le=1)] = 0.0,
                limit: Annotated[int, Field(ge=1, le=50)] = 10,
                order: Literal["recent", "chronological", "importance"] = "chronological", minute: Minute = None) -> dict:
        return _ask(match_id, minute, "moments", types=types, team=team, player=player, since_minute=since_minute,
                    min_importance=min_importance, limit=limit, order=order)

    @mcp.tool(title="Explain a moment", description=TOOLS["explain_moment"]["description"])
    def explain_moment(match_id: MatchId, moment_id: Annotated[str, Field(description="from the moments tool")]) -> dict:
        return _ask(match_id, None, "explain_moment", moment_id=moment_id)

    @mcp.tool(title="Event", description=TOOLS["event"]["description"])
    def event(match_id: MatchId, event_id: str) -> dict:
        return _ask(match_id, None, "event", event_id=event_id)

    @mcp.tool(title="Options at a decision", description=TOOLS["options"]["description"] + " Needs tracking data: the demo "
              "match, or a season match re-played with `matchengine season --tracking`.")
    def options(match_id: MatchId, event_id: Annotated[str, Field(description="the shot or pass event")]) -> dict:
        # the stats layer keeps the last seconds of tracking, so replay to the instant of the decision
        try:
            from .feed import _events

            ev = next((e for e in _events(_bundle(match_id) / "events.jsonl") if e["event_id"] == event_id), None)
            if ev is None:
                return {"error": f"no event '{event_id}' in {match_id}"}
            return _queries(match_id, None, (ev["period"], ev["t"] + 0.5)).call("options", {"event_id": event_id})
        except QueryError as e:
            return {"error": str(e)}

    @mcp.tool(title="Win probability", description=TOOLS["win_probability"]["description"])
    def win_probability(match_id: MatchId, minute: Minute = None) -> dict:
        return _ask(match_id, minute, "win_probability")

    @mcp.tool(title="Pass network", description=TOOLS["pass_network"]["description"])
    def pass_network(match_id: MatchId, team: Annotated[str, Field(description="team name, code or 'home'/'away'")], minute: Minute = None) -> dict:
        return _ask(match_id, minute, "pass_network", team=team)

    @mcp.tool(title="Timeline", description=TOOLS["timeline"]["description"])
    def timeline(match_id: MatchId, series: Literal[tuple(SERIES)], since_minute: float | None = None, minute: Minute = None) -> dict:  # type: ignore[valid-type]
        return _ask(match_id, minute, "timeline", series=series, since_minute=since_minute)

    # ---- Club mode
    def _position(side: str, position: str | int) -> dict:
        b = _board()["sides"][side]["board"]
        pos = str(position).strip().upper()
        hit = next((p for p in b if pos in (str(p["slot"]), p["label"].upper())), None)
        if not hit:
            raise QueryError(f"unknown position '{position}': one of " + ", ".join(f"{p['label']} (slot {p['slot']})" for p in b))
        return hit

    Side = Annotated[Literal["home", "away"], Field(description="home = Riverside Athletic, away = Kingsport United")]
    Position = Annotated[str, Field(description="position label (GK, LB, CB, ST...) or slot number from tryout_board")]

    @mcp.tool(title="Recruitment board")
    def tryout_board(side: Side = "home", position: Position | None = None, limit: Annotated[int, Field(ge=1, le=30)] = 8,
                     max_age: Annotated[int | None, Field(ge=16, le=45, description="only candidates this age or younger")] = None) -> dict:
        """Club mode: for each position in the fixture Riverside v Kingsport, the starter and the league's candidates
        ranked by an attribute projection for that role. 'ready' candidates have a pre-simulated try-out
        (tryout_result); anyone else can be simulated with run_tryout."""
        try:
            B = _board()
            rows = B["sides"][side]["board"] if position is None else [_position(side, position)]
        except QueryError as e:
            return {"error": str(e)}
        return {"fixture": f"{B['home']} v {B['away']}", "side": side, "positions": [
            {"slot": p["slot"], "label": p["label"], "starter": p["name"], "starter_projected": p["incumbent_projected"],
             "candidates": [{k: c[k] for k in ("player_id", "name", "club", "age", "projected", "ready")}
                            for c in p["candidates"] if max_age is None or c["age"] <= max_age][:limit]}
            for p in rows]}

    @mcp.tool(title="Try-out result")
    def tryout_result(side: Side, position: Position, player: Annotated[str, Field(description="candidate name or id")]) -> dict:
        """Club mode: a pre-simulated try-out. The fixture played many times with the candidate in that position and
        as many with the real starter, on the same seeds: expected points, xG for and against, his per-90 line against
        the starter's, the difference with its margin of error, and the Scout agent's note."""
        try:
            pos = _position(side, position)
        except QueryError as e:
            return {"error": str(e)}
        q = player.strip().lower()
        for t in _board()["sides"][side]["tryouts"]:
            c = t["candidate"]
            if t["position"]["slot"] == pos["slot"] and q in (c["player_id"].lower(), c["name"].lower(), c["name"].split()[-1].lower()):
                return _tryout_view(t)
        return {"status": "not_simulated", "hint": "call run_tryout to simulate it (engine only, about 30 s per match per core)"}

    @mcp.tool(title="Run a try-out")
    async def run_tryout(side: Side, position: Position, player: Annotated[str, Field(description="candidate player id (from tryout_board)")],
                         matches: Annotated[int, Field(ge=2, le=12, description="matches each way; more = smaller margin")] = 6,
                         ctx: Context | None = None) -> dict:
        """Club mode: simulate the fixture with this player in the position and with the starter, on the same seeds,
        with the match engine ($0, no AI model). The starter's runs are cached and shared. Takes about 30 s per match
        per CPU core."""
        import anyio

        global _tryout_used
        try:
            pos = _position(side, position)
        except QueryError as e:
            return {"error": str(e)}
        from matchengine.tryout import find_player, run_tryout as engine_tryout

        pid = player.strip().lower()
        try:
            find_player(pid)
        except Exception:
            return {"error": f"unknown player id '{player}': use a player_id from tryout_board"}
        with _lock:
            if _tryout_used + 2 * matches > MAX_TRYOUT_MATCHES:
                return {"error": f"this server's try-out allowance ({MAX_TRYOUT_MATCHES} simulated matches) is used up; restart it to reset"}
            _tryout_used += 2 * matches
        _seed_cache()
        B = _board()
        loop_progress: list[dict] = []

        def go():
            return engine_tryout(B["home"], B["away"], side, pos["slot"], pid, n=matches, cache_dir=TRYOUT_CACHE,
                                 on_progress=lambda r: loop_progress.append(r["progress"]))

        if ctx is not None:
            await ctx.report_progress(0, None, f"simulating {pid} at {pos['label']}")
        try:
            r = await anyio.to_thread.run_sync(go)
        except ValueError as e:
            return {"error": str(e)}
        return _tryout_view(r)

    return mcp


def _tryout_view(t: dict) -> dict:
    keep = ("futures", "win", "draw", "loss", "xpts", "xpts_margin_95", "xg_for", "xg_against", "minutes")
    w, o, d = t["with_candidate"], t["with_incumbent"], t["difference"]
    out = {"position": t["position"]["label"], "candidate": {k: t["candidate"][k] for k in ("player_id", "name", "club", "age")},
           "starter": {k: t["incumbent"][k] for k in ("player_id", "name", "club", "age")},
           "with_candidate": {k: w.get(k) for k in keep}, "with_starter": {k: o.get(k) for k in keep},
           "difference": d, "candidate_per90": w.get("per90"), "starter_per90": o.get("per90"), "method": t.get("method")}
    if t.get("scout"):
        out["scout_note"] = {k: t["scout"].get(k) for k in ("headline", "report", "verdict")}
    return out


def _seed_cache() -> None:
    """New try-outs reuse the repo's pre-simulated runs (copied once into the user cache, so the repo stays clean)."""
    src = DATA / "demo" / "riv-kin-1119" / "tryouts"
    for f in src.glob("*/*.json"):
        dst = TRYOUT_CACHE / f.parent.name / f.name
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="matchstats mcp", description="Kairos MCP server")
    ap.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--data", default=None, help="the repo's data folder (default: found from this package)")
    a = ap.parse_args(argv)
    global DATA
    if a.data:
        DATA = Path(a.data)
    server = build_server()
    threading.Thread(target=_season, daemon=True).start()   # the season hub takes a few seconds: warm it up
    if a.http:
        server.run("streamable-http", host=a.host, port=a.port)
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
