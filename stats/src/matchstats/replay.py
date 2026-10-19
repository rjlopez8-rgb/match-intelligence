"""Build a self-contained 2D match replay page from a recorded bundle.

The page needs no server: tracking frames are packed into a base64 Int16 array,
and the stats layer's own outputs (moments, win probability, live team stats)
are embedded alongside, so what the viewer shows is exactly what the agents
will see.

    matchstats replay ../data/sample/riv-kin-7 --out replay.html
"""

from __future__ import annotations

import base64
import bisect
import html
import json
from importlib import resources
from pathlib import Path

import numpy as np

from .analyzer import MatchAnalyzer
from .feed import replay_bundle

FRAME_HZ_EXPECTED = 5
ABSENT = -32768
CAPTION_TYPES = {
    "pass",
    "shot",
    "save",
    "tackle",
    "interception",
    "foul",
    "card",
    "goal",
    "offside",
    "substitution",
    "dribble",
    "clearance",
    "aerial_duel",
    "block",
    "keeper_claim",
    "kick_off",
    "ball_out",
}
MARKER_TYPES = {"goal", "red_card", "yellow_card", "big_chance", "great_save", "penalty_awarded"}


def _caption(e: dict, ctx) -> str | None:
    t, d = e["type"], e.get("details") or {}
    who = ctx.player_name(e["player_id"]) if e.get("player_id") else None
    if t == "pass":
        kind = d.get("pass_kind", "")
        label = {
            "corner": "corner",
            "free_kick": "free kick",
            "free_kick_cross": "free kick",
            "throw_in": "throw-in",
            "goal_kick": "goal kick",
            "cross": "cross",
            "through": "through ball",
            "long": "long ball",
            "switch": "switch of play",
            "cutback": "cut-back",
        }.get(kind)
        to = (
            ctx.player_name(d["recipient_id"])
            if d.get("recipient_id") in ctx.players
            else d.get("recipient_name")
        )
        if d.get("outcome") == "complete" and to:
            return f"{who} {'(' + label + ') ' if label else ''}→ {to}"
        return f"{who} {label or 'pass'}, {d.get('outcome', 'lost')}"
    if t == "shot":
        return f"{who} shoots: {d.get('outcome', '').replace('_', ' ')}"
    if t == "goal":
        return f"GOAL! {d.get('scorer_name', who)}"
    if t == "save":
        return f"Save by {who}"
    if t == "card":
        return f"{(d.get('card') or '').replace('_', ' ').capitalize()} card: {who}"
    if t == "substitution":
        return f"Substitution: {d.get('player_in_name')} on for {d.get('player_out_name')}"
    if t == "foul":
        return f"Foul by {who}" + (" (penalty)" if d.get("penalty") else "")
    if t == "ball_out":
        return None
    if t == "kick_off":
        return "Kick-off"
    if t == "aerial_duel":
        return f"Aerial duel won by {who}" if d.get("outcome") == "won" else None
    if t in ("tackle", "dribble"):
        won = d.get("outcome") in ("won", "complete", "success")
        return f"{who} {'wins a tackle' if t == 'tackle' and won else t}" if who else None
    return f"{who}: {t.replace('_', ' ')}" if who else None


def build_replay(bundle: str | Path) -> dict:
    """Run the stats layer over a bundle and collect everything the viewer needs."""
    bundle = Path(bundle)
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    a = MatchAnalyzer()
    frames: list[dict] = []
    samples: list[list[float]] = []
    captions: list[tuple[int, str, str]] = []
    possession_s = [0.0, 0.0]

    ctx = None
    for kind, item in replay_bundle(bundle):
        a.process(kind, item)
        ctx = a.ctx
        if kind == "event":
            if item["type"] in CAPTION_TYPES and frames:
                text = _caption(item, ctx)
                if text:
                    side = ctx.side(item["team_id"]) if item.get("team_id") else ""
                    captions.append((len(frames), text, side))
        elif kind == "frame":
            if item.get("possession_team_id") and item["ball"]["state"] != "dead":
                possession_s[0 if item["possession_team_id"] == ctx.home_id else 1] += 1
            frames.append(item)
            if (len(frames) - 1) % FRAME_HZ_EXPECTED == 0:
                samples.append(_sample(a, possession_s))

    if ctx is None or not frames:
        raise SystemExit(f"{bundle} has no tracking frames; record it with frames to build a replay")

    # -- pack tracking -------------------------------------------------------------------
    slots: list[str] = []
    slot_of: dict[str, int] = {}
    for f in frames:
        for pid, *_ in f["players"]:
            if pid not in slot_of:
                slot_of[pid] = len(slots)
                slots.append(pid)
    width = 2 * len(slots) + 6  # players xy, ball xyz, owner slot, possession side, t (ds)
    arr = np.full((len(frames), width), ABSENT, dtype=np.int16)
    for i, f in enumerate(frames):
        for pid, x, y, _v in f["players"]:
            s = slot_of[pid]
            arr[i, 2 * s] = round(x * 10)
            arr[i, 2 * s + 1] = round(y * 10)
        b = f["ball"]
        base = 2 * len(slots)
        arr[i, base] = round(b["x"] * 10)
        arr[i, base + 1] = round(b["y"] * 10)
        arr[i, base + 2] = round(min(b.get("z", 0.0), 30) * 10)
        arr[i, base + 3] = slot_of.get(b.get("owner_id"), -1)
        poss = f.get("possession_team_id")
        arr[i, base + 4] = -1 if not poss else (0 if poss == ctx.home_id else 1)
        arr[i, base + 5] = min(32000, round(f["t"] * 10))
    period2 = next((i for i, f in enumerate(frames) if f["period"] == 2), len(frames))

    keys = [(f["period"], f["t"]) for f in frames]

    def frame_of(period: int, t: float) -> int:
        return min(len(frames) - 1, bisect.bisect_left(keys, (period, t)))

    players = []
    for pid in slots:
        p = ctx.players[pid]
        players.append(
            {
                "id": pid,
                "name": p.name,
                "short": p.short_name,
                "shirt": p.shirt,
                "pos": p.position,
                "side": ctx.side(p.team_id),
                "gk": p.position == "GK",
            }
        )

    moments = []
    for m in a.moments:
        moments.append(
            {
                "f": frame_of(m["period"], m["t"]),
                "id": m["moment_id"],
                "type": m["type"],
                "minute": m["minute_display"],
                "title": m["title"],
                "side": ctx.side(m["team_id"]) if m["team_id"] else "",
                "importance": m["importance"],
                "weight": m["moment_weight"],
                "facts": _brief_facts(m),
                "evidence": len(m["evidence"]),
            }
        )

    def team(side: str) -> dict:
        t = meta[side]
        return {
            "name": t["name"],
            "short": t["short_name"],
            "code": t["code"],
            "colors": t["colors"],
            "formation": t["formation"],
            "manager": t["manager"],
        }

    return {
        "match": {
            "id": meta["match_id"],
            "competition": meta.get("competition"),
            "season": meta.get("season"),
            "matchday": meta.get("matchday"),
            "venue": meta.get("venue"),
            "attendance": meta.get("attendance"),
            "referee": (meta.get("referee") or {}).get("name")
            if isinstance(meta.get("referee"), dict)
            else meta.get("referee"),
            "home": team("home"),
            "away": team("away"),
        },
        "hz": FRAME_HZ_EXPECTED,
        "nFrames": len(frames),
        "width": width,
        "period2": period2,
        "players": players,
        "frames": base64.b64encode(arr.astype("<i2").tobytes()).decode("ascii"),
        "sampleEvery": FRAME_HZ_EXPECTED,
        "sampleKeys": SAMPLE_KEYS,
        "samples": samples,
        "captions": captions,
        "moments": moments,
        "models": a.models.status(),
        "final": {"score": dict(ctx.score)},
    }


SAMPLE_KEYS = [
    "wpH",
    "wpD",
    "wpA",
    "mom",
    "possH",
    "shotsH",
    "shotsA",
    "sotH",
    "sotA",
    "xgH",
    "xgA",
    "passH",
    "passA",
    "cmpH",
    "cmpA",
    "cornH",
    "cornA",
    "foulH",
    "foulA",
    "yelH",
    "yelA",
    "redH",
    "redA",
    "chaos",
    "goalsH",
    "goalsA",
]


def _sample(a: MatchAnalyzer, possession_s: list[float]) -> list[float]:
    ctx = a.ctx
    h, w = ctx.home_id, ctx.away_id
    wp = a.metric("win_probability").current
    if not isinstance(wp, dict):
        wp = {"home": wp[0], "draw": wp[1], "away": wp[2]}
    sh, ps, sm = a.metric("shooting").teams, a.metric("passing").teams, a.metric("summary").teams
    total = possession_s[0] + possession_s[1]
    chaos = a.metric("control").current().get("chaos_index", 0.0)
    return [
        round(wp["home"], 3),
        round(wp["draw"], 3),
        round(wp["away"], 3),
        a.metric("momentum").index(),
        round(possession_s[0] / total, 3) if total else 0.5,
        sh[h]["shots"],
        sh[w]["shots"],
        sh[h]["on_target"],
        sh[w]["on_target"],
        round(sh[h]["xg"], 2),
        round(sh[w]["xg"], 2),
        ps[h]["passes"],
        ps[w]["passes"],
        ps[h]["completed"],
        ps[w]["completed"],
        sm[h]["corners"],
        sm[w]["corners"],
        sm[h]["fouls"],
        sm[w]["fouls"],
        sm[h]["yellow_cards"],
        sm[w]["yellow_cards"],
        sm[h]["red_cards"],
        sm[w]["red_cards"],
        round(chaos, 1),
        ctx.score["home"],
        ctx.score["away"],
    ]


def _brief_facts(m: dict) -> list[str]:
    f = m["facts"]
    out = []
    if f.get("xg") is not None:
        out.append(f"xG {f['xg']:.2f}")
    if f.get("distance_m"):
        out.append(f"{f['distance_m']:.0f} m out")
    if f.get("shot_speed_kmh"):
        out.append(f"{f['shot_speed_kmh']:.0f} km/h strike")
    if f.get("speed_kmh"):
        out.append(f"{f['speed_kmh']} km/h")
    if f.get("difficulty"):
        out.append(f"difficulty {f['difficulty']:.1f}/10")
    if f.get("lines_broken"):
        out.append(f"{f['lines_broken']} opponents bypassed")
    if f.get("passes"):
        out.append(f"{f['passes']} passes")
    if f.get("possession_share"):
        out.append(f"{f['possession_share']:.0%} possession")
    if f.get("distance_km"):
        out.append(f"{f['distance_km']} km")
    if f.get("goalkeeper"):
        out.append(f"saved by {f['goalkeeper']}")
    if f.get("assist"):
        out.append(f"assist {f['assist']}")
    if f.get("chaos_index"):
        out.append(f"chaos {f['chaos_index']:.0f}/100")
    return out[:3]


def render_html(data: dict, standalone: bool = True) -> str:
    """The viewer page with the data embedded. ``standalone`` adds the document skeleton."""
    template = resources.files("matchstats").joinpath("viewer/replay.html").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    title = f"{data['match']['home']['short']} v {data['match']['away']['short']} Replay"
    page = template.replace('"__REPLAY_DATA__"', payload).replace("__TITLE__", html.escape(title))
    if not standalone:
        return page
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + page + "\n</body>\n</html>\n"
    )
