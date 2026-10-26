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
import re
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
# Shot flights for the 3D view: the tracking feed keeps the ball on the ground during shots,
# so the viewer rebuilds each flight from the event's start, true end point (with height),
# speed and outcome. Codes: 0 goal, 1 saved, 2 off target, 3 blocked, 4 post.
SHOT_CODES = {"goal": 0, "saved": 1, "off_target": 2, "blocked": 3, "post": 4}


SHOT_KINDS = {"open_play": 0, "header": 1, "free_kick": 2, "penalty": 3}
FEET = {"right_foot": 0, "left_foot": 1, "head": 2}
SAVE_KINDS = {"catch": 0, "parry": 1}


def _shot_row(f: float, e: dict, slot_of: dict, save_kind: str | None = None) -> list:
    """[frame, start x, y, end x, y, z, outcome, speed m/s, keeper slot, shooter slot, kind, foot, save kind]"""
    d = e.get("details") or {}
    start, end = e["location"], d.get("end_location") or e["location"]
    return [
        f,
        round(start["x"], 2),
        round(start["y"], 2),
        round(end["x"], 2),
        round(end["y"], 2),
        round(float(end.get("z") or 0.0), 2),
        SHOT_CODES.get(d.get("outcome"), 2),
        round(float(d.get("speed_kmh") or 80.0) / 3.6, 1),
        slot_of.get(d.get("goalkeeper_id"), -1),
        slot_of.get(e.get("player_id"), -1),
        SHOT_KINDS.get(d.get("shot_kind"), 0),
        FEET.get(d.get("body_part"), 0),
        SAVE_KINDS.get(save_kind, -1),
    ]


def _frac_frame(keys: list[tuple], period: int, t: float) -> float:
    """The event's time as a fractional frame index, so kicks and shots line up to the hundredth of a second."""
    i = bisect.bisect_left(keys, (period, t))
    if i <= 0:
        return 0.0
    if i >= len(keys):
        return float(len(keys) - 1)
    (p0, t0), (p1, t1) = keys[i - 1], keys[i]
    if p0 != period or p1 != period or t1 <= t0:
        return float(i)
    return round(i - 1 + (t - t0) / (t1 - t0), 2)


def _touch_code(e: dict) -> int:
    if e["type"] == "pass" and (e.get("details") or {}).get("pass_kind") == "throw_in":
        return 4
    if e["type"] == "shot" and (e.get("details") or {}).get("body_part") == "head":
        return 5
    return TOUCH_CODES[e["type"]]


# body animations in the 3D view: 1 = kick (pass, clearance), 2 = shot, 3 = goalkeeper save, 4 = throw-in, 5 = header
TOUCH_CODES = {"pass": 1, "clearance": 1, "shot": 2, "save": 3, "keeper_claim": 3}
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


def build_replay(bundle: str | Path, segments: list[dict] | None = None) -> dict:
    """Run the stats layer over a bundle and collect everything the viewer needs.

    ``segments`` are the AI desk's outputs (``matchagents ... --out``); when given, the 3D
    view plays each one when its moment happens: commentary, cameras, drawings, ghost futures.
    """
    bundle = Path(bundle)
    meta = json.loads((bundle / "meta.json").read_text(encoding="utf-8"))
    a = MatchAnalyzer()
    frames: list[dict] = []
    samples: list[list[float]] = []
    captions: list[tuple[int, str, str]] = []
    raw_touches: list[dict] = []
    raw_shots: list[dict] = []
    raw_actions: list[dict] = []
    save_kind: dict[str, str] = {}
    last_shot: dict | None = None
    possession_s = [0.0, 0.0]

    ctx = None
    for kind, item in replay_bundle(bundle):
        a.process(kind, item)
        ctx = a.ctx
        if kind == "event":
            if item["type"] == "shot":
                raw_shots.append(item)
                last_shot = item
            elif item["type"] == "save" and last_shot is not None:
                save_kind[last_shot["event_id"]] = (item.get("details") or {}).get("kind")
            if item["type"] in TOUCH_CODES and item.get("player_id"):
                raw_touches.append(item)
            if item["type"] in ACTION_TYPES and item.get("player_id"):
                raw_actions.append(item)
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
                "who": m.get("player_names") or [],
                "team": m.get("team_name"),
                "score": [m["score"]["home"], m["score"]["away"]] if m.get("score") else None,
                "x": {k: v for k, v in (m.get("facts") or {}).items() if k in RAW_FACTS and isinstance(v, (int, float))},
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
        "touches": [
            [_frac_frame(keys, e["period"], e["t"]), slot_of[e["player_id"]], _touch_code(e)]
            for e in raw_touches
            if e["player_id"] in slot_of
        ],
        "shots": [
            _shot_row(_frac_frame(keys, e["period"], e["t"]), e, slot_of, save_kind.get(e["event_id"])) for e in raw_shots
        ],
        "shotXg": [_xg(ctx, e) for e in raw_shots],
        "moments": moments,
        "actions": _actions(raw_actions, keys, slot_of, ctx),
        "models": a.models.status(),
        "final": {"score": dict(ctx.score)},
        "segments": _viewer_segments(segments or [], frame_of, slot_of, ctx),
        "club": _viewer_club(bundle, slot_of, ctx),
        **_match_logs(raw_shots, raw_actions, slot_of, ctx),
    }


def _match_logs(shots: list[dict], actions: list[dict], slot_of: dict, ctx) -> dict:
    """What the analyst graphics draw (shot map, xG race, pass network), stamped with match time (period, t) rather
    than frames, so a highlights reel can show the whole match so far. The page only ever reads entries up to the
    current moment: no future knowledge, as everywhere else in the broadcast.

    shotLog: [period, t, x, y, xg, outcome code, side 0/1, shooter slot]
    passLog: [period, t, passer slot, receiver slot, x, y] for completed passes between players on the pitch
    """
    shot_log = []
    for e in shots:
        d = e.get("details") or {}
        shot_log.append([e["period"], round(e["t"], 1), round(e["location"]["x"], 1), round(e["location"]["y"], 1),
                         _xg(ctx, e) or 0.0, SHOT_CODES.get(d.get("outcome"), 2), 0 if ctx.side(e["team_id"]) == "home" else 1,
                         slot_of.get(e.get("player_id"), -1)])
    pass_log = []
    for e in actions:
        d = e.get("details") or {}
        if e["type"] != "pass" or d.get("outcome") != "complete":
            continue
        a, b = slot_of.get(e.get("player_id")), slot_of.get(d.get("recipient_id"))
        if a is None or b is None:
            continue
        pass_log.append([e["period"], round(e["t"], 1), a, b, round(e["location"]["x"], 1), round(e["location"]["y"], 1)])
    return {"shotLog": shot_log, "passLog": pass_log}


ACTION_TYPES = {"pass", "shot", "tackle", "interception", "dribble", "foul", "save", "aerial_duel", "ball_recovery",
                "pressure", "block", "clearance"}
RAW_FACTS = ("xg", "distance_m", "shot_speed_kmh", "speed_kmh", "difficulty", "lines_broken", "passes",
             "possession_share", "distance_km", "chaos_index", "moment_weight")
# what each player did, for the player-focus stat line, alerts and timeline marks
ACT = {"pass_ok": 1, "pass_bad": 2, "shot": 3, "shot_on": 4, "goal": 5, "chance": 6, "tackle": 7, "interception": 8,
       "dribble": 9, "foul": 10, "save": 11, "aerial": 12, "recovery": 13, "assist": 14, "pressure": 15, "block": 16,
       "clearance": 17}


def _xg(ctx, e: dict) -> float | None:
    v = ctx.ann.get(e["event_id"], {}).get("xg")
    return round(v, 3) if isinstance(v, (int, float)) else None


def _actions(events: list[dict], keys: list[tuple], slot_of: dict, ctx) -> list[list]:
    key_pass, assists = {}, set()
    for eid, ann in ctx.ann.items():
        kp = ann.get("key_pass_event_id")
        if kp:
            key_pass[kp] = eid
    goals_from = {e["event_id"] for e in events if e["type"] == "shot" and (e.get("details") or {}).get("outcome") == "goal"}
    for kp, shot in key_pass.items():
        if shot in goals_from:
            assists.add(kp)
    out = []
    for e in events:
        s = slot_of.get(e["player_id"])
        if s is None:
            continue
        t, d = e["type"], e.get("details") or {}
        oc = d.get("outcome")
        if t == "pass":
            code = ACT["assist"] if e["event_id"] in assists else ACT["chance"] if e["event_id"] in key_pass else (
                ACT["pass_ok"] if oc == "complete" else ACT["pass_bad"])
        elif t == "shot":
            code = ACT["goal"] if oc == "goal" else ACT["shot_on"] if oc == "saved" else ACT["shot"]
        elif t == "tackle":
            if oc != "won":
                continue
            code = ACT["tackle"]
        elif t == "dribble":
            if oc != "success":
                continue
            code = ACT["dribble"]
        elif t == "aerial_duel":
            code = ACT["aerial"]
        else:
            code = ACT.get({"ball_recovery": "recovery"}.get(t, t))
            if code is None:
                continue
        out.append([round(_frac_frame(keys, e["period"], e["t"]), 1), s, code])
    return out


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


def render_html(data: dict, standalone: bool = True, view: str = "replay") -> str:
    """A viewer page with the data embedded: ``view`` is "replay" (2D) or "broadcast" (3D).
    ``standalone`` adds the document skeleton."""
    template = resources.files("matchstats").joinpath(f"viewer/{view}.html").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    suffix = "3D broadcast" if view == "broadcast" else "Replay"
    title = f"{data['match']['home']['short']} v {data['match']['away']['short']} · Kairos {suffix}"
    page = template.replace('"__REPLAY_DATA__"', payload).replace("__TITLE__", html.escape(title))
    if not standalone:
        return page
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + page + "\n</body>\n</html>\n"
    )


def render_viewer_shell(title: str = "Kairos · 3D broadcast") -> str:
    """The shared 3D page with no match inside: it loads ``matches/<id>.<hl|full>.json`` named in its URL
    (``watch.html?m=<id>&v=hl``), so the site ships one copy of the viewer instead of one per match."""
    template = resources.files("matchstats").joinpath("viewer/broadcast.html").read_text(encoding="utf-8")
    page = template.replace("__TITLE__", html.escape(title))
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + page + "\n</body>\n</html>\n"
    )


# ------------------------------------------------------------------- highlights
HIGHLIGHT_TYPES = ("goal", "great_save", "woodwork", "big_chance", "penalty_awarded", "save", "chance")
HIGHLIGHT_RANK = {t: i for i, t in enumerate(HIGHLIGHT_TYPES)}   # which moment names a clip that holds several
CLIP_BEFORE_S = 15.0                                             # the build-up
CLIP_AFTER_S = {"goal": 11.0}                                    # goals keep the celebration and the auto replay
CLIP_AFTER_DEFAULT_S = 6.0
CLIP_JOIN_GAP_S = 4.0                                            # clips closer than this play as one
# beyond the stats layer's key moments, the shots worth seeing again: a save from a real chance, or any good chance
SAVE_MIN_XG, CHANCE_MIN_XG = 0.05, 0.10
MAX_CLIPS = 10


def highlight_clips(data: dict) -> list[dict]:
    """Where the highlights are, in the full match's frames: one window per goal, save or chance, overlapping
    windows joined, never across half-time."""
    hz, n, p2 = data["hz"], data["nFrames"], data["period2"]
    picks = [m for m in data["moments"] if m["type"] in HIGHLIGHT_RANK]
    extra = []
    for row, xg in zip(data["shots"], data.get("shotXg") or []):
        oc, xg = row[6], xg or 0.0
        if oc == 1 and xg >= SAVE_MIN_XG or oc != 0 and xg >= CHANCE_MIN_XG:   # goals are already key moments
            extra.append({"f": int(row[0]), "type": "save" if oc == 1 else "chance", "importance": xg,
                          "minute": _minute_at(data, int(row[0])), "side": _side_of_slot(data, row[9])})
    # the reel stays short: keep the key moments, then the best extra chances
    room = max(0, MAX_CLIPS - len(picks))
    picks += sorted(extra, key=lambda e: -e["importance"])[:room]
    wins = []
    for m in picks:
        f = m["f"]
        lo, hi = (0, p2 - 1) if f < p2 else (p2, n - 1)
        a = max(lo, int(round(f - CLIP_BEFORE_S * hz)))
        b = min(hi, int(round(f + CLIP_AFTER_S.get(m["type"], CLIP_AFTER_DEFAULT_S) * hz)))
        wins.append({"a": a, "b": b, "period": 1 if f < p2 else 2, "moments": [m]})
    wins.sort(key=lambda w: w["a"])
    clips: list[dict] = []
    for w in wins:
        last = clips[-1] if clips else None
        if last and last["period"] == w["period"] and w["a"] <= last["b"] + CLIP_JOIN_GAP_S * hz:
            last["b"] = max(last["b"], w["b"])
            last["moments"] += w["moments"]
        else:
            clips.append(w)
    for c in clips:
        lead = min(c["moments"], key=lambda m: (HIGHLIGHT_RANK[m["type"]], -m["importance"]))
        c["lead"] = lead
    return clips[:MAX_CLIPS] if len(clips) <= MAX_CLIPS else sorted(
        sorted(clips, key=lambda c: (HIGHLIGHT_RANK[c["lead"]["type"]], -c["lead"]["importance"]))[:MAX_CLIPS], key=lambda c: c["a"])


def _side_of_slot(data: dict, slot: int) -> str:
    return data["players"][slot]["side"] if 0 <= slot < len(data["players"]) else ""


def _minute_at(data: dict, f: int) -> str:
    """The broadcast minute of a full-match frame: 1' to 45', 45+2', 46' to 90', 90+4'."""
    raw = np.frombuffer(base64.b64decode(data["frames"]), dtype="<i2")
    t = raw[f * data["width"] + data["width"] - 1] / 10
    period = 1 if f < data["period2"] else 2
    if t < 2700:
        return f"{int(t // 60) + 1 + 45 * (period - 1)}'"
    return f"{45 * period}+{int((t - 2700) // 60) + 1}'"


def highlights(data: dict) -> dict:
    """A highlights reel from a full match's viewer data (``build_replay``): only the frames of its goals, saves and
    chances, played back to back. Every frame keeps its real match clock; ``clips`` tells the 3D page where each cut
    is, so motion, ball flights and stats never blend across one. The AI desk and Club mode stay with the full match."""
    hz, w, n = data["hz"], data["width"], data["nFrames"]
    clips = highlight_clips(data)
    if not clips:   # a 0-0 with nothing on target: the opening and the closing minute
        clips = [{"a": 0, "b": min(n - 1, 30 * hz), "period": 1, "moments": [], "lead": None},
                 {"a": max(data["period2"], n - 1 - 30 * hz), "b": n - 1, "period": 2, "moments": [], "lead": None}]
    full = np.frombuffer(base64.b64decode(data["frames"]), dtype="<i2").reshape(n, w)
    starts, rows = [], []
    for c in clips:
        starts.append(sum(len(r) for r in rows))
        rows.append(full[c["a"]: c["b"] + 1])
    arr = np.concatenate(rows)
    m_new = len(arr)

    def remap(f):
        """A full-match frame (fractional allowed) in reel frames, or None when it is not in any clip."""
        for c, s0 in zip(clips, starts):
            if c["a"] <= f <= c["b"] + 0.999:
                return round(s0 + (f - c["a"]), 2)
        return None

    def orig(k: int) -> int:
        for c, s0 in zip(reversed(clips), reversed(starts)):
            if k >= s0:
                return c["a"] + (k - s0)
        return clips[0]["a"]

    def keep_rows(rows_in: list, at: int = 0) -> list:
        out = []
        for r in rows_in:
            f = remap(r[at])
            if f is not None:
                out.append([f, *r[at + 1:]] if at == 0 else r)
        return out

    every, samples = data["sampleEvery"], data["samples"]
    shots, xgs = [], []
    for row, xg in zip(data["shots"], data["shotXg"]):
        f = remap(row[0])
        if f is not None:
            shots.append([f, *row[1:]])
            xgs.append(xg)
    moments = []
    for m in data["moments"]:
        f = remap(m["f"])
        if f is not None:
            moments.append({**m, "f": int(f)})
    out = {
        **data,
        "nFrames": m_new,
        "frames": base64.b64encode(arr.astype("<i2").tobytes()).decode("ascii"),
        "period2": next((s0 for c, s0 in zip(clips, starts) if c["period"] == 2), m_new),
        "samples": [samples[min(len(samples) - 1, orig(k) // every)] for k in range(0, m_new, every)],
        "captions": [[int(remap(f)), t, s] for f, t, s in data["captions"] if remap(f) is not None],
        "touches": keep_rows(data["touches"]),
        "shots": shots,
        "shotXg": xgs,
        "moments": moments,
        "actions": keep_rows(data["actions"]),
        "segments": [],
        "club": None,
        "clips": [
            [s0, s0 + c["b"] - c["a"], c["period"], c["lead"]["minute"] if c["lead"] else "",
             c["lead"]["type"] if c["lead"] else "", c["lead"]["side"] if c["lead"] else ""]
            for c, s0 in zip(clips, starts)
        ],
        "highlights": True,
    }
    return out


# ------------------------------------------------------------------- AI desk segments
def _fan_line(curve: list[list], frame_of) -> list[int]:
    """One future's win-probability swing as [frame, home-minus-away in points, ...] for the fan chart."""
    out: list[int] = []
    for period, t, h, a in curve:
        out += [frame_of(period, t), round((h - a) * 100)]
    return out


CLUB_PER90 = ("goals", "xg", "shots", "assists", "xa", "key_passes", "passes_completed", "progressive_passes",
               "dribbles_completed", "tackles_won", "interceptions", "ball_recoveries", "clearances", "aerials_won",
               "dispossessed", "saves", "goals_conceded", "xg_faced", "distance_km", "sprints")


def _compact_tryout(r: dict) -> dict:
    def side(x: dict) -> dict:
        keep = ("futures", "win", "draw", "loss", "xwin", "xdraw", "xloss", "result_margin_95", "xpts", "xpts_margin_95", "xg_for",
                "xg_for_margin_95", "xg_against", "xg_against_margin_95", "minutes")
        return {**{k: x.get(k) for k in keep}, "per90": {k: x["per90"].get(k, 0) for k in CLUB_PER90}}

    fe = r.get("featured") or {}
    return {
        "candidate": {k: r["candidate"][k] for k in ("player_id", "name", "club", "role", "age")},
        "incumbent": {k: r["incumbent"][k] for k in ("player_id", "name", "club", "role", "age")},
        "position": r["position"],
        "with": side(r["with_candidate"]),
        "without": side(r["with_incumbent"]),
        "diff": r["difference"],
        "featured": {k: fe.get(k) for k in ("kind", "minute", "seed")} if fe else None,
        "method": r.get("method"),
        "report": r.get("scout"),
    }


def _viewer_club(bundle: Path, slot_of: dict, ctx) -> dict | None:
    """The recruitment board for this fixture (``tryouts/board.json`` from ``matchengine board``), with the
    pre-simulated try-outs and each one's featured moment packed like a ghost clip."""
    path = bundle / "tryouts" / "board.json"
    if not path.exists():
        return None
    import gzip

    data = json.loads(path.read_text(encoding="utf-8"))
    sides, clips = {}, {}
    for side, block in data.get("sides", {}).items():
        positions = []
        for pos in block["board"]:
            positions.append({
                **{k: pos[k] for k in ("slot", "label", "role", "player_id", "name", "incumbent_projected")},
                "candidates": [
                    {k: c[k] for k in ("player_id", "name", "club", "club_code", "role", "age", "projected", "ready", "tag")}
                    for c in pos["candidates"]
                ],
            })
        tryouts = {}
        for r in block.get("tryouts", []):
            key = f"{r['position']['slot']}:{r['candidate']['player_id']}"
            tryouts[key] = _compact_tryout(r)
            ref = (r.get("featured") or {}).get("clip")
            if ref and (bundle / "tryouts" / ref).exists():
                with gzip.open(bundle / "tryouts" / ref, "rt", encoding="utf-8") as fh:
                    clip = json.load(fh)
                alias = {**slot_of, r["candidate"]["player_id"]: slot_of.get(r["incumbent"]["player_id"], -1)}
                packed = _pack_clip({**clip, "kind": "tryout"}, alias, ctx)
                packed["moment"] = clip.get("moment")
                clips[f"{side}:{key}"] = packed
                tryouts[key]["clip"] = f"{side}:{key}"
        sides[side] = {"positions": positions, "tryouts": tryouts}
    return {"home": data.get("home"), "away": data.get("away"), "n": data.get("n"), "sides": sides, "clips": clips}


def _pack_clip(clip: dict, slot_of: dict, ctx) -> dict:
    """A moment of a simulated future in the main frame layout, with its shots, touches and goals,
    so the 3D view animates it exactly like the real match."""
    frames = clip["frames"]
    n_slots = max(slot_of.values(), default=-1) + 1  # aliases (a try-out's candidate) share a slot
    arr = np.full((len(frames), 2 * n_slots + 6), ABSENT, dtype=np.int16)
    keys = []
    for i, f in enumerate(frames):
        for pid, x, y, *_ in f["players"]:
            s = slot_of.get(pid)
            if s is not None:
                arr[i, 2 * s] = round(x * 10)
                arr[i, 2 * s + 1] = round(y * 10)
        b = f["ball"]
        base = 2 * n_slots
        arr[i, base] = round(b["x"] * 10)
        arr[i, base + 1] = round(b["y"] * 10)
        arr[i, base + 2] = round(min(b.get("z", 0.0), 30) * 10)
        arr[i, base + 3] = slot_of.get(b.get("owner_id"), -1)
        poss = f.get("possession_team_id")
        arr[i, base + 4] = -1 if not poss else (0 if poss == ctx.home_id else 1)
        arr[i, base + 5] = min(32000, round(f["t"] * 10))
        keys.append((f["period"], f["t"]))

    def local(e: dict) -> float:
        return _frac_frame(keys, e["period"], e["t"])

    events = clip.get("events") or []
    saves: dict[int, str] = {}
    last = None
    for e in events:
        if e["type"] == "shot":
            last = e
        elif e["type"] == "save" and last is not None:
            saves[id(last)] = (e.get("details") or {}).get("kind")
    return {
        "kind": clip["kind"],
        "minute": clip.get("minute"),
        "scorer": clip.get("scorer"),
        "side": clip.get("side"),
        "score0": clip.get("score0") or {"home": 0, "away": 0},
        "n": len(frames),
        "p2": 0 if frames[0]["period"] == 2 else len(frames),
        "frames": base64.b64encode(arr.astype("<i2").tobytes()).decode("ascii"),
        "shots": [_shot_row(local(e), e, slot_of, saves.get(id(e))) for e in events if e["type"] == "shot"],
        "touches": [
            [local(e), slot_of[e["player_id"]], _touch_code(e)]
            for e in events
            if e["type"] in TOUCH_CODES and e.get("player_id") in slot_of
        ],
        "goals": [
            [local(e), 0 if ctx.side(e["team_id"]) == "home" else 1] for e in events if e["type"] == "goal"
        ],
    }


def _pack_director(d: dict, slot, frame_of_event) -> dict:
    return {
        "live": d.get("live_camera"),
        "replay": bool(d.get("replay")),
        "fromF": frame_of_event(d.get("replay_from_event_id")),
        "graphic": d.get("graphic"),
        "tele": bool(d.get("show_telestration")),
        "shots": [
            {"cam": x["camera"], "sec": x["seconds"], "slow": x["slow_motion"], "focus": slot(x.get("focus_player")),
             "purpose": x["purpose"]}
            for x in (d.get("shots") or [])[:4]
        ],
    }


_RX_NUM = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)")
_RX_SCORE = re.compile(r"(?<!\d)(\d{1,2})\s?[-–]\s?(\d{1,2})(?!\d)")


def _leaves(obj, path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:40]):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def _same_value(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return str(a) == str(b)


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in (args or {}).items())


def _receipts(s: dict, frame_of_event, ctx) -> dict:
    """Where every fact on air came from: the facts, the tool call and result field that produced each one, the match
    events it cites; plus the simulation numbers when there was a what-if."""
    brief = s.get("brief") or {}
    results = s.get("tool_results") or []
    calls = [{"tool": r["tool"], "args": _fmt_args(r.get("args")), "keys": []} for r in results]
    facts = []
    for f in brief.get("facts") or []:
        found = None
        for ci, r in enumerate(results):
            if r["tool"] != f.get("source_tool"):
                continue
            for path, val in _leaves(r.get("result")):
                if isinstance(val, (int, float, str)) and not isinstance(val, bool) and _same_value(val, f.get("value")):
                    found = (ci, path)
                    break
            if found:
                break
        if found and [found[1], f.get("value")] not in calls[found[0]]["keys"]:
            calls[found[0]]["keys"].append([found[1], f.get("value")])
        ev = []
        for eid in (f.get("evidence") or [])[:4]:
            e = ctx.by_id.get(eid)
            if e:
                who = ctx.player_name(e["player_id"]) if e.get("player_id") else ""
                ev.append([frame_of_event(eid), f"{e['minute_display']} {e['type'].replace('_', ' ')}{' · ' + who if who else ''}"])
        facts.append({"s": f["statement"], "v": f.get("value"), "tool": f.get("source_tool"),
                      "call": found[0] if found else None, "at": found[1] if found else None, "ev": ev})
    res = ((s.get("scenario") or {}).get("result") or {})
    sim = None
    if res.get("status") == "ok" and res.get("with_change"):
        sim = {"n": res["with_change"].get("futures"), "with": {k: res["with_change"].get(k) for k in ("home_win", "draw", "away_win")},
               "as": {k: (res.get("as_it_happened") or {}).get(k) for k in ("home_win", "draw", "away_win")},
               "margin": res["with_change"].get("margin_95")}
    return {"facts": facts, "calls": calls, "sim": sim}


def _numbers_in(line: str, facts: list[dict], sim: dict | None, score: tuple | None) -> list[list]:
    """Each number in an on-air line, matched to its source: a fact index, "sim", "score", or None (unmatched)."""
    text = line.replace("\u00a0", " ").replace("\u202f", " ")
    out, taken = [], set()
    for a, b in _RX_SCORE.findall(text):
        out.append([f"{a}-{b}", "score" if score and (int(a), int(b)) in (score, score[::-1]) else None])
        taken.update((a, b))
    text = _RX_SCORE.sub(" ", text)
    sim_vals = []
    if sim:
        sim_vals = [sim["n"], sim.get("margin")] + list(sim["with"].values()) + list(sim["as"].values())
    for raw in _RX_NUM.findall(text):
        x = float(raw.replace(",", "."))
        dec = len(raw.replace(",", ".").split(".")[1]) if ("." in raw or "," in raw) else 0
        tol = max(0.5 * 10 ** -dec, 0.5 if dec == 0 else 0)

        def close(v) -> bool:
            try:
                v = abs(float(v))
            except (TypeError, ValueError):
                return False
            return any(abs(c - x) <= tol + 1e-9 for c in ((v, v * 100) if v <= 1 else (v,)))
        src = next((i for i, f in enumerate(facts) if close(f.get("v")) or any(close(n) for n in _RX_NUM.findall(f["s"] or ""))), None)
        if src is None and any(close(v) for v in sim_vals if v is not None):
            src = "sim"
        out.append([raw, src])
    return out


def _checks(checks: list[dict]) -> list[dict]:
    out = []
    for c in checks or []:
        mv = c.get("model_verdict")
        out.append({"a": c.get("attempt"), "code": [[i.get("claim"), i.get("problem")] for i in (c.get("code_issues") or [])],
                    "model": None if mv is None else bool(mv.get("approved")),
                    "mi": [[i.get("line"), i.get("claim"), i.get("problem")] for i in ((mv or {}).get("issues") or [])],
                    "ok": bool(c.get("approved"))})
    return out


def _line_receipts(lines: list[str], drafts: list[dict], checks: list[dict], rx: dict, score, fallback: bool = False) -> dict:
    """Per edition: which facts each line used, its numbers matched to sources, every check, and rejected drafts."""
    last = (drafts or [{}])[-1].get("lines") or []
    fu = [[i for i in (last[k].get("facts_used") or []) if 0 <= i < len(rx["facts"])] if k < len(last) and isinstance(last[k], dict) else []
          for k in range(len(lines))]
    return {"fu": fu, "nm": [_numbers_in(t, rx["facts"], rx["sim"], score) for t in lines], "chk": _checks(checks),
            "rej": [[(x.get("text") if isinstance(x, dict) else x) for x in (d.get("lines") or [])]
                    for d in ((drafts or []) if fallback else (drafts or [])[:-1])]}


def _viewer_segments(segments: list[dict], frame_of, slot_of: dict, ctx) -> list[dict]:
    names: dict[str, str] = {}
    for pid, p in ctx.players.items():
        for alias in (p.name, p.short_name, p.name.split()[-1]):
            names.setdefault(alias.lower(), pid)

    def slot(ref: str | None) -> int:
        if not ref:
            return -1
        pid = ref if ref in slot_of else names.get(ref.lower())
        return slot_of.get(pid, -1) if pid else -1

    def frame_of_event(eid: str | None) -> int | None:
        e = ctx.by_id.get(eid) if eid else None
        return frame_of(e["period"], e["t"]) if e else None

    out = []
    for s in segments:
        if not s.get("aired"):
            continue
        plan, d = s.get("plan") or {}, s.get("director") or {}
        brief, checks = s.get("brief") or {}, s.get("factcheck") or []
        seg = {
            "id": s["moment_id"],
            "f": frame_of(s["period"], s["t"]),
            "type": s["type"],
            "minute": s["minute"],
            "title": s["title"],
            "plan": {k: plan.get(k) for k in ("segment", "angle", "priority", "what_if")},
            "lines": s["commentary"]["lines"],
            "fallback": s["commentary"]["fallback"],
            "check": {
                "attempts": len(checks),
                "approved": bool(checks and checks[-1].get("approved")),
                "codeIssues": sum(len(c.get("code_issues") or []) for c in checks),
            },
            "facts": [
                {"s": f["statement"], "tool": f["source_tool"], "ev": len(f.get("evidence") or [])}
                for f in (brief.get("facts") or [])[:5]
            ],
            "insight": brief.get("insight"),
            "director": {
                "live": d.get("live_camera"),
                "replay": bool(d.get("replay")),
                "fromF": frame_of_event(d.get("replay_from_event_id")),
                "graphic": d.get("graphic"),
                "shots": [
                    {
                        "cam": x["camera"],
                        "sec": x["seconds"],
                        "slow": x["slow_motion"],
                        "focus": slot(x.get("focus_player")),
                        "purpose": x["purpose"],
                    }
                    for x in (d.get("shots") or [])[:4]
                ],
            }
            if d
            else None,
            "desk": {
                "usd": s["cost"]["usd"],
                "calls": s["cost"]["model_calls"],
                "seconds": s["cost"]["seconds"],
                "agents": s["cost"]["by_agent"],
                "receipts": len(s.get("receipts") or []),
                "tools": sorted(
                    {r["tool"] for r in (s.get("receipts") or []) if r.get("agent") == "analyst"}
                ),
            },
        }
        rx = _receipts(s, frame_of_event, ctx)
        m_score = _RX_SCORE.search(s.get("title") or "")       # the score at the moment, from its own title
        score = (int(m_score.group(1)), int(m_score.group(2))) if m_score else None
        seg["rx"] = rx
        com = s.get("commentary") or {}
        seg["lrx"] = _line_receipts([x if isinstance(x, str) else x.get("text") for x in com.get("lines") or []],
                                    com.get("drafts") or [], checks, rx, score, bool(com.get("fallback")))
        if s.get("editions"):
            seg["ed"] = {
                k: {"l": v["lines"], "h": v.get("headline"), "c": v.get("caption"), "q": v.get("question"),
                    "fb": v["fallback"], "att": len(v.get("checks") or []),
                    "ok": bool(v.get("checks") and v["checks"][-1].get("approved")),
                    "code": sum(len(c.get("code_issues") or []) for c in (v.get("checks") or [])),
                    "rx": _line_receipts(v["lines"], v.get("drafts") or [], v.get("checks") or [], rx, score, bool(v["fallback"]))}
                for k, v in s["editions"].items()
            }
            seg["dirs"] = {st: _pack_director(dd, slot, frame_of_event) for st, dd in (s.get("directors") or {}).items() if dd}
            seg["desk"]["editions"] = len(s["editions"])
        dec = s.get("decision")
        if dec and dec.get("teammates"):
            best = dec["teammates"][0]
            seg["decision"] = {
                "player": dec["event"]["player"], "better": bool(dec.get("better_option_exists")),
                "choice": {k: dec["choice"].get(k) for k in ("action", "xg", "value", "space_m", "defenders_in_lane")},
                "from": [dec["choice"]["location"]["x"], dec["choice"]["location"]["y"]],
                "best": {"name": best["name"], "slot": slot_of.get(best["player_id"], -1),
                         "at": [best["location"]["x"], best["location"]["y"]],
                         **{k: best[k] for k in ("p_complete", "xg_from_there", "space_m", "value", "opponents_near_pass_line")}},
            }
        tele = s.get("telestration")
        if tele and tele.get("shapes"):
            fz = tele.get("freeze") or {}
            seg["tele"] = {
                "f": frame_of(fz.get("period", s["period"]), fz.get("t", s["t"])),
                "caption": tele.get("caption"),
                "shapes": [
                    {
                        "kind": sh["kind"],
                        "color": sh["color"],
                        "label": sh.get("label"),
                        "slot": slot_of.get(sh.get("player_id"), -1),
                        **{k: round(sh[k], 2) for k in ("x", "y", "x2", "y2", "r") if sh.get(k) is not None},
                    }
                    for sh in tele["shapes"]
                ],
            }
        sc = s.get("scenario") or {}
        spec, res = sc.get("spec"), sc.get("result") or {}
        if spec:
            ghost = {
                "question": spec["question"],
                "intervention": spec["intervention"],
                "status": res.get("status"),
            }
            if res.get("status") == "ok":
                ghost["shotF"] = frame_of_event(res.get("event_id"))
                ghost["teamSide"] = res.get("team_side")
                if spec["intervention"] == "player_swap":
                    ghost["scout"] = {
                        k: res.get(k)
                        for k in (
                            "taker",
                            "best",
                            "candidates_evaluated",
                            "win_probability_if_scored",
                            "win_probability_if_missed",
                        )
                    }
                else:
                    ghost["actual"], ghost["changed"] = res.get("actual_outcome"), res.get("changed_outcome")
                    ghost["withChange"], ghost["asItHappened"] = (
                        res.get("with_change"),
                        res.get("as_it_happened"),
                    )
                    curves = res.get("curves") or {}
                    ghost["fan"] = [_fan_line(c, frame_of) for c in curves.get("with_change") or []]
                    ghost["featured"] = [
                        {
                            "final": fe["final"],
                            "goalsAfter": fe["goals_after"],
                            "clips": [_pack_clip(c, slot_of, ctx) for c in fe["clips"] if c.get("frames")],
                        }
                        for fe in res.get("featured") or []
                    ]
            else:
                ghost["reason"] = res.get("reason")
            seg["ghost"] = ghost
        out.append(seg)
    return sorted(out, key=lambda x: x["f"])
