"""The season hub's data: league table (actual and expected), results, match reports and player standings.

``build_season(root)`` reads the bundles written by ``matchengine season`` and runs the stats layer over every
match (key moments, win probability through the match). Box scores come from the engine's summaries; xG,
chances created and the expected table come from the events. ``render_season`` embeds it all in one
self-contained page (``matchstats season data/season --out season.html``).
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from importlib import resources
from pathlib import Path

from matchengine.league import get_club
from matchengine.tryout import expected_points, player_stats

from .analyzer import MatchAnalyzer

MOMENT_TYPES = {"goal", "big_chance", "great_save", "woodwork", "penalty_awarded", "red_card", "momentum_shift"}
PLAYER_KEYS = ("goals", "assists", "xg", "xa", "shots", "shots_on_target", "key_passes", "passes", "passes_completed",
               "progressive_passes", "dribbles_completed", "tackles", "tackles_won", "interceptions", "ball_recoveries",
               "clearances", "blocks", "aerials_won", "saves", "xg_faced", "goals_conceded", "fouls", "dispossessed")


def _events(folder: Path) -> list[dict]:
    path = folder / "events.jsonl.gz"
    opener = gzip.open if path.exists() else open
    path = path if path.exists() else folder / "events.jsonl"
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _club(code: str) -> dict:
    c = get_club(code)
    return {
        "code": c.code, "name": c.name, "short": c.short_name, "city": c.city, "stadium": c.stadium,
        "capacity": c.capacity, "founded": c.founded, "manager": c.manager, "formation": c.formation,
        "colors": dict(c.colors), "identity": c.identity,
    }


def _player_profile(p, club_code: str) -> dict:
    return {"id": p.player_id, "name": p.name, "short": p.short_name, "shirt": p.shirt, "club": club_code,
            "role": p.role, "pos": p.preferred_position, "age": p.age, "nat": p.nationality, "foot": p.foot,
            "bio": p.bio, "attr": p.attributes.as_dict()}


def _match(folder: Path, fx: dict) -> tuple[dict, dict[str, dict]]:
    """One match report, and every player's line in it."""
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    events = _events(folder)
    home, away = meta["home"], meta["away"]
    team_of = {home["team_id"]: "home", away["team_id"]: "away"}
    code_of = {home["team_id"]: home["code"], away["team_id"]: away["code"]}

    # stats layer: key moments and the win-probability story, minute by minute
    a = MatchAnalyzer()
    a.process("meta", meta)
    wp: list[list] = []
    for e in events:
        a.process("event", e)
        if e["type"] in ("match_start", "period_start"):
            continue
        minute = int(e["t"] // 60) + (45 if e["period"] == 2 else 0)
        cur = a.metric("win_probability").current
        cur = cur if isinstance(cur, dict) else {"home": cur[0], "draw": cur[1], "away": cur[2]}
        row = [minute, round(cur["home"], 3), round(cur["away"], 3)]
        if wp and wp[-1][0] >= minute:
            wp[-1] = [wp[-1][0], row[1], row[2]]   # the state at the end of each minute
        else:
            wp.append(row)
    moments = sorted((m for m in a.moments if m["type"] in MOMENT_TYPES), key=lambda m: -m["importance"])[:8]
    moments = sorted(moments, key=lambda m: (m["period"], m["t"]))

    xg = {"home": [], "away": []}
    goals, cards = [], []
    for e in events:
        side = team_of.get(e.get("team_id"))
        d = e.get("details") or {}
        if e["type"] == "shot" and side:
            xg[side].append(float((d.get("truth") or {}).get("xg") or 0.0))
        elif e["type"] == "goal" and side:
            goals.append({"min": e["minute_display"], "side": side, "scorer": d.get("scorer_id"), "scorer_name": d.get("scorer_name"),
                          "assist": d.get("assist_id"), "assist_name": d.get("assist_name"),
                          "pen": d.get("shot_kind") == "penalty", "og": bool(d.get("own_goal"))})
        elif e["type"] == "card" and side:
            cards.append({"min": e["minute_display"], "side": side, "player": e.get("player_id"),
                          "name": e.get("player_name"), "card": d.get("card")})
    hw, hd, hl, hxp = expected_points(xg["home"], xg["away"])
    _aw, _ad, _al, axp = expected_points(xg["away"], xg["home"])

    lines: dict[str, dict] = {}
    lineup_ids = {p["player_id"] for p in home["lineup"]} | {p["player_id"] for p in away["lineup"]}
    for pid, row in summary["players"].items():
        team_id = row["team_id"]
        st = player_stats(events, pid, team_id)
        st.pop("_conceded_team", None)
        side = team_of[team_id]
        if row.get("position") == "GK" or (pid in lineup_ids and any(p["player_id"] == pid and p["role"] == "GK" for p in (home["lineup"] + away["lineup"]))):
            other = "away" if side == "home" else "home"
            st["goals_conceded"] = summary["final_score"][other] if row["minutes"] >= 60 else 0
            st["clean_sheet"] = 1 if row["minutes"] >= 60 and summary["final_score"][other] == 0 else 0
        lines[pid] = {
            "club": code_of[team_id], "side": side, "min": row["minutes"], "start": pid in lineup_ids, "pos": row.get("position"),
            "dist": row.get("distance_km", 0), "sprints": row.get("sprints", 0), "top": row.get("top_speed_kmh", 0),
            **{k: round(float(v), 3) for k, v in st.items() if k in PLAYER_KEYS or k == "clean_sheet"},
        }
        for c in cards:
            if c["player"] == pid:
                lines[pid]["yc" if c["card"] == "yellow" else "rc"] = lines[pid].get("yc" if c["card"] == "yellow" else "rc", 0) + 1
                if c["card"] == "second_yellow":
                    lines[pid]["yc"] = lines[pid].get("yc", 0) + 1
    # player of the match: goals and assists first, then chances and the work that stops goals
    def potm_score(r: dict) -> float:
        return (3 * r.get("goals", 0) + 2 * r.get("assists", 0) + r.get("xg", 0) + r.get("xa", 0) + 0.25 * r.get("key_passes", 0)
                + 0.3 * r.get("saves", 0) + 0.2 * (r.get("tackles_won", 0) + r.get("interceptions", 0)) + 0.15 * r.get("dribbles_completed", 0))
    potm = max(lines, key=lambda k: potm_score(lines[k])) if lines else None

    th, ta = summary["teams"]["home"], summary["teams"]["away"]
    keys = ("possession", "xg", "shots", "shots_on_target", "passes", "pass_completion", "corners", "fouls", "yellow_cards",
            "red_cards", "tackles_won", "saves", "distance_km")
    report = {
        "id": fx["match_id"], "md": fx["matchday"], "date": fx["date"], "home": home["code"], "away": away["code"],
        "score": dict(summary["final_score"]), "venue": meta["venue"]["stadium"], "att": meta.get("attendance"),
        "ref": (meta.get("referee") or {}).get("name") if isinstance(meta.get("referee"), dict) else meta.get("referee"),
        "xg": [round(sum(xg["home"]), 2), round(sum(xg["away"]), 2)],
        "xpts": [round(hxp, 2), round(axp, 2)], "xres": [round(hw, 3), round(hd, 3), round(hl, 3)],
        "stats": {k: [th.get(k, 0), ta.get(k, 0)] for k in keys},
        "goals": goals, "cards": cards, "wp": wp, "potm": potm,
        "moments": [{"min": m["minute_display"], "type": m["type"], "title": m["title"],
                     "side": team_of.get(m.get("team_id"), "")} for m in moments],
        "lineups": {s: [p["player_id"] for p in meta[s]["lineup"]] for s in ("home", "away")},
        "subs": [{"min": e["minute_display"], "side": team_of.get(e.get("team_id")), "out": (e.get("details") or {}).get("player_out_id"),
                  "in": (e.get("details") or {}).get("player_in_id")} for e in events if e["type"] == "substitution"],
        "demo": bool(fx.get("demo")),
    }
    return report, lines


def build_season(root: str | Path, links: dict[str, str] | None = None, board: str | Path | None = None) -> dict:
    root = Path(root)
    index = json.loads((root / "season.json").read_text(encoding="utf-8"))
    clubs = {c: _club(c) for c in index["clubs"]}
    players: dict[str, dict] = {}
    for c in index["clubs"]:
        for p in get_club(c).squad:
            players[p.player_id] = _player_profile(p, c)

    matches, log = [], defaultdict(list)
    for fx in index["matches"]:
        report, lines = _match(root / fx["match_id"], fx)
        report["link"] = (links or {}).get(fx["match_id"])
        matches.append(report)
        for pid, ln in lines.items():
            log[pid].append({"m": fx["match_id"], **ln})

    # league table, actual and expected
    tab = {c: defaultdict(float, {k: 0.0 for k in ("p", "w", "d", "l", "gf", "ga", "pts", "xgf", "xga", "xpts")}) for c in clubs}
    form = defaultdict(list)
    race = {c: [0] * (max(m["md"] for m in matches) + 1) for c in clubs}
    for m in sorted(matches, key=lambda m: m["md"]):
        h, a = m["home"], m["away"]
        gh, ga = m["score"]["home"], m["score"]["away"]
        for code, gf, gaa, xgf, xga, xp, side in ((h, gh, ga, m["xg"][0], m["xg"][1], m["xpts"][0], "H"),
                                                    (a, ga, gh, m["xg"][1], m["xg"][0], m["xpts"][1], "A")):
            t = tab[code]
            t["p"] += 1
            t["gf"] += gf
            t["ga"] += gaa
            t["xgf"] += xgf
            t["xga"] += xga
            t["xpts"] += xp
            res = "W" if gf > gaa else "D" if gf == gaa else "L"
            t[res.lower()] += 1
            t["pts"] += 3 if res == "W" else 1 if res == "D" else 0
            form[code].append({"r": res, "m": m["id"], "opp": a if code == h else h, "ha": side, "s": f"{gf}-{gaa}"})
        for code in (h, a):
            race[code][m["md"]] = int(tab[code]["pts"])
    for code in race:   # a club without a game on a matchday keeps its points
        for k in range(1, len(race[code])):
            race[code][k] = max(race[code][k], race[code][k - 1])
    table = []
    for code, t in tab.items():
        table.append({"club": code, **{k: (round(v, 2) if isinstance(v, float) and not v.is_integer() else int(v)) for k, v in t.items()},
                      "gd": int(t["gf"] - t["ga"]), "form": form[code][-5:], "race": race[code]})
    table.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], r["club"]))
    for i, r in enumerate(table, 1):
        r["pos"] = i
    for i, r in enumerate(sorted(table, key=lambda r: -r["xpts"]), 1):
        r["xpos"] = i

    # season totals per player
    for pid, rows in log.items():
        tot = defaultdict(float)
        for r in rows:
            tot["apps"] += 1
            tot["starts"] += bool(r["start"])
            tot["min"] += r["min"]
            for k in PLAYER_KEYS + ("clean_sheet", "yc", "rc", "dist", "sprints"):
                tot[k] += r.get(k, 0) or 0
            tot["top"] = max(tot["top"], r.get("top", 0) or 0)
        players.setdefault(pid, {"id": pid, "name": pid, "club": rows[0]["club"]})
        players[pid]["season"] = {k: round(v, 2) for k, v in tot.items()}
        players[pid]["log"] = rows
        players[pid]["potm"] = sum(1 for m in matches if m["potm"] == pid)

    out = {"competition": index["competition"], "season": index["season"], "clubs": clubs, "table": table,
           "matches": matches, "players": players}
    if board and Path(board).exists():
        out["tryouts"] = _tryouts(Path(board))
    return out


def _tryouts(board: Path) -> list[dict]:
    """Club mode results to show on player pages (who was tried where, and how it went)."""
    data = json.loads(board.read_text(encoding="utf-8"))
    rows = []
    for side, block in data.get("sides", {}).items():
        for r in block.get("tryouts", []):
            rows.append({"player": r["candidate"]["player_id"], "for": r["fixture"]["team"], "pos": r["position"]["label"],
                         "instead": r["incumbent"]["name"], "xpts": r["difference"]["xpts"],
                         "margin": r["difference"]["xpts_margin_95"], "clear": r["difference"].get("clear", False),
                         "n": r["with_candidate"]["futures"], "verdict": (r.get("scout") or {}).get("verdict")})
    return rows


def render_season(data: dict) -> str:
    tpl = resources.files("matchstats").joinpath("viewer/season.html").read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = tpl.replace('"__SEASON_DATA__"', blob).replace("__TITLE__", f"Kairos · {data['competition']} {data['season']}")
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            "</head>\n<body>\n" + page + "\n</body>\n</html>\n")
