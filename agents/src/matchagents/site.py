"""The website: the season hub plus every season match in 3D, sharing one viewer page.

    site/index.html                 season hub (table, results, match reports, players, clubs)
    site/watch.html                 the 3D broadcast, one copy; it loads the match named in its URL
    site/matches/<id>.hl.json       highlights of every match: goals, saves and chances, back to back
    site/matches/<id>.full.json     the whole match: the match of the week on each matchday, and the demo
                                    match with its AI desk and Club mode
    site/<demo id>.html             old link to the demo broadcast, now a redirect

3D needs tracking, which the committed season does not keep. ``matchengine season --tracking DIR`` re-plays every
match on its seed (checked event for event against the committed season) and ``--tracking DIR`` here reads them.
Without it only the demo match gets 3D. Everything is static, so it deploys to Static Web Apps as it is.
"""

from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SWA_LIMIT_MB = 250         # Azure Static Web Apps (Free and Standard): total app size
SWA_WARN_MB = 225


def _drama(report: dict) -> float:
    """How much the result swung (total movement of home-minus-away win probability, minute by minute), lifted 15%
    per goal: the match of the week should be in doubt for long and still have something to watch."""
    wp = report.get("wp") or []
    swing = sum(abs((b[1] - b[2]) - (a[1] - a[2])) for a, b in zip(wp, wp[1:]))
    goals = sum((report.get("score") or {}).values()) if isinstance(report.get("score"), dict) else 0
    return swing * (1 + 0.15 * goals)


def matches_of_the_week(season: dict, demo_id: str | None) -> dict[int, str]:
    """One match per matchday. The demo match is its matchday's pick: it is the one with the AI desk."""
    best: dict[int, tuple[float, str]] = {}
    for m in season["matches"]:
        score = float("inf") if m["id"] == demo_id else _drama(m)
        if m["md"] not in best or score > best[m["md"]][0]:
            best[m["md"]] = (score, m["id"])
    return {md: mid for md, (_s, mid) in sorted(best.items())}


def _write_json(path: Path, data: dict) -> int:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _watch(mid: str, v: str) -> str:
    return f"watch.html?m={mid}&v={v}"


def _match_files(job: tuple) -> dict:
    """Viewer data for one match: highlights always, the full match when it is a match of the week."""
    from matchstats.replay import build_replay, highlights

    mid, bundle, full, segments, out, api = job
    data = build_replay(bundle, segments)
    if api and data.get("club"):
        data["club"]["api"] = api     # Club mode's live try-outs on the hosted engine server
    hub = f"index.html#/match/{mid}"
    hl = highlights(data)
    hl["hub"] = hub
    hl["alt"] = {"kind": "full", "href": _watch(mid, "full")} if full else None
    sizes = {"hl": _write_json(Path(out) / f"{mid}.hl.json", hl), "clips": len(hl["clips"])}
    if full:
        data["hub"] = hub
        data["alt"] = {"kind": "hl", "href": _watch(mid, "hl")}
        sizes["full"] = _write_json(Path(out) / f"{mid}.full.json", data)
    return {"id": mid, **sizes}


def _recap_fields(season: dict, season_dir: Path, recaps: Path | None) -> int:
    """Each match report gets its recap (the Recap agent's, or code-written from the stats when there is none) and
    its "what decided it?" simulation. Returns how many recaps came from the agent."""
    from .recap import TURNING_FILE, evidence, fallback_recap

    tp_path = season_dir / TURNING_FILE
    research = json.loads(tp_path.read_text(encoding="utf-8"))["matches"] if tp_path.exists() else {}
    written = json.loads(recaps.read_text(encoding="utf-8")) if recaps and recaps.exists() else {}
    agent = 0
    for m in season["matches"]:
        r = research.get(m["id"]) or {}
        rec = written.get(m["id"])
        if rec and not rec.get("fallback") and rec.get("by") not in (None, "code", "offline"):
            agent += 1
        else:
            rec = {**fallback_recap(evidence(m, season, r)), "by": "code"}
        m["recap"] = {k: rec[k] for k in ("headline", "recap", "decided")} | {
            "by": rec.get("by"), "reviewed": bool(rec.get("reviewed"))}
        if r.get("status") == "ok":
            tp = r["turning_point"]
            m["whatif"] = {"minute": tp["minute"], "player": tp["player"], "type": tp["type"], "side": tp["side"],
                           "xg": tp["xg"], "change": tp["change"], "n": r["futures"],
                           "actual": [r["as_it_happened"][k] for k in ("home_win", "draw", "away_win")],
                           "flipped": [r["with_change"][k] for k in ("home_win", "draw", "away_win")],
                           "margin": r["with_change"]["margin_95"],
                           "score": (r["with_change"].get("most_common_scores") or [[None]])[0][0]}
    return agent


def build_site(season_dir: Path, demo: Path, out: Path, segments: Path | None = None, tracking: Path | None = None,
               workers: int | None = None, recaps: Path | None = None, tryout_api: str | None = None, progress=print) -> dict:
    from matchstats.replay import render_viewer_shell
    from matchstats.season import build_season, render_season

    out.mkdir(parents=True, exist_ok=True)
    mdir = out / "matches"
    if mdir.exists():
        shutil.rmtree(mdir)             # a rebuild never leaves stale matches behind
    mdir.mkdir()
    demo_id = json.loads((demo / "meta.json").read_text(encoding="utf-8"))["match_id"]
    board = demo / "tryouts" / "board.json"
    season = build_season(season_dir, links={demo_id: _watch(demo_id, "full")}, board=board if board.exists() else None)
    motw = matches_of_the_week(season, demo_id)
    full_ids = set(motw.values())

    jobs, missing = [], []
    segs = json.loads(segments.read_text(encoding="utf-8")) if segments else None
    for m in season["matches"]:
        mid = m["id"]
        if mid == demo_id:
            bundle = demo
        elif tracking and (tracking / mid / "tracking.jsonl.gz").exists():
            bundle = tracking / mid
        else:
            missing.append(mid)
            continue
        jobs.append((mid, str(bundle), mid in full_ids, segs if mid == demo_id else None, str(mdir), tryout_api))

    n = max(1, min(len(jobs), workers or int(os.environ.get("MATCH_WORKERS") or (os.cpu_count() or 2))))
    progress(f"3D data for {len(jobs)} matches ({len([j for j in jobs if j[2]])} full) on {n} worker(s)")
    done: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=n) as ex:
        for r in ex.map(_match_files, jobs):
            done[r["id"]] = r

    for m in season["matches"]:
        r = done.get(m["id"])
        m["motw"] = m["id"] in full_ids
        m["w3d"] = {"hl": _watch(m["id"], "hl"), "full": _watch(m["id"], "full") if "full" in r else None,
                    "clips": r["clips"]} if r else None
    season["motw"] = motw
    n_agent = _recap_fields(season, Path(season_dir), recaps)
    (out / "index.html").write_text(render_season(season), encoding="utf-8")
    (out / "watch.html").write_text(render_viewer_shell(), encoding="utf-8")
    # Azure Static Web Apps: cache the match files (they only change with a new build), basic hardening headers
    (out / "staticwebapp.config.json").write_text(json.dumps({
        "routes": [{"route": "/matches/*", "headers": {"Cache-Control": "public, max-age=3600"}}],
        "mimeTypes": {".json": "application/json"},
        "globalHeaders": {"X-Content-Type-Options": "nosniff", "Referrer-Policy": "strict-origin-when-cross-origin"},
    }, indent=1), encoding="utf-8")
    # the demo's old address keeps working
    (out / f"{demo_id}.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>Redirecting…</title>'
        f'<meta http-equiv="refresh" content="0; url={_watch(demo_id, "full")}">'
        f'<script>location.replace("{_watch(demo_id, "full")}" + location.hash)</script>'
        f'<a href="{_watch(demo_id, "full")}">Open the 3D broadcast</a>', encoding="utf-8")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    report = {
        "matches": len(done), "full": sorted(i for i in done if "full" in done[i]), "missing": missing,
        "mb": round(total / 1e6, 1), "recaps_by_agent": n_agent,
        "hl_mb": round(sum(r["hl"] for r in done.values()) / 1e6, 1),
        "full_mb": round(sum(r.get("full", 0) for r in done.values()) / 1e6, 1),
    }
    if total / 1e6 > SWA_WARN_MB:
        progress(f"WARNING: the site is {report['mb']} MB; Static Web Apps allows {SWA_LIMIT_MB} MB in total.")
    return report
