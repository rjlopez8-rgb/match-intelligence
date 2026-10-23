"""A full Meridian League season: every club plays every other twice (14 matchdays, 56 matches).

``matchengine season --out data/season --demo data/demo/riv-kin-1119`` simulates every fixture and writes one
small bundle per match (meta, gzipped events, box-score summary; no tracking, which would be ~5 MB a match).
The demo match slots into the schedule as Riverside v Kingsport, so the 3D broadcast is one of the season's
real games. Matches are independent, so they run in parallel and a re-run only simulates what's missing.
"""

from __future__ import annotations

import argparse

import datetime as dt
import gzip
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .engine import ENGINE_VERSION, MatchEngine
from .league import CLUB_SPECS, get_club
from .models import MatchConfig
from .summary import build_summary

SEASON = "2026/27"
SEASON_SEED = 202627


def fixtures(codes: list[str]) -> list[tuple[int, str, str]]:
    """Double round robin by the circle method: (matchday, home, away). Second half mirrors the first."""
    teams = list(codes)
    n = len(teams)
    rounds = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = teams[i], teams[n - 1 - i]
            pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    out = [(r + 1, h, a) for r, pairs in enumerate(rounds) for h, a in pairs]
    out += [(r + n, a, h) for r, pairs in enumerate(rounds) for h, a in pairs]
    return out


def schedule(demo_home: str = "RIV", demo_away: str = "KIN", demo_date: str = "2026-10-24") -> list[dict]:
    """Every fixture with its date: one matchday a week, with the demo match on its real kick-off date."""
    codes = [s.code for s in CLUB_SPECS]
    fx = fixtures(codes)
    demo_md = next(md for md, h, a in fx if (h, a) == (demo_home, demo_away))
    anchor = dt.date.fromisoformat(demo_date)
    out = []
    for k, (md, h, a) in enumerate(fx):
        day = anchor + dt.timedelta(days=7 * (md - demo_md))
        out.append({"matchday": md, "date": day.isoformat(), "home": h, "away": a,
                    "seed": SEASON_SEED * 100 + k, "match_id": f"md{md:02d}-{h}-{a}".lower()})
    return out


def _write(out: Path, meta: dict, events: list[dict], summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    with gzip.open(out / "events.jsonl.gz", "wt", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e, ensure_ascii=False, separators=(",", ":")) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")


def _play(args: tuple) -> dict:
    fx, root = args
    cfg = MatchConfig(home=get_club(fx["home"]), away=get_club(fx["away"]), seed=fx["seed"], match_id=fx["match_id"],
                      season=SEASON, matchday=fx["matchday"], kickoff_utc=f"{fx['date']}T14:00:00Z")
    eng = MatchEngine(cfg, emit_frames=False)
    meta = eng.meta()
    for _ in eng.run():
        pass
    summary = build_summary(eng, meta)
    _write(Path(root) / fx["match_id"], meta, eng.events, summary)
    return {**fx, "score": {"home": eng.teams[0].score, "away": eng.teams[1].score}}


def simulate_season(root: str | Path, demo: str | Path | None = None, workers: int | None = None,
                    progress=print) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    demo = Path(demo) if demo else None
    demo_meta = json.loads((demo / "meta.json").read_text(encoding="utf-8")) if demo else None
    dh = demo_meta["home"]["code"] if demo_meta else "RIV"
    da = demo_meta["away"]["code"] if demo_meta else "KIN"
    sched = schedule(dh, da, (demo_meta or {}).get("kickoff_utc", "2026-10-24")[:10])
    jobs, done = [], []
    for fx in sched:
        out = root / fx["match_id"]
        if demo_meta and (fx["home"], fx["away"]) == (dh, da):
            fx["match_id"] = demo_meta["match_id"]
            fx["demo"] = True
            out = root / fx["match_id"]
            if not (out / "summary.json").exists():
                out.mkdir(parents=True, exist_ok=True)
                meta = {**demo_meta, "matchday": fx["matchday"], "season": SEASON}
                (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
                with (demo / "events.jsonl").open(encoding="utf-8") as src, gzip.open(out / "events.jsonl.gz", "wt", encoding="utf-8") as dst:
                    shutil.copyfileobj(src, dst)
                shutil.copy(demo / "summary.json", out / "summary.json")
            s = json.loads((out / "summary.json").read_text(encoding="utf-8"))["final_score"]
            done.append({**fx, "score": {"home": s["home"], "away": s["away"]}})
            continue
        if (out / "summary.json").exists():
            s = json.loads((out / "summary.json").read_text(encoding="utf-8"))["final_score"]
            done.append({**fx, "score": {"home": s["home"], "away": s["away"]}})
        else:
            jobs.append(fx)
    n = max(1, min(len(jobs) or 1, workers or int(os.environ.get("MATCH_WORKERS") or (os.cpu_count() or 2))))
    if jobs:
        with ProcessPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(_play, (fx, str(root))) for fx in jobs]
            for k, f in enumerate(as_completed(futs), 1):
                r = f.result()
                done.append(r)
                progress(f"  {k}/{len(jobs)}  MD{r['matchday']:>2} {r['home']} {r['score']['home']}-{r['score']['away']} {r['away']}")
    done.sort(key=lambda r: (r["matchday"], r["match_id"]))
    index = {"competition": "Meridian League", "season": SEASON, "engine": ENGINE_VERSION,
             "clubs": [s.code for s in CLUB_SPECS], "matches": done}
    (root / "season.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="matchengine season", description="Simulate a full league season")
    ap.add_argument("--out", required=True)
    ap.add_argument("--demo", default=None, help="a recorded match bundle to slot into the schedule")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args(argv)
    idx = simulate_season(a.out, a.demo, a.workers)
    print(f"{len(idx['matches'])} matches -> {a.out}/season.json")


if __name__ == "__main__":
    main()


