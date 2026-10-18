"""Calibration: simulate many matches and compare against typical top-flight averages.

Targets are per team per match and reflect widely published league-wide averages
for top European divisions in recent seasons (rounded ranges, not any single
provider's data). The engine is "calibrated" when the simulated mean of every
metric lands inside its range.
"""

from __future__ import annotations

import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations

from .engine import MatchEngine
from .league import load_league
from .models import MatchConfig
from .params import DEFAULT_PARAMS, EngineParams


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    low: float
    high: float
    per: str = "team"  # "team" or "match"


TARGETS: tuple[Target, ...] = (
    Target("goals", "Goals", 1.2, 1.65),
    Target("xg", "Expected goals (xG)", 1.2, 1.65),
    Target("shots", "Shots", 11.0, 14.5),
    Target("shots_on_target", "Shots on target", 3.6, 5.2),
    Target("conversion", "Goals per shot", 0.09, 0.125),
    Target("passes", "Passes", 380.0, 560.0),
    Target("pass_completion", "Pass completion", 0.77, 0.86),
    Target("crosses", "Crosses (open play + set piece)", 12.0, 22.0),
    Target("corners", "Corners", 4.2, 6.2),
    Target("fouls", "Fouls committed", 9.0, 12.5),
    Target("yellow_cards", "Yellow cards", 1.4, 2.3),
    Target("offsides", "Offsides", 1.2, 2.6),
    Target("tackles", "Tackles", 14.0, 22.0),
    Target("interceptions", "Interceptions", 7.0, 14.0),
    Target("clearances", "Clearances", 12.0, 26.0),
    Target("pressures", "Pressures", 120.0, 190.0),
    Target("dribbles", "Take-ons attempted", 7.0, 15.0),
    Target("aerials_won", "Aerial duels won", 12.0, 22.0),
    Target("distance_km", "Distance covered (km, incl. subs)", 104.0, 118.0),
    Target("sprints", "Sprints (>25.2 km/h, >1 s)", 110.0, 260.0),
    Target("ball_in_play_min", "Ball in play (minutes)", 52.0, 60.0, per="match"),
    Target("added_time_total", "Added time (minutes, both halves)", 5.0, 11.0, per="match"),
)


def _run_one(args: tuple[str, str, int, dict]) -> dict:
    home, away, seed, overrides = args
    league = load_league()
    params = EngineParams(**{**DEFAULT_PARAMS.__dict__, **overrides}) if overrides else DEFAULT_PARAMS
    cfg = MatchConfig(home=league[home], away=league[away], seed=seed)
    eng = MatchEngine(cfg, params=params, emit_frames=False)
    res = eng.simulate(keep_frames=False)
    s = res.summary
    out = {
        "teams": [],
        "ball_in_play_min": s["ball_in_play_min"],
        "added_time_total": sum(x or 0 for x in s["added_time"]),
        "fixture": f"{home}-{away}",
        "seed": seed,
        "score": s["final_score"],
    }
    for side in ("home", "away"):
        t = s["teams"][side]
        shots = t.get("shots", 0)
        out["teams"].append(
            {
                **{
                    k: t.get(k, 0)
                    for k in (
                        "goals",
                        "xg",
                        "shots",
                        "shots_on_target",
                        "passes",
                        "pass_completion",
                        "crosses",
                        "corners",
                        "fouls",
                        "yellow_cards",
                        "offsides",
                        "tackles",
                        "interceptions",
                        "clearances",
                        "pressures",
                        "dribbles",
                        "aerials_won",
                        "distance_km",
                        "sprints",
                        "possession",
                        "red_cards",
                    )
                },
                "conversion": (t.get("goals", 0) / shots) if shots else 0.0,
                "team": t["team_id"],
            }
        )
    return out


def run_calibration(
    n_matches: int = 24, workers: int = 2, overrides: dict | None = None, seed0: int = 1000
) -> dict:
    league = load_league()
    codes = list(league)
    fixtures = list(combinations(codes, 2))
    jobs = []
    for k in range(n_matches):
        h, a = fixtures[k % len(fixtures)]
        if k % 2:
            h, a = a, h
        jobs.append((h, a, seed0 + k, overrides or {}))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_run_one, jobs))
    else:
        results = [_run_one(j) for j in jobs]
    return summarise(results)


def summarise(results: list[dict]) -> dict:
    rows = []
    for tgt in TARGETS:
        if tgt.per == "team":
            vals = [t[tgt.key] for r in results for t in r["teams"]]
        else:
            vals = [r[tgt.key] for r in results]
        if tgt.key == "conversion":
            goals = sum(t["goals"] for r in results for t in r["teams"])
            shots = sum(t["shots"] for r in results for t in r["teams"])
            mean = goals / shots if shots else 0.0
        else:
            mean = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        rows.append(
            {
                "key": tgt.key,
                "label": tgt.label,
                "mean": round(mean, 3),
                "sd": round(sd, 3),
                "low": tgt.low,
                "high": tgt.high,
                "ok": tgt.low <= mean <= tgt.high,
            }
        )
    total_goals = [r["score"]["home"] + r["score"]["away"] for r in results]
    home_w = sum(r["score"]["home"] > r["score"]["away"] for r in results)
    draws = sum(r["score"]["home"] == r["score"]["away"] for r in results)
    return {
        "matches": len(results),
        "passed": sum(r["ok"] for r in rows),
        "total": len(rows),
        "rows": rows,
        "results": {
            "home_win": round(home_w / len(results), 3),
            "draw": round(draws / len(results), 3),
            "away_win": round(1 - (home_w + draws) / len(results), 3),
            "goals_per_match": round(statistics.fmean(total_goals), 2),
            "max_goals": max(total_goals),
        },
        "matches_detail": [
            {"fixture": r["fixture"], "seed": r["seed"], "score": r["score"]} for r in results
        ],
    }


def format_table(report: dict) -> str:
    lines = [f"{'metric':38s} {'sim mean':>9s} {'sd':>7s}   target", "-" * 72]
    for r in report["rows"]:
        flag = "ok" if r["ok"] else "<<"
        lines.append(f"{r['label']:38s} {r['mean']:9.3f} {r['sd']:7.2f}   {r['low']}-{r['high']}  {flag}")
    lines.append("-" * 72)
    lines.append(
        f"{report['passed']}/{report['total']} within range over {report['matches']} matches; "
        f"results {json.dumps(report['results'])}"
    )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(format_table(run_calibration(n)))
