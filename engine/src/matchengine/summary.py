"""Ground-truth box score built from the engine's own events and physical counters.

This is used for calibration and tests. The analytics layer computes its own
statistics from the raw stream; comparing the two is a free correctness check.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .engine import MatchEngine


TEAM_KEYS = (
    "goals",
    "xg",
    "shots",
    "shots_on_target",
    "shots_blocked",
    "shots_off_target",
    "big_chances",
    "passes",
    "passes_completed",
    "crosses",
    "crosses_completed",
    "long_passes",
    "through_balls",
    "corners",
    "fouls",
    "yellow_cards",
    "red_cards",
    "offsides",
    "tackles",
    "tackles_won",
    "interceptions",
    "clearances",
    "pressures",
    "dribbles",
    "dribbles_completed",
    "aerials_won",
    "saves",
    "carries",
    "ball_recoveries",
    "substitutions",
)


def _team_stats(events: list[dict], team_id: str) -> dict:
    s: dict = defaultdict(float)
    for e in events:
        if e["team_id"] != team_id:
            continue
        t, d = e["type"], e["details"]
        if t == "pass":
            s["passes"] += 1
            if d["outcome"] == "complete":
                s["passes_completed"] += 1
            if d["pass_kind"] in ("cross", "free_kick_cross") or (
                d["pass_kind"] == "corner" and d["height"] != "ground"
            ):
                s["crosses"] += 1
                if d["outcome"] == "complete":
                    s["crosses_completed"] += 1
            if d["pass_kind"] in ("long", "switch") or d["length_m"] >= 32:
                s["long_passes"] += 1
            if d["pass_kind"] == "through":
                s["through_balls"] += 1
            if d.get("restart_type") == "corner":
                s["corners"] += 1
        elif t == "shot":
            s["shots"] += 1
            s["xg"] += d["truth"]["xg"]
            if d["on_target"]:
                s["shots_on_target"] += 1
            if d["outcome"] == "blocked":
                s["shots_blocked"] += 1
            if d["outcome"] in ("off_target", "post"):
                s["shots_off_target"] += 1
            if d["truth"]["big_chance"]:
                s["big_chances"] += 1
        elif t == "goal":
            s["goals"] += 1
        elif t == "foul":
            s["fouls"] += 1
        elif t == "card":
            s["yellow_cards" if d["card"] == "yellow" else "red_cards"] += 1
            if d["card"] == "second_yellow":
                s["yellow_cards"] += 1
        elif t == "offside":
            s["offsides"] += 1
        elif t == "tackle":
            s["tackles"] += 1
            if d["outcome"] == "won":
                s["tackles_won"] += 1
        elif t == "interception":
            s["interceptions"] += 1
        elif t == "clearance":
            s["clearances"] += 1
        elif t == "pressure":
            s["pressures"] += 1
        elif t == "dribble":
            s["dribbles"] += 1
            if d["outcome"] == "success":
                s["dribbles_completed"] += 1
        elif t == "aerial_duel":
            s["aerials_won"] += 1
        elif t == "save":
            s["saves"] += 1
        elif t == "carry":
            s["carries"] += 1
        elif t == "ball_recovery":
            s["ball_recoveries"] += 1
        elif t == "substitution":
            s["substitutions"] += 1
    for key in TEAM_KEYS:
        s[key] += 0  # make every stat present, even when zero
    out = {k: (round(v, 3) if isinstance(v, float) and not v.is_integer() else int(v)) for k, v in s.items()}
    out["pass_completion"] = round(s["passes_completed"] / s["passes"], 3) if s["passes"] else 0.0
    return out


def build_summary(eng: MatchEngine, meta: dict) -> dict:
    events = eng.events
    for i in range(len(eng.pl)):
        if eng.pl[i] is not None and eng.pl[i].on_pitch:
            eng.sync_stats(i)
    teams = {}
    players = {}
    total_poss = sum(eng.possession_s) or 1.0
    for side, team in enumerate(eng.teams):
        st = _team_stats(events, team.team_id)
        st["possession"] = round(eng.possession_s[side] / total_poss, 3)
        everyone = list(team.players) + list(team.used_players)
        st["distance_km"] = round(sum(p.distance for p in everyone) / 1000, 2)
        st["sprints"] = sum(p.sprints for p in everyone)
        st["score"] = team.score
        teams["home" if side == 0 else "away"] = {"team_id": team.team_id, "name": team.club.name, **st}
        end_s = 45 * 60 * (len(eng.period_lengths) - 1) + (
            eng.period_lengths[-1] if eng.period_lengths else 0.0
        )
        for p in everyone:
            start = p.minutes_from
            stop = p.minutes_to if p.minutes_to is not None else end_s
            players[p.player_id] = {
                "name": p.profile.name,
                "team_id": team.team_id,
                "position": p.slot.label,
                "minutes": round(max(0.0, stop - start) / 60.0, 1),
                "distance_km": round(p.distance / 1000, 2),
                "hsr_m": round(p.hsr_distance),
                "sprint_m": round(p.sprint_distance),
                "sprints": p.sprints,
                "top_speed_kmh": round(p.max_speed * 3.6, 1),
                "energy_end": round(p.energy, 2),
            }
    per_player = defaultdict(lambda: defaultdict(float))
    for e in events:
        pid = e.get("player_id")
        if not pid:
            continue
        t, d = e["type"], e["details"]
        pp = per_player[pid]
        if t == "pass":
            pp["passes"] += 1
            pp["passes_completed"] += d["outcome"] == "complete"
        elif t == "shot":
            pp["shots"] += 1
            pp["xg"] += d["truth"]["xg"]
        elif t == "goal":
            pp["goals"] += 1
            if d.get("assist_id"):
                per_player[d["assist_id"]]["assists"] += 1
        elif t in ("tackle", "interception", "pressure", "dribble", "foul"):
            pp[t + "s"] += 1
    for pid, vals in per_player.items():
        if pid in players:
            players[pid].update(
                {k: (round(v, 3) if not float(v).is_integer() else int(v)) for k, v in vals.items()}
            )
    return {
        "match_id": eng.match_id,
        "final_score": {"home": eng.teams[0].score, "away": eng.teams[1].score},
        "goals": eng.goals,
        "teams": teams,
        "players": players,
        "ball_in_play_min": round(eng.ball_in_play_s / 60.0, 1),
        "added_time": eng.added_minutes,
        "events": len(events),
    }
