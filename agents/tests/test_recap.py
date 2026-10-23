import json
from pathlib import Path

from conftest import run

from matchagents import recap as R

DEMO = Path(__file__).resolve().parents[2] / "data" / "demo" / "riv-kin-1119"

EV = {
    "competition": "Meridian League", "matchday": 4, "date": "2026-10-03", "venue": "The Orchard",
    "home_team": "Elmhurst Albion", "away_team": "Kingsport United", "final_score": {"home": 0, "away": 1},
    "goals": [{"minute": "9'", "team": "Kingsport United", "scorer": "Rafael Duarte", "assist": "Yuto Arakawa",
               "penalty": False, "own_goal": False}],
    "red_cards": [],
    "expected_goals": {"home_team": 1.32, "away_team": 1.05},
    "possession_percent": {"home_team": 68, "away_team": 32},
    "shots": {"home_team": 8, "away_team": 13}, "shots_on_target": {"home_team": 3, "away_team": 6},
    "result_the_chances_deserved_percent": {"home_team_win": 44, "draw": 30, "away_team_win": 26},
    "what_decided_it": {
        "moment": "Rafael Duarte's goal at 9'", "minute": "9'", "player": "Rafael Duarte", "team": "Kingsport United",
        "chance_quality_xg": 0.1, "simulation": "...", "if_it": "had been saved", "simulated_futures_each_way": 20,
        "as_it_happened_percent": {"home_team_win": 0, "draw": 5, "away_team_win": 95},
        "if_flipped_percent": {"home_team_win": 40, "draw": 20, "away_team_win": 40},
        "margin_of_error_percent": 22, "most_common_final_score_if_flipped": "1-0",
    },
}
NAMES = {"Rafael Duarte", "Duarte", "Yuto Arakawa", "Arakawa", "Kwame Mensah", "Mensah"}


def test_turning_point_is_the_biggest_swing_with_its_shot():
    meta = json.loads((DEMO / "meta.json").read_text(encoding="utf-8"))
    events = [json.loads(x) for x in (DEMO / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    tp = R.turning_point(meta, events)
    shot = next(e for e in events if e["event_id"] == tp["event_id"])
    assert shot["type"] == "shot" and tp["player_id"] == shot["player_id"]
    assert tp["change"] == ("shot_misses" if tp["outcome"] == "goal" else "shot_scores")
    assert tp["type"] in R.TURNING_TYPES and tp["weight_pts"] is not None


def test_code_recap_passes_its_own_checks():
    rec = R.fallback_recap(EV)
    assert rec["headline"] == "Kingsport United beat Elmhurst Albion 1-0"
    assert "40%" in rec["decided"] and "95%" in rec["decided"]
    assert R.check_recap(rec, EV, NAMES) == []


def test_invented_numbers_and_players_are_caught():
    liar = {"headline": "Kingsport thrash Elmhurst 3-0", "recap": "Mensah scored for Kingsport, who had 71% of the ball.",
            "decided": "Duarte's goal changed everything: 88% of futures."}
    claims = {i["claim"] for i in R.check_recap(liar, EV, NAMES)}
    assert {"3-0", "71%", "88%", "Mensah"} <= claims


def test_offline_recap_is_approved_and_checked():
    rec = run(R.Recapper(offline_mode=True).recap(EV, NAMES))
    assert rec["checked"] and not rec.get("fallback") and rec["attempts"] == 1 and rec["reviewed"]


def test_a_lying_writer_gets_one_rewrite_then_the_code_recap():
    a = R.Recapper(offline_mode=True)
    a.agent.client.responder = lambda agent, m, o: json.dumps(
        {"headline": "Kingsport win 2-0", "recap": "A rout.", "decided": "Nothing mattered."})
    rec = run(a.recap(EV, NAMES))
    assert rec["fallback"] and rec["by"] == "code" and rec["headline"] == R.fallback_recap(EV)["headline"]
    assert len(a.agent.client.requests) == 2          # first draft + the one rewrite


def test_the_fact_checker_can_send_a_draft_back():
    a = R.Recapper(offline_mode=True)
    a.reviewer.client.responder = lambda agent, m, o: json.dumps(
        {"approved": False, "issues": [{"line": 1, "claim": "Duarte", "problem": "wrong team"}]})
    rec = run(a.recap(EV, NAMES))
    assert rec["fallback"]                             # rejected twice by the model review
    assert len(a.reviewer.client.requests) == 2
    first_rewrite = "".join(getattr(c, "text", "") or "" for msg in a.agent.client.requests[1]["messages"] for c in msg.contents)
    assert "wrong team" in first_rewrite               # the writer is told exactly what was wrong


def test_without_review_only_the_code_checks_run():
    a = R.Recapper(offline_mode=True, review=False)
    rec = run(a.recap(EV, NAMES))
    assert a.reviewer is None and not rec["reviewed"] and not rec.get("fallback")
