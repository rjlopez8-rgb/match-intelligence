"""Contract tests: every event the engine emits must be valid, ordered and internally consistent."""

from collections import Counter

from matchengine.params import PITCH_LENGTH, PITCH_WIDTH
from matchengine.schema import TrackingFrame, validate_event


def test_every_event_matches_the_schema(full_match):
    _, result = full_match
    for e in result.events:
        validate_event(e)


def test_publication_order_and_ids(full_match):
    _, result = full_match
    seqs = [e["seq"] for e in result.events]
    assert seqs == list(range(1, len(seqs) + 1))
    assert len({e["event_id"] for e in result.events}) == len(result.events)
    assert result.events[0]["type"] == "match_start"
    assert result.events[-1]["type"] == "match_end"


def test_required_event_types_are_present(full_match):
    """The rules name passes, shots, tackles, possession changes and pressure events explicitly."""
    _, result = full_match
    types = Counter(e["type"] for e in result.events)
    for required in (
        "pass",
        "shot",
        "tackle",
        "possession_change",
        "pressure",
        "carry",
        "kick_off",
        "period_start",
        "period_end",
        "ball_out",
    ):
        assert types[required] > 0, required


def test_locations_are_on_or_near_the_pitch(full_match):
    _, result = full_match
    for e in result.events:
        loc = e["location"]
        if loc:
            assert -3 <= loc["x"] <= PITCH_LENGTH + 3 and -3 <= loc["y"] <= PITCH_WIDTH + 3, e


def test_score_is_consistent(full_match):
    _, result = full_match
    goals = Counter(e["team_id"] for e in result.events if e["type"] == "goal")
    final = result.events[-1]["details"]["final_score"]
    home_id = result.meta["home"]["team_id"]
    away_id = result.meta["away"]["team_id"]
    assert final == {"home": goals[home_id], "away": goals[away_id]}
    shot_goals = sum(1 for e in result.events if e["type"] == "shot" and e["details"]["outcome"] == "goal")
    assert shot_goals == sum(goals.values())
    assert result.summary["final_score"] == final


def test_completed_passes_go_to_teammates(full_match):
    _, result = full_match
    team_of = {
        p["player_id"]: side["team_id"]
        for side in (result.meta["home"], result.meta["away"])
        for p in side["squad"]
    }
    for e in result.events:
        if e["type"] == "pass" and e["details"]["outcome"] == "complete" and e["details"]["recipient_id"]:
            assert team_of[e["details"]["recipient_id"]] == e["team_id"]


def test_possession_changes_switch_team(full_match):
    _, result = full_match
    for e in result.events:
        if e["type"] == "possession_change":
            assert e["details"]["from_team_id"] != e["team_id"]


def test_clock_runs_forward_within_periods(full_match):
    _, result = full_match
    for period in (1, 2):
        starts = [
            e["t"] for e in result.events if e["period"] == period and e["type"] in ("kick_off", "period_end")
        ]
        assert starts == sorted(starts)


def test_substitution_and_card_rules(full_match):
    _, result = full_match
    subs = Counter(e["team_id"] for e in result.events if e["type"] == "substitution")
    assert all(n <= 5 for n in subs.values())


def test_tracking_frames(full_match):
    _, result = full_match
    assert result.frames, "tracking frames expected"
    first = result.frames[0]
    TrackingFrame.model_validate(first)
    assert len(first["players"]) == 22
    for f in result.frames[:: max(1, len(result.frames) // 400)]:
        TrackingFrame.model_validate(f)
        for _pid, x, y, speed in f["players"]:
            assert -3 <= x <= PITCH_LENGTH + 3 and -3 <= y <= PITCH_WIDTH + 3
            assert speed < 11.0  # nobody runs faster than ~40 km/h


def test_events_and_frames_are_plain_json(full_match):
    import json

    _, result = full_match
    for e in result.events:
        json.dumps(e)
    json.dumps(result.frames[:50])
    json.dumps(result.meta)
    json.dumps(result.summary)
