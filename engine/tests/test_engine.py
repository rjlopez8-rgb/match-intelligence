"""Behavioural tests: determinism, realism bounds, scenarios and streaming."""

import itertools

import numpy as np

from matchengine import Incident, MatchConfig, MatchEngine, Scenario, SyntheticFeed, get_club, load_league
from matchengine.params import DEFAULT_PARAMS
from matchengine.probability import ground_pass_speed, xg_unblocked

from .conftest import run_until


def _cfg(seed=5, home="RIV", away="KIN"):
    return MatchConfig(home=get_club(home), away=get_club(away), seed=seed)


def test_same_seed_same_match():
    a = [i for k, i in run_until(MatchEngine(_cfg(), emit_frames=False), 300) if k == "event"]
    b = [i for k, i in run_until(MatchEngine(_cfg(), emit_frames=False), 300) if k == "event"]
    assert a == b


def test_different_seed_different_match():
    a = [i["type"] for k, i in run_until(MatchEngine(_cfg(1), emit_frames=False), 300) if k == "event"]
    b = [i["type"] for k, i in run_until(MatchEngine(_cfg(2), emit_frames=False), 300) if k == "event"]
    assert a != b


def test_league_is_fictional_and_deterministic():
    league = load_league()
    assert len(league) == 8
    ids = [p.player_id for c in league.values() for p in c.squad]
    names = [p.name for c in league.values() for p in c.squad]
    assert len(ids) == len(set(ids))
    assert len(names) == len(set(names))
    assert league["RIV"].player("riv-10").name == "Kwame Mensah"
    assert load_league() is league  # cached


def test_match_summary_is_in_realistic_ranges(full_match):
    """Wide bounds for a single match; the calibration suite checks averages over many matches."""
    _, result = full_match
    s = result.summary
    for side in ("home", "away"):
        t = s["teams"][side]
        assert 150 <= t["passes"] <= 900
        assert 0.6 <= t["pass_completion"] <= 0.95
        assert 2 <= t["shots"] <= 35
        assert t["goals"] <= 8
        assert 85 <= t["distance_km"] <= 135
    assert 40 <= s["ball_in_play_min"] <= 75
    full = [p for p in s["players"].values() if p["minutes"] > 85 and p["position"] != "GK"]
    assert 7.5 <= np.median([p["distance_km"] for p in full]) <= 13.5
    assert all(p["top_speed_kmh"] < 38 for p in full)


def test_scenario_red_card_is_applied():
    sc = Scenario(name="t", incidents=(Incident(minute=4, kind="red_card", team="away", player_id="kin-03"),))
    eng = MatchEngine(_cfg(), scenario=sc, emit_frames=False)
    events = [i for k, i in run_until(eng, 1500) if k == "event"]
    cards = [e for e in events if e["type"] == "card"]
    assert any(e["player_id"] == "kin-03" and e["details"]["card"] == "red" for e in cards)
    assert not eng.active[eng.teams[1].players[2].idx]  # Bell is off the pitch
    assert len(eng.active_indices(1)) == 10


def test_scenario_tactical_change_is_logged():
    sc = Scenario(
        name="t",
        incidents=(
            Incident(
                minute=2,
                kind="tactics",
                team="home",
                changes={"pressing": 0.95},
                reason="test",
                description="press higher",
            ),
        ),
    )
    events = [
        i for k, i in run_until(MatchEngine(_cfg(), scenario=sc, emit_frames=False), 200) if k == "event"
    ]
    tc = [e for e in events if e["type"] == "tactical_change"]
    assert tc and tc[0]["details"]["after"]["pressing"] == 0.95


def test_stream_yields_meta_first_then_events():
    feed = SyntheticFeed(_cfg(), speed=0, frame_hz=2)
    items = list(itertools.islice(iter(feed), 60))
    assert items[0][0] == "meta"
    kinds = {k for k, _ in items[1:]}
    assert kinds <= {"event", "frame"} and "event" in kinds and "frame" in kinds


def test_xg_model_behaves_sensibly():
    p = DEFAULT_PARAMS
    kw = dict(header=False, pressure=0.0, finishing=72, composure=70, keeper=75, big_chance=False)
    close = xg_unblocked(p, 99.0, 34.0, **kw)
    edge = xg_unblocked(p, 87.0, 34.0, **kw)
    far = xg_unblocked(p, 75.0, 34.0, **kw)
    wide = xg_unblocked(p, 99.0, 12.0, **kw)
    assert close > edge > far
    assert close > wide
    assert 0.7 < xg_unblocked(p, 94.0, 34.0, **{**kw, "big_chance": False}, kind="penalty") < 0.85
    assert xg_unblocked(p, 93.0, 34.0, **{**kw, "header": True}) < xg_unblocked(p, 93.0, 34.0, **kw)


def test_pass_speeds_are_realistic():
    p = DEFAULT_PARAMS
    assert 30 < ground_pass_speed(p, 10.0) * 3.6 < 55
    assert 45 < ground_pass_speed(p, 35.0) * 3.6 < 75
