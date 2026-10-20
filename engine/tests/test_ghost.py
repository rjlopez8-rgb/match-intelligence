"""Ghost futures: exact replay to a shot, paired futures, and the Ghost Scout shot model."""

import json
import pickle
from pathlib import Path

import pytest

from matchengine.ghost import (
    MISSED,
    SCORED,
    fork_at_shot,
    league_candidates,
    load_checkpoint,
    play_futures,
    summarise,
)

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample" / "riv-kin-7"
PENALTY_GOAL = "riv-kin-7-01533"  # Duarte, 74', scored


@pytest.fixture(scope="module")
def recorded():
    meta = json.loads((SAMPLE / "meta.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (SAMPLE / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    return meta, events


@pytest.fixture(scope="module")
def fork():
    """The frozen state shipped with the match (recorded once, valid on every machine)."""
    f = load_checkpoint(SAMPLE, PENALTY_GOAL)
    assert f is not None, "run: python -m matchengine.ghost data/sample/riv-kin-7"
    return f


def test_checkpoints_cover_every_shot(recorded):
    import gzip

    _, events = recorded
    with gzip.open(SAMPLE / "ghost_checkpoints.pkl.gz", "rb") as fh:
        forks = pickle.load(fh)["forks"]
    assert set(forks) == {e["event_id"] for e in events if e["type"] == "shot"}


def test_checkpoint_matches_the_recorded_match(fork, recorded):
    _, events = recorded
    eng = pickle.loads(fork.state)
    replayed = [(e["type"], e["t"], e["player_id"]) for e in eng.events]
    assert replayed == [(e["type"], e["t"], e["player_id"]) for e in events[: len(replayed)]]
    assert fork.actual == "goal" and fork.shooter_id == "kin-11" and fork.kind == "penalty"


def test_shot_model_is_re_evaluated_per_player(fork, recorded):
    _, events = recorded
    shot = next(e for e in events if e["event_id"] == PENALTY_GOAL)
    # the taker's own value matches what the engine recorded for the real shot
    assert abs(fork.xg_by_player["kin-11"] - shot["details"]["truth"]["xg"]) < 1e-3
    assert len(set(fork.xg_by_player.values())) > 2  # different ratings, different chances


def test_live_replay_matches_where_the_platform_allows(recorded):
    """Exact replay needs the same floating-point behaviour as the recording machine."""
    meta, events = recorded
    shot = next(e for e in events if e["event_id"] == PENALTY_GOAL)
    try:
        f = fork_at_shot(meta, shot, league_candidates(("ST",))[:3])
    except ValueError:
        pytest.skip("this machine's floating point differs from the recording machine (checkpoints cover it)")
    assert f.actual == "goal"


def test_futures_are_reproducible_and_the_change_applies(fork):
    a = play_futures(fork, MISSED, n=1, seed=42, parallel=False)[0]
    b = play_futures(fork, MISSED, n=1, seed=42, parallel=False)[0]
    assert a["final"] == b["final"]
    assert not a["fork_shot_scored"]
    kept = play_futures(fork, None, n=1, seed=42, parallel=False, record=1, record_s=3)[0]
    assert kept["fork_shot_scored"]
    assert kept["frames"] and kept["frames"][0]["players"]
    assert SCORED == "goal"


def test_summary_adds_up():
    futures = [
        {"final": {"home": h, "away": a}, "goals_after": []} for h, a in ((1, 0), (0, 0), (0, 1), (0, 2))
    ]
    s = summarise(futures)
    assert s["home_win"] + s["draw"] + s["away_win"] == pytest.approx(1.0)
    assert s["away_win"] == 0.5
