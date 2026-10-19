import json

from matchstats import MatchAnalyzer
from matchstats.schema import Moment, Snapshot

from .conftest import strip_truth


def test_snapshot_and_moments_match_schema(analyzer):
    snap = analyzer.snapshot()
    Snapshot.model_validate(snap)
    for m in analyzer.moments:
        Moment.model_validate(m)
    json.dumps(snap)  # fully JSON-serialisable


def test_models_never_see_ground_truth(items, analyzer):
    """Stripping every simulator truth field must not change a single number."""
    blind = MatchAnalyzer().run(strip_truth(items))
    assert blind.snapshot() == analyzer.snapshot()
    assert blind.moments == analyzer.moments


def test_final_score_and_goal_moments(analyzer):
    ctx = analyzer.ctx
    goals = [m for m in analyzer.moments if m["type"] == "goal"]
    assert len(goals) == ctx.score["home"] + ctx.score["away"]
    assert ctx.finished
    assert [m["title"].split(":")[0] for m in analyzer.moments if m["type"] == "period_summary"] == [
        "Half-time",
        "Full-time",
    ]


def test_evidence_exists_and_precedes_moment(analyzer):
    ctx = analyzer.ctx
    for m in analyzer.moments:
        for eid in m["evidence"]:
            e = ctx.by_id[eid]
            assert (e["period"], e["t"]) <= (m["period"], m["t"] + 1e-6), (m["moment_id"], eid)


def test_penalty_goal_receipts_start_at_the_foul(analyzer):
    for m in analyzer.moments:
        if m["type"] == "goal" and m["facts"].get("shot_kind") == "penalty":
            assert analyzer.ctx.by_id[m["evidence"][0]]["type"] == "foul"


def test_win_probabilities_are_distributions(analyzer):
    for p in analyzer.metric("win_probability").snapshot()["series"]:
        assert abs(p["home"] + p["draw"] + p["away"] - 1) < 1e-3
    for m in analyzer.moments:
        wp = m["win_probability"]
        assert abs(wp["home"] + wp["draw"] + wp["away"] - 1) < 1e-3
        assert -1 <= m["momentum"] <= 1


def test_streaming_matches_batch(items, analyzer):
    live = MatchAnalyzer()
    streamed = list(live.stream(items))
    assert streamed == analyzer.moments


def test_partial_snapshot_mid_match(items):
    a = MatchAnalyzer()
    for kind, item in items[: len(items) // 2]:
        a.process(kind, item)
    snap = a.snapshot()
    Snapshot.model_validate(snap)
    assert not snap["match"]["finished"]
    assert snap["clock"]["period"] in (1, 2)


def test_events_only_mode(items):
    a = MatchAnalyzer(frames=False).run((k, i) for k, i in items if k != "frame")
    snap = a.snapshot()
    assert snap["metrics"]["physical"] == {"available": False}
    shares = snap["metrics"]["possession"]["teams"]
    assert abs(sum(t["possession"] for t in shares.values()) - 1) < 0.01


def test_stat_sanity(analyzer):
    snap = analyzer.snapshot()["metrics"]
    for t in snap["passing"]["teams"].values():
        assert 0.5 < t["accuracy"] < 1.0
        assert 0 <= t["avg_difficulty"] <= 10
    for t in snap["physical"]["teams"].values():
        assert 80 < t["distance_km"] < 140  # a squad covers ~100-120 km
    assert snap["physical"]["records"]["fastest_player_kmh"] < 38
    assert 0 <= snap["control"]["chaos_index"] <= 100
