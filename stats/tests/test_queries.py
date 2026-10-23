import pytest

from matchstats.queries import TOOLS, MatchQueries, QueryError

SAMPLE_ARGS = {
    "player_stats": {"player": "Duarte"},
    "top_players": {"stat": "xg", "team": "home"},
    "pass_network": {"team": "Kingsport"},
    "timeline": {"series": "momentum", "since_minute": 60},
}


@pytest.fixture(scope="module")
def q(analyzer):
    return MatchQueries(analyzer)


def test_every_tool_answers(q, analyzer):
    for tool in TOOLS:
        if tool in ("explain_moment", "event", "options"):
            continue
        result = q.call(tool, SAMPLE_ARGS.get(tool, {}))
        assert result["as_of"]["finished"] is True, tool


def test_explain_moment_returns_readable_evidence(q, analyzer):
    goal = q.moments(types=["goal"])["moments"][0]
    x = q.explain_moment(goal["moment_id"])
    assert x["evidence"] and all("minute" in e and "type" in e for e in x["evidence"])
    assert x["evidence"][-1]["type"] == "goal"


def test_name_resolution(q):
    assert q.player("duarte") == q.player("Rafael Duarte") == "kin-11"
    assert q.player("Duartte") == "kin-11"  # typo-tolerant
    assert q.team("away") == q.team("KIN") == q.team("kingsport") == "kingsport-united"


def test_bad_input_is_a_clear_error(q):
    with pytest.raises(QueryError, match="unknown"):
        q.player_stats("Nobody At All")
    with pytest.raises(QueryError, match="choose from"):
        q.top_players("charisma")
    with pytest.raises(QueryError):
        q.explain_moment("nope")


def test_filters(q):
    goals = q.moments(types=["goal"], order="chronological")["moments"]
    assert all(m["type"] == "goal" for m in goals)
    late = q.moments(since_minute=80, order="chronological")["moments"]
    assert all(int(m["minute"].split("'")[0].split("+")[0]) >= 79 for m in late)
    top = q.moments(order="importance", limit=3)["moments"]
    assert [m["importance"] for m in top] == sorted((m["importance"] for m in top), reverse=True)


def test_tool_specs_are_valid_json_schema_objects():
    for name, spec in TOOLS.items():
        params = spec["parameters"]
        assert params["type"] == "object" and set(params["required"]) <= set(params["properties"]), name
        assert hasattr(MatchQueries, name)


def test_options_at_a_shot(items):
    """The decision tool needs the tracking frame at the action, so stop the feed just after a shot."""
    from matchstats import MatchAnalyzer

    a, shot = MatchAnalyzer(), None
    for kind, item in items:
        a.process(kind, item)
        if kind == "event" and item["type"] == "shot" and shot is None:
            shot = item
        if shot and kind == "frame" and item["t"] > shot["t"] + 1:
            break
    q = MatchQueries(a)
    o = q.options(shot["event_id"])
    assert o["choice"]["action"] == "shot" and 0 <= o["choice"]["xg"] <= 1
    assert 1 <= len(o["teammates"]) <= 5
    vals = [r["value"] for r in o["teammates"]]
    assert vals == sorted(vals, reverse=True)
    for r in o["teammates"]:
        assert 0 <= r["p_complete"] <= 1 and 0 <= r["xg_from_there"] <= 1 and r["space_m"] >= 0
    assert o["better_option_exists"] == (o["best_option"] is not None)
    with pytest.raises(QueryError):
        MatchQueries(a).options("nope")
