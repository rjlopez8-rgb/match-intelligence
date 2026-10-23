"""The whole desk on the sample match, offline (rule-based agents, real tools, $0)."""

import pytest
from conftest import GOAL, MISSED_PENALTY, SAMPLE, run

from matchagents.desk import Desk, workflow_mermaid
from matchagents.session import MatchSession


@pytest.fixture(scope="module")
def segments():
    session = MatchSession(SAMPLE)
    desk = Desk(session, offline_mode=True)

    async def cover_all():
        return [await desk.cover(m) for m in session.moments(0.45)]

    return {s["moment_id"]: s for s in run(cover_all())}


def test_every_big_moment_gets_a_verified_segment(segments):
    assert GOAL in segments and MISSED_PENALTY in segments
    for s in segments.values():
        assert s["errors"] == []
        assert s["aired"] and s["commentary"]["lines"]
        assert s["factcheck"] and (s["factcheck"][-1]["approved"] or s["commentary"]["fallback"])
        assert s["cost"]["usd"] == 0  # offline runs are free


def test_goal_segment_is_complete(segments):
    g = segments[GOAL]
    assert g["plan"]["priority"] == "breaking"
    assert g["director"]["replay"] and g["director"]["shots"]
    shapes = g["telestration"]["shapes"]
    assert shapes and all(0 <= s["x"] <= 105 and 0 <= s["y"] <= 68 for s in shapes)
    if g["scenario"]["spec"]:  # one what-if per half: an earlier moment may already have used it
        assert g["scenario"]["spec"]["intervention"] == "shot_misses"
    assert any(s["scenario"]["spec"] for s in segments.values() if s["period"] == g["period"])
    tools = [r["tool"] for r in g["receipts"] if r["agent"] == "analyst"]
    assert "explain_moment" in tools  # the analyst really called the stats tools


def test_only_one_what_if_per_half(segments):
    per_half = {}
    for s in segments.values():
        if s["plan"] and s["plan"]["segment"] == "what_if":
            per_half[s["period"]] = per_half.get(s["period"], 0) + 1
    assert per_half and all(n == 1 for n in per_half.values())


def test_agents_cannot_see_the_future():
    session = MatchSession(SAMPLE)
    session.advance_to(MISSED_PENALTY)  # 8th minute; the match ends 0-1
    board = session.queries.scoreboard()
    assert board["as_of"]["score"] == {"home": 0, "away": 0} and board["goals"] == []


def test_workflow_diagram_names_all_seven_agents():
    session = MatchSession(SAMPLE)
    text = workflow_mermaid(Desk(session, offline_mode=True))
    for agent in ("producer", "analyst", "scenario", "telestrator", "director", "commentator", "factchecker"):
        assert agent in text


def test_editions_tell_one_moment_every_way_and_resume_for_free():
    from matchagents.desk import Edition

    session = MatchSession(SAMPLE)
    moment = session.advance_to(GOAL)
    desk = Desk(session, offline_mode=True)
    eds = [Edition("fan", "neutral", "en"), Edition("fan", "away", "es"), Edition("academy", "neutral", "fr"),
           Edition("analyst", "neutral", "en")]
    seg = run(desk.cover_editions(moment, eds))
    assert set(seg["editions"]) == {e.key for e in eds} and seg["aired"]
    for v in seg["editions"].values():
        assert v["lines"] and not v["fallback"] and v["checks"][-1]["approved"] and v["headline"]
    texts = {k: " ".join(v["lines"]) for k, v in seg["editions"].items()}
    assert texts["fan.away.es"] != texts["fan.neutral.en"] and "¡" in texts["fan.away.es"]
    assert "xG" in texts["analyst.neutral.en"]
    d = seg["directors"]
    assert set(d) == {"fan", "academy", "analyst"}
    assert [s["camera"] for s in d["fan"]["shots"]] != [s["camera"] for s in d["analyst"]["shots"]]
    assert seg["decision"] and seg["decision"]["choice"]["action"] == "shot"
    # resuming with everything done makes no model calls at all
    session2 = MatchSession(SAMPLE)
    m2 = session2.advance_to(GOAL)
    desk2 = Desk(session2, offline_mode=True)
    again = run(desk2.cover_editions(m2, eds, seg))
    assert again["cost"]["model_calls"] == seg["cost"]["model_calls"] and len(desk2.trace.calls) == 0
    assert again["editions"] == seg["editions"]


def test_editions_workflow_has_a_desk_per_style():
    from matchagents.desk import Edition

    eds = [Edition("fan"), Edition("academy", lang="fr")]
    text = workflow_mermaid(Desk(MatchSession(SAMPLE), offline_mode=True), eds)
    assert "fan_desk" in text and "academy_desk" in text
