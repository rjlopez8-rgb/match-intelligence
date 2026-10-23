import asyncio
import json

import pytest

pytest.importorskip("mcp", minversion="2.0")

from matchstats import mcp_server  # noqa: E402


def _call(srv, name, args):
    r = asyncio.run(srv.call_tool(name, args))
    return json.loads(r.content[0].text)


@pytest.fixture(scope="module")
def srv():
    return mcp_server.build_server()


def test_every_tool_is_listed_with_a_description(srv):
    tools = asyncio.run(srv.list_tools())
    names = {t.name for t in tools}
    assert {"list_matches", "league_table", "what_decided_it", "scoreboard", "moments", "options", "tryout_board",
            "tryout_result", "run_tryout"} <= names
    assert all(t.description for t in tools)
    stat = next(t for t in tools if t.name == "top_players").input_schema["properties"]["stat"]
    assert "goals" in stat["enum"]


def test_match_tools_answer_as_of_a_minute(srv):
    at60 = _call(srv, "scoreboard", {"match_id": "riv-kin-1119", "minute": 60})
    full = _call(srv, "scoreboard", {"match_id": "riv-kin-1119"})
    assert at60["as_of"]["finished"] is False and full["as_of"]["finished"] is True
    assert full["home"]["goals"] + full["away"]["goals"] == 3       # Riverside 1-2 Kingsport


def test_options_replays_to_the_decision(srv):
    r = _call(srv, "options", {"match_id": "riv-kin-1119", "event_id": "riv-kin-1119-00022"})
    assert r["choice"]["action"] == "shot" and r["teammates"]


def test_bad_ids_are_refused_not_read(srv):
    assert "error" in _call(srv, "scoreboard", {"match_id": "../../etc"})
    assert "error" in _call(srv, "tryout_result", {"side": "home", "position": "XX", "player": "nobody"})


def test_club_mode_reads_the_board(srv):
    b = _call(srv, "tryout_board", {"side": "home", "position": "ST"})
    assert b["positions"][0]["starter"] and b["positions"][0]["candidates"]
    ready = next(c for c in b["positions"][0]["candidates"] if c["ready"])
    r = _call(srv, "tryout_result", {"side": "home", "position": "ST", "player": ready["player_id"]})
    assert r["with_candidate"]["futures"] >= 2 and "xpts_margin_95" in r["difference"]
