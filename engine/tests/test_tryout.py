"""Player try-outs: the pieces that don't need full simulations (those run in ``matchengine board``)."""

import json
from pathlib import Path

import pytest

from matchengine import tryout as T
from matchengine.league import get_club

DEMO = Path(__file__).resolve().parents[2] / "data" / "demo" / "riv-kin-1119"


def test_expected_points_is_a_proper_distribution():
    w, d, l, xp = T.expected_points([0.5, 0.3], [0.1])
    assert abs(w + d + l - 1) < 1e-9
    assert xp == pytest.approx(3 * w + d)
    assert T.expected_points([], []) == pytest.approx((0.0, 1.0, 0.0, 1.0))
    # more and better chances, more expected points
    assert T.expected_points([0.4, 0.4], [0.1])[3] > T.expected_points([0.1], [0.4, 0.4])[3]


def test_lineup_and_candidates():
    lu = T.lineup("RIV", "KIN")
    assert [p["label"] for p in lu["home"]][:2] == ["GK", "RB"] and len(lu["away"]) == 11
    st = next(p for p in lu["home"] if p["label"] == "ST")
    cands = T.candidates("RIV", "KIN", "home", st["slot"])
    ids = {c["player_id"] for c in cands}
    kin = {p.player_id for p in get_club("KIN").squad}
    assert not ids & kin                      # never the opponent's players
    assert st["player_id"] not in ids         # nor the man already there
    assert cands[0]["fit"] == 0               # natural strikers first


def test_swap_puts_the_candidate_in_the_slot():
    riv = get_club("RIV")
    player, _ = T.find_player("wes-20")
    club = T._club_with(riv, 9, player)
    assert club.starting_xi[9] == "wes-20" and club.player("wes-20").name == player.name
    assert riv.starting_xi[9] != "wes-20"     # the real club is untouched


def test_player_stats_from_a_real_match():
    events = [json.loads(line) for line in (DEMO / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    s = T.player_stats(events, "riv-10", "riverside-athletic")   # Mensah
    assert s["goals"] == 1 and s["shots"] >= 1 and s["xg"] > 0
    gk = T.player_stats(events, "kin-01", "kingsport-united")    # Fairclough
    assert gk["saves"] >= 1 and gk["xg_faced"] > 0


def test_summaries_and_paired_difference():
    def run(seed, xpts, xgf, result, goals):
        return {"seed": seed, "result": result, "xpts": xpts, "xg_for": xgf, "xg_against": 1.0, "xwin": 0.4, "xdraw": 0.3,
                "player": {"minutes": 90.0, "goals": goals, "xg": 0.4}}

    base = [run(1, 1.0, 1.0, "D", 0), run(2, 1.2, 1.1, "W", 1)]
    trial = [run(1, 1.5, 1.4, "W", 1), run(2, 1.6, 1.5, "W", 1)]
    s = T.summarise_runs(trial)
    assert s["futures"] == 2 and s["win"] == 1.0 and s["per90"]["goals"] == 1.0
    assert s["xwin"] == 0.4 and s["xloss"] == 0.3
    d = T.compare(base, trial)
    assert d["pairs"] == 2 and d["xpts"] == pytest.approx(0.45)


def test_board_marks_ready_players():
    b = T.board("RIV", "KIN", "home", ready=2)
    assert len(b) == 11
    for pos in b:
        ready = [c for c in pos["candidates"] if c["ready"]]
        assert len(ready) == 2 and ready[0]["tag"] == "Top projected"
        assert pos["candidates"] == sorted(pos["candidates"], key=lambda c: -c["projected"])
