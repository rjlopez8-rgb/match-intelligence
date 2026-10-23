import base64
import json
from pathlib import Path

from matchstats.replay import build_replay, render_html

from .conftest import SAMPLE


def test_replay_page_is_self_contained():
    data = build_replay(SAMPLE)
    raw = base64.b64decode(data["frames"])
    assert len(raw) == data["nFrames"] * data["width"] * 2
    assert len(data["samples"]) == -(-data["nFrames"] // data["sampleEvery"])
    assert data["samples"][-1][-2:] == [data["final"]["score"]["home"], data["final"]["score"]["away"]]
    page = render_html(data)
    assert page.startswith("<!doctype html>") and "__REPLAY_DATA__" not in page and "__TITLE__" not in page
    assert "<script src=" not in page  # no external scripts: works offline


def test_desk_segments_are_packed_for_the_3d_view():
    """A segment from the AI desk lands on the right frame, with drawings and ghost tracking decoded."""
    seg = {
        "moment_id": "riv-kin-7-m044",
        "aired": True,
        "period": 2,
        "t": 1681.5,
        "type": "goal",
        "minute": "74'",
        "title": "GOAL",
        "plan": {"segment": "what_if", "angle": "a", "priority": "breaking", "what_if": "q"},
        "commentary": {"lines": ["one", "two"], "fallback": False},
        "brief": {
            "facts": [{"statement": "s", "source_tool": "explain_moment", "evidence": ["x"]}],
            "insight": "i",
        },
        "factcheck": [{"approved": True, "code_issues": []}],
        "director": {
            "live_camera": "sweep",
            "replay": True,
            "replay_from_event_id": "riv-kin-7-01532",
            "graphic": "xg",
            "shots": [
                {
                    "camera": "behind",
                    "seconds": 3,
                    "slow_motion": True,
                    "focus_player": "Duarte",
                    "purpose": "p",
                }
            ],
        },
        "telestration": {
            "freeze": {"period": 2, "t": 1681.1},
            "caption": "c",
            "shapes": [
                {
                    "kind": "ring",
                    "color": "highlight",
                    "label": "Duarte",
                    "player_id": "kin-11",
                    "x": 93.3,
                    "y": 33.9,
                    "r": 1.6,
                }
            ],
        },
        "scenario": {
            "spec": {"question": "q", "intervention": "shot_misses"},
            "result": {
                "status": "ok",
                "event_id": "riv-kin-7-01533",
                "team_side": "away",
                "actual_outcome": "goal",
                "changed_outcome": "saved",
                "with_change": {"futures": 2},
                "as_it_happened": {"futures": 2},
                "curves": {"with_change": [[[2, 1681.5, 0.4, 0.3], [2, 2400.0, 0.7, 0.1]]]},
                "featured": [
                    {
                        "final": {"home": 1, "away": 0},
                        "goals_after": [{"minute": "86'", "side": "home", "scorer": "Brekke"}],
                        "clips": [
                            {
                                "kind": "fork",
                                "score0": {"home": 0, "away": 0},
                                "frames": [
                                    {
                                        "period": 2,
                                        "t": 1681.3,
                                        "ball": {"x": 90, "y": 34, "z": 0.5, "owner_id": None},
                                        "possession_team_id": None,
                                        "players": [["kin-11", 93.0, 34.0, 1.0]],
                                    },
                                    {
                                        "period": 2,
                                        "t": 1681.5,
                                        "ball": {"x": 95, "y": 34, "z": 0.8, "owner_id": None},
                                        "possession_team_id": None,
                                        "players": [["kin-11", 93.1, 34.0, 1.0]],
                                    },
                                ],
                                "events": [
                                    {
                                        "type": "shot",
                                        "period": 2,
                                        "t": 1681.4,
                                        "player_id": "kin-11",
                                        "team_id": "kingsport-united",
                                        "location": {"x": 93.5, "y": 34},
                                        "details": {
                                            "outcome": "saved",
                                            "speed_kmh": 100,
                                            "end_location": {"x": 105, "y": 33, "z": 0.6},
                                            "goalkeeper_id": "riv-01",
                                        },
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        },
        "receipts": [{"agent": "analyst", "tool": "explain_moment"}],
        "cost": {"usd": 0.01, "model_calls": 8, "seconds": 30, "by_agent": {}},
    }
    data = build_replay(SAMPLE, [seg, {**seg, "moment_id": "x", "aired": False}])
    assert len(data["segments"]) == 1
    s = data["segments"][0]
    goal = next(m for m in data["moments"] if m["id"] == "riv-kin-7-m044")
    assert s["f"] == goal["f"]
    assert s["director"]["shots"][0]["focus"] >= 0 and s["director"]["fromF"] < s["f"]
    assert s["tele"]["shapes"][0]["slot"] >= 0
    gh = s["ghost"]
    assert gh["fan"] == [[gh["fan"][0][0], 10, gh["fan"][0][2], 60]]
    clip = gh["featured"][0]["clips"][0]
    raw = base64.b64decode(clip["frames"])
    assert len(raw) == clip["n"] * data["width"] * 2  # same layout as the real match
    assert clip["p2"] == 0 and clip["shots"][0][6] == 1  # second half; the shot is a save


def test_recruitment_board_is_packed_for_the_3d_view(tmp_path):
    import shutil

    from matchstats.replay import build_replay

    src = Path(__file__).resolve().parents[2] / "data" / "sample" / "riv-kin-7"
    for f in src.iterdir():
        if f.is_file():
            shutil.copy(f, tmp_path / f.name)
    side = lambda xpts: {"futures": 2, "win": 0.5, "draw": 0.5, "loss": 0.0, "result_margin_95": 0.7, "xpts": xpts,
                         "xpts_margin_95": 0.3, "xg_for": 1.2, "xg_for_margin_95": 0.2, "xg_against": 0.9,
                         "xg_against_margin_95": 0.2, "minutes": 95.0, "per90": {"goals": 0.5, "xg": 0.6}}
    tryout = {"candidate": {"player_id": "wes-20", "name": "Ante Babić", "club": "Westmere Town", "role": "ST", "age": 28},
              "incumbent": {"player_id": "riv-10", "name": "Kwame Mensah", "club": "Riverside Athletic", "role": "ST", "age": 27},
              "position": {"slot": 9, "label": "ST", "role": "ST"}, "with_candidate": side(1.6), "with_incumbent": side(1.4),
              "difference": {"pairs": 2, "xpts": 0.2, "xpts_margin_95": 0.3, "clear": False}, "featured": None,
              "scout": {"headline": "Close call", "report": "Too close to call.", "verdict": "shortlist"}}
    pos = {"slot": 9, "label": "ST", "role": "ST", "player_id": "riv-10", "name": "Kwame Mensah", "incumbent_projected": 79.6,
           "candidates": [{"player_id": "wes-20", "name": "Ante Babić", "club": "Westmere Town", "club_code": "WES",
                           "role": "ST", "age": 28, "projected": 74.0, "ready": True, "tag": "Top projected"}]}
    (tmp_path / "tryouts").mkdir()
    (tmp_path / "tryouts" / "board.json").write_text(json.dumps(
        {"home": "RIV", "away": "KIN", "n": 2, "sides": {"home": {"board": [pos], "tryouts": [tryout]}}}), encoding="utf-8")
    club = build_replay(tmp_path)["club"]
    t = club["sides"]["home"]["tryouts"]["9:wes-20"]
    assert t["diff"]["xpts"] == 0.2 and t["report"]["verdict"] == "shortlist"
    assert club["sides"]["home"]["positions"][0]["candidates"][0]["ready"]


def test_personal_edition_data_is_packed():
    """Editions (style x channel x language), per-style Director cuts, the decision and the player-focus inputs."""
    seg = {
        "moment_id": "riv-kin-7-m044", "aired": True, "period": 2, "t": 1681.5, "type": "goal", "minute": "74'",
        "title": "GOAL", "plan": {"segment": "replay_breakdown", "angle": "a", "priority": "breaking", "what_if": None},
        "commentary": {"lines": ["one"], "fallback": False}, "brief": {"facts": [], "insight": "i"},
        "factcheck": [{"approved": True, "code_issues": []}], "director": None, "telestration": None,
        "scenario": {"spec": None, "result": None}, "receipts": [],
        "cost": {"usd": 0.0, "model_calls": 0, "seconds": 0.0, "by_agent": {}},
        "editions": {"fan.home.es": {"lines": ["¡Gol!"], "headline": "¡GOL!", "caption": "La definición", "question": None,
                                     "fallback": False, "checks": [{"approved": True, "code_issues": []}]}},
        "directors": {"academy": {"live_camera": "behind", "replay": True, "replay_from_event_id": None, "graphic": "player_card",
                                  "show_telestration": True, "shots": [{"camera": "player", "seconds": 4, "slow_motion": True,
                                                                        "focus_player": "Duarte", "purpose": "the decision"}]}},
        "decision": {"event": {"player": "Duarte"}, "better_option_exists": False,
                     "choice": {"action": "shot", "xg": 0.74, "value": 0.74, "space_m": 2.0, "defenders_in_lane": 0,
                                "location": {"x": 94.0, "y": 34.0}},
                     "teammates": [{"name": "Sarr", "player_id": "kin-07", "location": {"x": 90.0, "y": 20.0}, "p_complete": 0.7,
                                    "xg_from_there": 0.05, "space_m": 3.0, "value": 0.035, "opponents_near_pass_line": 1}]},
    }
    data = build_replay(SAMPLE, [seg])
    s = data["segments"][0]
    ed = dict(s["ed"]["fan.home.es"])
    rx = ed.pop("rx")
    assert ed == {"l": ["¡Gol!"], "h": "¡GOL!", "c": "La definición", "q": None, "fb": False, "att": 1, "ok": True, "code": 0}
    assert rx["nm"] == [[]] and rx["chk"][0]["ok"]           # receipts: no numbers in "¡Gol!", approved first time
    assert s["dirs"]["academy"]["shots"][0]["cam"] == "player" and s["dirs"]["academy"]["shots"][0]["focus"] >= 0
    assert s["decision"]["best"]["name"] == "Sarr" and s["decision"]["from"] == [94.0, 34.0] and not s["decision"]["better"]
    assert len(data["shotXg"]) == len(data["shots"]) and all(x is None or 0 <= x <= 1 for x in data["shotXg"])
    goal = next(m for m in data["moments"] if m["type"] == "goal")
    assert goal["who"] and goal["score"] and "xg" in goal["x"]
    acts = data["actions"]
    assert acts and all(0 <= a[1] < len(data["players"]) for a in acts)
    assert any(a[2] == 5 for a in acts)          # the goal is in his action list
    assert [a[0] for a in acts] == sorted(a[0] for a in acts)


def test_highlights_reel_keeps_only_the_clips_and_the_real_clock():
    """Goals, saves and chances back to back: frames cut from the full match, every frame index remapped,
    the match clock untouched, and nothing (moments, shots, touches) pointing outside the reel."""
    from matchstats.replay import highlight_clips, highlights

    full = build_replay(SAMPLE)
    reel = highlights(full)
    wins = highlight_clips(full)
    n, w = reel["nFrames"], reel["width"]
    assert n == sum(c["b"] - c["a"] + 1 for c in wins) < full["nFrames"] / 10
    assert len(base64.b64decode(reel["frames"])) == n * w * 2
    clips = reel["clips"]
    assert clips[0][0] == 0 and clips[-1][1] == n - 1
    assert all(a[1] + 1 == b[0] for a, b in zip(clips, clips[1:]))       # contiguous, one after another
    assert {c[4] for c in clips} <= {"goal", "great_save", "woodwork", "big_chance", "penalty_awarded", "save", "chance"}
    assert len(clips) <= 10
    goals = [m for m in full["moments"] if m["type"] == "goal"]
    assert sum(1 for m in reel["moments"] if m["type"] == "goal") == len(goals)   # every goal is in
    for row in reel["touches"] + reel["actions"] + reel["shots"]:
        assert 0 <= row[0] <= n - 1
    assert len(reel["shots"]) == len(reel["shotXg"])
    assert len(reel["samples"]) == -(-n // reel["sampleEvery"])
    # the clock: a reel frame shows the same match time as the full-match frame it was cut from
    import numpy as np

    f_full = np.frombuffer(base64.b64decode(full["frames"]), "<i2").reshape(-1, w)
    f_reel = np.frombuffer(base64.b64decode(reel["frames"]), "<i2").reshape(-1, w)
    for c, (s0, s1, period, *_rest) in zip(wins, clips):
        assert (f_reel[s0:s1 + 1] == f_full[c["a"]:c["b"] + 1]).all()
        assert period == (1 if c["a"] < full["period2"] else 2)
    assert reel["segments"] == [] and reel["club"] is None and reel["highlights"] is True


def test_viewer_shell_loads_its_match_from_the_url():
    from matchstats.replay import render_viewer_shell

    page = render_viewer_shell()
    assert page.startswith("<!doctype html>") and '"__REPLAY_DATA__"' in page and "__TITLE__" not in page
    assert "matches/${id}.${v}.json" in page


def test_receipts_trace_every_fact_and_number_on_air():
    """Every fact names the exact tool result field it came from, and every number in every edition's lines matches
    a source (a fact, the simulation or the score): the page can show a receipt for anything said on air."""
    demo = Path(__file__).resolve().parents[2] / "data" / "demo" / "riv-kin-1119"
    segs = json.loads((demo / "editions-offline.json").read_text(encoding="utf-8"))
    data = build_replay(demo, segs)
    seen = 0
    for sg, src in zip(data["segments"], [s for s in segs if s.get("aired")]):
        results = src.get("tool_results") or []
        for f in sg["rx"]["facts"]:
            if f["call"] is None:
                continue
            node = results[f["call"]]["result"]
            for part in f["at"].replace("]", "").replace("[", ".").split("."):
                node = node[int(part)] if isinstance(node, list) else node[part]
            assert abs(float(node) - float(f["v"])) < 1e-6           # the receipt points at the real value
        for ed in sg["ed"].values():
            assert len(ed["rx"]["nm"]) == len(ed["l"]) and ed["rx"]["chk"]
            for line in ed["rx"]["nm"]:
                for _num, source in line:
                    assert source is not None
                    seen += 1
    assert seen > 50


def test_match_logs_for_the_analyst_boards():
    data = build_replay(SAMPLE)
    shots, passes = data["shotLog"], data["passLog"]
    assert len(shots) == len(data["shots"])
    assert [(r[0], r[1]) for r in shots] == sorted((r[0], r[1]) for r in shots)
    assert all(r[6] in (0, 1) for r in shots) and all(0 <= r[4] <= 1 for r in shots)
    sides = {i: p["side"] for i, p in enumerate(data["players"])}
    assert passes and all(sides[a] == sides[b] for _p, _t, a, b, _x, _y in passes)   # completed passes stay in the team
