import base64

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
