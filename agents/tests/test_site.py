from matchagents.site import _drama, matches_of_the_week


def _m(mid, md, wp):
    return {"id": mid, "md": md, "wp": wp}


def test_match_of_the_week_is_the_one_that_swung_most():
    calm = _m("md01-a-b", 1, [[0, 0.40, 0.30], [45, 0.42, 0.28], [90, 0.45, 0.25]])
    wild = _m("md01-c-d", 1, [[0, 0.40, 0.30], [30, 0.80, 0.05], [60, 0.20, 0.60], [90, 0.90, 0.02]])
    other = _m("md02-a-c", 2, [[0, 0.5, 0.2], [90, 0.5, 0.2]])
    assert _drama(wild) > _drama(calm)
    assert matches_of_the_week({"matches": [calm, wild, other]}, None) == {1: "md01-c-d", 2: "md02-a-c"}


def test_the_demo_match_is_always_its_matchdays_pick():
    calm = _m("riv-kin-1119", 7, [[0, 0.4, 0.3], [90, 0.4, 0.3]])
    wild = _m("md07-c-d", 7, [[0, 0.4, 0.3], [30, 0.9, 0.05], [90, 0.1, 0.8]])
    assert matches_of_the_week({"matches": [wild, calm]}, "riv-kin-1119") == {7: "riv-kin-1119"}


def test_goals_lift_a_matchday_pick():
    swing = [[0, 0.4, 0.3], [45, 0.6, 0.2], [90, 0.4, 0.3]]
    nil = {"id": "md08-a-b", "md": 8, "wp": swing, "score": {"home": 0, "away": 0}}
    four = {"id": "md08-c-d", "md": 8, "wp": swing, "score": {"home": 4, "away": 0}}
    assert matches_of_the_week({"matches": [nil, four]}, None) == {8: "md08-c-d"}
