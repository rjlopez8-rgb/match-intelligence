from collections import Counter

from matchengine.season import fixtures, schedule


def test_double_round_robin():
    codes = ["A", "B", "C", "D", "E", "F", "G", "H"]
    fx = fixtures(codes)
    assert len(fx) == 56 and max(md for md, _, _ in fx) == 14
    pairs = Counter((h, a) for _, h, a in fx)
    assert all(v == 1 for v in pairs.values()) and len(pairs) == 56     # each ordered pair once: home and away
    for md in range(1, 15):
        teams = [t for m, h, a in fx if m == md for t in (h, a)]
        assert sorted(teams) == codes                                   # everyone plays once a matchday
    homes = Counter(h for _, h, _ in fx)
    assert set(homes.values()) == {7}


def test_demo_match_keeps_its_date():
    s = schedule("RIV", "KIN", "2026-10-24")
    demo = next(x for x in s if x["home"] == "RIV" and x["away"] == "KIN")
    assert demo["date"] == "2026-10-24"
    assert len({x["seed"] for x in s}) == 56
