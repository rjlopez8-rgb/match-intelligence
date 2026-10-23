import json
import shutil
from pathlib import Path

from matchstats.season import build_season, render_season

DATA = Path(__file__).resolve().parents[2] / "data"


def test_season_hub_from_two_matches(tmp_path):
    shutil.copytree(DATA / "demo" / "riv-kin-1119", tmp_path / "riv-kin-1119", ignore=shutil.ignore_patterns("tracking*", "*.pkl.gz", "tryouts"))
    shutil.copytree(DATA / "sample" / "riv-kin-7", tmp_path / "riv-kin-7", ignore=shutil.ignore_patterns("tracking*", "*.pkl.gz"))
    (tmp_path / "season.json").write_text(json.dumps({"competition": "Meridian League", "season": "2026/27",
        "clubs": ["RIV", "KIN"], "matches": [
            {"matchday": 1, "date": "2026-10-17", "home": "RIV", "away": "KIN", "match_id": "riv-kin-7"},
            {"matchday": 2, "date": "2026-10-24", "home": "RIV", "away": "KIN", "match_id": "riv-kin-1119", "demo": True}]}), encoding="utf-8")
    d = build_season(tmp_path, links={"riv-kin-1119": "riv-kin-1119.html"})
    t = {r["club"]: r for r in d["table"]}
    # riv-kin-7 ended 0-1 and riv-kin-1119 ended 1-2: Kingsport won both
    assert t["KIN"]["pts"] == 6 and t["RIV"]["pts"] == 0 and t["KIN"]["pos"] == 1
    assert t["KIN"]["gf"] == 3 and t["RIV"]["gd"] == -2
    assert t["KIN"]["race"] == [0, 3, 6]
    demo = next(m for m in d["matches"] if m["id"] == "riv-kin-1119")
    assert demo["link"] == "riv-kin-1119.html" and len(demo["goals"]) == 3 and demo["wp"]
    mensah = d["players"]["riv-10"]["season"]
    assert mensah["apps"] == 2 and mensah["goals"] >= 1
    assert "__SEASON_DATA__" not in render_season(d)
