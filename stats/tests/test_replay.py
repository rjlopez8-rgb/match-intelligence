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
