import copy
from pathlib import Path

import pytest

from matchstats import MatchAnalyzer, replay_bundle

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample" / "riv-kin-7"


@pytest.fixture(scope="session")
def items():
    """The sample match as a list of feed items (meta, events and 5 Hz frames, in time order)."""
    return list(replay_bundle(SAMPLE))


@pytest.fixture(scope="session")
def analyzer(items):
    return MatchAnalyzer().run(items)


def strip_truth(items):
    """Remove every simulator-only ground-truth field from a feed."""
    out = []
    for kind, item in items:
        if kind == "event" and "truth" in (item.get("details") or {}):
            item = copy.deepcopy(item)
            item["details"].pop("truth")
        out.append((kind, item))
    return out
