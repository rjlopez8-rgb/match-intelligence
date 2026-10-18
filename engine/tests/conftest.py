import pytest

from matchengine import MatchConfig, MatchEngine, get_club


@pytest.fixture(scope="session")
def full_match():
    """One complete simulated match (events + 2 Hz tracking), shared by tests."""
    cfg = MatchConfig(home=get_club("RIV"), away=get_club("KIN"), seed=11)
    eng = MatchEngine(cfg, frame_hz=2)
    result = eng.simulate(keep_frames=True)
    return eng, result


def run_until(engine: MatchEngine, seconds: float) -> list[tuple[str, dict]]:
    items = []
    for kind, item in engine.run():
        items.append((kind, item))
        if engine.period == 1 and engine.t >= seconds:
            break
    return items
