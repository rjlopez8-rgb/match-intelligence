import pytest

from matchagents.budget import UNKNOWN_MODEL_PRICE, BudgetExceeded, BudgetGuard, price_for


def test_prices_match_deployment_names():
    assert price_for("gpt-6-sol-1") == price_for("analyst-sol")
    assert price_for("gpt-6-luna-1")[0] < price_for("gpt-6-sol")[0]
    assert price_for("mystery-model") == UNKNOWN_MODEL_PRICE


def test_guard_stops_before_the_cap(tmp_path):
    g = BudgetGuard(daily_cap_usd=0.01, path=tmp_path / "usage.json")
    g.check("gpt-6-sol")  # nothing spent yet: allowed
    g.record("gpt-6-sol", input_tokens=2000, output_tokens=500)  # ~ $0.011
    with pytest.raises(BudgetExceeded):
        g.check("gpt-6-sol")
    assert g.remaining() == 0


def test_guard_persists_across_instances(tmp_path):
    p = tmp_path / "usage.json"
    BudgetGuard(daily_cap_usd=1, path=p).record("gpt-6-luna", 1_000_000, 0)
    assert abs(BudgetGuard(daily_cap_usd=1, path=p).spent_today() - 0.15) < 1e-9
