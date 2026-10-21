import asyncio
from pathlib import Path

import pytest

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample" / "riv-kin-7"
GOAL = "riv-kin-7-m044"  # Duarte 74'
MISSED_PENALTY = "riv-kin-7-m007"


@pytest.fixture(autouse=True)
def _no_real_azure(monkeypatch, tmp_path):
    """Tests never touch a real endpoint or the real usage file."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test-resource.services.ai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for var in ("MATCH_MODEL", "MATCH_MODEL_ANALYST", "MATCH_DAILY_CAP_USD"):
        monkeypatch.delenv(var, raising=False)


def run(coro):
    return asyncio.run(coro)
