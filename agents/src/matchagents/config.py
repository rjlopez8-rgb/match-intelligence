"""Which model each agent uses, and how much it may say.

Every agent reads its deployment from an environment variable, so swapping a model is a
settings change, not a code change:

    MATCH_MODEL                 default deployment for every agent (gpt-5-mini-1)
    MATCH_MODEL_<AGENT>         override for one agent, e.g. MATCH_MODEL_ANALYST=gpt-6-sol-1

The defaults below are starting points; the evaluation harness decides the final choice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

AGENTS = ("producer", "analyst", "scenario", "telestrator", "director", "commentator", "factchecker", "scout")
DEFAULT_DEPLOYMENT = "gpt-5-mini-1"


@dataclass(frozen=True)
class AgentSettings:
    name: str
    deployment: str
    max_tokens: int  # includes reasoning tokens on reasoning models, so leave headroom
    reasoning_effort: str | None


# (max_tokens, reasoning effort). Routing and formatting jobs think little; analysis thinks more.
_LIMITS: dict[str, tuple[int, str | None]] = {
    "producer": (1500, "low"),
    "analyst": (4000, "low"),
    "scenario": (2000, "low"),
    "telestrator": (2500, "low"),
    "director": (1500, "low"),
    "commentator": (2500, "low"),
    "factchecker": (2500, "low"),
    "scout": (2500, "low"),
}


def settings_for(agent: str) -> AgentSettings:
    if agent not in AGENTS:
        raise ValueError(f"unknown agent '{agent}'")
    deployment = os.environ.get(f"MATCH_MODEL_{agent.upper()}") or os.environ.get("MATCH_MODEL") or DEFAULT_DEPLOYMENT
    max_tokens, effort = _LIMITS[agent]
    effort = os.environ.get(f"MATCH_EFFORT_{agent.upper()}", effort) or None
    return AgentSettings(agent, deployment, max_tokens, effort)


def all_settings() -> dict[str, AgentSettings]:
    return {a: settings_for(a) for a in AGENTS}
