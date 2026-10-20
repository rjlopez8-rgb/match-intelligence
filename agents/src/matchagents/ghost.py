"""Runs the Scenario agent's plan through the match engine (ghost futures and Ghost Scout).

The Scenario agent decides *what* to simulate; this runner does it with the real engine and
returns numbers only. Those numbers join the fact-check evidence, so the Commentator can say
"in 20 simulated futures..." and have it verified like any other claim.
"""

from __future__ import annotations

import os
import time

from matchengine.ghost import load_checkpoint, scout_shot, what_if_shot

from .schemas import ScenarioSpec
from .session import MatchSession


def _futures(requested: int) -> int:
    cap = int(os.environ.get("MATCH_GHOST_FUTURES_MAX", "40"))
    return max(6, min(cap, requested))


class GhostRunner:
    """Implements the desk's ScenarioRunner with the matchengine ghost module."""

    def __init__(self, record_ghosts: int = 3, parallel: bool = True):
        self.record_ghosts = record_ghosts
        self.parallel = parallel

    def _shot_event(self, spec: ScenarioSpec, session: MatchSession) -> dict | None:
        ev = session.ctx.by_id.get(spec.event_id)
        if ev and ev["type"] == "goal":  # the agent may point at the goal: use the shot behind it
            ev = session.ctx.by_id.get((ev.get("details") or {}).get("shot_event_id", ""))
        return ev if ev and ev["type"] == "shot" else None

    def run(self, spec: ScenarioSpec, session: MatchSession) -> dict:
        t0 = time.perf_counter()
        shot = self._shot_event(spec, session)
        if shot is None:
            return {"status": "unsupported", "reason": f"{spec.event_id} is not a shot the engine can replay"}
        n = _futures(spec.simulations)
        # the frozen state recorded with the match (exact on every machine); else replay locally
        fork = load_checkpoint(session.bundle, shot["event_id"])
        try:
            if spec.intervention == "player_swap":
                result = scout_shot(session.meta, shot, n=n, parallel=self.parallel, fork=fork)
            elif spec.intervention in ("shot_scores", "shot_misses", "goal_disallowed"):
                change = "shot_scores" if spec.intervention == "shot_scores" else "shot_misses"
                result = what_if_shot(
                    session.meta, shot, change, n=n, record=self.record_ghosts, parallel=self.parallel, fork=fork
                )
            else:
                return {"status": "unsupported", "reason": f"'{spec.intervention}' is not simulated yet"}
        except ValueError as exc:
            return {"status": "unsupported", "reason": str(exc)}
        return {
            "status": "ok",
            "intervention": spec.intervention,
            "seconds": round(time.perf_counter() - t0, 1),
            **result,
        }


def for_agents(result: dict | None) -> dict | None:
    """The scenario result without bulky ghost frames, for prompts and the fact-check."""
    if not result:
        return result
    return {k: v for k, v in result.items() if k != "ghosts"}
