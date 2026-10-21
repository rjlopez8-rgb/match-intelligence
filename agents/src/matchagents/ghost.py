"""Runs the Scenario agent's plan through the match engine (ghost futures and Ghost Scout).

The Scenario agent decides *what* to simulate; this runner does it with the real engine and
returns numbers only. Those numbers join the fact-check evidence, so the Commentator can say
"in 20 simulated futures..." and have it verified like any other claim.
"""

from __future__ import annotations

import os
import time

from matchengine.ghost import load_checkpoint, scout_shot, what_if_shot
from matchstats.models import default_models

from .schemas import ScenarioSpec
from .session import MatchSession


def _futures(requested: int) -> int:
    fixed = os.environ.get("MATCH_GHOST_FUTURES")  # set by --futures: an exact count for this run
    if fixed:
        return max(6, int(fixed))
    cap = int(os.environ.get("MATCH_GHOST_FUTURES_MAX", "40"))
    return max(6, min(cap, requested))


class GhostRunner:
    """Implements the desk's ScenarioRunner with the matchengine ghost module."""

    def __init__(self, record_ghosts: int = 1, parallel: bool = True):
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
        if "states" in result:
            result["curves"] = {k: [win_curve(s) for s in v] for k, v in result.pop("states").items()}
        return {
            "status": "ok",
            "intervention": spec.intervention,
            "seconds": round(time.perf_counter() - t0, 1),
            **result,
        }


def win_curve(states: list[list]) -> list[list]:
    """One future's live win probability over time: [period, t, home, away], from the trained model."""
    models = default_models()
    out = []
    for period, t, gh, ga, xgh, xga, rh, ra in states:
        minute = min(t / 60.0, 49.0) + (45.0 if period == 2 else 0.0)
        h, _d, a = models.win_probabilities(minute, gh - ga, xgh - xga, rh - ra)
        out.append([period, t, round(h, 3), round(a, 3)])
    return out


def for_agents(result: dict | None) -> dict | None:
    """The scenario result without bulky ghost frames, for prompts and the fact-check."""
    if not result:
        return result
    return {k: v for k, v in result.items() if k not in ("ghosts", "featured", "curves", "states")}
