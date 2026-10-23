"""The Scout agent: turns a player try-out (simulated by the match engine) into a scouting note.

Same safety net as the commentary: every number in the note must appear in the try-out data (code check,
which cannot be talked round), one revision if it fails, and a code-written fallback note if the second
draft fails too. ``matchagents scout BOARD.json`` writes a note into every pre-simulated try-out on a board.
"""

from __future__ import annotations

import json
from typing import Any

from agent_framework import Agent

from .budget import BudgetGuard
from .checks import check_numbers, check_style
from .config import settings_for
from .llm import ScriptedChatClient, Trace, foundry_client
from .prompts import INSTRUCTIONS
from .schemas import Commentary, Line, ScoutReport


def evidence(t: dict) -> dict:
    """The numbers the Scout may use, in spoken-friendly form (rounded once, here)."""
    w, o, d = t["with"], t["without"], t["diff"]

    def r2(x: float) -> float:
        return round(float(x), 2)

    return {
        "fixture_team": t.get("team"),
        "position": t["position"]["label"],
        "candidate": t["candidate"],
        "incumbent": t["incumbent"],
        "matches_each_way": w["futures"],
        "expected_points_per_match": {"with_candidate": r2(w["xpts"]), "with_incumbent": r2(o["xpts"]),
                                      "difference": r2(d["xpts"]), "margin_of_error": r2(d["xpts_margin_95"]),
                                      "difference_is_clear": d.get("clear", False)},
        "team_xg_for": {"with_candidate": r2(w["xg_for"]), "with_incumbent": r2(o["xg_for"]), "difference": r2(d["xg_for"])},
        "team_xg_against": {"with_candidate": r2(w["xg_against"]), "with_incumbent": r2(o["xg_against"]),
                            "difference": r2(d["xg_against"])},
        "results_with_candidate": {"win": r2(w["win"]), "draw": r2(w["draw"]), "loss": r2(w["loss"])},
        "results_with_incumbent": {"win": r2(o["win"]), "draw": r2(o["draw"]), "loss": r2(o["loss"])},
        "per90_candidate": {k: r2(v) for k, v in w["per90"].items()},
        "per90_incumbent": {k: r2(v) for k, v in o["per90"].items()},
    }


def fallback_note(t: dict) -> ScoutReport:
    """Written by code from the numbers alone: always true, never exciting."""
    d, c, i = t["diff"], t["candidate"]["name"], t["incumbent"]["name"]
    pts, m = round(d["xpts"], 2), round(d["xpts_margin_95"], 2)
    if d.get("clear") and pts > 0:
        verdict, head = "sign", f"{c} makes the team better at {t['position']['label']}"
    elif pts > 0:
        verdict, head = "shortlist", f"{c} looks promising, but it is too close to call"
    else:
        verdict, head = "pass", f"{c} is no upgrade on {i}"
    sign = "+" if pts >= 0 else "-"
    report = (f"Over {t['with']['futures']} simulated matches each way, the team averaged {sign}{abs(pts):.2f} expected "
              f"points per match with {c} instead of {i} (margin of error {m:.2f}).")
    return ScoutReport(headline=head, report=report, verdict=verdict)


def _offline_responder(agent: str, messages, options) -> str:
    from .offline import payload

    p = payload(messages)
    note = fallback_note(p["_tryout"])
    return note.model_dump_json()


class Scout:
    def __init__(self, offline_mode: bool = False, guard: BudgetGuard | None = None, http_client=None):
        self.offline = offline_mode
        self.trace = Trace()
        self.guard = None if offline_mode else (guard or BudgetGuard())
        s = settings_for("scout")
        client = (ScriptedChatClient("scout", _offline_responder, self.trace) if offline_mode
                  else foundry_client(s, self.trace, self.guard, http_client=http_client))
        options: dict[str, Any] = {"model": s.deployment, "max_tokens": s.max_tokens}
        if s.reasoning_effort and not offline_mode:
            options["reasoning_effort"] = s.reasoning_effort
        self.agent = Agent(client=client, name="scout", instructions=INSTRUCTIONS["scout"], default_options=options)

    async def _ask(self, t: dict, ev: dict, issues: list[dict] | None) -> ScoutReport:
        task = "Write the scouting note for this try-out."
        if issues:
            task += " Your previous draft failed the fact-check; fix exactly these problems: " + json.dumps(issues)
        data = {**ev, "_tryout": t} if self.offline else ev
        r = await self.agent.run(f"{task}\n\nDATA:\n{json.dumps(data, ensure_ascii=False)}",
                                 options={"response_format": ScoutReport})
        return r.value if isinstance(r.value, ScoutReport) else ScoutReport.model_validate_json(r.text)

    async def note(self, t: dict) -> dict:
        ev = evidence(t)
        issues: list[dict] | None = None
        for attempt in (1, 2):
            rep = await self._ask(t, ev, issues)
            as_lines = Commentary(lines=[Line(text=rep.headline, facts_used=[]), Line(text=rep.report, facts_used=[])], headline="", caption="", question=None)
            issues = check_numbers(as_lines, [ev]) + check_style(as_lines)
            if not issues:
                return {**rep.model_dump(), "checked": True, "attempts": attempt}
        fb = fallback_note(t)
        return {**fb.model_dump(), "checked": True, "attempts": 2, "fallback": True}
