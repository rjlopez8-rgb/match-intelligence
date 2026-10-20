"""The broadcast desk: seven agents orchestrated as a Microsoft Agent Framework workflow.

For every key moment the stats layer detects:

    Producer ──► Analyst ──────┐
       │    ├──► Telestrator ──┼──► (join) ──► Director ─────────────────────┐
       │    └──► Scenario ─────┘          └──► Commentator ⇄ Fact-checker ───┴──► (join) ──► Segment
       └──► (skip: not worth airtime)

- The Producer decides whether the moment airs, the angle, and which specialists to call.
- Analyst, Telestrator and Scenario agent work in parallel.
- The Commentator writes only from the Analyst's facts. The Fact-checker gates every line: first
  code (every number must exist in the stats layer's results), then a model review. Rejected lines
  go back to the Commentator once; if they still fail, the stats layer's own factual title airs.
  Nothing unverified reaches the screen.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_framework import Agent, Executor, WorkflowBuilder, WorkflowContext, handler

from . import offline
from .budget import BudgetExceeded, BudgetGuard
from .checks import check_numbers, check_style, resolve_telestration
from .config import AGENTS, AgentSettings, settings_for
from .ghost import for_agents
from .llm import ScriptedChatClient, Trace, foundry_client
from .prompts import INSTRUCTIONS
from .schemas import (
    AnalystBrief,
    Commentary,
    DirectorCue,
    ProducerPlan,
    ScenarioSpec,
    Telestration,
    Verdict,
    ViewerProfile,
)
from .session import MatchSession
from .tools import Ledger, Receipt, stats_tools

MAX_REVISIONS = 1
EVIDENCE_KEYS = ("event_id", "minute", "type", "team", "player", "location", "details")


class ScenarioRunner(Protocol):
    """Runs a ScenarioSpec through the match engine and returns the distribution of results."""

    def run(self, spec: ScenarioSpec, session: MatchSession) -> dict: ...


# ----------------------------------------------------------------------------- state
@dataclass
class DeskState:
    moment: dict
    profile: ViewerProfile
    evidence: list[dict]
    t0: float = field(default_factory=time.perf_counter)
    plan: ProducerPlan | None = None
    brief: AnalystBrief | None = None
    telestration: Telestration | None = None
    telestration_resolved: dict | None = None
    scenario: ScenarioSpec | None = None
    scenario_result: dict | None = None
    director: DirectorCue | None = None
    drafts: list[Commentary] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    final_lines: list[str] = field(default_factory=list)
    fallback: bool = False
    errors: list[str] = field(default_factory=list)
    halted: bool = False


@dataclass
class Revision:
    state: DeskState
    issues: list[dict]


def _data(task: str, data: dict) -> str:
    return f"{task}\n\nDATA:\n{json.dumps(data, ensure_ascii=False, default=str)}"


# --------------------------------------------------------------------------- the desk
class Desk:
    """Holds the seven agents for one match and covers moments one at a time."""

    def __init__(
        self,
        session: MatchSession,
        profile: ViewerProfile | None = None,
        *,
        offline_mode: bool = False,
        guard: BudgetGuard | None = None,
        scenario_runner: ScenarioRunner | None = None,
        http_client=None,
    ):
        self.session = session
        self.profile = profile or ViewerProfile()
        self.offline = offline_mode
        self.guard = None if offline_mode else (guard or BudgetGuard())
        self.scenario_runner = scenario_runner
        self.trace = Trace()
        self.ledger = Ledger()
        self.rundown: list[dict] = []
        self.settings: dict[str, AgentSettings] = {a: settings_for(a) for a in AGENTS}
        self.agents: dict[str, Agent] = {a: self._agent(a, http_client) for a in AGENTS}

    def _agent(self, name: str, http_client) -> Agent:
        s = self.settings[name]
        if self.offline:
            client = ScriptedChatClient(name, offline.responder, self.trace)
        else:
            client = foundry_client(s, self.trace, self.guard, http_client=http_client)
        tools = self._analyst_tools() if name == "analyst" else None
        options: dict[str, Any] = {"model": s.deployment, "max_tokens": s.max_tokens}
        if s.reasoning_effort and not self.offline:
            options["reasoning_effort"] = s.reasoning_effort
        return Agent(client=client, name=name, instructions=INSTRUCTIONS[name], tools=tools, default_options=options)

    def _analyst_tools(self):
        desk = self

        class _Queries:  # resolves to the session's live queries at call time
            def call(self, tool, args):
                return desk.session.queries.call(tool, args)

        class _LedgerRef:
            def add(self, receipt):
                desk.ledger.add(receipt)

        return stats_tools(_Queries(), _LedgerRef(), "analyst")  # type: ignore[arg-type]

    async def ask(self, agent: str, task: str, data: dict, schema):
        """Run one agent and return its validated structured output."""
        self.trace.step(agent, "start")
        r = await self.agents[agent].run(_data(task, data), options={"response_format": schema})
        value = r.value if isinstance(r.value, schema) else schema.model_validate_json(r.text)
        self.trace.step(agent, "done")
        return value

    # -------------------------------------------------------------------------- covering
    def _evidence(self, moment: dict) -> list[dict]:
        ex = self.session.queries.explain_moment(moment["moment_id"])
        self.ledger.add(Receipt("desk", "explain_moment", {"moment_id": moment["moment_id"]}, ex))
        return [{k: e.get(k) for k in EVIDENCE_KEYS} for e in ex["evidence"]]

    async def cover(self, moment: dict) -> dict:
        """Run the full desk on one moment and return the segment that airs (or the reason it doesn't)."""
        self.ledger = Ledger()
        start_call = len(self.trace.calls)
        m = dict(moment)
        m["team_side"] = self.session.ctx.side(m.get("team_id"))
        state = DeskState(m, self.profile, self._evidence(m))
        if self.guard is not None and self.guard.remaining() <= 0:
            state.halted = True
            state.errors.append("daily budget reached before this moment")
        workflow = build_workflow(self)
        result = await workflow.run(state)
        outputs = result.get_outputs()
        final: DeskState = outputs[-1] if outputs else state
        segment = self._segment(final, start_call)
        if final.plan and final.plan.cover:
            self.rundown.append(
                {
                    "minute": m["minute_display"],
                    "period": m["period"],
                    "type": m["type"],
                    "segment": final.plan.segment,
                    "said": final.final_lines[:1],
                }
            )
        return segment

    def _segment(self, s: DeskState, start_call: int) -> dict:
        calls = self.trace.calls[start_call:]
        m = s.moment
        return {
            "moment_id": m["moment_id"],
            "minute": m["minute_display"],
            "period": m["period"],
            "t": m["t"],
            "type": m["type"],
            "title": m["title"],
            "aired": bool(s.plan and s.plan.cover and s.final_lines),
            "plan": s.plan.model_dump() if s.plan else None,
            "commentary": {
                "lines": s.final_lines,
                "fallback": s.fallback,
                "drafts": [d.model_dump() for d in s.drafts],
            },
            "brief": s.brief.model_dump() if s.brief else None,
            "factcheck": s.checks,
            "director": s.director.model_dump() if s.director else None,
            "telestration": s.telestration_resolved,
            "scenario": {
                "spec": s.scenario.model_dump() if s.scenario else None,
                "result": s.scenario_result,
            },
            "receipts": self.ledger.summary(),
            "cost": {
                "usd": round(sum(c.usd for c in calls), 6),
                "model_calls": len(calls),
                "input_tokens": sum(c.input_tokens for c in calls),
                "output_tokens": sum(c.output_tokens for c in calls),
                "seconds": round(time.perf_counter() - s.t0, 2),
                "by_agent": _by_agent(calls),
            },
            "errors": s.errors,
        }


def _by_agent(calls) -> dict:
    out: dict[str, dict] = {}
    for c in calls:
        a = out.setdefault(c.agent, {"model": c.model, "calls": 0, "usd": 0.0, "seconds": 0.0})
        a["calls"] += 1
        a["usd"] = round(a["usd"] + c.usd, 6)
        a["seconds"] = round(a["seconds"] + c.seconds, 3)
    return out


async def _safely(state: DeskState, agent: str, coro):
    """Run one agent step; failures degrade the segment instead of crashing the broadcast."""
    if state.halted:
        coro.close()
        return None
    try:
        return await coro
    except BudgetExceeded as exc:
        state.halted = True
        state.errors.append(f"{agent}: {exc}")
    except Exception as exc:  # a model or network error must never take the broadcast down
        state.errors.append(f"{agent}: {type(exc).__name__}: {exc}")
    return None


# -------------------------------------------------------------------------- executors
class ProducerStep(Executor):
    def __init__(self, desk: Desk):
        super().__init__(id="producer")
        self.desk = desk

    @handler
    async def run(self, state: DeskState, ctx: WorkflowContext[DeskState, DeskState]) -> None:
        m = state.moment
        data = {
            "moment": m,
            "profile": state.profile.model_dump(),
            "rundown": self.desk.rundown[-8:],
            "score": m["score"],
        }
        state.plan = await _safely(state, "producer", self.desk.ask("producer", "Plan coverage.", data, ProducerPlan))
        if state.plan is None or not state.plan.cover:
            await ctx.yield_output(state)
            return
        await ctx.send_message(state)


class AnalystStep(Executor):
    def __init__(self, desk: Desk):
        super().__init__(id="analyst")
        self.desk = desk

    @handler
    async def run(self, state: DeskState, ctx: WorkflowContext[DeskState]) -> None:
        data = {
            "moment_id": state.moment["moment_id"],
            "moment": state.moment,
            "angle": state.plan.angle,
            "viewer_style": state.profile.style,
        }
        state.brief = await _safely(
            state, "analyst", self.desk.ask("analyst", "Research this moment with the tools.", data, AnalystBrief)
        )
        await ctx.send_message(state)


class TelestratorStep(Executor):
    def __init__(self, desk: Desk):
        super().__init__(id="telestrator")
        self.desk = desk

    @handler
    async def run(self, state: DeskState, ctx: WorkflowContext[DeskState]) -> None:
        if state.plan.telestrate:
            data = {"moment": state.moment, "angle": state.plan.angle, "evidence": state.evidence}
            state.telestration = await _safely(
                state, "telestrator", self.desk.ask("telestrator", "Draw on the replay.", data, Telestration)
            )
            if state.telestration:
                state.telestration_resolved = resolve_telestration(state.telestration, self.desk.session, state.moment)
        await ctx.send_message(state)


class ScenarioStep(Executor):
    def __init__(self, desk: Desk):
        super().__init__(id="scenario")
        self.desk = desk

    @handler
    async def run(self, state: DeskState, ctx: WorkflowContext[DeskState]) -> None:
        if state.plan.what_if:
            data = {"what_if": state.plan.what_if, "moment": state.moment, "evidence": state.evidence}
            state.scenario = await _safely(
                state, "scenario", self.desk.ask("scenario", "Design the simulation.", data, ScenarioSpec)
            )
            if state.scenario:
                state.scenario.simulations = max(6, min(40, state.scenario.simulations))
                runner = self.desk.scenario_runner
                self.desk.trace.step("scenario", "simulating", futures=state.scenario.simulations)
                try:  # CPU-heavy: run off the event loop so the other agents keep working
                    state.scenario_result = (
                        await asyncio.to_thread(runner.run, state.scenario, self.desk.session)
                        if runner
                        else {"status": "not_run", "reason": "ghost futures disabled for this run"}
                    )
                except Exception as exc:  # a simulation failure must not take the segment down
                    state.errors.append(f"scenario runner: {type(exc).__name__}: {exc}")
        await ctx.send_message(state)


class Gather(Executor):
    """Waits for every parallel branch, then passes the shared state on."""

    @handler
    async def run(self, states: list[DeskState], ctx: WorkflowContext[DeskState, DeskState]) -> None:
        if self.id == "on_air":
            await ctx.yield_output(states[0])
        else:
            await ctx.send_message(states[0])


class DirectorStep(Executor):
    def __init__(self, desk: Desk):
        super().__init__(id="director")
        self.desk = desk

    @handler
    async def run(self, state: DeskState, ctx: WorkflowContext[DeskState]) -> None:
        data = {
            "moment": state.moment,
            "plan": state.plan.model_dump(),
            "evidence": [{k: e[k] for k in ("event_id", "minute", "type", "player")} for e in state.evidence],
            "telestration_caption": state.telestration.caption if state.telestration else None,
        }
        state.director = await _safely(
            state, "director", self.desk.ask("director", "Call the cameras.", data, DirectorCue)
        )
        await ctx.send_message(state)


class CommentatorStep(Executor):
    def __init__(self, desk: Desk):
        super().__init__(id="commentator")
        self.desk = desk

    async def _write(self, state: DeskState, issues: list[dict] | None) -> Commentary | None:
        if state.brief is None:
            return None
        data = {
            "angle": state.plan.angle,
            "segment": state.plan.segment,
            "profile": state.profile.model_dump(),
            "headline": state.brief.headline,
            "facts": [{"index": i, **f.model_dump()} for i, f in enumerate(state.brief.facts)],
            "insight": state.brief.insight,
            "what_if": state.plan.what_if,
            "what_if_result": _usable(state.scenario_result),
        }
        task = "Write the on-air lines."
        if issues:
            data["previous_draft"] = state.drafts[-1].model_dump()
            data["fact_check_issues"] = issues
            task = "Rewrite the lines to fix every fact-check issue."
        return await _safely(state, "commentator", self.desk.ask("commentator", task, data, Commentary))

    @handler
    async def first(self, state: DeskState, ctx: WorkflowContext[DeskState]) -> None:
        draft = await self._write(state, None)
        if draft:
            state.drafts.append(draft)
        await ctx.send_message(state)

    @handler
    async def revise(self, rev: Revision, ctx: WorkflowContext[DeskState]) -> None:
        draft = await self._write(rev.state, rev.issues)
        if draft:
            rev.state.drafts.append(draft)
        await ctx.send_message(rev.state)


class FactCheckStep(Executor):
    def __init__(self, desk: Desk):
        super().__init__(id="factchecker")
        self.desk = desk

    @handler
    async def run(self, state: DeskState, ctx: WorkflowContext[DeskState | Revision]) -> None:
        attempt = len(state.checks)
        draft = state.drafts[-1] if state.drafts and len(state.drafts) > attempt else None
        if draft is None:  # nothing (new) to check: air the stats layer's own factual title
            state.final_lines, state.fallback = [state.moment["title"]], True
            await ctx.send_message(state, target_id="on_air")
            return
        sources = [*self.desk.ledger.results(), state.moment, for_agents(state.scenario_result) or {}]
        code_issues = check_numbers(draft, sources) + check_style(draft)
        verdict = None
        if not code_issues:
            data = {
                "commentary": draft.model_dump(),
                "facts": [f.model_dump() for f in state.brief.facts],
                "tool_results": _unique_results(self.desk.ledger),
                "simulation_result": _usable(state.scenario_result),
            }
            verdict = await _safely(
                state, "factchecker", self.desk.ask("factchecker", "Check the lines.", data, Verdict)
            )
        approved = not code_issues and verdict is not None and verdict.approved
        issues = code_issues or ([i.model_dump() for i in verdict.issues] if verdict else [])
        state.checks.append(
            {
                "attempt": attempt + 1,
                "code_issues": code_issues,
                "model_verdict": verdict.model_dump() if verdict else None,
                "approved": approved,
            }
        )
        if approved:
            state.final_lines = [line.text for line in draft.lines]
            await ctx.send_message(state, target_id="on_air")
        elif attempt < MAX_REVISIONS and not state.halted:
            await ctx.send_message(Revision(state, issues), target_id="commentator")
        else:
            state.final_lines, state.fallback = [state.moment["title"]], True
            await ctx.send_message(state, target_id="on_air")


def _usable(result: dict | None) -> dict | None:
    return for_agents(result) if result and result.get("status") == "ok" else None


def _unique_results(ledger: Ledger) -> list[dict]:
    seen, out = set(), []
    for r in ledger.receipts:
        key = (r.tool, json.dumps(r.args, sort_keys=True))
        if r.error is None and key not in seen:
            seen.add(key)
            out.append({"tool": r.tool, "args": r.args, "result": r.result})
    return out


def build_workflow(desk: Desk):
    producer = ProducerStep(desk)
    analyst, telestrator, scenario = AnalystStep(desk), TelestratorStep(desk), ScenarioStep(desk)
    research = Gather(id="research_done")
    director, commentator, checker = DirectorStep(desk), CommentatorStep(desk), FactCheckStep(desk)
    on_air = Gather(id="on_air")
    return (
        WorkflowBuilder(name="broadcast-desk", start_executor=producer, output_from=[producer, on_air])
        .add_fan_out_edges(producer, [analyst, telestrator, scenario])
        .add_fan_in_edges([analyst, telestrator, scenario], research)
        .add_fan_out_edges(research, [director, commentator])
        .add_edge(commentator, checker)
        .add_edge(checker, commentator)
        .add_fan_in_edges([director, checker], on_air)
        .build()
    )


def workflow_mermaid(desk: Desk) -> str:
    from agent_framework import WorkflowViz

    return WorkflowViz(build_workflow(desk)).to_mermaid()
