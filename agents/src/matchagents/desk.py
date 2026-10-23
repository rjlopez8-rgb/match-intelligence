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

Editions (``Desk.cover_editions``): one moment told many ways. The research runs once (Producer, Analyst,
Telestrator, Scenario, one set of simulations), then one edition desk per viewer style works in parallel:

    Producer ──► Analyst / Telestrator / Scenario ──► (join) ──► Fan desk ───────┐
                                                           ├──► Tactician desk ──┤
                                                           ├──► Analyst desk ────┼──► (join) ──► Segment
                                                           └──► Academy desk ────┘

Each edition desk has its own Director cut, then a Commentator ⇄ Fact-checker loop for every channel
(neutral, or a club's own channel) and language (English, Spanish, French), checked in that language.
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
    LANGUAGES,
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
MAX_PARALLEL = int(__import__("os").environ.get("MATCH_PARALLEL_CALLS", "4"))  # model calls in flight at once
EVIDENCE_KEYS = ("event_id", "minute", "type", "team", "player", "location", "details")


class ScenarioRunner(Protocol):
    """Runs a ScenarioSpec through the match engine and returns the distribution of results."""

    def run(self, spec: ScenarioSpec, session: MatchSession) -> dict: ...


# ----------------------------------------------------------------------------- state
@dataclass(frozen=True)
class Edition:
    """One way of telling a moment: a viewer style, a channel (neutral or a club's own) and a language."""

    style: str
    channel: str = "neutral"  # neutral | home | away
    lang: str = "en"

    @property
    def key(self) -> str:
        return f"{self.style}.{self.channel}.{self.lang}"

    @staticmethod
    def parse(key: str) -> Edition:
        style, channel, lang = key.split(".")
        return Edition(style, channel, lang)


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
    decision: dict | None = None
    editions: list[Edition] | None = None
    variants: dict[str, dict] = field(default_factory=dict)
    directors: dict[str, dict] = field(default_factory=dict)


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
        self.scenario_cache: dict[tuple[str, str], dict] = {}  # one set of simulations serves every edition
        self._sem: asyncio.Semaphore | None = None

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

    def _decision(self, evidence: list[dict]) -> dict | None:
        """The options at the key shot (teammates' space, pass odds, xG): the academy's coaching point."""
        shot = next((e for e in reversed(evidence) if e["type"] == "shot"), None)
        if not shot:
            return None
        try:
            res = self.session.queries.options(shot["event_id"])
        except Exception:  # noqa: BLE001 - no frame, no decision: the segment still airs
            return None
        self.ledger.add(Receipt("desk", "options", {"event_id": shot["event_id"]}, res))
        return res

    def _state(self, moment: dict, editions: list[Edition] | None = None) -> DeskState:
        m = dict(moment)
        m["team_side"] = self.session.ctx.side(m.get("team_id"))
        state = DeskState(m, self.profile, self._evidence(m), editions=editions)
        state.decision = self._decision(state.evidence)
        return state

    def team_name(self, channel: str) -> str | None:
        return None if channel == "neutral" else self.session.meta[channel]["short_name"]

    def sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(MAX_PARALLEL)
        return self._sem

    # ------------------------------------------------------------- commentary and checking
    def _moment_brief(self, m: dict) -> dict:
        return {k: m.get(k) for k in ("type", "minute_display", "team_name", "player_names", "score", "title")}

    async def commentary(self, state: DeskState, profile: ViewerProfile, issues: list[dict] | None,
                         previous: Commentary | None) -> Commentary | None:
        if state.brief is None:
            return None
        data = {
            "angle": state.plan.angle,
            "segment": state.plan.segment,
            "profile": profile.model_dump(),
            "moment": self._moment_brief(state.moment),
            "teams": {side: self.session.meta[side]["short_name"] for side in ("home", "away")},
            "headline": state.brief.headline,
            "facts": [{"index": i, **f.model_dump()} for i, f in enumerate(state.brief.facts)],
            "insight": state.brief.insight,
            "decision": state.brief.decision,
            "what_if": state.plan.what_if,
            "what_if_result": _usable(state.scenario_result),
        }
        task = "Write the on-air lines."
        if issues and previous is not None:
            data["previous_draft"] = previous.model_dump()
            data["fact_check_issues"] = issues
            task = "Rewrite the lines to fix every fact-check issue."
        async with self.sem():
            return await _safely(state, "commentator", self.ask("commentator", task, data, Commentary))

    async def review(self, state: DeskState, draft: Commentary, attempt: int) -> tuple[bool, list[dict], dict]:
        """Code check first (numbers must exist in the receipts), then the model's review."""
        sources = [*self.ledger.results(), state.moment, for_agents(state.scenario_result) or {}]
        code_issues = check_numbers(draft, sources) + check_style(draft)
        verdict = None
        if not code_issues:
            data = {
                "commentary": draft.model_dump(),
                "facts": [f.model_dump() for f in state.brief.facts],
                "tool_results": _unique_results(self.ledger),
                "simulation_result": _usable(state.scenario_result),
            }
            async with self.sem():
                ask = self.ask("factchecker", "Check the lines.", data, Verdict)
                verdict = await _safely(state, "factchecker", ask)
        approved = not code_issues and verdict is not None and verdict.approved
        issues = code_issues or ([i.model_dump() for i in verdict.issues] if verdict else [])
        record = {
            "attempt": attempt + 1,
            "code_issues": code_issues,
            "model_verdict": verdict.model_dump() if verdict else None,
            "approved": approved,
        }
        return approved, issues, record

    async def edition(self, state: DeskState, ed: Edition) -> dict:
        """Commentary for one edition, fact-checked in its language, with one rewrite allowed."""
        profile = ViewerProfile(style=ed.style, favourite_team=self.team_name(ed.channel), language=LANGUAGES[ed.lang])
        drafts: list[Commentary] = []
        checks: list[dict] = []
        issues = None
        for attempt in range(MAX_REVISIONS + 1):
            draft = await self.commentary(state, profile, issues, drafts[-1] if drafts else None)
            if draft is None:
                break
            drafts.append(draft)
            ok, issues, record = await self.review(state, draft, attempt)
            checks.append(record)
            if ok:
                return {"lines": [ln.text for ln in draft.lines], "headline": draft.headline, "caption": draft.caption,
                        "question": draft.question, "fallback": False, "checks": checks,
                        "drafts": [d.model_dump() for d in drafts]}
            if state.halted:
                break
        return {"lines": [state.moment["title"]], "headline": None, "caption": None, "question": None,
                "fallback": True, "checks": checks, "drafts": [d.model_dump() for d in drafts]}

    async def direct(self, state: DeskState, style: str) -> DirectorCue | None:
        data = {
            "moment": state.moment,
            "viewer_style": style,
            "plan": state.plan.model_dump(),
            "evidence": [{k: e[k] for k in ("event_id", "minute", "type", "player")} for e in state.evidence],
            "telestration_caption": state.telestration.caption if state.telestration else None,
        }
        async with self.sem():
            return await _safely(state, "director", self.ask("director", "Call the cameras.", data, DirectorCue))

    async def edition_desk(self, state: DeskState, style: str) -> None:
        """One style's desk: its Director cut, then every channel and language of its commentary, in parallel."""
        todo = [ed for ed in (state.editions or []) if ed.style == style and ed.key not in state.variants]
        if style not in state.directors:
            cue = await self.direct(state, style)
            if cue:
                state.directors[style] = cue.model_dump()
        results = await asyncio.gather(*(self.edition(state, ed) for ed in todo))
        for ed, r in zip(todo, results, strict=True):
            # a budget stop or a failed model call is not a verdict: leave it out so the next run retries it
            if not (r["fallback"] and (state.halted or not r["checks"])):
                state.variants[ed.key] = r

    # ---------------------------------------------------------------------------- editions
    async def cover_editions(self, moment: dict, editions: list[Edition], saved: dict | None = None) -> dict:
        """Tell one moment every way in ``editions``. ``saved`` is this moment's earlier output: its research
        and finished editions are reused, so an interrupted run resumes where it stopped and pays only once."""
        self.ledger = Ledger()
        start_call = len(self.trace.calls)
        state = self._state(moment, editions)
        if self.guard is not None and self.guard.remaining() <= 0:
            state.halted = True
            state.errors.append("daily budget reached before this moment")
        if saved and saved.get("plan") and saved.get("brief"):
            _restore(state, saved, self.ledger)
            styles = sorted({ed.style for ed in editions}, key=lambda st: [ed.style for ed in editions].index(st))
            await asyncio.gather(*(self.edition_desk(state, st) for st in styles))
            final = state
        else:
            if saved:
                state.variants.update(saved.get("editions") or {})
            result = await build_workflow(self, editions=editions).run(state)
            outputs = result.get_outputs()
            final = outputs[-1] if outputs else state
        seg = self._segment(final, start_call)
        if final.plan and final.plan.cover:
            self.rundown.append({"minute": moment["minute_display"], "period": moment["period"], "type": moment["type"],
                                 "segment": final.plan.segment, "said": [final.plan.angle]})
        default = next((k for k in (e.key for e in editions) if k in final.variants), None)
        if default:
            v = final.variants[default]
            seg["commentary"] = {"lines": v["lines"], "fallback": v["fallback"], "drafts": v["drafts"]}
            seg["factcheck"] = v["checks"]
            seg["aired"] = bool(final.plan and v["lines"])
        seg["editions"] = final.variants
        seg["directors"] = final.directors
        seg["director"] = final.directors.get(editions[0].style) if editions else seg["director"]
        seg["tool_results"] = _unique_results(self.ledger)
        seg["telestration_raw"] = final.telestration.model_dump() if final.telestration else None
        if saved and saved.get("cost"):
            for k in ("usd", "model_calls", "input_tokens", "output_tokens"):
                seg["cost"][k] = round(seg["cost"][k] + saved["cost"].get(k, 0), 6)
        return seg

    async def cover(self, moment: dict) -> dict:
        """Run the full desk on one moment and return the segment that airs (or the reason it doesn't)."""
        self.ledger = Ledger()
        start_call = len(self.trace.calls)
        m = dict(moment)
        m["team_side"] = self.session.ctx.side(m.get("team_id"))
        state = self._state(moment)
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
                "headline": s.drafts[-1].headline if s.drafts and not s.fallback else None,
                "caption": s.drafts[-1].caption if s.drafts and not s.fallback else None,
                "question": s.drafts[-1].question if s.drafts and not s.fallback else None,
                "fallback": s.fallback,
                "drafts": [d.model_dump() for d in s.drafts],
            },
            "brief": s.brief.model_dump() if s.brief else None,
            "factcheck": s.checks,
            "director": s.director.model_dump() if s.director else None,
            "telestration": s.telestration_resolved,
            "decision": s.decision,
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


def _restore(state: DeskState, saved: dict, ledger: Ledger) -> None:
    """Rebuild a moment's shared research from an earlier run's output (no model calls)."""
    state.plan = ProducerPlan.model_validate(saved["plan"])
    state.brief = AnalystBrief.model_validate(saved["brief"])
    if saved.get("telestration_raw"):
        state.telestration = Telestration.model_validate(saved["telestration_raw"])
    state.telestration_resolved = saved.get("telestration")
    sc = saved.get("scenario") or {}
    state.scenario = ScenarioSpec.model_validate(sc["spec"]) if sc.get("spec") else None
    state.scenario_result = sc.get("result")
    state.decision = saved.get("decision") or state.decision
    state.variants.update(saved.get("editions") or {})
    state.directors.update(saved.get("directors") or {})
    for r in saved.get("tool_results") or []:
        ledger.add(Receipt("analyst", r["tool"], r["args"], r["result"]))


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
        profile = state.profile.model_dump()
        if state.editions:
            profile = {"style": "every style at once (fan, tactician, analyst, academy)", "favourite_team": None,
                       "languages": sorted({LANGUAGES[e.lang] for e in state.editions})}
        data = {
            "moment": m,
            "profile": profile,
            "rundown": self.desk.rundown[-8:],
            "score": m["score"],
        }
        state.plan = await _safely(state, "producer", self.desk.ask("producer", "Plan coverage.", data, ProducerPlan))
        if state.plan is not None and state.editions and not state.plan.cover:
            state.plan.cover = True  # editions are for moments a person picked: they always air
            self.desk.trace.step("producer", "cover overridden: moment chosen for editions")
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
            "viewer_style": "all" if state.editions else state.profile.style,
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
                key = (state.scenario.event_id, state.scenario.intervention)
                cached = self.desk.scenario_cache.get(key)
                if cached and cached.get("status") == "ok":
                    state.scenario_result = cached
                    self.desk.trace.step("scenario", "reused simulations", event_id=key[0])
                else:
                    self.desk.trace.step("scenario", "simulating", futures=state.scenario.simulations)
                    try:  # CPU-heavy: run off the event loop so the other agents keep working
                        state.scenario_result = (
                            await asyncio.to_thread(runner.run, state.scenario, self.desk.session)
                            if runner
                            else {"status": "not_run", "reason": "ghost futures disabled for this run"}
                        )
                        if state.scenario_result.get("status") == "ok":
                            self.desk.scenario_cache[key] = state.scenario_result
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
        state.director = await self.desk.direct(state, state.profile.style)
        await ctx.send_message(state)


class CommentatorStep(Executor):
    def __init__(self, desk: Desk):
        super().__init__(id="commentator")
        self.desk = desk

    @handler
    async def first(self, state: DeskState, ctx: WorkflowContext[DeskState]) -> None:
        draft = await self.desk.commentary(state, state.profile, None, None)
        if draft:
            state.drafts.append(draft)
        await ctx.send_message(state)

    @handler
    async def revise(self, rev: Revision, ctx: WorkflowContext[DeskState]) -> None:
        st = rev.state
        draft = await self.desk.commentary(st, st.profile, rev.issues, st.drafts[-1] if st.drafts else None)
        if draft:
            st.drafts.append(draft)
        await ctx.send_message(st)


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
        approved, issues, record = await self.desk.review(state, draft, attempt)
        state.checks.append(record)
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


class EditionStep(Executor):
    """One viewer style's desk: Director cut + Commentator ⇄ Fact-checker for each channel and language."""

    def __init__(self, desk: Desk, style: str):
        super().__init__(id=f"{style}_desk")
        self.desk, self.style = desk, style

    @handler
    async def run(self, state: DeskState, ctx: WorkflowContext[DeskState]) -> None:
        if not state.halted:
            await self.desk.edition_desk(state, self.style)
        await ctx.send_message(state)


def build_workflow(desk: Desk, editions: list[Edition] | None = None):
    producer = ProducerStep(desk)
    analyst, telestrator, scenario = AnalystStep(desk), TelestratorStep(desk), ScenarioStep(desk)
    research = Gather(id="research_done")
    if editions:
        styles = list(dict.fromkeys(ed.style for ed in editions))
        desks = [EditionStep(desk, st) for st in styles]
        on_air = Gather(id="on_air")
        return (
            WorkflowBuilder(name="broadcast-desk-editions", start_executor=producer, output_from=[producer, on_air])
            .add_fan_out_edges(producer, [analyst, telestrator, scenario])
            .add_fan_in_edges([analyst, telestrator, scenario], research)
            .add_fan_out_edges(research, desks)
            .add_fan_in_edges(desks, on_air)
            .build()
        )
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


def workflow_mermaid(desk: Desk, editions: list[Edition] | None = None) -> str:
    from agent_framework import WorkflowViz

    return WorkflowViz(build_workflow(desk, editions)).to_mermaid()
