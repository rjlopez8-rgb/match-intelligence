"""Chat clients for the agents: Microsoft Foundry for real runs, a scripted client for free ones.

Every client carries ``BudgetMiddleware``. It runs around each individual model request:
before the request it asks the budget guard for permission (a refusal raises
``BudgetExceeded`` and nothing is sent), and after it records the real token counts.
Each request is also written to the run's ``Trace``, which the control room and the
"why each agent uses its model" table are built from.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

from agent_framework import BaseChatClient, ChatContext, ChatMiddleware, ChatResponse, Content, Message, UsageDetails
from agent_framework._middleware import ChatMiddlewareLayer
from agent_framework._tools import FunctionInvocationLayer

from .budget import BudgetGuard, price_for
from .config import AgentSettings


# ----------------------------------------------------------------------------------- trace
@dataclass
class CallRecord:
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    usd: float
    seconds: float
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class Trace:
    """Everything one run did: model calls (with cost) and workflow steps, in order."""

    calls: list[CallRecord] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)

    @property
    def usd(self) -> float:
        return round(sum(c.usd for c in self.calls), 6)

    @property
    def tokens(self) -> tuple[int, int]:
        return sum(c.input_tokens for c in self.calls), sum(c.output_tokens for c in self.calls)

    def step(self, agent: str, what: str, **data: Any) -> None:
        self.steps.append({"agent": agent, "what": what, "t": round(time.time(), 3), **data})

    def to_dict(self) -> dict:
        i, o = self.tokens
        return {
            "usd": self.usd,
            "input_tokens": i,
            "output_tokens": o,
            "calls": [asdict(c) for c in self.calls],
            "steps": self.steps,
        }


def _estimate_tokens(messages: list[Message], options: dict) -> int:
    """Rough input size (about 3.5 characters per token), used only for the pre-call budget check."""
    chars = len(str(options.get("instructions") or ""))
    for m in messages:
        for c in m.contents:
            chars += len(getattr(c, "text", None) or "") + len(str(getattr(c, "arguments", "") or ""))
            chars += len(str(getattr(c, "result", "") or ""))
    tools = options.get("tools") or []
    chars += 400 * (len(tools) if isinstance(tools, list) else 1)
    return int(chars / 3.5) + 50


class BudgetMiddleware(ChatMiddleware):
    """Ask permission before every model request; record its true cost after."""

    def __init__(self, agent: str, trace: Trace, guard: BudgetGuard | None):
        self.agent, self.trace, self.guard = agent, trace, guard

    async def process(self, context: ChatContext, call_next) -> None:
        options = dict(context.options or {})
        model = str(options.get("model") or getattr(context.client, "model", None) or "unknown")
        max_out = int(options.get("max_tokens") or 2000)
        if self.guard is not None:
            self.guard.check(model, _estimate_tokens(list(context.messages), options), max_out)
        t0 = time.perf_counter()
        try:
            await call_next()
        except Exception as exc:
            self.trace.calls.append(CallRecord(self.agent, model, 0, 0, 0.0, time.perf_counter() - t0, error=str(exc)))
            raise
        result = context.result
        usage = getattr(result, "usage_details", None) or {}
        n_in, n_out = int(usage.get("input_token_count") or 0), int(usage.get("output_token_count") or 0)
        if isinstance(context.client, ScriptedChatClient):
            model, usd = "offline", 0.0
        elif self.guard is not None:
            usd = self.guard.record(model, n_in, n_out)
        else:
            p_in, p_out = price_for(model)
            usd = (n_in * p_in + n_out * p_out) / 1e6
        calls = [
            c.name
            for m in (getattr(result, "messages", None) or [])
            for c in m.contents
            if getattr(c, "type", "") == "function_call"
        ]
        self.trace.calls.append(
            CallRecord(self.agent, model, n_in, n_out, round(usd, 6), round(time.perf_counter() - t0, 3), calls)
        )


# ------------------------------------------------------------------------- Foundry client
def foundry_endpoint() -> str:
    """The resource host from AZURE_OPENAI_ENDPOINT, however it was copied from the portal."""
    raw = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    parts = urlsplit(raw if "://" in raw else f"https://{raw}")
    if not parts.netloc:
        raise RuntimeError("Set AZURE_OPENAI_ENDPOINT (see agents/README.md).")
    return f"https://{parts.netloc}"


def foundry_client(settings: AgentSettings, trace: Trace, guard: BudgetGuard, http_client=None):
    """A chat client for one agent on Microsoft Foundry (OpenAI v1 API on the Foundry resource)."""
    from agent_framework.openai import OpenAIChatCompletionClient
    from openai import AsyncOpenAI

    key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("Set AZURE_OPENAI_API_KEY (see agents/README.md).")
    raw = AsyncOpenAI(base_url=f"{foundry_endpoint()}/openai/v1/", api_key=key, http_client=http_client, max_retries=2)
    return OpenAIChatCompletionClient(
        settings.deployment, async_client=raw, middleware=[BudgetMiddleware(settings.name, trace, guard)]
    )


# ------------------------------------------------------------------------ scripted client
# A responder gets (agent name, messages, options) and returns either final text (str) or a
# list of tool calls [(tool_name, args_dict), ...] for the framework to execute.
Responder = Callable[[str, list[Message], dict], "str | list[tuple[str, dict]]"]


class ScriptedChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    """Runs the full agent machinery (tools, middleware, structured output) without a model.

    Used by the tests and by ``--offline`` runs, so the whole desk can be exercised for $0.
    """

    def __init__(self, agent: str, responder: Responder, trace: Trace, **kwargs):
        super().__init__(middleware=[BudgetMiddleware(agent, trace, None)], **kwargs)
        self.agent, self.responder, self.model = agent, responder, "offline"
        self.requests: list[dict] = []

    async def _inner_get_response(self, *, messages, stream, options, **kwargs):
        msgs = list(messages)
        self.requests.append({"messages": msgs, "options": dict(options)})
        out = self.responder(self.agent, msgs, dict(options))
        n_in = sum(
            len(getattr(c, "text", None) or str(getattr(c, "result", "") or "")) for m in msgs for c in m.contents
        )
        if isinstance(out, list):
            contents = [
                Content.from_function_call(call_id=f"call-{len(self.requests)}-{i}", name=n, arguments=json.dumps(a))
                for i, (n, a) in enumerate(out)
            ]
        else:
            contents = [Content.from_text(out)]
        usage = UsageDetails(input_token_count=n_in // 4, output_token_count=len(str(out)) // 4)
        return ChatResponse(
            messages=[Message(role="assistant", contents=contents)], usage_details=usage, model="offline"
        )
