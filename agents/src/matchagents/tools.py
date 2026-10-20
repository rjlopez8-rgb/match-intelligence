"""The stats layer's 11 queries, handed to agents as tools, with a receipt for every call.

Agents never calculate a number themselves: they ask these tools. Each call and its exact
result goes into the ``Ledger``, and the fact-checker only accepts numbers that appear in
the ledger (or in the moment the stats layer detected). That is what "every claim has
receipts" means in practice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent_framework import FunctionTool
from matchstats.queries import TOOLS, MatchQueries, QueryError

MAX_RESULT_CHARS = 14_000


@dataclass
class Receipt:
    agent: str
    tool: str
    args: dict
    result: Any
    error: str | None = None


@dataclass
class Ledger:
    receipts: list[Receipt] = field(default_factory=list)

    def add(self, r: Receipt) -> None:
        self.receipts.append(r)

    def results(self) -> list[Any]:
        return [r.result for r in self.receipts if r.error is None]

    def summary(self) -> list[dict]:
        return [{"agent": r.agent, "tool": r.tool, "args": r.args, "ok": r.error is None} for r in self.receipts]


def _compact(obj: Any) -> str:
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + '..."(truncated: ask a narrower question)"'
    return text


def stats_tools(queries: MatchQueries, ledger: Ledger, agent: str, only: tuple[str, ...] | None = None) -> list:
    """FunctionTools for an agent; ``only`` limits which queries it may use."""
    tools = []
    for name, spec in TOOLS.items():
        if only and name not in only:
            continue

        def run(_name=name, **kwargs):
            args = {k: v for k, v in kwargs.items() if v is not None}
            try:
                result = queries.call(_name, args)
            except (QueryError, TypeError, ValueError) as exc:
                ledger.add(Receipt(agent, _name, args, None, str(exc)))
                return _compact({"error": str(exc)})
            ledger.add(Receipt(agent, _name, args, result))
            return _compact(result)

        tools.append(FunctionTool(name=name, description=spec["description"], func=run, input_model=spec["parameters"]))
    return tools
