"""The exact HTTP requests the desk would send to Microsoft Foundry, captured without a network.

A fake Foundry endpoint (httpx MockTransport) answers like the real Chat Completions API,
using the offline agents for content. This proves the wire format (URL, model deployment,
token limit, reasoning effort, strict JSON schema, tools) and that the budget guard records
every request and refuses to send once the cap is reached.
"""

import json

import httpx2 as httpx
from agent_framework import Content, Message
from conftest import GOAL, SAMPLE, run

from matchagents import offline
from matchagents.budget import BudgetGuard
from matchagents.desk import Desk
from matchagents.session import MatchSession


def _to_framework(messages: list[dict]) -> list[Message]:
    out = []
    for m in messages:
        if m["role"] == "user":
            text = m["content"] if isinstance(m["content"], str) else "".join(p.get("text", "") for p in m["content"])
            out.append(Message(role="user", contents=[text]))
        elif m["role"] == "assistant" and m.get("tool_calls"):
            out.append(
                Message(
                    role="assistant",
                    contents=[
                        Content.from_function_call(
                            call_id=c["id"], name=c["function"]["name"], arguments=c["function"]["arguments"]
                        )
                        for c in m["tool_calls"]
                    ],
                )
            )
        elif m["role"] == "tool":
            out.append(
                Message(
                    role="tool", contents=[Content.from_function_result(call_id=m["tool_call_id"], result=m["content"])]
                )
            )
    return out


AGENT_BY_PROMPT = {
    "You are the Producer": "producer",
    "You are the Analyst": "analyst",
    "You are the Scenario agent": "scenario",
    "You are the Telestrator": "telestrator",
    "You are the Director": "director",
    "You are the Commentator": "commentator",
    "You are the Fact-checker": "factchecker",
}


class FakeFoundry:
    def __init__(self):
        self.requests: list[tuple[httpx.Request, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request, body))
        system = body["messages"][0]["content"]
        system = system if isinstance(system, str) else "".join(p.get("text", "") for p in system)
        agent = next(a for prefix, a in AGENT_BY_PROMPT.items() if system.startswith(prefix))
        out = offline.responder(agent, _to_framework(body["messages"]), {})
        if isinstance(out, list):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                    for i, (n, a) in enumerate(out)
                ],
            }
            finish = "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": out}, "stop"
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(self.requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200},
            },
        )


def _desk(fake, cap=5.0, tmp=None):
    session = MatchSession(SAMPLE)
    moment = session.advance_to(GOAL)
    guard = BudgetGuard(daily_cap_usd=cap, path=tmp / "usage.json")
    desk = Desk(session, guard=guard, http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake)))
    return desk, moment, guard


def test_wire_format_and_budget_records(tmp_path):
    fake = FakeFoundry()
    desk, moment, guard = _desk(fake, tmp=tmp_path)
    seg = run(desk.cover(moment))

    assert seg["aired"] and not seg["commentary"]["fallback"], seg["errors"]
    # the analyst makes 3 calls: stats tools, then options on the shot, then the brief
    assert len(fake.requests) == seg["cost"]["model_calls"] == 9
    for req, body in fake.requests:
        assert str(req.url) == "https://test-resource.services.ai.azure.com/openai/v1/chat/completions"
        assert req.headers["authorization"] == "Bearer test-key"
        assert body["model"] == "gpt-5-mini-1"
        assert body["max_completion_tokens"] >= 1500 and "max_tokens" not in body
        assert body["reasoning_effort"] == "low"
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
    analyst = [b for _, b in fake.requests if "tools" in b]
    assert analyst and {t["function"]["name"] for t in analyst[0]["tools"]} >= {"explain_moment", "win_probability"}
    # every request was priced and recorded before the next one
    assert abs(guard.spent_today() - seg["cost"]["usd"]) < 1e-9
    assert seg["cost"]["usd"] > 0


def test_budget_cap_stops_requests_and_the_broadcast_survives(tmp_path):
    fake = FakeFoundry()
    desk, moment, guard = _desk(fake, cap=0.012, tmp=tmp_path)  # enough for a few calls only
    seg = run(desk.cover(moment))
    assert any("budget" in e.lower() for e in seg["errors"])
    assert 0 < len(fake.requests) < 8  # it stopped before sending the rest
    assert guard.spent_today() < 0.012  # the cap held
    # the stats layer's own factual title airs instead of anything unverified
    assert seg["commentary"]["lines"] == [moment["title"]] and seg["commentary"]["fallback"]


def test_model_per_agent_is_a_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("MATCH_MODEL_ANALYST", "gpt-6-sol-1")
    fake = FakeFoundry()
    desk, moment, _ = _desk(fake, tmp=tmp_path)
    run(desk.cover(moment))
    models = {b["model"] for _, b in fake.requests if "tools" in b}
    others = {b["model"] for _, b in fake.requests if "tools" not in b}
    assert models == {"gpt-6-sol-1"} and others == {"gpt-5-mini-1"}
