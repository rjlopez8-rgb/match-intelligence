"""Rule-based stand-ins for the seven agents, used by ``--offline`` runs and the tests.

They read the same inputs the models get and return the same structured outputs, built only
from real stats-layer data. That makes the whole desk runnable for $0: useful for building the
UI and the demo while the real models are reserved for what matters. The Analyst even makes
real tool calls, so the tool and receipt machinery is exercised too.
"""

from __future__ import annotations

import contextlib
import json

from agent_framework import Message

SHOT_TYPES = {"goal", "big_chance", "great_save"}


def payload(messages: list[Message]) -> dict:
    for m in messages:
        if m.role == "user":
            text = "".join(getattr(c, "text", None) or "" for c in m.contents)
            if "DATA:\n" in text:
                return json.loads(text.split("DATA:\n", 1)[1])
    return {}


def tool_results(messages: list[Message]) -> dict[str, dict]:
    calls, out = {}, {}
    for m in messages:
        for c in m.contents:
            if getattr(c, "type", "") == "function_call":
                calls[c.call_id] = c.name
            elif getattr(c, "type", "") == "function_result":
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    out[calls.get(c.call_id, "?")] = json.loads(
                        c.result if isinstance(c.result, str) else str(c.result)
                    )
    return out


def _producer(p: dict) -> dict:
    m = p["moment"]
    typ, imp = m["type"], m["importance"]
    aired_what_if = any(r.get("segment") == "what_if" and r.get("period") == m["period"] for r in p.get("rundown", []))
    what_if = None
    if typ in ("goal", "big_chance") and not aired_what_if and (m.get("moment_weight") or 0) >= 20:
        verb = "had not scored" if typ == "goal" else "had scored"
        what_if = (
            f"What if {m['player_names'][0] if m['player_names'] else m['team_name']} {verb} at {m['minute_display']}?"
        )
    segment = (
        "what_if"
        if what_if
        else "replay_breakdown"
        if typ in SHOT_TYPES | {"penalty_awarded", "red_card"}
        else "stat_drop"
        if typ in ("dominance", "period_summary", "top_speed")
        else "live_call"
    )
    priority = "breaking" if typ in ("goal", "red_card") else "high" if imp >= 0.7 else "normal"
    return {
        "cover": imp >= 0.45,
        "segment": segment,
        "angle": m["title"],
        "priority": priority,
        "telestrate": typ in SHOT_TYPES | {"penalty_awarded"},
        "what_if": what_if,
    }


def _flat_numbers(prefix: str, obj, out: list):
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flat_numbers(f"{prefix} {k}".strip(), v, out)


def _spoken(value):
    return round(value, 2) if isinstance(value, float) else value


def _analyst(p: dict, results: dict) -> list[tuple[str, dict]] | dict:
    if "explain_moment" not in results:
        return [("explain_moment", {"moment_id": p["moment_id"]}), ("win_probability", {})]
    ex = results["explain_moment"]
    evidence = [e["event_id"] for e in ex.get("evidence", [])][-3:]
    facts = []
    nums: list = []
    _flat_numbers("", ex.get("facts", {}), nums)
    for name, value in nums[:4]:
        facts.append(
            {
                "statement": f"{name.replace('_', ' ')} was {_spoken(value)}",
                "value": str(value),
                "source_tool": "explain_moment",
                "evidence": evidence,
            }
        )
    if ex.get("moment_weight"):
        facts.append(
            {
                "statement": f"the moment moved win probability by {ex['moment_weight']} points",
                "value": str(ex["moment_weight"]),
                "source_tool": "explain_moment",
                "evidence": evidence,
            }
        )
    wp = results.get("win_probability", {}).get("current")
    if wp:
        facts.append(
            {
                "statement": f"home win probability is now {round(wp['home'] * 100)}%",
                "value": str(wp["home"]),
                "source_tool": "win_probability",
                "evidence": [],
            }
        )
    return {"headline": ex["title"], "facts": facts, "insight": f"{ex['title']}."}


def _telestrator(p: dict) -> dict:
    m, ev = p["moment"], p.get("evidence", [])
    key = ev[-1] if ev else None
    passes = [e for e in ev if e["type"] == "pass" and e.get("location")]
    shapes = []
    if m["player_names"]:
        shapes.append(
            {
                "kind": "ring",
                "player": m["player_names"][0],
                "event_id": None,
                "to_player": None,
                "to_event_id": None,
                "label": m["player_names"][0],
                "color": "highlight",
            }
        )
    if passes:
        shapes.append(
            {
                "kind": "arrow",
                "player": None,
                "event_id": passes[-1]["event_id"],
                "to_player": None,
                "to_event_id": None,
                "label": "the pass",
                "color": "home" if m.get("team_side") == "home" else "away",
            }
        )
    if key and key.get("location"):
        shapes.append(
            {
                "kind": "zone",
                "player": None,
                "event_id": key["event_id"],
                "to_player": None,
                "to_event_id": None,
                "label": None,
                "color": "warning",
            }
        )
    return {"freeze_event_id": key["event_id"] if key else None, "shapes": shapes, "caption": m["title"]}


def _director(p: dict) -> dict:
    m, plan, ev = p["moment"], p["plan"], p.get("evidence", [])
    typ = m["type"]
    replay = plan["segment"] in ("replay_breakdown", "what_if")
    live = (
        "behind"
        if typ in SHOT_TYPES | {"penalty_awarded"}
        else "tactical"
        if typ in ("dominance", "momentum_shift", "tactical_change", "pressing_surge")
        else "broadcast"
    )
    shots = []
    if replay:
        focus = m["player_names"][0] if m["player_names"] else None
        shots = [
            {
                "camera": "tactical",
                "focus_player": None,
                "seconds": 4.0,
                "slow_motion": False,
                "purpose": "the build-up",
            },
            {
                "camera": "behind",
                "focus_player": focus,
                "seconds": 3.0,
                "slow_motion": True,
                "purpose": "the decisive touch",
            },
            {
                "camera": "sweep" if typ == "goal" else "player",
                "focus_player": focus,
                "seconds": 4.0,
                "slow_motion": False,
                "purpose": "the reaction",
            },
        ]
    graphic = {
        "goal": "win_probability",
        "big_chance": "xg",
        "great_save": "xg",
        "dominance": "momentum",
        "momentum_shift": "momentum",
        "period_summary": "scoreboard",
    }.get(typ, "none")
    return {
        "live_camera": live,
        "replay": replay,
        "replay_from_event_id": ev[0]["event_id"] if replay and ev else None,
        "shots": shots,
        "show_telestration": bool(plan.get("telestrate")),
        "graphic": graphic,
    }


def _commentator(p: dict) -> dict:
    facts = p["facts"]
    lines = [{"text": p["headline"] + "!", "facts_used": []}]
    for i, f in enumerate(facts[:2]):
        lines.append({"text": f"{f['statement'][0].upper()}{f['statement'][1:]}.", "facts_used": [i]})
    return {"lines": lines}


def _scenario(p: dict) -> dict:
    ev = [e for e in p.get("evidence", []) if e["type"] in ("shot", "goal")]
    target = ev[0]["event_id"] if ev else (p.get("evidence") or [{"event_id": "unknown"}])[-1]["event_id"]
    scored = p["moment"]["type"] == "goal"
    return {
        "question": p["what_if"],
        "intervention": "shot_misses" if scored else "shot_scores",
        "event_id": target,
        "simulations": 20,
        "hypothesis": "The result would likely change, because the moment carried a large swing in win probability.",
    }


def responder(agent: str, messages: list[Message], options: dict):
    p = payload(messages)
    if agent == "producer":
        out = _producer(p)
    elif agent == "analyst":
        out = _analyst(p, tool_results(messages))
        if isinstance(out, list):
            return out
    elif agent == "telestrator":
        out = _telestrator(p)
    elif agent == "director":
        out = _director(p)
    elif agent == "commentator":
        out = _commentator(p)
    elif agent == "factchecker":
        out = {"approved": True, "issues": []}
    elif agent == "scenario":
        out = _scenario(p)
    else:
        raise ValueError(agent)
    return json.dumps(out)
