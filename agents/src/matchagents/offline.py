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

SHOT_TYPES = {"goal", "big_chance", "great_save", "woodwork"}


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
    swing = max(m.get("moment_weight") or 0, (m.get("facts") or {}).get("potential_swing_pts") or 0)
    if typ in ("goal", "big_chance", "great_save", "woodwork") and not aired_what_if and swing >= 20:
        verb = "had not scored" if typ == "goal" else "had scored"
        who = (m.get("facts") or {}).get("shooter") or (m["player_names"][0] if m["player_names"] else m["team_name"])
        what_if = f"What if {who} {verb} at {m['minute_display']}?"
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
    shot = next((e for e in reversed(ex.get("evidence", [])) if e["type"] == "shot"), None)
    if shot and "options" not in results:
        return [("options", {"event_id": shot["event_id"]})]
    evidence = [e["event_id"] for e in ex.get("evidence", [])][-3:]
    f = ex.get("facts", {}) or {}
    facts = []

    def add(statement, value, tool="explain_moment", ev=evidence):
        facts.append({"statement": statement, "value": str(value), "source_tool": tool, "evidence": ev})

    # canonical openings: the offline Commentator recognises them to write in every language
    if f.get("xg") is not None:
        add(f"The chance was worth {f['xg']:.2f} xG", f["xg"])
    if f.get("distance_m"):
        add(f"The shot came from {f['distance_m']:.0f} metres", f["distance_m"])
    if f.get("shot_speed_kmh"):
        add(f"The ball flew at {f['shot_speed_kmh']:.0f} km/h", f["shot_speed_kmh"])
    if ex.get("moment_weight"):
        add(f"The moment moved win probability by {ex['moment_weight']:.0f} points", ex["moment_weight"])
    for side in ("home", "away"):
        t = f.get(side) if isinstance(f.get(side), dict) else None
        if t and t.get("possession") is not None:
            add(f"Possession {t['team']}: {round(t['possession'] * 100)}%", t["possession"])
            add(f"Expected goals {t['team']}: {t['xg']:.2f}", t["xg"])
    if not facts:
        nums: list = []
        _flat_numbers("", f, nums)
        for name, value in nums[:4]:
            add(f"{name.replace('_', ' ')} was {_spoken(value)}", value)
    wp = results.get("win_probability", {}).get("current")
    if wp:
        add(f"Home win probability is now {round(wp['home'] * 100)}%", wp["home"], "win_probability", [])
    decision = ""
    o = results.get("options")
    if o and isinstance(o, dict) and o.get("choice"):
        ch, best = o["choice"], (o.get("teammates") or [None])[0]
        if o.get("better_option_exists") and best:
            decision = (f"{o['event']['player']} shot at {ch.get('xg')} xG; {best['name']} was free with "
                        f"{best['space_m']} m of space: a pass there arrives {round(best['p_complete'] * 100)}% of "
                        f"the time and he would shoot at {best['xg_from_there']} xG.")
            pc = round(best["p_complete"] * 100)
            add(f"Better option: {best['name']}, pass {pc}%", best["p_complete"], "options", [])
        elif best:
            decision = (f"{o['event']['player']} was right to shoot: his chance was {ch.get('xg')} xG and the best "
                        f"pass, to {best['name']}, led to only {best['xg_from_there']} xG.")
            x = best["xg_from_there"]
            add(f"Right call: best alternative only {x:.2f} xG", x, "options", [])
    return {"headline": ex["title"], "facts": facts, "insight": f"{ex['title']}.", "decision": decision}


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


DIRECTOR_CUTS = {  # (camera, slow motion, purpose) per style: the same moment, cut for its audience
    "fan": [("sweep", False, "the reaction"), ("player", True, "the hero")],
    "analyst": [("skycam", False, "the build-up"), ("behind", True, "the shot"), ("tactical", False, "the shape")],
    "tactician": [("tactical", False, "the shape"), ("skycam", False, "the move"), ("behind", True, "the finish")],
    "academy": [("player", True, "the decision"), ("behind", True, "the outcome"), ("tactical", False, "the options")],
}
GRAPHIC = {"fan": "scoreboard", "analyst": "xg", "tactician": "momentum", "academy": "player_card"}


def _director(p: dict) -> dict:
    m, plan, ev = p["moment"], p["plan"], p.get("evidence", [])
    typ, style = m["type"], p.get("viewer_style") or "fan"
    replay = plan["segment"] in ("replay_breakdown", "what_if")
    live = (
        "behind"
        if typ in SHOT_TYPES | {"penalty_awarded"}
        else "tactical"
        if typ in ("dominance", "momentum_shift", "tactical_change", "pressing_surge", "period_summary")
        else "broadcast"
    )
    if style == "fan" and typ == "goal":
        live = "sweep"
    elif style == "tactician" and live == "broadcast":
        live = "tactical"
    focus = m["player_names"][0] if m["player_names"] else None
    shots = []
    if replay:
        for cam, slow, purpose in DIRECTOR_CUTS.get(style, DIRECTOR_CUTS["fan"]):
            shots.append({"camera": cam, "focus_player": focus if cam in ("player", "behind", "sweep") else None,
                          "seconds": 4.0 if slow else 3.5, "slow_motion": slow, "purpose": purpose})
    graphic = GRAPHIC.get(style, "none") if typ in SHOT_TYPES else {
        "dominance": "momentum", "momentum_shift": "momentum", "period_summary": "scoreboard"}.get(typ, "none")
    return {
        "live_camera": live,
        "replay": replay,
        "replay_from_event_id": ev[0]["event_id"] if replay and ev else None,
        "shots": shots,
        "show_telestration": bool(plan.get("telestrate")) or style in ("tactician", "academy"),
        "graphic": graphic,
    }


from . import phrasebook  # noqa: E402  (kept beside the responders that use it)


def _commentator(p: dict) -> dict:
    return phrasebook.commentary(p)


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
