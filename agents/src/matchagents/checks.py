"""The parts of the desk that are code, not model: number checking and drawing coordinates.

``check_numbers`` is the first half of the fact-check. Every number, percentage, minute and
scoreline in the commentary must match a number the stats layer actually produced (in the
ledger of tool results or the detected moment). It cannot be talked round, which is why it
runs before, and independently of, the model-based fact-checker.

``resolve_telestration`` turns the Telestrator's references ("ring Duarte", "arrow along event
...-00412") into pitch coordinates from tracking data, so drawings land where players really were.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from matchstats.queries import QueryError

from .schemas import Commentary, Telestration

# ----------------------------------------------------------------------- number checking
_SCORE = re.compile(r"(?<!\d)(?<!\d\.)(\d{1,2})\s?[-–]\s?(\d{1,2})(?!\d|\.\d)")
_PCT = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s?(?:%|percent\b|per cent\b|por ciento\b|pour cent\b)")
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")
_NUM_IN_STR = re.compile(r"\d+(?:\.\d+)?")
STRUCTURAL = {45.0, 90.0}  # halves and full time are not claims


def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def evidence_numbers(sources: Iterable[Any]) -> tuple[list[float], set[tuple[int, int]]]:
    """All numbers and scorelines present in the evidence."""
    nums: list[float] = []
    scores: set[tuple[int, int]] = set()
    for src in sources:
        for node in _walk(src):
            if isinstance(node, bool) or node is None:
                continue
            if isinstance(node, (int, float)):
                nums.append(float(node))
            elif isinstance(node, str):
                nums.extend(float(x) for x in _NUM_IN_STR.findall(node))
                for a, b in _SCORE.findall(node):
                    scores.add((int(a), int(b)))
        # score dicts {"home": h, "away": a}
        stack = [src]
        while stack:
            n = stack.pop()
            if isinstance(n, dict):
                h, a = n.get("home"), n.get("away")
                if isinstance(h, int) and isinstance(a, int) and not isinstance(h, bool):
                    scores.add((h, a))
                stack.extend(n.values())
            elif isinstance(n, list):
                stack.extend(n)
    return nums, scores


def _matches(written: str, values: list[float], percent: bool) -> bool:
    x = float(written)
    decimals = len(written.split(".")[1]) if "." in written else 0
    tol = 0.5 * 10**-decimals + 1e-9
    if not percent and decimals == 0:
        tol = max(tol, 0.5)
    for v in values:
        v = abs(v)  # the sign is read as words ("swung by 0.43 towards..."); the model check covers direction
        candidates = [v]
        if percent and v <= 1.0:
            candidates.append(v * 100)
        if any(abs(c - x) <= tol for c in candidates):
            return True
    return False


def _normalise(text: str) -> str:
    """Read numbers the way the viewer's language writes them: 0,74 is 0.74 and 1,234 is 1234."""
    text = re.sub(r"(?<![\d.,])([1-9]\d{0,2}),(?=\d{3}(?!\d))", r"\1", text)  # thousands separators
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)  # decimal commas
    return text.replace("\u00a0", " ").replace("\u202f", " ")


def _texts(commentary: Commentary) -> list[tuple[int | str, str]]:
    """Everything the viewer will read: the lines, and the on-screen headline, caption and what-if question."""
    out: list[tuple[int | str, str]] = [(i, line.text) for i, line in enumerate(commentary.lines)]
    for field in ("headline", "caption", "question"):
        text = getattr(commentary, field, None)
        if text:
            out.append((field, text))
    return out


def check_numbers(commentary: Commentary, sources: Iterable[Any]) -> list[dict]:
    """Issues for every number in the commentary that the evidence does not contain."""
    values, scores = evidence_numbers(sources)
    issues = []
    for i, raw in _texts(commentary):
        text = _normalise(raw)
        for a, b in _SCORE.findall(text):
            if (int(a), int(b)) not in scores and (int(b), int(a)) not in scores:
                issues.append({"line": i, "claim": f"{a}-{b}", "problem": "scoreline not in the evidence"})
        text = _SCORE.sub(" ", text)
        for p in _PCT.findall(text):
            if not _matches(p, values, percent=True):
                issues.append({"line": i, "claim": f"{p}%", "problem": "percentage not in the evidence"})
        text = _PCT.sub(" ", text)
        for n in _NUM.findall(text):
            if float(n) in STRUCTURAL:
                continue
            if not _matches(n, values, percent=False):
                issues.append({"line": i, "claim": n, "problem": "number not in the evidence"})
    return issues


_DATA = re.compile(r"[{}\[\]]|'(?:home|away|draw)'|\b(?:home|away|draw)\s*:|\b[a-z]+_[a-z_]+\b")
_LONG_DECIMAL = re.compile(r"(?<![\w.])\d+\.\d{3,}")


def check_style(commentary: Commentary, max_numbers: int = 3) -> list[dict]:
    """Issues for lines that would sound wrong on air, even if every number is true. ``max_numbers`` is per line:
    3 for spoken commentary; written recaps allow more (a what-if sentence carries a minute, two shares and a count)."""
    issues = []
    for i, text in _texts(commentary):
        if m := _DATA.search(text):
            issues.append({"line": i, "claim": m.group(0), "problem": "pasted data: say it in words"})
        for d in _LONG_DECIMAL.findall(text):
            issues.append({"line": i, "claim": d, "problem": "unrounded number: round it for speech"})
        if len(_NUM.findall(_SCORE.sub(" ", _normalise(text)))) > max_numbers:
            issues.append({"line": i, "claim": text[:40], "problem": "too many numbers in one line"})
    return issues


# ------------------------------------------------------------------- telestration resolver
def _loc(event: dict | None, key: str = "location") -> tuple[float, float] | None:
    if not event:
        return None
    loc = event.get(key) if key == "location" else (event.get("details") or {}).get(key)
    if not loc:
        return None
    return float(loc["x"]), float(loc["y"])


def resolve_telestration(tele: Telestration, session, moment: dict) -> dict:
    """Pitch coordinates (x 0-105, y 0-68, same as events) for every shape that can be placed."""
    ctx, q = session.ctx, session.queries
    freeze = ctx.by_id.get(tele.freeze_event_id or "")
    period, t = (freeze["period"], freeze["t"]) if freeze else (moment["period"], moment["t"])

    def pid(ref: str | None) -> str | None:
        if not ref:
            return None
        try:
            return q.player(ref)
        except QueryError:
            return None

    def pos(player_ref: str | None, event_id: str | None, end: bool = False):
        p = pid(player_ref)
        if p:
            xy = session.player_xy(p, period, t)
            if xy:
                return xy, p
        ev = ctx.by_id.get(event_id or "")
        if ev:
            return (_loc(ev, "end_location") if end else None) or _loc(ev), p or ev.get("player_id")
        return None, p

    shapes, dropped = [], []
    for s in tele.shapes:
        start, who = pos(s.player, s.event_id)
        out = {"kind": s.kind, "color": s.color, "label": s.label, "player_id": who}
        if s.kind in ("arrow", "run"):
            if s.to_player or s.to_event_id:
                end, _ = pos(s.to_player, s.to_event_id)
            else:
                end = _loc(ctx.by_id.get(s.event_id or ""), "end_location")
            if s.kind == "arrow" and s.event_id and not s.player:
                start = _loc(ctx.by_id.get(s.event_id))
            if start and end and math.dist(start, end) > 0.5:
                out.update(x=start[0], y=start[1], x2=end[0], y2=end[1])
                shapes.append(out)
                continue
        elif start:
            out.update(x=start[0], y=start[1], r=6.0 if s.kind == "zone" else 1.6)
            shapes.append(out)
            continue
        dropped.append(s.model_dump())
    return {
        "freeze": {"event_id": tele.freeze_event_id if freeze else None, "period": period, "t": t},
        "caption": tele.caption,
        "shapes": shapes,
        "dropped": dropped,
    }
