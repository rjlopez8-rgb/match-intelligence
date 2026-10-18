"""Event vocabulary and clock formatting shared by the engine and the schema."""

from __future__ import annotations

from enum import Enum

SCHEMA_VERSION = "1.0.0"


class EventType(str, Enum):
    MATCH_START = "match_start"
    PERIOD_START = "period_start"
    KICK_OFF = "kick_off"
    PASS = "pass"
    CARRY = "carry"
    DRIBBLE = "dribble"
    PRESSURE = "pressure"
    TACKLE = "tackle"
    INTERCEPTION = "interception"
    BALL_RECOVERY = "ball_recovery"
    MISCONTROL = "miscontrol"
    CLEARANCE = "clearance"
    BLOCK = "block"
    AERIAL_DUEL = "aerial_duel"
    SHOT = "shot"
    SAVE = "save"
    KEEPER_CLAIM = "keeper_claim"
    GOAL = "goal"
    FOUL = "foul"
    CARD = "card"
    BALL_OUT = "ball_out"
    OFFSIDE = "offside"
    POSSESSION_CHANGE = "possession_change"
    SUBSTITUTION = "substitution"
    TACTICAL_CHANGE = "tactical_change"
    ADDED_TIME = "added_time"
    PERIOD_END = "period_end"
    MATCH_END = "match_end"


PERIOD_LENGTH_S = 45 * 60


def match_seconds(period: int, t: float) -> float:
    """Seconds on the broadcast clock (second half starts at 45:00)."""
    return (period - 1) * PERIOD_LENGTH_S + t


def clock_strings(period: int, t: float) -> dict:
    """Return minute, second, broadcast clock and minute label for a moment in the match."""
    if t > PERIOD_LENGTH_S:
        base = 45 * period
        extra = t - PERIOD_LENGTH_S
        return {
            "minute": base + int(extra // 60),
            "second": int(extra % 60),
            "clock": f"{base}:00+{int(extra // 60)}:{int(extra % 60):02d}",
            "minute_display": f"{base}+{int(extra // 60) + 1}'",
        }
    ms = match_seconds(period, t)
    return {
        "minute": int(ms // 60),
        "second": int(ms % 60),
        "clock": f"{int(ms // 60):02d}:{int(ms % 60):02d}",
        "minute_display": f"{int(ms // 60) + 1}'",
    }
