"""Pitch geometry shared by every metric.

Feed coordinates are absolute (105 x 68 m, origin bottom-left). Most football
metrics are easier in the *attacking frame* of the team on the ball, where that
team always attacks towards x = 105. The direction of play comes from the match
metadata (home attacks right in period 1).
"""

from __future__ import annotations

import math

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
BOX_DEPTH = 16.5
BOX_HALF_WIDTH = 20.16
GOAL_HALF_WIDTH = 3.66


def attacking_frame(x: float, y: float, attacks_right: bool) -> tuple[float, float]:
    return (x, y) if attacks_right else (PITCH_LENGTH - x, PITCH_WIDTH - y)


def in_box(x_tf: float, y_tf: float) -> bool:
    return x_tf >= PITCH_LENGTH - BOX_DEPTH and abs(y_tf - PITCH_WIDTH / 2) <= BOX_HALF_WIDTH


def third(x_tf: float) -> str:
    return "defensive" if x_tf < 35.0 else ("middle" if x_tf < 70.0 else "attacking")


def lane(y_tf: float) -> str:
    """Lane from the attacking team's perspective (facing x = 105, small y is the right flank)."""
    return "right" if y_tf < 22.67 else ("centre" if y_tf < 45.33 else "left")


def zone_id(x_tf: float, y_tf: float, cols: int = 6, rows: int = 3) -> int:
    cx = min(cols - 1, max(0, int(x_tf / (PITCH_LENGTH / cols))))
    cy = min(rows - 1, max(0, int(y_tf / (PITCH_WIDTH / rows))))
    return cy * cols + cx


def distance_to_goal(x_tf: float, y_tf: float) -> float:
    return math.hypot(PITCH_LENGTH - x_tf, PITCH_WIDTH / 2 - y_tf)


def goal_angle(x_tf: float, y_tf: float) -> float:
    dx = max(PITCH_LENGTH - x_tf, 0.05)
    a1 = math.atan2(PITCH_WIDTH / 2 + GOAL_HALF_WIDTH - y_tf, dx)
    a2 = math.atan2(PITCH_WIDTH / 2 - GOAL_HALF_WIDTH - y_tf, dx)
    return abs(a1 - a2)


def threat(x_tf: float, y_tf: float) -> float:
    """Expected-threat style value of having the ball at a location (0 to ~0.3).

    Our own transparent approximation: almost flat in your own half and rising
    steeply towards the centre of the opponent's box.
    """
    d = distance_to_goal(x_tf, y_tf)
    angle_factor = min(1.0, max(0.25, goal_angle(x_tf, y_tf) / 0.6))
    return 0.005 + 0.015 * (x_tf / PITCH_LENGTH) + 0.28 * math.exp(-d / 10.5) * angle_factor
