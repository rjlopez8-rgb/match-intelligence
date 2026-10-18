"""Geometry helpers. All coordinates are metres on a 105 x 68 pitch."""

from __future__ import annotations

import math

import numpy as np

from .params import GOAL_HALF_WIDTH, PENALTY_BOX_DEPTH, PENALTY_BOX_HALF_WIDTH, PITCH_LENGTH, PITCH_WIDTH


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def to_team_frame(x: float, y: float, direction: int) -> tuple[float, float]:
    """Convert absolute coords to a frame where the team attacks towards x = 105."""
    if direction == 1:
        return x, y
    return PITCH_LENGTH - x, PITCH_WIDTH - y


def from_team_frame(x: float, y: float, direction: int) -> tuple[float, float]:
    return to_team_frame(x, y, direction)  # the transform is its own inverse


def dist_to_goal(x: float, y: float) -> float:
    """Distance from a team-frame point to the centre of the goal being attacked."""
    return math.hypot(PITCH_LENGTH - x, PITCH_WIDTH / 2 - y)


def goal_angle(x: float, y: float) -> float:
    """Angle (radians) subtended by the goal mouth from a team-frame point."""
    dx = PITCH_LENGTH - x
    if dx <= 0.05:
        return 0.05 if abs(y - PITCH_WIDTH / 2) > GOAL_HALF_WIDTH else math.pi * 0.95
    a1 = math.atan2(PITCH_WIDTH / 2 + GOAL_HALF_WIDTH - y, dx)
    a2 = math.atan2(PITCH_WIDTH / 2 - GOAL_HALF_WIDTH - y, dx)
    return abs(a1 - a2)


def in_attacking_box(x: float, y: float) -> bool:
    """True if a team-frame point is inside the penalty area being attacked."""
    return x >= PITCH_LENGTH - PENALTY_BOX_DEPTH and abs(y - PITCH_WIDTH / 2) <= PENALTY_BOX_HALF_WIDTH


def in_own_box(x: float, y: float) -> bool:
    return x <= PENALTY_BOX_DEPTH and abs(y - PITCH_WIDTH / 2) <= PENALTY_BOX_HALF_WIDTH


def in_pitch(x: float, y: float) -> bool:
    return 0.0 <= x <= PITCH_LENGTH and 0.0 <= y <= PITCH_WIDTH


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (distance, t) where t in [0, 1] is the projection of p onto segment ab."""
    ab = b - a
    denom = float(ab @ ab)
    if denom < 1e-9:
        return float(np.linalg.norm(p - a)), 0.0
    t = clamp(float((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj)), t


def time_to_reach(
    distance: float, top_speed: float, accel: float, current_speed: float = 0.0, reaction: float = 0.2
) -> float:
    """Time for a player to cover ``distance`` from a standing-ish start."""
    if distance <= 0:
        return reaction
    v0 = min(current_speed, top_speed)
    d_acc = (top_speed**2 - v0**2) / (2 * accel)
    if distance <= d_acc:
        t = (-v0 + math.sqrt(v0 * v0 + 2 * accel * distance)) / accel
    else:
        t = (top_speed - v0) / accel + (distance - d_acc) / top_speed
    return reaction + t


def boundary_crossing(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, str] | None:
    """If segment start->end leaves the pitch, return the exit point and which line it crossed.

    The line is one of ``"left_goal_line"``, ``"right_goal_line"``, ``"bottom_touchline"``, ``"top_touchline"``.
    """
    if in_pitch(float(end[0]), float(end[1])):
        return None
    d = end - start
    best_t, best_line = 2.0, ""
    for coord, bound, name in (
        (0, 0.0, "left_goal_line"),
        (0, PITCH_LENGTH, "right_goal_line"),
        (1, 0.0, "bottom_touchline"),
        (1, PITCH_WIDTH, "top_touchline"),
    ):
        if abs(d[coord]) < 1e-9:
            continue
        t = (bound - start[coord]) / d[coord]
        if 0.0 <= t <= 1.0 and t < best_t:
            p = start + t * d
            other = 1 - coord
            lim = PITCH_WIDTH if other == 1 else PITCH_LENGTH
            if -1e-6 <= p[other] <= lim + 1e-6:
                best_t, best_line = t, name
    if not best_line:
        return np.clip(end, [0, 0], [PITCH_LENGTH, PITCH_WIDTH]), "bottom_touchline"
    p = start + best_t * d
    p = np.clip(p, [0, 0], [PITCH_LENGTH, PITCH_WIDTH])
    return p, best_line


def angle_deg(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx))
