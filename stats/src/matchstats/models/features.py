"""Feature definitions shared by training and live inference.

These functions only read *observable* feed fields. They never touch
``details.truth`` (simulator ground truth), which is reserved for evaluating the
models. A test enforces this.
"""

from __future__ import annotations

import math

from ..pitch import PITCH_LENGTH, distance_to_goal, goal_angle, in_box

SHOT_FEATURES = (
    "intercept",
    "distance",
    "distance_sq",
    "angle",
    "header",
    "penalty",
    "free_kick",
    "under_pressure",
    "defenders_in_lane",
    "distance_x_angle",
)

PASS_FEATURES = (
    "intercept",
    "length",
    "length_sq",
    "high",
    "through",
    "cross",
    "long",
    "back",
    "set_piece",
    "under_pressure",
    "opponent_proximity",
    "progression",
    "end_depth",
    "end_in_box",
    "release_speed",
)

WINPROB_FEATURES = (
    "intercept",
    "elapsed",
    "lateness",
    "goal_diff",
    "goal_diff_x_lateness",
    "abs_goal_diff",
    "abs_goal_diff_x_lateness",
    "xg_diff",
    "xg_diff_x_remaining",
    "red_card_diff",
    "red_card_diff_x_remaining",
)

SET_PIECE_KINDS = {"throw_in", "goal_kick", "kick_off", "free_kick", "keeper_distribution"}
CROSS_KINDS = {"cross", "corner", "free_kick_cross"}


def shot_features(d: dict, x_tf: float, y_tf: float) -> list[float]:
    dist = distance_to_goal(x_tf, y_tf)
    ang = goal_angle(x_tf, y_tf)
    return [
        1.0,
        dist / 10.0,
        (dist / 10.0) ** 2,
        ang,
        1.0 if d.get("body_part") == "head" else 0.0,
        1.0 if d.get("shot_kind") == "penalty" else 0.0,
        1.0 if d.get("shot_kind") == "free_kick" else 0.0,
        1.0 if d.get("under_pressure") else 0.0,
        float(min(d.get("defenders_in_lane", 0), 3)),
        dist / 10.0 * ang,
    ]


def pass_features(d: dict, x_tf: float, y_tf: float, end_x_tf: float, end_y_tf: float) -> list[float]:
    length = float(d.get("length_m", 0.0))
    kind = d.get("pass_kind", "ground")
    nearest = min(float(d.get("nearest_opponent_m", 10.0)), 10.0)
    return [
        1.0,
        length / 10.0,
        (length / 10.0) ** 2,
        1.0 if d.get("height") == "high" else 0.0,
        1.0 if kind == "through" else 0.0,
        1.0 if kind in CROSS_KINDS else 0.0,
        1.0 if kind in ("long", "switch") else 0.0,
        1.0 if kind == "back" else 0.0,
        1.0 if kind in SET_PIECE_KINDS else 0.0,
        1.0 if d.get("under_pressure") else 0.0,
        1.0 - nearest / 10.0,
        float(d.get("progression_m", 0.0)) / 10.0,
        end_x_tf / PITCH_LENGTH,
        1.0 if in_box(end_x_tf, end_y_tf) else 0.0,
        float(d.get("release_speed_kmh", 40.0)) / 50.0,
    ]


def winprob_features(minute: float, goal_diff: int, xg_diff: float, red_diff: int) -> list[float]:
    """Game-state features from the home team's perspective.

    ``lateness`` = 1 / sqrt(minutes left + 1) captures how quickly a lead hardens as time runs out
    (it accelerates sharply in the final minutes); ``remaining`` fades signals that need time to
    turn into goals (chance quality, an extra player).
    """
    f = min(max(minute / 95.0, 0.0), 1.0)
    late = 1.0 / math.sqrt(max(0.0, 94.0 - minute) + 1.0)
    remaining = 1.0 - f
    return [
        1.0,
        f,
        late,
        float(goal_diff),
        goal_diff * late,
        float(abs(goal_diff)),
        abs(goal_diff) * late,
        xg_diff,
        xg_diff * remaining,
        float(red_diff),
        red_diff * remaining,
    ]


def angle_deg(x: float, y: float) -> float:
    return math.degrees(math.atan2(y, x))
