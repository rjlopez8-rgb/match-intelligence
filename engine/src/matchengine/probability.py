"""Probability models used by the engine.

These are the engine's *ground truth* models: they decide what actually happens.
Downstream analytics never see them directly (only their outcomes), which means
our own xG and pass-difficulty models can be evaluated against them later.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import clamp, dist_to_goal, goal_angle, sigmoid
from .params import PITCH_LENGTH, PITCH_WIDTH, EngineParams


def position_value(x: float, y: float) -> float:
    """Value of having the ball at a team-frame location (roughly: chance a goal follows).

    Shaped like expected-threat grids: almost flat in your own half, rising sharply
    towards the centre of the opponent's box.
    """
    d = dist_to_goal(x, y)
    ang = goal_angle(x, y)
    angle_factor = clamp(ang / 0.6, 0.25, 1.0)
    return 0.006 + 0.016 * (x / PITCH_LENGTH) + 0.30 * math.exp(-d / 10.5) * angle_factor


def position_value_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = PITCH_LENGTH - x
    dy = PITCH_WIDTH / 2 - y
    d = np.hypot(dx, dy)
    a1 = np.arctan2(PITCH_WIDTH / 2 + 3.66 - y, np.maximum(dx, 0.05))
    a2 = np.arctan2(PITCH_WIDTH / 2 - 3.66 - y, np.maximum(dx, 0.05))
    ang = np.abs(a1 - a2)
    af = np.clip(ang / 0.6, 0.25, 1.0)
    return 0.006 + 0.016 * (x / PITCH_LENGTH) + 0.30 * np.exp(-d / 10.5) * af


def xg_unblocked(
    p: EngineParams,
    x: float,
    y: float,
    *,
    header: bool,
    pressure: float,
    finishing: int,
    composure: int,
    keeper: int,
    big_chance: bool,
    kind: str = "open_play",
) -> float:
    """Probability that an unblocked shot from team-frame (x, y) is scored."""
    if kind == "penalty":
        return clamp(0.76 + (finishing - 75) / 500 + (composure - 70) / 600 - (keeper - 75) / 600, 0.62, 0.88)
    d = dist_to_goal(x, y)
    ang = goal_angle(x, y)
    z = p.xg_intercept + p.xg_dist * d + p.xg_angle * ang
    if header:
        z += p.xg_header
    if kind == "free_kick":
        z -= 0.35
    z += p.xg_pressure * pressure
    z += (finishing - 72) / 45.0 + (composure - 70) / 90.0 * (0.5 + pressure)
    z -= (keeper - 75) / 55.0
    if big_chance:
        z += 0.4
    return clamp(sigmoid(z), 0.003, 0.92)


def player_times_to_points(
    pos: np.ndarray,
    vel: np.ndarray,
    top: np.ndarray,
    accel: np.ndarray,
    points: np.ndarray,
    reach: float,
    reaction: float = 0.22,
) -> np.ndarray:
    """Seconds for each player (n) to get within ``reach`` of each point (m). Shape (n, m)."""
    diff = points[None, :, :] - pos[:, None, :]
    d = np.linalg.norm(diff, axis=2)
    d = np.maximum(d - reach, 0.0)
    # Component of current velocity towards the point gives a head start.
    unit = diff / np.maximum(np.linalg.norm(diff, axis=2, keepdims=True), 1e-6)
    v_along = np.clip(np.einsum("nmk,nk->nm", unit, vel), 0.0, None)
    v0 = np.minimum(v_along, top[:, None])
    a = accel[:, None]
    vmax = top[:, None]
    d_acc = (vmax**2 - v0**2) / (2 * a)
    t_acc = (-v0 + np.sqrt(v0**2 + 2 * a * d)) / a
    t_cruise = (vmax - v0) / a + (d - d_acc) / vmax
    t = np.where(d <= d_acc, t_acc, t_cruise)
    return reaction + t


def ground_ball_time(v0: float, decel: float, dist: np.ndarray) -> np.ndarray:
    """Time for a decelerating ground ball to travel ``dist`` metres."""
    disc = np.maximum(v0 * v0 - 2 * decel * dist, 0.05)
    return (v0 - np.sqrt(disc)) / decel


def ground_pass_speed(p: EngineParams, distance: float, arrival: float | None = None) -> float:
    arr = p.pass_arrival_speed if arrival is None else arrival
    return math.sqrt(arr * arr + 2 * p.ground_decel * distance)


def aerial_profile(distance: float, kind: str) -> tuple[float, float]:
    """Return (horizontal speed m/s, peak height m) for a lofted ball."""
    if kind in ("cross", "corner"):
        return 17.0 + 0.08 * distance, clamp(2.5 + 0.12 * distance, 3.0, 7.0)
    if kind == "clearance":
        return 19.0 + 0.06 * distance, clamp(0.22 * distance, 6.0, 18.0)
    return 13.5 + 0.12 * distance, clamp(0.17 * distance, 3.5, 14.0)


def intercept_probabilities(
    p: EngineParams, ball_times: np.ndarray, opp_times: np.ndarray, interceptable: np.ndarray
) -> np.ndarray:
    """Per-opponent probability of cutting out a pass.

    ``ball_times`` (m,) is when the ball reaches each sample point; ``opp_times`` (n, m) when each
    opponent can get there; ``interceptable`` (m,) masks points where the ball is low enough.
    """
    margin = (ball_times[None, :] - opp_times - p.intercept_margin) / p.intercept_temp
    probs = 1.0 / (1.0 + np.exp(-np.clip(margin, -30, 30)))
    probs = probs * interceptable[None, :]
    return probs.max(axis=1) if probs.size else np.zeros(opp_times.shape[0])


def duel_probability(
    attacker_skill: float, defender_skill: float, scale: float = 18.0, bias: float = 0.0
) -> float:
    return sigmoid((attacker_skill - defender_skill) / scale + bias)
