"""On-ball decision making.

The ball carrier evaluates every option it can see (passes to feet, lofted passes,
through balls, crosses, shots, carries, take-ons, holding, clearing), estimates each
option's success probability with *perception noise* that shrinks with the player's
vision, scores them by expected possession value, and picks one with a softmax.
Low-vision or tired players therefore make worse choices: exactly the texture the
analytics layer should be able to detect and explain.

Pass options are evaluated in one vectorised batch for speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .geometry import clamp, dist_to_goal, goal_angle, point_segment_distance, sigmoid
from .params import GOAL_HALF_WIDTH, PENALTY_BOX_DEPTH, PENALTY_BOX_HALF_WIDTH, PITCH_LENGTH, PITCH_WIDTH
from .probability import (
    duel_probability,
    player_times_to_points,
    position_value,
    position_value_vec,
    xg_unblocked,
)
from .tactics import opponent_line_tf, to_tf

if TYPE_CHECKING:  # pragma: no cover
    from .engine import MatchEngine

SAMPLES = np.array([0.12, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
CARRY_ANGLES = np.radians([-60.0, -30.0, 0.0, 30.0, 60.0])
AERIAL_KINDS = ("long", "switch", "cross", "corner", "free_kick_cross", "clearance", "goal_kick")


@dataclass
class Action:
    kind: str  # pass | shot | carry | dribble | hold | clear
    target: np.ndarray | None = None  # absolute coordinates
    receiver: int | None = None
    pass_kind: str = "ground"  # ground | through | long | switch | cross | cutback | back
    aerial: bool = False
    utility: float = 0.0
    p_est: float = 1.0
    opponent: int | None = None
    arrival: float | None = None
    extra: dict = field(default_factory=dict)


def opp_value(x_tf: float, y_tf: float) -> float:
    """Value for the opponent if they win the ball at our team-frame (x, y)."""
    return position_value(PITCH_LENGTH - x_tf, PITCH_WIDTH - y_tf)


def _aerial_profile_vec(dist: np.ndarray, kinds: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vh = np.empty_like(dist)
    peak = np.empty_like(dist)
    for k, kind in enumerate(kinds):
        d = dist[k]
        if kind in ("cross", "corner", "free_kick_cross"):
            vh[k], peak[k] = 17.0 + 0.08 * d, clamp(2.5 + 0.12 * d, 3.0, 7.0)
        elif kind == "clearance":
            vh[k], peak[k] = 19.0 + 0.06 * d, clamp(0.22 * d, 6.0, 18.0)
        else:
            vh[k], peak[k] = 13.5 + 0.12 * d, clamp(0.17 * d, 3.5, 14.0)
    return vh, peak


def evaluate_passes(
    eng: MatchEngine,
    kicker: int,
    receivers: list[int | None],
    ends: np.ndarray,
    aerial: np.ndarray,
    kinds: list[str],
    noise: float = 0.0,
    arrivals: np.ndarray | None = None,
) -> dict:
    """Vectorised success model for K candidate passes from the kicker's position.

    Returns arrays of shape (K,) plus per-sample details used when executing a pass.
    """
    p = eng.p
    side = eng.pl[kicker].side
    start = eng.pos[kicker]
    K = len(ends)
    vec = ends - start[None, :]
    dist = np.maximum(np.hypot(vec[:, 0], vec[:, 1]), 0.5)
    pts = start[None, None, :] + vec[:, None, :] * SAMPLES[None, :, None]  # (K, S, 2)

    arr = np.full(K, p.pass_arrival_speed) if arrivals is None else arrivals
    v0_ground = np.sqrt(arr * arr + 2 * p.ground_decel * dist)
    s_dist = SAMPLES[None, :] * dist[:, None]
    disc = np.maximum(v0_ground[:, None] ** 2 - 2 * p.ground_decel * s_dist, 0.05)
    tb_ground = (v0_ground[:, None] - np.sqrt(disc)) / p.ground_decel
    vh, peak = _aerial_profile_vec(dist, kinds)
    tb_air = s_dist / vh[:, None]
    4 * peak[:, None] * SAMPLES[None, :] * (1 - SAMPLES[None, :])
    tb = np.where(aerial[:, None], tb_air, tb_ground)
    # Lofted balls can only be cut out right after the kick; the landing is contested separately.
    inter = np.where(aerial[:, None], (SAMPLES[None, :] <= 0.12).astype(float), 1.0)

    opps = eng.active_indices(1 - side)
    n = len(opps)
    flat = pts.reshape(-1, 2)
    ot = player_times_to_points(
        eng.pos[opps],
        eng.vel[opps],
        eng.top_eff[opps],
        eng.accel[opps],
        flat,
        p.reach_radius,
        reaction=p.intercept_reaction,
    ).reshape(n, K, len(SAMPLES))
    inter_full = np.broadcast_to(inter, (n, K, len(SAMPLES))).copy()
    gk = eng.goalkeeper(1 - side)
    if gk is not None:
        k = np.where(opps == gk)[0]
        if len(k):
            k = int(k[0])
            d = eng.teams[1 - side].direction
            ptf = to_tf(flat, d).reshape(K, len(SAMPLES), 2)
            in_box = (ptf[..., 0] <= PENALTY_BOX_DEPTH) & (
                np.abs(ptf[..., 1] - PITCH_WIDTH / 2) <= PENALTY_BOX_HALF_WIDTH
            )
            ot[k] = np.where(in_box, ot[k] - 0.25, ot[k] + 0.4)
    if noise > 0 and n:
        ot = ot + eng.rng.normal(0.0, noise, ot.shape)
    margin = (tb[None, :, :] - ot - p.intercept_margin) / p.intercept_temp
    probs = 1.0 / (1.0 + np.exp(-np.clip(margin, -30, 30))) * inter_full
    pint = probs.max(axis=2) if n else np.zeros((0, K))  # (n, K)
    p_no_int = np.prod(1.0 - pint, axis=0) if n else np.ones(K)

    p_reach = np.ones(K)
    p_ctrl = np.ones(K)
    has_r = np.array([r is not None for r in receivers])
    if has_r.any():
        ridx = np.array([r if r is not None else kicker for r in receivers])
        rpos, rvel = eng.pos[ridx], eng.vel[ridx]
        diff = ends - rpos
        rd = np.maximum(np.hypot(diff[:, 0], diff[:, 1]) - p.reach_radius, 0.0)
        unit = diff / np.maximum(np.hypot(diff[:, 0], diff[:, 1]), 1e-6)[:, None]
        v_al = np.clip(np.einsum("kd,kd->k", unit, rvel), 0, None)
        top, acc = eng.top_eff[ridx], eng.accel[ridx]
        v_al = np.minimum(v_al, top)
        d_acc = (top**2 - v_al**2) / (2 * acc)
        t_acc = (-v_al + np.sqrt(v_al**2 + 2 * acc * rd)) / acc
        t_cr = (top - v_al) / acc + (rd - d_acc) / top
        rt = 0.15 + np.where(rd <= d_acc, t_acc, t_cr)
        total = tb[:, -1]
        p_reach = np.where(has_r, 1.0 / (1.0 + np.exp((rt - total) / 0.35)), 1.0)
        opp_end = ot[:, :, -1].min(axis=0) if n else np.full(K, 9.0)
        end_pressure = np.clip(1.0 - (opp_end - total) / 1.6, 0.0, 1.0)
        ft = np.array([eng.pl[i].a.first_touch for i in ridx], dtype=float)
        hd = np.array([eng.pl[i].a.heading for i in ridx], dtype=float)
        arr_speed = np.maximum(v0_ground - p.ground_decel * total, 0.0)
        ctrl_ground = (
            p.first_touch_base
            + (ft - 70) / 700.0
            - 0.055 * end_pressure
            - np.maximum(0.0, arr_speed - 11.0) * 0.012
        )
        duel = 1.0 / (1.0 + np.exp(-((opp_end - rt) / 0.45 + (hd - 65) / 40.0)))
        ctrl_air = np.clip(0.12 + 0.5 * duel + (ft - 70) / 500.0, 0.1, 0.7)
        p_ctrl = np.where(has_r, np.where(aerial, ctrl_air, np.clip(ctrl_ground, 0.6, 0.995)), 1.0)
    return {
        "p": p_no_int * p_reach * p_ctrl,
        "p_no_int": p_no_int,
        "p_reach": p_reach,
        "p_ctrl": p_ctrl,
        "pint": pint,
        "opps": opps,
        "tb": tb,
        "ot": ot,
        "pts": pts,
        "dist": dist,
        "v0": np.where(aerial, vh, v0_ground),
        "vh": vh,
        "peak": np.where(aerial, peak, 0.0),
        "decel": np.where(aerial, 0.0, p.ground_decel),
    }


def evaluate_pass(
    eng: MatchEngine,
    kicker: int,
    receiver: int | None,
    start: np.ndarray,
    end: np.ndarray,
    aerial: bool,
    kind: str,
    noise: float = 0.0,
    arrival: float | None = None,
) -> dict:
    """Single-pass convenience wrapper around :func:`evaluate_passes` (start must be the kicker)."""
    arrivals = None if arrival is None else np.array([arrival])
    b = evaluate_passes(
        eng,
        kicker,
        [receiver],
        np.asarray(end, dtype=float)[None, :],
        np.array([aerial]),
        [kind],
        noise=noise,
        arrivals=arrivals,
    )
    return {
        "p": float(b["p"][0]),
        "p_no_int": float(b["p_no_int"][0]),
        "p_reach": float(b["p_reach"][0]),
        "p_ctrl": float(b["p_ctrl"][0]),
        "pint": b["pint"][:, 0],
        "opps": b["opps"],
        "tb": b["tb"][0],
        "ot": b["ot"][:, 0, :],
        "pts": b["pts"][0],
        "v0": float(b["v0"][0]),
        "vh": float(b["vh"][0]),
        "decel": float(b["decel"][0]),
        "peak": float(b["peak"][0]),
        "dist": float(b["dist"][0]),
        "total": float(b["tb"][0, -1]),
    }


def shot_lane(eng: MatchEngine, shooter: int, pos_abs: np.ndarray) -> tuple[float, int, list[int]]:
    """Estimate block probability from defenders between the shooter and goal."""
    side = eng.pl[shooter].side
    d = eng.teams[side].direction
    goal_abs = to_tf(np.array([PITCH_LENGTH, PITCH_WIDTH / 2]), d)
    p_clear = 1.0
    blockers = []
    gk = eng.goalkeeper(1 - side)
    for j in eng.active_indices(1 - side):
        if j == gk:
            continue
        dist, t = point_segment_distance(eng.pos[j], pos_abs, goal_abs)
        if 0.03 < t < 0.97 and dist < 1.6:
            near = clamp(1.0 - dist / 1.6, 0.0, 1.0)
            closeness = 1.4 if t < 0.35 else 1.0
            p_clear *= 1.0 - eng.p.shot_block_per_defender * 2.2 * near * closeness
            blockers.append(int(j))
    return 1.0 - clamp(p_clear, 0.02, 1.0), len(blockers), blockers


def shot_model(eng: MatchEngine, shooter: int, header: bool = False, kind: str = "open_play") -> dict:
    side = eng.pl[shooter].side
    d = eng.teams[side].direction
    pos = eng.pos[shooter]
    x, y = to_tf(pos, d)
    a = eng.pl[shooter].a
    level, nearest = eng.pressure_level(shooter)
    p_block, n_lane, blockers = (0.0, 0, []) if kind == "penalty" else shot_lane(eng, shooter, pos)
    gk = eng.goalkeeper(1 - side)
    keeper = eng.pl[gk].a.goalkeeping if gk is not None else 30
    big = n_lane == 0 and dist_to_goal(x, y) < 14 and level < 0.35 and kind == "open_play"
    if gk is not None and kind == "open_play":
        gk_tf = to_tf(eng.pos[gk], d)
        # Keeper badly out of position (rushed out, wrong side) raises the chance.
        off_line = abs(gk_tf[1] - PITCH_WIDTH / 2 - (y - PITCH_WIDTH / 2) * 0.25)
        if (off_line > 6 or gk_tf[0] < PITCH_LENGTH - 16) and dist_to_goal(x, y) < 16 and level < 0.5:
            big = True
    p_unblocked = xg_unblocked(
        eng.p,
        x,
        y,
        header=header,
        pressure=level,
        finishing=a.finishing,
        composure=a.composure,
        keeper=keeper,
        big_chance=big,
        kind=kind,
    )
    if header:
        p_unblocked *= clamp(0.75 + (a.heading - 60) / 80, 0.5, 1.2)
    return {
        "x": x,
        "y": y,
        "p_block": p_block,
        "p_unblocked": p_unblocked,
        "xg": (1.0 - p_block) * p_unblocked,
        "pressure": level,
        "nearest": nearest,
        "n_lane": n_lane,
        "blockers": blockers,
        "big_chance": big,
        "distance": dist_to_goal(x, y),
        "angle": goal_angle(x, y),
        "goalkeeper": gk,
    }


def decide(
    eng: MatchEngine,
    c: int,
    allowed: set[str] | None = None,
    max_pass: float = 62.0,
    pass_kind_override: str | None = None,
) -> Action:
    p = eng.p
    P = eng.pl[c]
    a = P.a
    side = P.side
    team = eng.teams[side]
    tac = team.tactics
    d = team.direction
    rng = eng.rng
    energy = float(eng.energy[c])
    pos_abs = eng.pos[c].copy()
    cx, cy = to_tf(pos_abs, d)
    here = np.array([cx, cy])
    level, nearest = eng.pressure_level(c)
    vision_noise = 0.05 + 0.3 * (1.0 - a.vision / 100.0) + 0.1 * (1.0 - energy)
    risk_scale = p.turnover_cost * (1.25 - 0.5 * tac.risk) * (1.0 - 0.15 * team.confidence)
    premium = p.possession_premium * (1.35 - tac.directness) * (1.15 - 0.3 * tac.risk)
    if team.escalation > 0:
        premium *= 0.75
    elif team.escalation < 0:
        premium *= 1.3
    v_here = position_value(cx, cy)
    options: list[Action] = []

    def allow(k: str) -> bool:
        return allowed is None or k in allowed

    mates = [int(i) for i in eng.active_indices(side) if i != c]
    sign = 1 if d == 1 else -1

    # ---- Passes (batched) -------------------------------------------------------------
    if allow("pass") and mates:
        recv: list[int] = []
        ends_tf: list[np.ndarray] = []
        kinds: list[str] = []
        air: list[bool] = []
        arrivals: list[float] = []
        offs: list[bool] = []
        for m in mates:
            M = eng.pl[m]
            if M.slot.role == "GK" and (cx > 45 or level < 0.35):
                continue
            m_tf = to_tf(eng.pos[m], d)
            v_tf = eng.vel[m] * sign
            offside = eng.is_offside(m, side)
            if offside and rng.random() < 0.6 + 0.3 * a.vision / 100:
                continue
            feet = np.clip(m_tf + v_tf * 0.45, [0.8, 0.8], [PITCH_LENGTH - 0.8, PITCH_WIDTH - 0.8])
            dist = float(np.hypot(*(feet - here)))
            if 4.0 <= dist <= max_pass:
                dx = feet[0] - cx
                if dist <= 40:
                    recv.append(m), ends_tf.append(feet), air.append(False), offs.append(offside)
                    kinds.append("back" if dx < -4 else "ground"), arrivals.append(p.pass_arrival_speed)
                if dist > 30 or (dist > 20 and level > 0.6):
                    recv.append(m), ends_tf.append(feet), air.append(True), offs.append(offside)
                    kinds.append("switch" if abs(feet[1] - cy) > 28 else "long")
                    arrivals.append(p.pass_arrival_speed)
            if np.hypot(*v_tf) > 3.2 and v_tf[0] > 1.5 and M.slot.role != "GK":
                through = np.clip(
                    m_tf + v_tf * 1.05 + np.array([3.0, 0.0]), [1, 1], [PITCH_LENGTH - 1.5, PITCH_WIDTH - 1]
                )
                dist = float(np.hypot(*(through - here)))
                if 6.0 <= dist <= min(max_pass, 45):
                    recv.append(m), ends_tf.append(through), air.append(False), offs.append(offside)
                    kinds.append("through"), arrivals.append(p.pass_arrival_speed + 1.5)
        if recv:
            ends_tf_a = np.array(ends_tf)
            ends_abs = to_tf(ends_tf_a, d)
            air_a = np.array(air)
            ev = evaluate_passes(
                eng, c, recv, ends_abs, air_a, kinds, noise=vision_noise, arrivals=np.array(arrivals)
            )
            pr = ev["p"]
            vec = ends_tf_a - here[None, :]
            value = position_value_vec(ends_tf_a[:, 0], ends_tf_a[:, 1])
            fail = here[None, :] + vec * 0.65
            loss = position_value_vec(PITCH_LENGTH - fail[:, 0], PITCH_WIDTH - fail[:, 1])
            u = pr * value - (1 - pr) * (loss + premium) * risk_scale
            u += p.forward_bias * vec[:, 0] / 10.0
            u -= np.where(vec[:, 0] < -5, p.back_pass_penalty * (0.6 + tac.directness), 0.0)
            u += np.where(air_a, 0.004 * (tac.directness - 0.5) - p.aerial_pass_cost, 0.0)
            for k in range(len(recv)):
                uk = float(u[k])
                if recv[k] == P.received_from and vec[k, 0] < 3:
                    uk -= p.return_pass_penalty
                if kinds[k] == "through":
                    uk += 0.002 * tac.risk
                if eng.pl[recv[k]].slot.role == "GK":
                    uk -= 0.003
                prk = float(pr[k]) * (0.15 if offs[k] else 1.0)
                options.append(
                    Action(
                        "pass",
                        ends_abs[k],
                        recv[k],
                        pass_kind_override or kinds[k],
                        bool(air_a[k]),
                        uk,
                        prk,
                        arrival=arrivals[k],
                        extra={"offside": offs[k]},
                    )
                )

    # ---- Crosses ------------------------------------------------------------------------
    if allow("cross") and cx > 66 and (cy < 23 or cy > 45):
        near_y = 30.0 if cy < PITCH_WIDTH / 2 else 38.0
        far_y = 39.0 if cy < PITCH_WIDTH / 2 else 29.0
        zones = [
            (np.array([99.5, near_y]), "near"),
            (np.array([94.0, 34.0]), "spot"),
            (np.array([98.5, far_y]), "far"),
        ]
        if cx > 95:
            zones.append((np.array([88.0, 34.0]), "cutback"))
        att_pos = to_tf(eng.pos[mates], d) if mates else np.zeros((0, 2))
        opp_idx = eng.active_indices(1 - side)
        opp_pos = to_tf(eng.pos[opp_idx], d)
        for z_tf, zname in zones:
            n_att = int(np.sum(np.hypot(*(att_pos - z_tf).T) < 7.5)) if len(att_pos) else 0
            n_def = int(np.sum(np.hypot(*(opp_pos - z_tf).T) < 6.5))
            if n_att == 0:
                continue
            p_win = sigmoid((n_att - 0.85 * n_def) / 1.3 - 0.5 + (a.crossing - 65) / 40.0)
            if zname == "cutback":
                p_win = sigmoid((n_att - 0.6 * n_def) / 1.2 + 0.3)
            gain = 0.10 if zname != "cutback" else 0.09
            u = (
                p_win * gain
                - (1 - p_win) * (0.012 + premium) * risk_scale
                + 0.008 * (tac.crossing - 0.5)
                + p.cross_bias
            )
            options.append(
                Action(
                    "pass",
                    to_tf(z_tf, d),
                    None,
                    "cutback" if zname == "cutback" else "cross",
                    zname != "cutback",
                    u,
                    p_win,
                    extra={"zone": zname},
                )
            )

    # ---- Shot ---------------------------------------------------------------------------
    if allow("shot") and dist_to_goal(cx, cy) < 34.0 and goal_angle(cx, cy) > 0.08:
        sm = shot_model(eng, c)
        est = sm["xg"] * math.exp(rng.normal(0, 0.22 * (1.2 - a.composure / 100)))
        if est >= p.min_shot_xg:
            bias = p.shot_bias * (1 + 0.15 * team.confidence) * (1.25 if team.escalation > 0 else 1.0)
            # Shooting ends the possession even when it fails, so it competes with keeping the ball.
            u = est * bias - (1 - est) * premium * 0.35
            options.append(Action("shot", None, None, utility=u, p_est=sm["xg"]))

    # ---- Carries (batched) --------------------------------------------------------------
    opp_idx = eng.active_indices(1 - side)
    if allow("carry"):
        speed = max(eng.top_eff[c] * p.dribble_speed_factor, 3.0)
        angles = list(CARRY_ANGLES)
        if level > 0.5:
            angles += [math.radians(130), math.radians(-130)]
        ang = np.array(angles)
        L = np.where(np.abs(ang) < 1.6, 6.0, 4.0)
        ends = here[None, :] + np.stack([L * np.cos(ang), L * np.sin(ang)], axis=1)
        ok = (
            (ends[:, 0] > 1)
            & (ends[:, 0] < PITCH_LENGTH - 1)
            & (ends[:, 1] > 1)
            & (ends[:, 1] < PITCH_WIDTH - 1)
        )
        ends, L = ends[ok], L[ok]
        if len(ends) and len(opp_idx):
            mids = (here[None, :] + ends) / 2
            pts_tf = np.concatenate([mids, ends])
            ot = player_times_to_points(
                eng.pos[opp_idx],
                eng.vel[opp_idx],
                eng.top_eff[opp_idx],
                eng.accel[opp_idx],
                to_tf(pts_tf, d),
                1.0,
            )
            m_n = len(ends)
            tc_mid = L / 2 / speed + 0.15
            tc_end = L / speed + 0.15
            margin = np.minimum(
                (ot[:, :m_n] - tc_mid[None, :]).min(axis=0), (ot[:, m_n:] - tc_end[None, :]).min(axis=0)
            )
            margin = margin + rng.normal(0, vision_noise, m_n)
            p_keep = 1.0 / (1.0 + np.exp(-(margin / 0.42 + (a.dribbling - 68) / 35.0 + 0.8)))
            val = position_value_vec(ends[:, 0], ends[:, 1])
            lossv = position_value_vec(PITCH_LENGTH - ends[:, 0], PITCH_WIDTH - ends[:, 1])
            u = p_keep * val - (1 - p_keep) * (lossv + premium) * risk_scale
            u += p.forward_bias * (ends[:, 0] - cx) / 10.0 * 0.6 - p.carry_cost
            # Near goal the box is crowded: dribbling further in is rarely on.
            u -= np.where(
                np.hypot(PITCH_LENGTH - ends[:, 0], PITCH_WIDTH / 2 - ends[:, 1]) < 24.0,
                p.box_carry_cost,
                0.0,
            )
            for k in range(m_n):
                options.append(
                    Action("carry", to_tf(ends[k], d), None, utility=float(u[k]), p_est=float(p_keep[k]))
                )

    # ---- Take-on ------------------------------------------------------------------------
    if allow("dribble") and len(opp_idx):
        o_tf = to_tf(eng.pos[opp_idx], d)
        rel = o_tf - here
        dist = np.hypot(rel[:, 0], rel[:, 1])
        in_front = (rel[:, 0] > 0.3) & (np.abs(rel[:, 1]) < 3.0) & (dist < 3.4)
        if np.any(in_front):
            k = int(np.argmin(np.where(in_front, dist, 99)))
            j = int(opp_idx[k])
            D = eng.pl[j].a
            pw = duel_probability(
                a.dribbling * 0.7 + a.pace * 0.15 + a.acceleration * 0.15,
                D.tackling * 0.6 + D.positioning * 0.25 + D.pace * 0.15,
                15.0,
                bias=-0.1 + p.dribble_attempt_bias,
            )
            end_tf = np.array([min(cx + 7.0, PITCH_LENGTH - 2), cy + rng.uniform(-2, 2)])
            u = (
                pw * position_value(*end_tf) * p.dribble_utility_scale
                - (1 - pw) * (opp_value(cx, cy) + premium) * risk_scale
            )
            u += 0.002 * (a.dribbling - 70) / 20 - p.dribble_utility_cost
            options.append(Action("dribble", to_tf(end_tf, d), None, utility=u, p_est=pw, opponent=j))

    # ---- Hold / shield ------------------------------------------------------------------
    if allow("hold"):
        p_keep = 1.0 - 0.32 * level * (1.0 - a.strength / 130.0)
        u = (
            p_keep * v_here
            - (1 - p_keep) * (opp_value(cx, cy) + premium) * risk_scale
            - p.hold_cost * (0.5 + tac.tempo)
        )
        options.append(Action("hold", None, None, utility=u, p_est=p_keep))

    # ---- Clearance ----------------------------------------------------------------------
    if allow("clear") and cx < 34 and level > 0.35 and P.slot.role != "GK":
        if (cy < 22 or cy > PITCH_WIDTH - 22) and rng.random() < 0.45:
            # Wide and under pressure: put it out for a throw-in.
            out_y = -6.0 if cy < PITCH_WIDTH / 2 else PITCH_WIDTH + 6.0
            target_tf = np.array([clamp(cx + rng.uniform(15, 35), 20, 70), out_y])
        else:
            target_tf = np.array(
                [
                    clamp(cx + rng.uniform(38, 55), 40, 85),
                    clamp(cy + rng.uniform(-18, 18), 6, PITCH_WIDTH - 6),
                ]
            )
        u = 0.45 * position_value(62, 34) - 0.55 * opp_value(43, 34) + 0.009 * (1 - tac.risk) - premium * 0.3
        options.append(Action("clear", to_tf(target_tf, d), None, aerial=True, utility=u))

    if not options:
        return Action("hold", None, None)

    temp = p.decision_temperature * (1.4 - a.vision / 100.0) * (1.0 + 0.6 * (1.0 - energy))
    utils = np.array([o.utility for o in options])
    z = (utils - utils.max()) / max(temp, 1e-4)
    w = np.exp(z)
    w /= w.sum()
    choice = int(rng.choice(len(options), p=w))
    return options[choice]


def goal_mouth_point(direction: int, y_off: float) -> np.ndarray:
    gx = PITCH_LENGTH if direction == 1 else 0.0
    gy = PITCH_WIDTH / 2 + (y_off if direction == 1 else -y_off)
    return np.array([gx, gy])


__all__ = [
    "Action",
    "decide",
    "evaluate_pass",
    "evaluate_passes",
    "shot_model",
    "goal_mouth_point",
    "GOAL_HALF_WIDTH",
    "AERIAL_KINDS",
    "opponent_line_tf",
]
