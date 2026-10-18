"""Off-ball positioning: team shape, pressing, marking, support runs and set-piece setups.

Targets are computed in each team's attacking frame and converted back to
absolute pitch coordinates. Urgency codes: 0 = pace set by distance,
1 = at least running pace, 2 = sprint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .geometry import clamp
from .params import PITCH_LENGTH, PITCH_WIDTH
from .state import BallState

if TYPE_CHECKING:  # pragma: no cover
    from .engine import MatchEngine

FLIP = np.array([PITCH_LENGTH, PITCH_WIDTH])


def to_tf(points: np.ndarray, direction: int) -> np.ndarray:
    return points if direction == 1 else FLIP - points


def opponent_line_tf(eng: MatchEngine, side: int) -> float:
    """x (in ``side``'s attacking frame) of the second-last opponent: the offside line."""
    opp = 1 - side
    idx = eng.active_indices(opp)
    if len(idx) < 2:
        return PITCH_LENGTH
    xs = np.sort(to_tf(eng.pos[idx], eng.teams[side].direction)[:, 0])
    return float(max(xs[-2], PITCH_LENGTH / 2))


def shape_targets(eng: MatchEngine, side: int, ball_abs: np.ndarray, in_possession: bool) -> None:
    team = eng.teams[side]
    tac = team.tactics
    d = team.direction
    bx, by = to_tf(ball_abs, d)
    idx = eng.active_indices(side)
    if len(idx) == 0:
        return
    u = eng.slot_u[idx]
    sy = eng.slot_y[idx]
    is_gk = eng.is_gk[idx]
    conf = team.confidence

    if in_possession:
        depth = 40.0 + 8.0 * tac.width + 4.0 * tac.directness
        rear = clamp(bx - 0.62 * depth + 12.0 * (tac.line_height - 0.5) + 3.0 * conf, 14.0, 60.0)
        width_scale = 0.86 + 0.26 * tac.width
        # Rest defence: the back line stays goal-side of the opponent's most advanced player.
        opp_idx = eng.active_indices(1 - side)
        if len(opp_idx):
            opp_x = to_tf(eng.pos[opp_idx], d)[:, 0]
            rear = max(min(rear, float(opp_x.min()) - 3.0), 12.0)
        x = rear + u * depth
        y = PITCH_WIDTH / 2 + (sy - PITCH_WIDTH / 2) * width_scale + (by - PITCH_WIDTH / 2) * 0.14
        line = opponent_line_tf(eng, side)
        x = np.minimum(x, line - 0.8)
    else:
        depth = 31.0 + 4.0 * (1.0 - tac.pressing)
        lh = tac.line_height + 0.08 * conf
        rear = clamp(bx - 0.72 * depth - 5.0 + 16.0 * (lh - 0.5), 7.0 + 10.0 * lh, 38.0 + 14.0 * lh)
        # Once the ball is within ~40 m, the back line drops to stay goal-side of the deepest attacker.
        opp_idx = eng.active_indices(1 - side)
        if len(opp_idx) and bx < 42.0:
            opp_tf = to_tf(eng.pos[opp_idx], d)
            central = np.abs(opp_tf[:, 1] - PITCH_WIDTH / 2) < 24.0
            if np.any(central):
                deepest = float(np.min(opp_tf[central, 0]))
                rear = min(rear, max(deepest - 1.6, 4.5))
        x = rear + u * depth
        y = PITCH_WIDTH / 2 + (sy - PITCH_WIDTH / 2) * 0.64 + (by - PITCH_WIDTH / 2) * 0.36

    # Ambient drift keeps players moving naturally; defenders near their own goal stay disciplined.
    phase = eng.drift_phase[idx]
    amp = 1.6 if in_possession else clamp((bx - 20.0) / 50.0, 0.15, 0.9)
    x = x + 2.0 * amp * np.sin(eng.t * 0.42 + phase)
    y = y + 2.4 * amp * np.sin(eng.t * 0.35 + phase * 1.7)

    # Goalkeeper
    gk_x = clamp(2.0 + 0.11 * bx + (6.0 * tac.line_height if in_possession else 0.0), 2.0, 22.0)
    gk_y = PITCH_WIDTH / 2 + (by - PITCH_WIDTH / 2) * 0.2
    x = np.where(is_gk, gk_x, x)
    y = np.where(is_gk, gk_y, y)

    x = np.minimum(np.maximum(x, 1.0), PITCH_LENGTH - 1.0)
    y = np.minimum(np.maximum(y, 1.5), PITCH_WIDTH - 1.5)
    if d == 1:
        eng.target[idx, 0] = x
        eng.target[idx, 1] = y
    else:
        eng.target[idx, 0] = PITCH_LENGTH - x
        eng.target[idx, 1] = PITCH_WIDTH - y
    eng.urg[idx] = 0


def apply_marking(eng: MatchEngine, side: int) -> None:
    """Out of possession, outfield players tighten onto nearby opponents near their goal."""
    team = eng.teams[side]
    d = team.direction
    mine = [i for i in eng.active_indices(side) if eng.pl[i].slot.role != "GK"]
    theirs = eng.active_indices(1 - side)
    if not mine or len(theirs) == 0:
        return
    goal = to_tf(np.array([0.0, PITCH_WIDTH / 2]), d)
    opp_pos = eng.pos[theirs]
    opp_tf_x = to_tf(opp_pos, d)[:, 0]
    tgt = eng.target[mine]
    dmat = np.linalg.norm(tgt[:, None, :] - opp_pos[None, :, :], axis=2)
    assigned_m: set[int] = set()
    assigned_o: set[int] = set()
    order = np.dstack(np.unravel_index(np.argsort(dmat, axis=None), dmat.shape))[0]
    for mi, oi in order:
        if dmat[mi, oi] > 16.0:
            break
        if mi in assigned_m or oi in assigned_o:
            continue
        if theirs[oi] == eng.carrier:
            continue
        j = theirs[oi]
        o = opp_pos[oi]
        v = eng.vel[j]
        to_goal = goal - o
        n = float(np.hypot(*to_goal))
        unit = to_goal / n if n > 1e-6 else np.zeros(2)
        running_at_goal = float(v @ unit) > 3.5
        w = clamp((56.0 - opp_tf_x[oi]) / 30.0, 0.0, 0.9)
        if running_at_goal:
            w = max(w, 0.85)
            o = o + v * 0.5
        if w <= 0.02:
            continue
        mark_pt = o + unit * 1.6
        i = mine[mi]
        eng.target[i] = (1 - w) * eng.target[i] + w * mark_pt
        assigned_m.add(mi)
        assigned_o.add(oi)


def apply_pressing(eng: MatchEngine, side: int) -> None:
    if eng.carrier is None or eng.ball_state != BallState.HELD:
        return
    carrier = eng.carrier
    if eng.pl[carrier].side == side:
        return
    team = eng.teams[side]
    tac = team.tactics
    d = team.direction
    c_pos = eng.pos[carrier]
    bx = to_tf(c_pos, d)[0]
    press_intensity = clamp(tac.pressing + 0.1 * team.confidence, 0.0, 1.0)
    trigger = bx < 36.0 + 64.0 * press_intensity
    counter_press = (eng.t - team.lost_ball_at) < 3.0 and press_intensity > 0.5
    cands = [
        i for i in eng.active_indices(side) if eng.pl[i].slot.role != "GK" and eng.t >= eng.recover_until[i]
    ]
    if not cands:
        return
    dists = np.linalg.norm(eng.pos[cands] - c_pos, axis=1)
    order = np.argsort(dists)
    own_goal = to_tf(np.array([0.0, PITCH_WIDTH / 2]), d)
    to_goal = own_goal - c_pos
    to_goal /= max(np.linalg.norm(to_goal), 1e-6)

    p1 = cands[order[0]]
    jockey = eng.p.press_jockey_dist
    if trigger or counter_press or bx < 30.0:
        eng.target[p1] = c_pos + to_goal * jockey
        eng.urg[p1] = 3 if (counter_press and dists[order[0]] < 12.0) else 2
    else:
        # Outside the pressing zone: stay goal-side and track the carrier at a distance.
        eng.target[p1] = c_pos + to_goal * 4.0
        eng.urg[p1] = 2 if np.linalg.norm(eng.vel[carrier]) > 3.0 else 1
    eng.pressing_idx = p1

    # Danger zone: a goal-side defender steps out to block the route to goal.
    c_tf = to_tf(c_pos, d)
    dist_goal = float(np.hypot(c_tf[0], c_tf[1] - PITCH_WIDTH / 2))
    eng.cover_idx = None
    if dist_goal < 32.0 and len(order) > 1:
        block_pt = c_pos + to_goal * clamp(2.2 + (dist_goal - 12.0) * 0.08, 2.0, 3.6)
        others = [cands[k] for k in order[1:]]
        tf_x = to_tf(eng.pos[others], d)[:, 0]
        goal_side = [o for o, x in zip(others, tf_x, strict=True) if x < c_tf[0] + 1.0]
        if goal_side:
            bd = np.hypot(*(eng.pos[goal_side] - block_pt).T)
            cover = goal_side[int(np.argmin(bd))]
            eng.target[cover] = block_pt
            eng.urg[cover] = 2
            eng.cover_idx = cover

    if len(order) > 1 and (trigger or counter_press) and press_intensity > 0.55:
        p2 = cands[order[1]]
        if dists[order[1]] < 18.0 and p2 != eng.cover_idx:
            # Cut the passing lane to the carrier's nearest teammate.
            mates = [i for i in eng.active_indices(1 - side) if i != carrier]
            md = np.linalg.norm(eng.pos[mates] - c_pos, axis=1)
            lane_mate = mates[int(np.argmin(md))]
            eng.target[p2] = 0.55 * c_pos + 0.45 * eng.pos[lane_mate]
            eng.urg[p2] = 2 if counter_press else 1


def apply_support_and_runs(eng: MatchEngine, side: int) -> None:
    """In possession: make runs in behind / into the box and offer short options."""
    team = eng.teams[side]
    tac = team.tactics
    d = team.direction
    carrier = eng.carrier
    if carrier is None:
        return
    c_tf = to_tf(eng.pos[carrier], d)
    line = opponent_line_tf(eng, side)
    rng = eng.rng

    # Schedule a new run.
    if eng.t >= team.next_run_at and c_tf[0] > 35.0:
        team.next_run_at = eng.t + rng.uniform(1.0, 2.6) * (1.45 - tac.tempo)
        weights = {
            "ST": 3.0,
            "W": 2.6,
            "AM": 2.0,
            "CM": 0.9,
            "FB": 0.7 + tac.width,
            "DM": 0.2,
            "CB": 0.0,
            "GK": 0.0,
        }
        cands, w = [], []
        for i in eng.active_indices(side):
            p = eng.pl[i]
            if i == carrier or eng.t < p.run_until:
                continue
            wt = weights[p.slot.role] * (0.6 + p.a.pace / 100)
            if wt > 0:
                cands.append(i)
                w.append(wt)
        if cands:
            w_arr = np.array(w) / sum(w)
            r = int(rng.choice(cands, p=w_arr))
            p = eng.pl[r]
            r_tf = to_tf(eng.pos[r], d)
            wide_final_third = c_tf[0] > 70 and (c_tf[1] < 22 or c_tf[1] > 46)
            if wide_final_third and p.slot.role in ("ST", "W", "AM", "CM"):
                tgt, kind = np.array([rng.uniform(91, 100), rng.uniform(27, 41)]), "box"
            elif line < 92 and rng.random() < 0.6 and p.slot.role in ("ST", "W", "AM"):
                y = r_tf[1] + np.sign(PITCH_WIDTH / 2 - r_tf[1]) * rng.uniform(0, 7)
                tgt, kind = np.array([min(line + rng.uniform(6, 15), 101.0), y]), "in_behind"
            elif p.slot.role == "FB":
                touch = 4.0 if r_tf[1] < PITCH_WIDTH / 2 else PITCH_WIDTH - 4.0
                tgt, kind = np.array([min(c_tf[0] + rng.uniform(6, 14), 97.0), touch]), "overlap"
            else:
                tgt, kind = r_tf + (c_tf - r_tf) * 0.45, "check"
            tgt = np.clip(tgt, [2, 2], [PITCH_LENGTH - 2, PITCH_WIDTH - 2])
            p.run_target_tf = tgt
            p.run_kind = kind
            p.run_until = eng.t + rng.uniform(2.4, 4.6)

    for i in eng.active_indices(side):
        if i == carrier:
            continue
        p = eng.pl[i]
        if p.run_target_tf is not None and eng.t < p.run_until:
            tgt = to_tf(p.run_target_tf, d)
            eng.target[i] = tgt
            eng.urg[i] = 3 if p.run_kind == "in_behind" else (2 if p.run_kind in ("box", "overlap") else 1)
            if np.linalg.norm(eng.pos[i] - tgt) < 1.5:
                p.run_until = eng.t
        elif p.run_target_tf is not None:
            p.run_target_tf = None

    # Short support: the two nearest teammates come towards the ball if it is isolated.
    mates = [i for i in eng.active_indices(side) if i != carrier and eng.pl[i].slot.role != "GK"]
    if not mates:
        return
    md = np.linalg.norm(eng.pos[mates] - eng.pos[carrier], axis=1)
    opp_idx = eng.active_indices(1 - side)
    for k in np.argsort(md)[:3]:
        i = mates[int(k)]
        if eng.pl[i].run_target_tf is not None:
            continue
        tgt = eng.target[i].copy()
        if md[k] > 20.0:
            tgt = tgt + (eng.pos[carrier] - tgt) * 0.3
        # Step away from the nearest marker to show for the ball.
        od = np.linalg.norm(eng.pos[opp_idx] - tgt, axis=1)
        j = int(np.argmin(od))
        if od[j] < 3.5:
            away = tgt - eng.pos[opp_idx[j]]
            n = np.linalg.norm(away)
            if n > 1e-6:
                tgt = tgt + away / n * (3.5 - od[j])
        eng.target[i] = np.clip(tgt, [1, 1], [PITCH_LENGTH - 1, PITCH_WIDTH - 1])


def setup_targets(eng: MatchEngine) -> None:
    """Positions during a dead ball, depending on the restart type."""
    r = eng.restart
    if r is None:
        return
    kind, side, loc = r["kind"], r["side"], r["loc"]
    for s in (0, 1):
        shape_targets(eng, s, loc, in_possession=(s == side))
    if kind == "kick_off":
        for s in (0, 1):
            d = eng.teams[s].direction
            for i in eng.active_indices(s):
                tf = to_tf(eng.target[i], d)
                tf[0] = min(tf[0], PITCH_LENGTH / 2 - 1.0 - (0 if s == side else 8.5))
                eng.target[i] = to_tf(tf, d)
    elif kind == "corner":
        _corner_setup(eng, side)
    elif kind == "penalty":
        d = eng.teams[side].direction
        for s in (0, 1):
            for i in eng.active_indices(s):
                if eng.pl[i].slot.role == "GK":
                    continue
                tf = to_tf(eng.target[i], eng.teams[side].direction)
                tf[0] = min(tf[0], PITCH_LENGTH - 18.0)
                eng.target[i] = to_tf(tf, d)
    elif kind == "free_kick":
        _wall_setup(eng, side, loc)
    taker = r.get("taker")
    if taker is not None:
        eng.target[taker] = loc
        eng.urg[taker] = 1


def _corner_setup(eng: MatchEngine, side: int) -> None:
    rng = eng.rng
    d = eng.teams[side].direction
    att = [
        i for i in eng.active_indices(side) if eng.pl[i].slot.role != "GK" and i != eng.restart.get("taker")
    ]
    att.sort(key=lambda i: -(eng.pl[i].a.heading + eng.pl[i].a.strength * 0.5))
    in_box, stay = att[:5], att[5:][-2:]
    for k, i in enumerate(in_box):
        eng.target[i] = to_tf(np.array([rng.uniform(92, 100), 28 + k * 3.0 + rng.uniform(-1, 1)]), d)
        eng.urg[i] = 1
    for k, i in enumerate(stay):
        eng.target[i] = to_tf(np.array([50.0 + rng.uniform(-3, 3), 20.0 + 28.0 * k]), d)
    defenders = [i for i in eng.active_indices(1 - side) if eng.pl[i].slot.role != "GK"]
    defenders.sort(key=lambda i: -(eng.pl[i].a.heading))
    for k, i in enumerate(defenders[:9]):
        eng.target[i] = to_tf(np.array([rng.uniform(94, 101), 27 + k * 1.6]), d)
    for i in defenders[9:]:
        eng.target[i] = to_tf(np.array([75.0, 34.0]), d)


def _wall_setup(eng: MatchEngine, side: int, loc: np.ndarray) -> None:
    d = eng.teams[side].direction
    loc_tf = to_tf(loc, d)
    goal = np.array([PITCH_LENGTH, PITCH_WIDTH / 2])
    dist = np.linalg.norm(goal - loc_tf)
    if dist > 35:
        return
    unit = (goal - loc_tf) / max(dist, 1e-6)
    wall_center = loc_tf + unit * 9.15
    perp = np.array([-unit[1], unit[0]])
    defenders = [i for i in eng.active_indices(1 - side) if eng.pl[i].slot.role != "GK"]
    defenders.sort(key=lambda i: np.linalg.norm(eng.pos[i] - loc))
    n_wall = 4 if dist < 25 else 3
    for k, i in enumerate(defenders[:n_wall]):
        eng.target[i] = to_tf(wall_center + perp * (k - (n_wall - 1) / 2) * 0.8, d)
