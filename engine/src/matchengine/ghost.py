"""Ghost futures: copy a match at one moment, change one thing, and play it out many times.

The engine is deterministic for a seed, so a recorded match can be replayed exactly up to
any shot. At that instant the whole match state (positions, fatigue, cards, tactics, the
ball in flight) is copied. Each copy gets its own random stream and plays to full time:

- **What if it had gone in / been missed?** The shot's outcome is changed in each copy, and
  the same random streams are used with and without the change (common random numbers), so
  the difference between the two sets of futures is the effect of the change alone.
- **Ghost Scout: what if another player had taken it?** The engine's own shot model is
  re-evaluated for the exact same chance (same position, pressure, keeper and lane) with
  each candidate's ratings. Their goal probability is combined with the two sets of futures
  to give the result each candidate would most likely have produced.

Nothing is rigged: changed outcomes are applied the way the engine applies its own, and
every future then follows the normal match mechanics.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import os
import pickle
from collections import deque
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .decisions import shot_model
from .engine import ENGINE_VERSION, MatchEngine
from .events import EventType
from .league import CLUB_SPECS, get_club
from .models import MatchConfig, PlayerProfile

SCORED, MISSED = "goal", "saved"
ATTACKING_ROLES = ("ST", "W", "AM")


# --------------------------------------------------------------------------- driving
def _begin(eng: MatchEngine) -> None:
    cfg = eng.cfg
    eng.emit(
        EventType.MATCH_START,
        home_team_id=eng.teams[0].team_id,
        away_team_id=eng.teams[1].team_id,
        kickoff_utc=cfg.kickoff_utc,
        competition=cfg.competition,
        venue=cfg.home.stadium,
    )
    eng._start_period(1)
    eng._buffer.clear()


def _advance(eng: MatchEngine, stop: Callable[[MatchEngine], bool] | None = None, on_tick=None) -> bool:
    """Step the match the same way ``MatchEngine.run`` does. Returns True if ``stop`` fired."""
    while True:
        while not eng._period_over():
            eng._step()
            if on_tick is not None:
                on_tick(eng)
            eng._buffer.clear()
            if stop is not None and stop(eng):
                return True
        eng._end_period()
        eng._buffer.clear()
        if eng.period == 1:
            eng._start_period(2)
            eng._buffer.clear()
        else:
            return False


def config_from_meta(meta: dict) -> MatchConfig:
    """Rebuild the configuration of a recorded match from its meta.json."""
    eng = meta.get("engine") or {}
    if eng.get("scenario"):
        raise ValueError("matches generated with scripted scenarios cannot be replayed yet")
    return MatchConfig(
        home=get_club(meta["home"]["code"]),
        away=get_club(meta["away"]["code"]),
        seed=int(eng.get("seed", 7)),
        match_id=meta.get("match_id"),
        competition=meta.get("competition", "Meridian League"),
        season=meta.get("season", "2026/27"),
        matchday=meta.get("matchday", 1),
        kickoff_utc=meta.get("kickoff_utc", "2026-10-24T14:00:00Z"),
    )


# ---------------------------------------------------------------------------- forks
@dataclass
class ShotFork:
    """The match frozen the instant a shot has been struck (outcome decided, ball in flight)."""

    event: dict  # the recorded shot event
    actual: str  # its real outcome
    shooter_index: int
    shooter_id: str
    side: int  # 0 home, 1 away
    kind: str  # open_play / penalty / free_kick / header ...
    header: bool
    xg_by_player: dict[str, float]  # engine shot model for this exact chance, per candidate
    state: bytes = field(repr=False)  # pickled engine


def _same_shot(eng: MatchEngine, c: int, event: dict) -> bool:
    return (
        eng.period == event["period"]
        and abs(round(eng.t, 2) - event["t"]) < 0.051
        and eng.pl[c].player_id == event["player_id"]
    )


def fork_at_shot(meta: dict, event: dict, candidates: list[PlayerProfile] | None = None) -> ShotFork:
    """Replay a recorded match exactly up to a shot and freeze it there.

    While replaying, the engine's shot model is evaluated for the target chance with each
    candidate's ratings (a read-only calculation that does not touch the random stream, so
    the replay stays exact).
    """
    if event.get("type") != "shot":
        raise ValueError(f"event {event.get('event_id')} is a {event.get('type')}, not a shot")
    eng = MatchEngine(
        config_from_meta(meta), frame_hz=(meta.get("engine") or {}).get("frame_hz", 5), emit_frames=False
    )
    captured: dict = {}
    original = MatchEngine._shoot

    def shoot(c: int, header: bool = False, kind: str = "open_play") -> None:
        if not captured and _same_shot(eng, c, event):
            ps = eng.pl[c]
            own = ps.profile
            xg = {own.player_id: shot_model(eng, c, header=header, kind=kind)["xg"]}
            for cand in candidates or []:
                ps.profile = dataclasses.replace(own, attributes=cand.attributes)
                xg[cand.player_id] = shot_model(eng, c, header=header, kind=kind)["xg"]
            ps.profile = own
            captured.update(index=c, header=header, kind=kind, xg=xg)
        original(eng, c, header, kind)

    eng._shoot = shoot  # type: ignore[method-assign]
    _begin(eng)
    found = _advance(eng, stop=lambda e: bool(captured) and e.flight is not None and e.flight.kind == "shot")
    del eng._shoot
    if not found:
        raise ValueError(f"shot {event['event_id']} was not reproduced by the replay")
    c = captured["index"]
    return ShotFork(
        event=event,
        actual=eng.flight.outcome["type"],
        shooter_index=c,
        shooter_id=eng.pl[c].player_id,
        side=eng.pl[c].side,
        kind=captured["kind"],
        header=captured["header"],
        xg_by_player={k: round(float(v), 4) for k, v in captured["xg"].items()},
        state=pickle.dumps(eng),
    )


# ---------------------------------------------------------------------- checkpoints
CHECKPOINT_FILE = "ghost_checkpoints.pkl.gz"


def build_checkpoints(meta: dict, candidates: list[PlayerProfile] | None = None) -> dict[str, ShotFork]:
    """Freeze the match at every shot in one replay, with Ghost Scout values for every candidate.

    Exact replay depends on floating-point details that differ between machines (operating
    system, NumPy build). So the frozen states are recorded once, on the machine that recorded
    the match, and shipped with it; any machine can then play futures from them.
    """
    if candidates is None:
        candidates = league_candidates()
    eng = MatchEngine(
        config_from_meta(meta), frame_hz=(meta.get("engine") or {}).get("frame_hz", 5), emit_frames=False
    )
    original = MatchEngine._shoot
    pending_info: list[dict] = []

    def shoot(c: int, header: bool = False, kind: str = "open_play") -> None:
        ps = eng.pl[c]
        own = ps.profile
        xg = {own.player_id: shot_model(eng, c, header=header, kind=kind)["xg"]}
        for cand in candidates:
            ps.profile = dataclasses.replace(own, attributes=cand.attributes)
            xg[cand.player_id] = shot_model(eng, c, header=header, kind=kind)["xg"]
        ps.profile = own
        original(eng, c, header, kind)
        pending_info.append({"index": c, "header": header, "kind": kind, "xg": xg, "frozen": False})

    forks: list[tuple[dict, ShotFork]] = []

    def freeze_new_shots(e: MatchEngine) -> None:
        if not pending_info or pending_info[-1]["frozen"]:
            return
        info = pending_info[-1]
        info["frozen"] = True
        f = e.flight
        if f is None or f.kind != "shot":
            return
        del e._shoot  # a closure cannot be pickled: remove it while freezing
        state = pickle.dumps(e)
        e._shoot = shoot  # type: ignore[method-assign]
        c = info["index"]
        forks.append(
            (
                f.pending,  # filled in with its event id when the shot is published
                ShotFork(
                    event={},
                    actual=f.outcome["type"],
                    shooter_index=c,
                    shooter_id=e.pl[c].player_id,
                    side=e.pl[c].side,
                    kind=info["kind"],
                    header=info["header"],
                    xg_by_player={k: round(float(v), 4) for k, v in info["xg"].items()},
                    state=state,
                ),
            )
        )

    eng._shoot = shoot  # type: ignore[method-assign]
    _begin(eng)
    _advance(eng, on_tick=freeze_new_shots)
    out = {}
    for pending, fork in forks:
        if pending.get("event_id"):
            fork.event = dict(pending)
            out[pending["event_id"]] = fork
    return out


def save_checkpoints(bundle: str | Path, forks: dict[str, ShotFork] | None = None) -> Path:
    root = Path(bundle)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    forks = forks if forks is not None else build_checkpoints(meta)
    recorded = {
        json.loads(line)["event_id"]: json.loads(line)
        for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"type":"shot"' in line or '"type": "shot"' in line
    }
    mismatched = [
        eid for eid, f in forks.items() if eid not in recorded or recorded[eid]["t"] != f.event["t"]
    ]
    if mismatched or len(forks) != len(recorded):
        raise RuntimeError(
            "this machine does not reproduce the recorded match exactly; build checkpoints on the machine "
            "that generated it (or regenerate the match here)"
        )
    path = root / CHECKPOINT_FILE
    with gzip.open(path, "wb", compresslevel=6) as fh:
        pickle.dump({"engine_version": ENGINE_VERSION, "forks": forks}, fh)
    return path


def load_checkpoint(bundle: str | Path, event_id: str) -> ShotFork | None:
    """The frozen state at a shot, if the bundle ships checkpoints (only load bundles you trust)."""
    path = Path(bundle) / CHECKPOINT_FILE
    if not path.exists():
        return None
    with gzip.open(path, "rb") as fh:
        data = pickle.load(fh)
    return data["forks"].get(event_id)


# -------------------------------------------------------------------------- futures
def _set_outcome(eng: MatchEngine, outcome: str) -> None:
    """Change the outcome of the shot in flight, keeping its published event consistent."""
    f = eng.flight
    if outcome == MISSED and f.outcome.get("gk") is None:
        outcome = "off_target"
    f.outcome["type"] = outcome
    pending = getattr(f, "pending", None)
    if outcome == "goal" and f.outcome.get("gk") is not None:
        _beat_keeper(eng, f, pending)
    if pending is not None:
        pending["details"]["outcome"] = outcome
        pending["details"]["on_target"] = outcome in ("goal", "saved")


GOAL_HALF_W, GOAL_H = 3.66, 2.44


def _beat_keeper(eng: MatchEngine, f, pending: dict | None) -> None:
    """A shot the keeper reached, made to go in: it finds the corner away from him instead.

    The same placement for every future (no randomness), so the ghost timeline shows one shot.
    Only the flight's end point moves; the strike, its timing and the result being forced are unchanged.
    """
    gk = f.outcome["gk"]
    gy = float(eng.pos[gk][1])
    end = np.array(f.end, dtype=float)
    centre = 34.0
    off = end[1] - centre
    if abs(end[1] - gy) >= 1.6 and abs(off) >= 2.2:
        return  # already out of his reach: keep it
    side = np.sign(centre - gy) if abs(gy - centre) > 0.35 else (-np.sign(off) if abs(off) > 0.2 else 1.0)
    z0 = float(getattr(f, "end_z", 0.0) or (pending or {}).get("details", {}).get("end_location", {}).get("z") or 0.0)
    kind = (pending or {}).get("details", {}).get("shot_kind")
    z = 1.85 if kind == "free_kick" else (0.3 if z0 < 0.6 else min(1.9, max(1.2, z0)))
    end[1] = centre + side * 3.05
    f.end = end
    if pending is not None:
        d = pending["details"]
        d["end_location"] = {"x": round(float(end[0]), 2), "y": round(float(end[1]), 2), "z": round(z, 2)}
        gm = d.get("goal_mouth")
        if isinstance(gm, dict):
            flip = -1.0 if (gm.get("y_offset_m") or 0) * off < 0 else 1.0  # the mouth offset is in the attacking team's frame
            d["goal_mouth"] = {**gm, "y_offset_m": round(float(flip * side * 3.05), 2), "z_m": round(z, 2)}


FORK_CLIP_S = 7.0  # seconds of the changed moment kept for the ghost timeline
GOAL_PRE_S, GOAL_POST_S = 9.0, 3.5  # a goal in a future is kept from 9 s before to 3.5 s after
CLIP_EVENT_TYPES = {"pass", "shot", "save", "goal", "clearance", "keeper_claim", "kick_off", "block"}


def _states(eng: MatchEngine, period: int, t: float) -> list[list]:
    """The game state about once a minute from the fork to full time:
    [period, t, goals home, goals away, xG home, xG away, reds home, reds away]."""
    home = eng.teams[0].team_id
    g, xg, red = [0, 0], [0.0, 0.0], [0, 0]
    out: list[list] = []
    last = None

    def snap(e: dict) -> list:
        return [e["period"], round(e["t"], 1), g[0], g[1], round(xg[0], 3), round(xg[1], 3), red[0], red[1]]

    for e in eng.events:
        key = (e["period"], int(e["t"] // 60))
        if key != last and (e["period"], e["t"]) >= (period, t):
            out.append(snap(e))
            last = key
        side = 0 if e.get("team_id") == home else 1
        if e["type"] == "goal":
            g[side] += 1
        elif e["type"] == "shot":
            xg[side] += float((e["details"].get("truth") or {}).get("xg") or 0.0)
        elif e["type"] == "card" and e["details"].get("card") in ("red", "second_yellow"):
            red[side] += 1
    if eng.events:
        out.append(snap(eng.events[-1]))
    return out


def _future(args: tuple) -> dict:
    """One future (runs in a worker process).

    ``record_s`` keeps tracking frames for that many seconds after the fork; ``clips`` also keeps
    each later goal (frames and events), which the ghost timeline jumps to.
    """
    state, seed, outcome, record_s, clips_on = args
    eng: MatchEngine = pickle.loads(state)
    eng.rng = np.random.default_rng(seed)
    if outcome:
        _set_outcome(eng, outcome)
    start_events = len(eng.events)
    fork_key = (eng.period, eng.t)
    score0 = {"home": eng.teams[0].score, "away": eng.teams[1].score}
    shot_event = getattr(eng.flight, "pending", None) if eng.flight is not None else None
    frames: list[dict] = []
    clips: list[dict] = []
    t_end = (eng.period, eng.t + record_s)
    if record_s > 0 or clips_on:
        eng.emit_frames = True
        buf: deque = deque(maxlen=int(GOAL_PRE_S * 5) + 1)
        open_clip: dict | None = None

        def keep(e: MatchEngine) -> None:
            nonlocal open_clip
            for kind, item in e._buffer:
                if kind == "event" and clips_on and item["type"] == "goal":
                    own = shot_event is not None and item["details"].get("shot_event_id") == shot_event.get(
                        "event_id"
                    )
                    if not own and open_clip is None:
                        scored_home = item["team_id"] == e.teams[0].team_id
                        open_clip = {
                            "kind": "goal",
                            "score0": {
                                "home": e.teams[0].score - (1 if scored_home else 0),
                                "away": e.teams[1].score - (0 if scored_home else 1),
                            },
                            "frames": list(buf),
                            "until": (item["period"], item["t"] + GOAL_POST_S),
                            "minute": item["minute_display"],
                            "scorer": item["details"].get("scorer_name"),
                            "side": "home" if item["team_id"] == e.teams[0].team_id else "away",
                        }
                if kind != "frame":
                    continue
                key = (item["period"], item["t"])
                if record_s > 0 and key <= t_end:
                    frames.append(item)
                if clips_on:
                    buf.append(item)
                    if open_clip is not None:
                        open_clip["frames"].append(item)
                        if key >= open_clip["until"]:
                            clips.append(open_clip)
                            open_clip = None
            if not clips_on and (e.period, e.t) >= t_end:
                e.emit_frames = False

        _advance(eng, on_tick=keep)
        if open_clip is not None:
            clips.append(open_clip)
    else:
        _advance(eng)
    shot_id = shot_event.get("event_id") if shot_event else None
    goals, fork_scored = [], False
    for e in eng.events[start_events:]:
        if e["type"] != "goal":
            continue
        if shot_id and e["details"].get("shot_event_id") == shot_id:
            fork_scored = True  # the shot we forked on: not a goal "after" it
            continue
        goals.append(
            {
                "minute": e["minute_display"],
                "side": "home" if e["team_id"] == eng.teams[0].team_id else "away",
                "scorer": e["details"].get("scorer_name"),
            }
        )
    out = {
        "seed": seed,
        "final": {"home": eng.teams[0].score, "away": eng.teams[1].score},
        "fork_shot_scored": fork_scored,
        "goals_after": goals,
        "states": _states(eng, *fork_key),
        "frames": frames,
    }
    if clips_on:
        new_events = eng.events[start_events:]
        windows = [{"kind": "fork", "score0": score0, "frames": frames}] + clips
        for c in windows:
            c.pop("until", None)
            if not c["frames"]:
                c["events"] = []
                continue
            lo = (c["frames"][0]["period"], c["frames"][0]["t"] - 0.5)
            hi = (c["frames"][-1]["period"], c["frames"][-1]["t"] + 0.5)
            c["events"] = [
                e for e in new_events if e["type"] in CLIP_EVENT_TYPES and lo <= (e["period"], e["t"]) <= hi
            ]
        out["clips"] = windows
    return out


def _workers(n: int) -> int:
    return max(1, min(n, (os.cpu_count() or 2) - 1 or 1))


def play_futures(
    fork: ShotFork,
    outcome: str | None,
    n: int = 20,
    seed: int = 1000,
    record: int = 0,
    record_s: float = 12.0,
    parallel: bool = True,
    clips: bool = False,
    seeds: list[int] | None = None,
) -> list[dict]:
    """``n`` futures from the fork. ``outcome`` forces the shot (goal / saved), None keeps it.

    The first ``record`` futures also return tracking frames for ``record_s`` seconds (and, with
    ``clips``, every later goal). ``seeds`` replays chosen futures exactly.
    """
    seeds = seeds or [seed + i for i in range(n)]
    jobs = [
        (fork.state, s, outcome, record_s if i < record else 0.0, clips and i < record)
        for i, s in enumerate(seeds)
    ]
    if parallel and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=_workers(len(jobs))) as ex:
            return list(ex.map(_future, jobs, chunksize=max(1, len(jobs) // (4 * _workers(len(jobs))))))
    return [_future(j) for j in jobs]


def _result_of(final: dict) -> str:
    h, a = final["home"], final["away"]
    return "home_win" if h > a else "away_win" if a > h else "draw"


def summarise(futures: list[dict]) -> dict:
    n = len(futures)
    counts = {"home_win": 0, "draw": 0, "away_win": 0}
    for f in futures:
        counts[_result_of(f["final"])] += 1
    scores: dict[str, int] = {}
    for f in futures:
        key = f"{f['final']['home']}-{f['final']['away']}"
        scores[key] = scores.get(key, 0) + 1
    return {
        "futures": n,
        "home_win": round(counts["home_win"] / n, 3),
        "draw": round(counts["draw"] / n, 3),
        "away_win": round(counts["away_win"] / n, 3),
        "most_common_scores": sorted(scores.items(), key=lambda kv: -kv[1])[:3],
        "avg_goals_after": round(sum(len(f["goals_after"]) for f in futures) / n, 2),
        # 95% margin of error on any of the three shares (worst case p = 0.5): say it, don't overclaim
        "margin_95": round(0.98 / n**0.5, 3),
    }


# ----------------------------------------------------------------------- what if
def pick_featured(futures: list[dict], k: int = 1) -> list[dict]:
    """The futures worth showing: the most common result, told by one late goal if possible."""
    if not futures:
        return []
    counts: dict[str, int] = {}
    for f in futures:
        counts[_result_of(f["final"])] = counts.get(_result_of(f["final"]), 0) + 1
    modal = max(counts, key=counts.get)
    typical = [f for f in futures if _result_of(f["final"]) == modal]
    pool = [f for f in typical if f["goals_after"]] or typical  # always tell the most likely story
    pool = sorted(pool, key=lambda f: (len(f["goals_after"]) != 1, len(f["goals_after"]), f["seed"]))
    return pool[:k]


def what_if_shot(
    meta: dict,
    event: dict,
    change: str,
    n: int = 20,
    seed: int = 1000,
    record: int = 1,
    parallel: bool = True,
    fork: ShotFork | None = None,
) -> dict:
    """Settle "what if this shot had gone in / been missed?" with paired simulations.

    ``change`` is "shot_scores" or "shot_misses". Returns the result distribution with and without
    the change (same fork, same random streams), each future's game state over time (for the fan
    of futures), and ``record`` featured futures replayed with clips of the changed moment and of
    every later goal (for the ghost timeline).
    """
    fork = fork or fork_at_shot(meta, event)
    forced = SCORED if change == "shot_scores" else MISSED
    if (forced == SCORED) == (fork.actual == "goal"):
        raise ValueError(f"the shot already ended as '{fork.actual}': nothing to change")
    changed = play_futures(fork, forced, n, seed, parallel=parallel)
    actual = play_futures(fork, None, n, seed, parallel=parallel)
    chosen = pick_featured(changed, record)
    featured = play_futures(
        fork,
        forced,
        seeds=[f["seed"] for f in chosen],
        record=len(chosen),
        record_s=FORK_CLIP_S,
        clips=True,
        parallel=parallel,
    )
    return {
        "event_id": event["event_id"],
        "minute": event["minute_display"],
        "shooter_id": fork.shooter_id,
        "team_side": "home" if fork.side == 0 else "away",
        "actual_outcome": fork.actual,
        "changed_outcome": forced,
        "with_change": summarise(changed),
        "as_it_happened": summarise(actual),
        "states": {
            "with_change": [f["states"] for f in changed],
            "as_it_happened": [f["states"] for f in actual],
        },
        "featured": [
            {"seed": f["seed"], "final": f["final"], "goals_after": f["goals_after"], "clips": f["clips"]}
            for f in featured
        ],
        "method": f"{n} paired futures from the instant of the shot (same random streams with and without the change)",
    }


# --------------------------------------------------------------------- ghost scout
def league_candidates(
    roles: tuple[str, ...] = ATTACKING_ROLES, exclude: set[str] | None = None
) -> list[PlayerProfile]:
    out = []
    for spec in CLUB_SPECS:
        for p in get_club(spec.code).squad:
            if p.role in roles and p.player_id not in (exclude or set()):
                out.append(p)
    return out


def scout_shot(
    meta: dict,
    event: dict,
    n: int = 20,
    seed: int = 1000,
    roles: tuple[str, ...] = ATTACKING_ROLES,
    top: int = 5,
    parallel: bool = True,
    fork: ShotFork | None = None,
) -> dict:
    """Ghost Scout: who in the league would most likely have turned this chance into a better result?

    For each candidate: the engine's goal probability for this exact chance with their ratings,
    and the shooter's team win probability that implies, combining the futures where the chance
    went in with those where it did not.
    """
    cands = league_candidates(roles)
    by_id = {p.player_id: p for p in cands}
    fork = fork if fork is not None and len(fork.xg_by_player) > 1 else fork_at_shot(meta, event, cands)
    scored = summarise(play_futures(fork, SCORED, n, seed, parallel=parallel))
    missed = summarise(play_futures(fork, MISSED, n, seed, parallel=parallel))
    win_key = "home_win" if fork.side == 0 else "away_win"
    w_goal, w_miss = scored[win_key], missed[win_key]

    def row(pid: str, p_goal: float) -> dict:
        prof = by_id.get(pid)
        club = pid.split("-")[0].upper()
        return {
            "player_id": pid,
            "name": prof.name if prof else pid,
            "club": club,
            "role": prof.role if prof else None,
            "finishing": prof.attributes.finishing if prof else None,
            "composure": prof.attributes.composure if prof else None,
            "goal_probability": round(p_goal, 3),
            "team_win_probability": round(p_goal * w_goal + (1 - p_goal) * w_miss, 3),
        }

    own_p = fork.xg_by_player[fork.shooter_id]
    ranked = sorted(
        (row(pid, p) for pid, p in fork.xg_by_player.items() if pid != fork.shooter_id and pid in by_id),
        key=lambda r: -r["goal_probability"],
    )
    actual = row(fork.shooter_id, own_p)
    actual["name"] = event.get("player_name") or actual["name"]
    return {
        "event_id": event["event_id"],
        "minute": event["minute_display"],
        "team_side": "home" if fork.side == 0 else "away",
        "actual_outcome": fork.actual,
        "chance": {"kind": fork.kind, "header": fork.header},
        "taker": actual,
        "win_probability_if_scored": w_goal,
        "win_probability_if_missed": w_miss,
        "candidates_evaluated": len(ranked),
        "best": ranked[:top],
        "method": (
            "engine shot model re-evaluated for this exact chance with each player's ratings; "
            f"{n} paired futures each for scored and missed"
        ),
    }


if __name__ == "__main__":  # python -m matchengine.ghost <bundle>: record checkpoints for a match
    import sys

    from matchengine import ghost as _ghost  # pickle the classes under their real module name

    for b in sys.argv[1:]:
        print(_ghost.save_checkpoints(b))
