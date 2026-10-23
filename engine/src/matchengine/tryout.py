"""Player try-outs: any player in the league, in any position, for a full match, many times over.

A try-out plays the same fixture ``n`` times with the candidate in the chosen position and ``n`` times with
the player who really starts there, on the same seeds, so both sides of the comparison get the same luck to
begin with and the only difference is the player. Both play the full 90 (neither is substituted for
tactics or fatigue), so the comparison is like for like.

Football results are noisy, so the verdict leans on measures that settle quickly:

* **expected points** from the quality of every shot in the match (the chance of winning, drawing or losing
  given the shots each side took), not just the final score;
* **xG for and against** the team;
* **the player's own output per 90**: goals, xG, assists, chances created, take-ons, tackles, interceptions...

Every average comes with a 95% margin of error. Runs are cached per seed, so the incumbent's runs are
shared by every try-out in that position and a try-out can be extended later without starting again.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import math
import os
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .engine import ENGINE_VERSION, MatchEngine
from .formations import slots_for
from .league import CLUB_SPECS, get_club
from .models import ClubProfile, MatchConfig, PlayerProfile

SEED0 = 50_000  # try-out seeds are separate from any real match's seed
CLIP_PRE_S, CLIP_POST_S = 9.0, 3.5  # the featured moment: 9 s of build-up, 3.5 s after

# per-90 stats reported for the player (higher is better unless listed in LOWER_IS_BETTER)
PLAYER_KEYS = (
    "goals", "xg", "shots", "shots_on_target", "assists", "xa", "key_passes", "passes", "passes_completed",
    "progressive_passes", "crosses_completed", "dribbles", "dribbles_completed", "tackles", "tackles_won",
    "interceptions", "ball_recoveries", "clearances", "blocks", "aerials_won", "pressures", "fouls",
    "dispossessed", "saves", "goals_conceded", "xg_faced", "distance_km", "sprints",
)
LOWER_IS_BETTER = {"fouls", "dispossessed", "goals_conceded"}


# ----------------------------------------------------------------------------------- setup
def lineup(home: str, away: str) -> dict[str, list[dict]]:
    """Each side's starting positions: [{slot, label, role, player_id, name, player_role}]."""
    out = {}
    for side, code in (("home", home), ("away", away)):
        club = get_club(code)
        out[side] = [
            {"slot": k, "label": sl.label, "role": sl.role, "player_id": pid, "name": club.player(pid).name,
             "player_role": club.player(pid).role}
            for k, (sl, pid) in enumerate(zip(slots_for(club.formation), club.starting_xi))
        ]
    return out


def find_player(player_id: str) -> tuple[PlayerProfile, ClubProfile]:
    for spec in CLUB_SPECS:
        club = get_club(spec.code)
        for p in club.squad:
            if p.player_id == player_id:
                return p, club
    raise KeyError(f"No player {player_id!r} in the league")


def candidates(home: str, away: str, side: str, slot: int, roles: tuple[str, ...] | None = None) -> list[dict]:
    """League players who could be tried in that position (anyone except the two starting XIs and the
    opponent's squad). ``roles`` narrows it; by default players whose natural role fits the slot come first."""
    mine, theirs = (get_club(home), get_club(away)) if side == "home" else (get_club(away), get_club(home))
    role = slots_for(mine.formation)[slot].role
    near = {
        "GK": ("GK",), "CB": ("CB", "DM", "FB"), "FB": ("FB", "W", "CB"), "DM": ("DM", "CM", "CB"),
        "CM": ("CM", "DM", "AM"), "AM": ("AM", "CM", "W"), "W": ("W", "AM", "ST"), "ST": ("ST", "W", "AM"),
    }[role]
    exclude = set(mine.starting_xi) | {p.player_id for p in theirs.squad}
    out = []
    for spec in CLUB_SPECS:
        club = get_club(spec.code)
        for p in club.squad:
            if p.player_id in exclude or (roles and p.role not in roles):
                continue
            fit = 0 if p.role == role else 1 if p.role in near else 2
            out.append({"player_id": p.player_id, "name": p.name, "club": club.name, "club_code": club.code,
                        "role": p.role, "age": p.age, "fit": fit, "own_club": club.code == mine.code})
    return sorted(out, key=lambda c: (c["fit"], c["name"]))


def _club_with(club: ClubProfile, slot: int, player: PlayerProfile) -> ClubProfile:
    """The club with ``player`` starting in ``slot`` (and not on its bench)."""
    xi = list(club.starting_xi)
    xi[slot] = player.player_id
    squad = club.squad if any(p.player_id == player.player_id for p in club.squad) else (*club.squad, player)
    bench = tuple(b for b in club.bench if b != player.player_id)
    return dataclasses.replace(club, squad=squad, starting_xi=tuple(xi), bench=bench)


def _config(home: str, away: str, side: str, slot: int, candidate: str | None, seed: int) -> MatchConfig:
    h, a = get_club(home), get_club(away)
    if candidate:
        player, _ = find_player(candidate)
        if side == "home":
            h = _club_with(h, slot, player)
        else:
            a = _club_with(a, slot, player)
    return MatchConfig(home=h, away=a, seed=seed, match_id=f"tryout-{home}-{away}-{seed}".lower())


# ----------------------------------------------------------------------------------- one match
def _poisson_binomial(ps: list[float]) -> np.ndarray:
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return dist


def expected_points(xg_for: list[float], xg_against: list[float]) -> tuple[float, float, float, float]:
    """(win, draw, loss, expected points) from every shot's xG: each shot scores independently with its xG."""
    f, a = _poisson_binomial(xg_for), _poisson_binomial(xg_against)
    joint = np.outer(f, a)
    win = float(np.tril(joint, -1).sum())
    draw = float(np.trace(joint))
    loss = max(0.0, 1.0 - win - draw)
    return win, draw, loss, 3 * win + draw


def player_stats(events: list[dict], player_id: str, team_id: str) -> dict:
    s: dict = defaultdict(float)
    last_pass: dict | None = None
    for e in events:
        t, d, pid = e["type"], e.get("details") or {}, e.get("player_id")
        if t == "pass" and e.get("team_id") == team_id and d.get("outcome") == "complete":
            last_pass = e
        if t == "shot" and e.get("team_id") == team_id:
            xg = float((d.get("truth") or {}).get("xg") or 0.0)
            if pid == player_id:
                s["shots"] += 1
                s["xg"] += xg
                s["shots_on_target"] += bool(d.get("on_target"))
            lp = last_pass
            if lp and lp.get("player_id") == player_id and (lp.get("details") or {}).get("recipient_id") == pid \
                    and (e["period"], e["t"]) >= (lp["period"], lp["t"]) and e["t"] - lp["t"] < 8:
                s["key_passes"] += 1
                s["xa"] += xg
        if t == "shot" and e.get("team_id") != team_id and d.get("goalkeeper_id") == player_id:
            s["xg_faced"] += float((d.get("truth") or {}).get("xg") or 0.0)
        if t == "goal":
            if d.get("scorer_id") == player_id:
                s["goals"] += 1
            if d.get("assist_id") == player_id:
                s["assists"] += 1
            if e.get("team_id") != team_id:
                s["_conceded_team"] += 1
        if t == "tackle" and d.get("opponent_id") == player_id and d.get("outcome") == "won":
            s["dispossessed"] += 1
        if pid != player_id:
            continue
        if t == "pass":
            s["passes"] += 1
            if d.get("outcome") == "complete":
                s["passes_completed"] += 1
                if (d.get("progression_m") or 0) >= 10:
                    s["progressive_passes"] += 1
                if d.get("pass_kind") in ("cross", "free_kick_cross"):
                    s["crosses_completed"] += 1
        elif t == "dribble":
            s["dribbles"] += 1
            s["dribbles_completed"] += d.get("outcome") == "success"
            s["dispossessed"] += d.get("outcome") != "success"
        elif t == "tackle":
            s["tackles"] += 1
            s["tackles_won"] += d.get("outcome") == "won"
        elif t in ("interception", "ball_recovery", "clearance", "pressure", "foul"):
            s[{"interception": "interceptions", "ball_recovery": "ball_recoveries", "clearance": "clearances",
               "pressure": "pressures", "foul": "fouls"}[t]] += 1
        elif t == "block":
            s["blocks"] += 1
        elif t == "aerial_duel":
            s["aerials_won"] += d.get("outcome") == "won"
        elif t == "save":
            s["saves"] += 1
    return s


def _best_moment(events: list[dict], player_id: str, team_id: str) -> tuple[float, dict] | None:
    """The player's best moment in a match: a goal, an assist, a big chance, a key pass or a big save."""
    best = None
    last_pass = None
    for e in events:
        t, d, pid = e["type"], e.get("details") or {}, e.get("player_id")
        if t == "pass" and e.get("team_id") == team_id and d.get("outcome") == "complete":
            last_pass = e
        score, kind = 0.0, None
        if t == "goal" and d.get("scorer_id") == player_id:
            score, kind = 10 + float(d.get("xg") or 0), "goal"
        elif t == "goal" and d.get("assist_id") == player_id:
            score, kind = 8, "assist"
        elif t == "shot" and pid == player_id:
            xg = float((d.get("truth") or {}).get("xg") or 0)
            score, kind = 2 + 6 * xg, "chance"
        elif t == "shot" and e.get("team_id") == team_id and last_pass and last_pass.get("player_id") == player_id \
                and (last_pass.get("details") or {}).get("recipient_id") == pid:
            score, kind = 1.5 + 4 * float((d.get("truth") or {}).get("xg") or 0), "key_pass"
        elif t == "save" and pid == player_id:
            score, kind = 3 + float(d.get("shot_speed_kmh") or 0) / 60, "save"
        if kind and (best is None or score > best[0]):
            best = (score, {"kind": kind, "period": e["period"], "t": e["t"], "minute": e.get("minute_display"),
                            "event_id": e.get("event_id")})
    return best


def _play(args: tuple) -> dict:
    """One full match (runs in a worker process)."""
    home, away, side, slot, candidate, seed = args
    cfg = _config(home, away, side, slot, candidate, seed)
    eng = MatchEngine(cfg, emit_frames=False)
    s = 0 if side == "home" else 1
    idx = s * 11 + slot
    eng.protected = frozenset({idx})
    for _ in eng.run():
        pass
    player_id = candidate or get_club(home if side == "home" else away).starting_xi[slot]
    team_id = eng.teams[s].team_id
    xg_for, xg_against = [], []
    for e in eng.events:
        if e["type"] == "shot":
            xg = float((e["details"].get("truth") or {}).get("xg") or 0.0)
            (xg_for if e["team_id"] == team_id else xg_against).append(xg)
    win, draw, loss, xpts = expected_points(xg_for, xg_against)
    st = player_stats(eng.events, player_id, team_id)
    # physical output and minutes for whoever played in the slot (the protected player, barring injury)
    everyone = list(eng.teams[s].players) + list(eng.teams[s].used_players)
    me = next((p for p in everyone if p.player_id == player_id), None)
    for i in range(len(eng.pl)):
        if eng.pl[i] is not None and eng.pl[i].on_pitch:
            eng.sync_stats(i)
    end_s = 45 * 60 * (len(eng.period_lengths) - 1) + (eng.period_lengths[-1] if eng.period_lengths else 0.0)
    if me is not None:
        mins_to = me.minutes_to if me.minutes_to is not None else end_s
        st["minutes"] = max(1.0, (mins_to - (me.minutes_from or 0.0)) / 60.0)
        st["distance_km"] = me.distance / 1000.0
        st["sprints"] = me.sprints
    else:
        st["minutes"] = 0.0
    if eng.pl[idx].slot.role == "GK":
        st["goals_conceded"] = st.pop("_conceded_team", 0)
    else:
        st.pop("_conceded_team", None)
    gf, ga = eng.teams[s].score, eng.teams[1 - s].score
    best = _best_moment(eng.events, player_id, team_id)
    return {
        "seed": seed,
        "score": {"home": eng.teams[0].score, "away": eng.teams[1].score},
        "result": "W" if gf > ga else "D" if gf == ga else "L",
        "xg_for": round(sum(xg_for), 3), "xg_against": round(sum(xg_against), 3),
        "xwin": round(win, 4), "xdraw": round(draw, 4), "xpts": round(xpts, 4),
        "player": {k: round(float(v), 4) for k, v in st.items()},
        "best": ({"score": round(best[0], 3), **best[1]} if best else None),
    }


# ----------------------------------------------------------------------------------- aggregation
def _mean_ci(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = float(np.mean(xs))
    if n < 2:
        return m, 0.0
    return m, float(1.96 * np.std(xs, ddof=1) / math.sqrt(n))


def summarise_runs(runs: list[dict]) -> dict:
    n = len(runs)
    res = defaultdict(int)
    for r in runs:
        res[r["result"]] += 1
    xpts, xpts_m = _mean_ci([r["xpts"] for r in runs])
    xwin = float(np.mean([r["xwin"] for r in runs])) if n else 0.0
    xdraw = float(np.mean([r["xdraw"] for r in runs])) if n else 0.0
    xgf, xgf_m = _mean_ci([r["xg_for"] for r in runs])
    xga, xga_m = _mean_ci([r["xg_against"] for r in runs])
    mins = sum(r["player"].get("minutes", 0.0) for r in runs) or 1.0
    per90 = {k: round(sum(r["player"].get(k, 0.0) for r in runs) / mins * 90, 3) for k in PLAYER_KEYS}
    return {
        "futures": n,
        "win": round(res["W"] / n, 3) if n else 0.0, "draw": round(res["D"] / n, 3) if n else 0.0,
        "loss": round(res["L"] / n, 3) if n else 0.0,
        "result_margin_95": round(0.98 / math.sqrt(n), 3) if n else 1.0,
        # the same split from chance quality (every shot's xG): what the team deserved, far less noisy
        "xwin": round(xwin, 3), "xdraw": round(xdraw, 3), "xloss": round(max(0.0, 1 - xwin - xdraw), 3),
        "xpts": round(xpts, 3), "xpts_margin_95": round(xpts_m, 3),
        "xg_for": round(xgf, 3), "xg_for_margin_95": round(xgf_m, 3),
        "xg_against": round(xga, 3), "xg_against_margin_95": round(xga_m, 3),
        "minutes": round(mins / n, 1) if n else 0.0,
        "per90": per90,
    }


def compare(base: list[dict], trial: list[dict]) -> dict:
    """Paired differences on the seeds both have."""
    b = {r["seed"]: r for r in base}
    pairs = [(b[r["seed"]], r) for r in trial if r["seed"] in b]
    if not pairs:
        return {"pairs": 0}
    d_xpts, d_xpts_m = _mean_ci([t["xpts"] - o["xpts"] for o, t in pairs])
    d_xgf, d_xgf_m = _mean_ci([t["xg_for"] - o["xg_for"] for o, t in pairs])
    d_xga, d_xga_m = _mean_ci([t["xg_against"] - o["xg_against"] for o, t in pairs])
    return {
        "pairs": len(pairs),
        "xpts": round(d_xpts, 3), "xpts_margin_95": round(d_xpts_m, 3),
        "xg_for": round(d_xgf, 3), "xg_for_margin_95": round(d_xgf_m, 3),
        "xg_against": round(d_xga, 3), "xg_against_margin_95": round(d_xga_m, 3),
        # how sure we are the change is real: share of the margin the difference clears
        "clear": bool(abs(d_xpts) > d_xpts_m) if len(pairs) > 1 else False,
    }


# ----------------------------------------------------------------------------------- cache + runner
class RunCache:
    """Per-seed results on disk: ``<dir>/<home>-<away>/<side>-<slot>-<player or base>.json``."""

    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None

    def _path(self, home: str, away: str, side: str, slot: int, who: str) -> Path | None:
        if not self.root:
            return None
        return self.root / f"{home}-{away}".lower() / f"{side}-{slot}-{who}.json"

    def load(self, home, away, side, slot, who) -> dict[int, dict]:
        p = self._path(home, away, side, slot, who)
        if not p or not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("engine") != ENGINE_VERSION:
            return {}
        return {int(r["seed"]): r for r in data["runs"]}

    def save(self, home, away, side, slot, who, runs: dict[int, dict]) -> None:
        p = self._path(home, away, side, slot, who)
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"engine": ENGINE_VERSION, "runs": [runs[k] for k in sorted(runs)]}), encoding="utf-8")
        tmp.replace(p)


def _workers(n: int) -> int:
    return max(1, min(n, int(os.environ.get("MATCH_WORKERS") or (os.cpu_count() or 2))))


def run_tryout(
    home: str, away: str, side: str, slot: int, candidate: str, n: int = 20, cache_dir: str | Path | None = None,
    parallel: bool = True, on_progress: Callable[[dict], None] | None = None, seed0: int = SEED0,
) -> dict:
    """Try ``candidate`` in ``side``'s ``slot`` against the player who starts there, over ``n`` paired matches.

    ``on_progress`` is called with the partial result after every finished match (for live streaming).
    """
    cache = RunCache(cache_dir)
    incumbent = get_club(home if side == "home" else away).starting_xi[slot]
    if candidate == incumbent:
        raise ValueError("That player already starts there")
    seeds = list(range(seed0, seed0 + n))
    base = cache.load(home, away, side, slot, "base")
    trial = cache.load(home, away, side, slot, candidate)
    jobs = [(home, away, side, slot, None, s) for s in seeds if s not in base]
    jobs += [(home, away, side, slot, candidate, s) for s in seeds if s not in trial]

    def report(done: int) -> dict:
        b = [base[s] for s in seeds if s in base]
        t = [trial[s] for s in seeds if s in trial]
        out = assemble(home, away, side, slot, candidate, b, t)
        out["progress"] = {"done": done, "total": len(jobs), "cached": 2 * n - len(jobs)}
        return out

    if jobs:
        def take(job, r):
            (trial if job[4] else base)[r["seed"]] = r

        if parallel and len(jobs) > 1 and _workers(len(jobs)) > 1:
            with ProcessPoolExecutor(max_workers=_workers(len(jobs))) as ex:
                futs = {ex.submit(_play, j): j for j in jobs}
                for k, f in enumerate(as_completed(futs), 1):
                    take(futs[f], f.result())
                    if on_progress:
                        on_progress(report(k))
                    if k % 4 == 0:
                        cache.save(home, away, side, slot, "base", base)
                        cache.save(home, away, side, slot, candidate, trial)
        else:
            for k, j in enumerate(jobs, 1):
                take(j, _play(j))
                if on_progress:
                    on_progress(report(k))
        cache.save(home, away, side, slot, "base", base)
        cache.save(home, away, side, slot, candidate, trial)
    return report(len(jobs))


def assemble(home: str, away: str, side: str, slot: int, candidate: str, base: list[dict], trial: list[dict]) -> dict:
    lu = lineup(home, away)[side][slot]
    player, club = find_player(candidate)
    inc_p, inc_club = find_player(lu["player_id"])
    team = get_club(home if side == "home" else away)
    opp = get_club(away if side == "home" else home)
    with_ = summarise_runs(trial)
    as_is = summarise_runs(base)
    best = max((r for r in trial if r.get("best")), key=lambda r: r["best"]["score"], default=None)
    return {
        "fixture": {"home": home, "away": away, "side": side, "team": team.name, "opponent": opp.name},
        "position": {"slot": slot, "label": lu["label"], "role": lu["role"]},
        "candidate": {"player_id": player.player_id, "name": player.name, "club": club.name, "role": player.role,
                      "age": player.age, "attributes": player.attributes.as_dict()},
        "incumbent": {"player_id": inc_p.player_id, "name": inc_p.name, "club": inc_club.name, "role": inc_p.role,
                      "age": inc_p.age, "attributes": inc_p.attributes.as_dict()},
        "with_candidate": with_,
        "with_incumbent": as_is,
        "difference": compare(base, trial),
        "featured": ({"seed": best["seed"], **best["best"], "score_line": best["score"]} if best else None),
        "method": (f"{len(trial)} full matches of {team.name} v {opp.name} with {player.name} at {lu['label']} and "
                   f"{len(base)} with {inc_p.name}, paired on the same seeds; both play the full match. Expected "
                   "points come from the quality of every shot in each match."),
        "engine": ENGINE_VERSION,
    }


# ----------------------------------------------------------------------------------- featured clip
def featured_clip(home: str, away: str, side: str, slot: int, candidate: str, seed: int, moment: dict) -> dict | None:
    """Re-play one try-out match with tracking on and cut the clip around the candidate's best moment."""
    cfg = _config(home, away, side, slot, candidate, seed)
    eng = MatchEngine(cfg, emit_frames=True)
    eng.protected = frozenset({(0 if side == "home" else 1) * 11 + slot})
    lo = (moment["period"], moment["t"] - CLIP_PRE_S)
    hi = (moment["period"], moment["t"] + CLIP_POST_S)
    frames, score0 = [], None
    for kind, item in eng.run():
        if kind != "frame":
            continue
        key = (item["period"], item["t"])
        if key < lo:
            continue
        if score0 is None:
            score0 = {"home": eng.teams[0].score, "away": eng.teams[1].score}
        frames.append(item)
        if key >= hi:
            break
    if not frames:
        return None
    events = [e for e in eng.events if lo <= (e["period"], e["t"]) <= hi]
    hit = next((e for e in events if e.get("event_id") == moment.get("event_id")), None)
    if hit is None:  # the re-play didn't reproduce the match (a different machine): no clip
        return None
    return {"kind": "tryout", "moment": moment["kind"], "minute": moment.get("minute"), "score0": score0,
            "frames": frames, "events": events, "seed": seed, "candidate": candidate}


def save_clip(clip: dict, path: str | Path) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(clip, fh)


# ----------------------------------------------------------------------------------- projections + boards
# A quick, transparent first read for every player in every position: a weighted average of the attributes
# that role relies on. It ranks the board instantly; simulations then confirm or overturn it.
ROLE_WEIGHTS: dict[str, dict[str, float]] = {
    "GK": {"goalkeeping": 0.55, "positioning": 0.15, "composure": 0.1, "passing": 0.1, "first_touch": 0.05, "strength": 0.05},
    "CB": {"tackling": 0.25, "positioning": 0.2, "heading": 0.15, "strength": 0.15, "pace": 0.08, "passing": 0.08, "composure": 0.09},
    "FB": {"pace": 0.18, "tackling": 0.18, "stamina": 0.12, "crossing": 0.12, "positioning": 0.12, "acceleration": 0.1, "passing": 0.1, "dribbling": 0.08},
    "DM": {"tackling": 0.2, "positioning": 0.18, "passing": 0.18, "vision": 0.1, "stamina": 0.1, "strength": 0.1, "composure": 0.14},
    "CM": {"passing": 0.22, "vision": 0.18, "first_touch": 0.12, "stamina": 0.12, "dribbling": 0.1, "tackling": 0.1, "composure": 0.1, "shot_power": 0.06},
    "AM": {"vision": 0.2, "passing": 0.18, "dribbling": 0.16, "first_touch": 0.12, "finishing": 0.12, "composure": 0.12, "acceleration": 0.1},
    "W": {"pace": 0.18, "dribbling": 0.2, "acceleration": 0.12, "crossing": 0.14, "finishing": 0.12, "first_touch": 0.1, "passing": 0.08, "vision": 0.06},
    "ST": {"finishing": 0.28, "composure": 0.14, "positioning": 0.14, "first_touch": 0.1, "pace": 0.1, "heading": 0.1, "shot_power": 0.08, "dribbling": 0.06},
}


def projected(player: PlayerProfile, role: str) -> float:
    a = player.attributes.as_dict()
    w = ROLE_WEIGHTS[role]
    score = sum(a[k] * v for k, v in w.items()) / sum(w.values())
    if player.role != role and not (role != "GK" and player.role != "GK"):
        score -= 25  # an outfielder in goal (or a keeper outfield) is not a real option
    return round(score, 1)


def board(home: str, away: str, side: str, ready: int = 2) -> list[dict]:
    """Every starting position on ``side`` with all league candidates ranked by projection, and which
    ``ready`` players to pre-simulate: the top projected player and the best under-23 (the 'hidden gem')."""
    out = []
    for pos in lineup(home, away)[side]:
        inc, _ = find_player(pos["player_id"])
        cands = []
        for c in candidates(home, away, side, pos["slot"]):
            p, _club = find_player(c["player_id"])
            cands.append({**c, "projected": projected(p, pos["role"])})
        cands.sort(key=lambda c: -c["projected"])
        picks: list[str] = []
        if cands:
            picks.append(cands[0]["player_id"])
            young = [c for c in cands if c["age"] <= 23 and c["player_id"] not in picks]
            if ready > 1 and young:
                picks.append(young[0]["player_id"])
            for c in cands:
                if len(picks) >= ready:
                    break
                if c["player_id"] not in picks:
                    picks.append(c["player_id"])
        for c in cands:
            c["ready"] = c["player_id"] in picks
            c["tag"] = ("Top projected" if c["player_id"] == picks[0] else "Best under-23" if c["age"] <= 23 else "Next best") \
                if c["ready"] else None
        out.append({**pos, "incumbent_projected": projected(inc, pos["role"]), "candidates": cands})
    return out
