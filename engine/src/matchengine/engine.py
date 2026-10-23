"""The match engine: a 10 Hz agent-based football simulation.

Every tick, 22 players move towards tactically computed targets with realistic
acceleration, top speed and fatigue; the ball carrier makes decisions from what
it can see; duels, passes and shots are resolved by probability models; and the
referee, managers and clock run the match around them.

The engine yields two streams in time order:

* ``("event", dict)``: football events (passes, shots, tackles, pressure...),
  published once their outcome is known, like a live data feed.
* ``("frame", dict)``: tracking frames with every player's and the ball's position.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from . import tactics
from .decisions import Action, decide, evaluate_pass, goal_mouth_point, shot_model
from .events import SCHEMA_VERSION, EventType, clock_strings, match_seconds
from .formations import slots_for
from .geometry import boundary_crossing, clamp, in_pitch, sigmoid
from .models import MatchConfig, PlayerProfile
from .params import (
    CENTER,
    DEFAULT_PARAMS,
    DT,
    GOAL_HALF_WIDTH,
    PENALTY_BOX_DEPTH,
    PENALTY_BOX_HALF_WIDTH,
    PENALTY_SPOT_DIST,
    PERIOD_SECONDS,
    PITCH_LENGTH,
    PITCH_WIDTH,
    TICK_HZ,
    EngineParams,
)
from .state import BallState, Flight, PlayerState, TeamState
from .tactics import to_tf

ENGINE_VERSION = "1.0.0"
N_PLAYERS = 22
CROSSY = ("cross", "corner", "free_kick_cross")


@dataclass
class MatchResult:
    meta: dict
    events: list[dict]
    frames: list[dict]
    summary: dict
    timeline: list[dict] = field(default_factory=list)


class MatchEngine:
    def __init__(
        self,
        config: MatchConfig,
        params: EngineParams = DEFAULT_PARAMS,
        scenario=None,
        frame_hz: int = 10,
        emit_frames: bool = True,
    ):
        self.cfg = config
        self.p = params
        self.rng = np.random.default_rng(config.seed)
        self.match_id = config.resolved_match_id()
        self.scenario = scenario
        self.emit_frames = emit_frames
        self.frame_every = max(1, round(TICK_HZ / max(1, frame_hz)))

        self.pos = np.zeros((N_PLAYERS, 2))
        self.vel = np.zeros((N_PLAYERS, 2))
        self.target = np.zeros((N_PLAYERS, 2))
        self.urg = np.zeros(N_PLAYERS, dtype=int)
        self.top = np.zeros(N_PLAYERS)
        self.top_eff = np.zeros(N_PLAYERS)
        self.accel = np.zeros(N_PLAYERS)
        self.active = np.zeros(N_PLAYERS, dtype=bool)
        # Physical state and counters (synced into PlayerState on substitution / at full time).
        self.energy = np.ones(N_PLAYERS)
        self.s_dist = np.zeros(N_PLAYERS)
        self.s_hsr = np.zeros(N_PLAYERS)
        self.s_spr = np.zeros(N_PLAYERS)
        self.s_nspr = np.zeros(N_PLAYERS, dtype=int)
        self.s_maxsp = np.zeros(N_PLAYERS)
        self.s_above = np.zeros(N_PLAYERS)  # continuous seconds above sprint threshold
        self.s_counted = np.zeros(N_PLAYERS, dtype=bool)
        self.recover_until = np.zeros(N_PLAYERS)
        self.stam_mult = np.ones(N_PLAYERS)
        self.shape_smooth: np.ndarray | None = None
        self.slot_u = np.zeros(N_PLAYERS)
        self.slot_y = np.zeros(N_PLAYERS)
        self.is_gk = np.zeros(N_PLAYERS, dtype=bool)
        self.drift_phase = self.rng.uniform(0, 2 * math.pi, N_PLAYERS)
        self.pl: list[PlayerState] = [None] * N_PLAYERS  # type: ignore[list-item]
        self._active_cache: dict[int, np.ndarray] = {}

        self.teams = [self._init_team(0), self._init_team(1)]

        self.ball_pos = np.array(CENTER, dtype=float)
        self.ball_vel = np.zeros(2)
        self.ball_z = 0.0
        self.ball_state = BallState.DEAD
        self.carrier: int | None = None
        self.flight: Flight | None = None
        self.restart: dict | None = None
        self.keeper_holding = False
        self.pressing_idx: int | None = None
        self.cover_idx: int | None = None

        self.period = 1
        self.t = 0.0
        self.tick = 0
        self.seq = 0
        self.frame_no = 0
        self.possession_id = 0
        self.possession_side: int | None = None
        self.poss_start_t = 0.0
        self.poss_passes = 0
        self.play_pattern = "kick_off"
        self.last_touch_side: int | None = None
        self.carrier_spell = 0
        self._pressured: set[tuple[int, int]] = set()
        self.last_completed: dict | None = None
        self.stoppage = [0.0, 0.0]
        self.added_minutes: list[int | None] = [None, None]
        self.period_end_t = float(PERIOD_SECONDS)
        self.ball_in_play_s = 0.0
        self.possession_s = [0.0, 0.0]
        self.period_lengths: list[float] = []
        self.events: list[dict] = []
        self._buffer: list[tuple[str, dict]] = []
        self._incidents_done: set[int] = set()
        self._pending_injury: list[dict] = []
        self._next_minute_check = 60.0
        self.goals: list[dict] = []
        self._sub_plan = [self._plan_subs(), self._plan_subs()]

    # ------------------------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------------------------
    def _init_team(self, side: int) -> TeamState:
        club = self.cfg.home if side == 0 else self.cfg.away
        slots = list(slots_for(club.formation))
        players: list[PlayerState] = []
        for k, pid in enumerate(club.starting_xi):
            prof = club.player(pid)
            idx = side * 11 + k
            ps = PlayerState(profile=prof, side=side, idx=idx, slot=slots[k])
            players.append(ps)
            self._install(ps)
        bench = [club.player(pid) for pid in club.bench]
        team = TeamState(
            side=side,
            club=club,
            tactics=club.tactics,
            slots=slots,
            players=players,
            bench=bench,
            direction=1 if side == 0 else -1,
            confidence=0.08 if side == 0 else 0.0,
        )
        return team

    def _install(self, ps: PlayerState) -> None:
        a = ps.profile.attributes
        p = self.p
        i = ps.idx
        self.pl[i] = ps
        self.top[i] = p.top_speed_min + (p.top_speed_max - p.top_speed_min) * (a.pace / 99.0)
        self.accel[i] = p.accel_min + (p.accel_max - p.accel_min) * (a.acceleration / 99.0)
        self.energy[i] = ps.energy
        self.stam_mult[i] = 1.6 - a.stamina / 100.0
        self.set_slot(i, ps.slot)
        self.top_eff[i] = self.top[i] * (0.8 + 0.2 * ps.energy)
        self.active[i] = ps.on_pitch
        for arr in (self.s_dist, self.s_hsr, self.s_spr, self.s_maxsp, self.s_above, self.recover_until):
            arr[i] = 0.0
        self.s_nspr[i] = 0
        self.s_counted[i] = False
        self._active_cache.clear()

    def set_slot(self, i: int, slot) -> None:
        self.pl[i].slot = slot
        self.slot_u[i] = slot.u
        self.slot_y[i] = slot.y
        self.is_gk[i] = slot.role == "GK"

    def sync_stats(self, i: int) -> None:
        """Copy array-held physical counters into the PlayerState currently at index ``i``."""
        ps = self.pl[i]
        ps.energy = float(self.energy[i])
        ps.distance = float(self.s_dist[i])
        ps.hsr_distance = float(self.s_hsr[i])
        ps.sprint_distance = float(self.s_spr[i])
        ps.sprints = int(self.s_nspr[i])
        ps.max_speed = float(self.s_maxsp[i])

    def _plan_subs(self) -> list[tuple[float, int]]:
        r = self.rng
        n_windows = int(r.choice([2, 3, 3]))
        times = sorted(r.uniform(56, 87, n_windows))
        counts = {2: [2, 2], 3: [2, 2, 1]}[n_windows]
        if r.random() < 0.25:
            counts[-1] = max(0, counts[-1] - 1)
        return [(float(t) * 60.0, c) for t, c in zip(times, counts, strict=True)]

    # ------------------------------------------------------------------------------------
    # Helpers used by tactics and decisions
    # ------------------------------------------------------------------------------------
    def active_indices(self, side: int) -> np.ndarray:
        arr = self._active_cache.get(side)
        if arr is None:
            base = side * 11
            arr = np.array([i for i in range(base, base + 11) if self.active[i]], dtype=int)
            self._active_cache[side] = arr
        return arr

    def goalkeeper(self, side: int) -> int | None:
        for i in self.active_indices(side):
            if self.pl[i].slot.role == "GK":
                return int(i)
        return None

    def in_own_box(self, side: int, pt: np.ndarray) -> bool:
        """Is absolute point ``pt`` inside the penalty area that ``side`` defends?"""
        x, y = to_tf(np.asarray(pt, dtype=float), self.teams[side].direction)
        return x <= PENALTY_BOX_DEPTH and abs(y - PITCH_WIDTH / 2) <= PENALTY_BOX_HALF_WIDTH

    def pressure_level(self, c: int) -> tuple[float, float]:
        opp = self.active_indices(1 - self.pl[c].side)
        if len(opp) == 0:
            return 0.0, 99.0
        d = np.linalg.norm(self.pos[opp] - self.pos[c], axis=1)
        nearest = float(d.min())
        level = clamp((4.6 - nearest) / 3.6, 0.0, 1.0) + 0.15 * max(0, int(np.sum(d < 3.0)) - 1)
        return clamp(level, 0.0, 1.0), nearest

    def is_offside(self, m: int, side: int) -> bool:
        d = self.teams[side].direction
        mx = to_tf(self.pos[m], d)[0]
        bx = to_tf(self.ball_pos, d)[0]
        line = tactics.opponent_line_tf(self, side)
        return mx > line + 0.35 and mx > bx + 0.2 and mx > PITCH_LENGTH / 2

    def match_minute(self) -> float:
        return match_seconds(self.period, self.t) / 60.0

    def _team_id(self, side: int | None) -> str | None:
        return None if side is None else self.teams[side].team_id

    # ------------------------------------------------------------------------------------
    # Event publishing
    # ------------------------------------------------------------------------------------
    def _event(
        self,
        etype: EventType,
        side: int | None = None,
        player: int | None = None,
        loc: np.ndarray | None = None,
        t: float | None = None,
        **details,
    ) -> dict:
        t = self.t if t is None else t
        clk = clock_strings(self.period, t)
        ps = self.pl[player] if player is not None else None
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": None,
            "match_id": self.match_id,
            "seq": None,
            "type": etype.value,
            "period": self.period,
            "t": round(t, 2),
            **clk,
            "elapsed_s": round(self._elapsed(t), 2),
            "team_id": self._team_id(side if side is not None else (ps.side if ps else None)),
            "player_id": ps.player_id if ps else None,
            "player_name": ps.profile.name if ps else None,
            "location": None if loc is None else {"x": round(float(loc[0]), 2), "y": round(float(loc[1]), 2)},
            "possession": {
                "id": self.possession_id,
                "team_id": self._team_id(self.possession_side),
                "play_pattern": self.play_pattern,
            },
            "score": {"home": self.teams[0].score, "away": self.teams[1].score},
            "details": details,
        }

    def _publish(self, ev: dict) -> dict:
        ev["details"] = _native(ev["details"])
        self.seq += 1
        ev["seq"] = self.seq
        ev["event_id"] = f"{self.match_id}-{self.seq:05d}"
        ev["score"] = {"home": self.teams[0].score, "away": self.teams[1].score}
        self.events.append(ev)
        self._buffer.append(("event", ev))
        return ev

    def emit(
        self,
        etype: EventType,
        side: int | None = None,
        player: int | None = None,
        loc: np.ndarray | None = None,
        t: float | None = None,
        **details,
    ) -> dict:
        return self._publish(self._event(etype, side, player, loc, t, **details))

    def _elapsed(self, t: float) -> float:
        before = sum(self.period_lengths[: self.period - 1]) + (self.period - 1) * 15 * 60
        return before + t

    def _drain(self) -> Iterator[tuple[str, dict]]:
        buf, self._buffer = self._buffer, []
        yield from buf

    # ------------------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------------------
    def meta(self) -> dict:
        cfg = self.cfg
        return {
            "schema_version": SCHEMA_VERSION,
            "match_id": self.match_id,
            "competition": cfg.competition,
            "season": cfg.season,
            "matchday": cfg.matchday,
            "kickoff_utc": cfg.kickoff_utc,
            "venue": {"stadium": cfg.home.stadium, "city": cfg.home.city, "capacity": cfg.home.capacity},
            "attendance": cfg.attendance
            or int(cfg.home.capacity * (0.88 + 0.11 * ((cfg.seed * 7919) % 100) / 100)),
            "referee": cfg.referee,
            "weather": cfg.weather,
            "pitch": {
                "length_m": PITCH_LENGTH,
                "width_m": PITCH_WIDTH,
                "origin": "bottom-left corner flag; x along the length, y along the width",
            },
            "home": self._team_meta(0),
            "away": self._team_meta(1),
            "attacking_direction": {
                "period_1": {"home": "right", "away": "left"},
                "period_2": {"home": "left", "away": "right"},
                "note": "'right' means attacking towards x = 105",
            },
            "engine": {
                "name": "matchengine",
                "version": ENGINE_VERSION,
                "seed": cfg.seed,
                "tick_hz": TICK_HZ,
                "frame_hz": TICK_HZ // self.frame_every,
                "scenario": getattr(self.scenario, "name", None),
            },
        }

    def _team_meta(self, side: int) -> dict:
        team = self.teams[side]
        club = team.club
        lineup = []
        for ps in team.players:
            lineup.append(
                {
                    "player_id": ps.player_id,
                    "name": ps.profile.name,
                    "short_name": ps.profile.short_name,
                    "shirt": ps.profile.shirt,
                    "position": ps.slot.label,
                    "role": ps.slot.role,
                }
            )
        bench = [
            {
                "player_id": b.player_id,
                "name": b.name,
                "short_name": b.short_name,
                "shirt": b.shirt,
                "position": b.preferred_position,
                "role": b.role,
            }
            for b in team.bench
        ]
        return {
            "team_id": club.club_id,
            "name": club.name,
            "short_name": club.short_name,
            "code": club.code,
            "colors": club.colors,
            "manager": club.manager,
            "formation": club.formation,
            "tactics": club.tactics.to_dict(),
            "identity": club.identity,
            "lineup": lineup,
            "bench": bench,
            "squad": [p.to_dict() for p in club.squad],
        }

    def run(self) -> Iterator[tuple[str, dict]]:
        """Simulate the match, yielding ("event", e) and ("frame", f) items in order."""
        self.emit(
            EventType.MATCH_START,
            home_team_id=self.teams[0].team_id,
            away_team_id=self.teams[1].team_id,
            kickoff_utc=self.cfg.kickoff_utc,
            competition=self.cfg.competition,
            venue=self.cfg.home.stadium,
        )
        yield from self._drain()
        for period in (1, 2):
            self._start_period(period)
            yield from self._drain()
            while not self._period_over():
                self._step()
                yield from self._drain()
            self._end_period()
            yield from self._drain()
        self.emit(
            EventType.MATCH_END,
            final_score={"home": self.teams[0].score, "away": self.teams[1].score},
            winner=self._winner(),
        )
        yield from self._drain()

    def simulate(self, keep_frames: bool = True) -> MatchResult:
        from .summary import build_summary

        meta = self.meta()
        frames: list[dict] = []
        for kind, item in self.run():
            if kind == "frame" and keep_frames:
                frames.append(item)
        return MatchResult(meta=meta, events=self.events, frames=frames, summary=build_summary(self, meta))

    def _winner(self) -> str | None:
        h, a = self.teams[0].score, self.teams[1].score
        return None if h == a else (self.teams[0].team_id if h > a else self.teams[1].team_id)

    # ------------------------------------------------------------------------------------
    # Periods
    # ------------------------------------------------------------------------------------
    def _start_period(self, period: int) -> None:
        self.period = period
        self.t = 0.0
        self._next_minute_check = 60.0
        for team in self.teams:
            team.direction = (1 if team.side == 0 else -1) * (1 if period == 1 else -1)
            for ps in team.players:
                if period == 2:
                    self.energy[ps.idx] = min(1.0, self.energy[ps.idx] + 0.18)
                ps.run_target_tf = None
            self.recover_until[:] = 0.0
        kicker_side = 0 if period == 1 else 1
        self.added_minutes[period - 1] = None
        self.period_end_t = float(PERIOD_SECONDS)
        self.emit(
            EventType.PERIOD_START,
            period=period,
            attacking_right=self.teams[0].team_id if self.teams[0].direction == 1 else self.teams[1].team_id,
        )
        self._set_restart("kick_off", kicker_side, np.array(CENTER), (1.0, 1.0))
        tactics.setup_targets(self)
        self.pos[:] = self.target
        self.vel[:] = 0.0
        self.possession_side = None

    def _period_over(self) -> bool:
        if self.t >= PERIOD_SECONDS and self.added_minutes[self.period - 1] is None:
            self._announce_added_time()
        if self.t < self.period_end_t:
            return False
        if self.flight is not None and self.flight.kind == "shot":
            return False
        if self.t > self.period_end_t + 25.0:
            return True
        if self.ball_state == BallState.DEAD:
            return self.restart is None or self.restart["kind"] != "penalty"
        return all(not self.in_own_box(side, self.ball_pos) for side in (0, 1))

    def _announce_added_time(self) -> None:
        stop = self.stoppage[self.period - 1]
        base = 0.6 if self.period == 1 else 1.6
        added = int(clamp(round(base + stop / 60.0 * 0.38 + self.rng.uniform(-0.5, 0.8)), 1, 9))
        self.added_minutes[self.period - 1] = added
        self.period_end_t = PERIOD_SECONDS + added * 60 + self.rng.uniform(0, 30)
        self.emit(EventType.ADDED_TIME, minutes=added)

    def _end_period(self) -> None:
        if self.carrier is not None:
            self._release(self.carrier)
        self.flight = None
        self.restart = None
        self.keeper_holding = False
        self.ball_state = BallState.DEAD
        self.period_lengths.append(self.t)
        self.emit(
            EventType.PERIOD_END,
            period=self.period,
            score={"home": self.teams[0].score, "away": self.teams[1].score},
            duration_s=round(self.t, 1),
        )

    # ------------------------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------------------------
    def _step(self) -> None:
        self.pressing_idx = None
        self._scenario_tick()
        if self.t >= self._next_minute_check:
            self._next_minute_check += 60.0
            self._minute_tick()
        if self.tick % TICK_HZ == 0:
            self._second_tick()

        if self.ball_state == BallState.DEAD:
            self._dead_tick()
        else:
            self.ball_in_play_s += DT
            if self.possession_side is not None:
                self.possession_s[self.possession_side] += DT
            if self.ball_state == BallState.FLIGHT:
                self._flight_tick()
            elif self.ball_state == BallState.LOOSE:
                self._loose_tick()
            elif self.ball_state == BallState.HELD:
                self._carrier_tick()

        if self.tick % 2 == 0 or self.ball_state == BallState.DEAD:
            self._compute_targets()
        self._move()
        self._sync_ball()
        if self.emit_frames and self.tick % self.frame_every == 0:
            self._record_frame()
        self.t += DT
        self.tick += 1

    def _second_tick(self) -> None:
        self.top_eff[:] = self.top * (0.8 + 0.2 * self.energy)
        for team in self.teams:
            baseline = 0.08 if team.side == 0 else 0.0
            team.confidence += (baseline - team.confidence) * 0.006

    def _minute_tick(self) -> None:
        minute = self.match_minute()
        for team in self.teams:
            opp = self.teams[1 - team.side]
            diff = team.score - opp.score
            base = team.club.tactics
            if diff < 0 and minute >= 57 and team.escalation < 1:
                self._change_tactics(
                    team,
                    base.adjusted(
                        pressing=base.pressing + 0.14,
                        line_height=base.line_height + 0.12,
                        tempo=base.tempo + 0.12,
                        risk=base.risk + 0.12,
                        directness=base.directness + 0.08,
                        crossing=base.crossing + 0.1,
                    ),
                    1,
                    "chasing the game",
                    "Pushing higher and pressing harder to find a goal.",
                )
            elif diff < 0 and minute >= 78 and team.escalation < 2:
                self._change_tactics(
                    team,
                    base.adjusted(
                        pressing=base.pressing + 0.25,
                        line_height=base.line_height + 0.22,
                        tempo=base.tempo + 0.2,
                        risk=base.risk + 0.25,
                        directness=base.directness + 0.2,
                        crossing=base.crossing + 0.2,
                    ),
                    2,
                    "all-out attack",
                    "Throwing bodies forward; long balls into the box.",
                )
            elif 0 < diff <= 2 and minute >= 72 and team.escalation != -1:
                self._change_tactics(
                    team,
                    base.adjusted(
                        pressing=base.pressing - 0.16,
                        line_height=base.line_height - 0.18,
                        tempo=base.tempo - 0.14,
                        risk=base.risk - 0.14,
                    ),
                    -1,
                    "protecting the lead",
                    "Dropping deeper and slowing the game down.",
                )
            elif diff == 0 and team.escalation == -1:
                self._change_tactics(
                    team, base, 0, "back to the plan", "Lead gone; returning to the usual approach."
                )

    def _change_tactics(self, team: TeamState, new, level: int, reason: str, description: str) -> None:
        old = team.tactics
        team.tactics = new
        team.escalation = level
        keys = ("pressing", "line_height", "tempo", "risk", "directness", "crossing")
        self.emit(
            EventType.TACTICAL_CHANGE,
            side=team.side,
            reason=reason,
            description=description,
            manager=team.club.manager,
            before={k: round(getattr(old, k), 2) for k in keys},
            after={k: round(getattr(new, k), 2) for k in keys},
        )

    # ------------------------------------------------------------------------------------
    # Dead ball
    # ------------------------------------------------------------------------------------
    def _set_restart(
        self,
        kind: str,
        side: int,
        loc: np.ndarray,
        delay: tuple[float, float],
        taker: int | None = None,
        **extra,
    ) -> None:
        if self.carrier is not None:
            self._release(self.carrier)
        self.ball_state = BallState.DEAD
        self.flight = None
        self.keeper_holding = False
        self.ball_pos = np.array(loc, dtype=float)
        self.ball_vel[:] = 0.0
        self.ball_z = 0.0
        if taker is None or not self.active[taker]:
            taker = self._choose_taker(kind, side, loc)
        self.restart = {
            "kind": kind,
            "side": side,
            "loc": np.array(loc, dtype=float),
            "resume": self.t + float(self.rng.uniform(*delay)),
            "taker": taker,
            **extra,
        }

    def _choose_taker(self, kind: str, side: int, loc: np.ndarray) -> int:
        idx = [int(i) for i in self.active_indices(side)]
        gk = self.goalkeeper(side)
        outfield = [i for i in idx if i != gk] or idx
        if kind in ("goal_kick",) and gk is not None:
            return gk
        if kind == "corner":
            return max(outfield, key=lambda i: self.pl[i].a.crossing)
        if kind == "penalty":
            return max(outfield, key=lambda i: self.pl[i].a.finishing + self.pl[i].a.composure * 0.5)
        if kind == "kick_off":
            fwd = [i for i in outfield if self.pl[i].slot.role in ("ST", "AM", "CM")]
            return fwd[0] if fwd else outfield[0]
        if kind == "free_kick":
            d_goal = np.linalg.norm(to_tf(loc, self.teams[side].direction) - np.array([PITCH_LENGTH, 34.0]))
            if d_goal < 32:
                return max(
                    outfield,
                    key=lambda i: self.pl[i].a.shot_power + self.pl[i].a.finishing + self.pl[i].a.crossing,
                )
        return min(outfield, key=lambda i: float(np.linalg.norm(self.pos[i] - loc)))

    def _dead_tick(self) -> None:
        r = self.restart
        if r is None:
            return
        self._apply_pending_subs()
        if self.t >= r["resume"]:
            self._take_restart()

    def _take_restart(self) -> None:
        r = self.restart
        assert r is not None
        self.restart = None
        side, kind, loc = r["side"], r["kind"], r["loc"]
        taker = r["taker"]
        if not self.active[taker]:
            taker = self._choose_taker(kind, side, loc)
        self.pos[taker] = loc
        self.vel[taker] = 0.0
        self.ball_pos = loc.copy()
        self.play_pattern = {
            "kick_off": "from_kick_off",
            "throw_in": "from_throw_in",
            "goal_kick": "from_goal_kick",
            "corner": "from_corner",
            "free_kick": "from_free_kick",
            "penalty": "from_penalty",
            "offside": "from_free_kick",
        }.get(kind, "regular_play")
        self._give_ball(taker, kind, restart=True)
        d = self.teams[side].direction
        loc_tf = to_tf(loc, d)
        rng = self.rng
        tac = self.teams[side].tactics

        if kind == "kick_off":
            self.emit(EventType.KICK_OFF, side=side, player=taker, loc=loc)
            act = decide(self, taker, allowed={"pass"}, max_pass=22.0)
            self._execute(act, restart="kick_off")
        elif kind == "throw_in":
            act = decide(self, taker, allowed={"pass"}, max_pass=26.0)
            if act.kind == "pass":
                act.aerial = False
            self._execute(act, restart="throw_in")
        elif kind == "goal_kick":
            if rng.random() < 0.15 + 0.75 * (1 - tac.directness):
                act = decide(self, taker, allowed={"pass"}, max_pass=34.0)
                self._execute(act, restart="goal_kick")
            else:
                tgt_tf = np.array([rng.uniform(55, 78), rng.uniform(14, 54)])
                self._execute(Action("pass", to_tf(tgt_tf, d), None, "long", True), restart="goal_kick")
        elif kind == "corner":
            if rng.random() < self.p.corner_short:
                act = decide(self, taker, allowed={"pass"}, max_pass=16.0)
                self._execute(act, restart="corner")
            else:
                tgt_tf = np.array([rng.uniform(92.5, 101.0), rng.uniform(27.5, 40.5)])
                self._execute(Action("pass", to_tf(tgt_tf, d), None, "corner", True), restart="corner")
        elif kind == "free_kick":
            dist = float(np.linalg.norm(loc_tf - np.array([PITCH_LENGTH, 34.0])))
            ang_ok = abs(loc_tf[1] - 34.0) < 0.9 * (PITCH_LENGTH - loc_tf[0]) + 6
            if dist < 31 and ang_ok and rng.random() < 0.62:
                self._shoot(taker, kind="free_kick")
            elif loc_tf[0] > 70 and rng.random() < 0.7:
                tgt_tf = np.array([rng.uniform(91, 100), rng.uniform(28, 40)])
                self._execute(
                    Action("pass", to_tf(tgt_tf, d), None, "free_kick_cross", True), restart="free_kick"
                )
            else:
                act = decide(self, taker, allowed={"pass", "carry"})
                self._execute(act, restart="free_kick")
        elif kind == "penalty":
            self._shoot(taker, kind="penalty")
        else:  # offside / indirect free kick
            act = decide(self, taker, allowed={"pass"})
            self._execute(act, restart="free_kick")

    # ------------------------------------------------------------------------------------
    # Possession
    # ------------------------------------------------------------------------------------
    def _decision_delay(self, idx: int) -> float:
        p = self.p
        P = self.pl[idx]
        tac = self.teams[P.side].tactics
        base = p.decision_delay_min + (p.decision_delay_max - p.decision_delay_min) * (1 - tac.tempo) * 0.8
        level, _ = self.pressure_level(idx)
        delay = base * float(self.rng.uniform(0.5, 1.45)) * (1 - 0.45 * level)
        return max(0.15, delay)

    def _give_ball(self, idx: int, how: str, restart: bool = False) -> None:
        P = self.pl[idx]
        side = P.side
        self.ball_state = BallState.HELD
        self.carrier = idx
        self.flight = None
        self.ball_pos = self.pos[idx].copy()
        self.ball_z = 0.0
        self.ball_vel[:] = 0.0
        self.last_touch_side = side
        P.carry_start = self.pos[idx].copy()
        P.carry_t0 = self.t
        P.carry_max_speed = 0.0
        P.carry_target = None
        P.decide_at = self.t + self._decision_delay(idx)
        self.carrier_spell += 1
        if side != self.possession_side:
            old = self.possession_side
            if old is not None:
                self.teams[old].lost_ball_at = self.t
                self.emit(
                    EventType.POSSESSION_CHANGE,
                    side=side,
                    player=idx,
                    loc=self.pos[idx],
                    from_team_id=self._team_id(old),
                    how=how,
                    previous_duration_s=round(self.t - self.poss_start_t, 1),
                    previous_completed_passes=self.poss_passes,
                )
            self.possession_id += 1
            self.possession_side = side
            self.poss_start_t = self.t
            self.poss_passes = 0
            if not restart:
                self.play_pattern = "regular_play"
            self.last_completed = None

    def _release(self, idx: int) -> None:
        """The carrier's spell on the ball ends; publish a carry if they moved with it."""
        P = self.pl[idx]
        if P.carry_start is not None:
            dist = float(np.linalg.norm(self.pos[idx] - P.carry_start))
            dur = self.t - P.carry_t0
            if dist >= 3.0 and dur > 0.3:
                d = self.teams[P.side].direction
                prog = float(to_tf(self.pos[idx], d)[0] - to_tf(P.carry_start, d)[0])
                level, nearest = self.pressure_level(idx)
                self.emit(
                    EventType.CARRY,
                    side=P.side,
                    player=idx,
                    loc=P.carry_start,
                    t=P.carry_t0,
                    end_location=_loc(self.pos[idx]),
                    distance_m=round(dist, 2),
                    duration_s=round(dur, 2),
                    progression_m=round(prog, 2),
                    max_speed_kmh=round(P.carry_max_speed * 3.6, 1),
                    under_pressure=nearest < 3.0,
                )
        P.carry_start = None
        P.carry_target = None
        if self.carrier == idx:
            self.carrier = None
        self.keeper_holding = False

    def _scramble_velocity(self, at: np.ndarray, lo: float, hi: float) -> np.ndarray:
        """Random ricochet velocity; near a touchline the ball often squirts out of play."""
        ang = self.rng.uniform(0, 2 * math.pi)
        v = np.array([math.cos(ang), math.sin(ang)]) * self.rng.uniform(lo, hi)
        edge = min(at[1], PITCH_WIDTH - at[1])
        if edge < 12.0 and self.rng.random() < 0.6:
            push = 4.5 + edge * 0.7
            v[1] = -push if at[1] < PITCH_WIDTH / 2 else push
        return v

    def _make_loose(self, at: np.ndarray, vel: np.ndarray) -> None:
        if self.carrier is not None:
            self._release(self.carrier)
        self.ball_state = BallState.LOOSE
        self.flight = None
        self.ball_pos = np.array(at, dtype=float)
        self.ball_vel = np.array(vel, dtype=float)
        self.ball_z = 0.0

    # ------------------------------------------------------------------------------------
    # Ball carrier
    # ------------------------------------------------------------------------------------
    def _carrier_tick(self) -> None:
        c = self.carrier
        if c is None:
            return
        P = self.pl[c]
        if not self.keeper_holding:
            self._engage(c)
            if self.ball_state != BallState.HELD or self.carrier != c:
                return
        if P.carry_target is not None and np.linalg.norm(self.pos[c] - P.carry_target) < 1.0:
            P.carry_target = None
            P.decide_at = min(P.decide_at, self.t + 0.1)
        if self.t < P.decide_at:
            return
        if self.keeper_holding:
            self._keeper_distribution(c)
            return
        act = decide(self, c)
        self._execute(act)

    def _keeper_distribution(self, c: int) -> None:
        tac = self.teams[self.pl[c].side].tactics
        d = self.teams[self.pl[c].side].direction
        self.keeper_holding = False
        if self.rng.random() < 0.2 + 0.6 * (1 - tac.directness):
            act = decide(self, c, allowed={"pass"}, max_pass=40.0)
            if act.kind == "pass":
                act.aerial = act.aerial and float(np.linalg.norm(act.target - self.pos[c])) > 30
            self._execute(act, restart="keeper")
        else:
            tgt_tf = np.array([self.rng.uniform(58, 80), self.rng.uniform(12, 56)])
            self._execute(Action("pass", to_tf(tgt_tf, d), None, "long", True), restart="keeper")

    def _execute(self, act: Action, restart: str | None = None) -> None:
        c = self.carrier
        if c is None:
            return
        P = self.pl[c]
        if restart and act.kind not in ("pass", "shot", "carry"):
            mates = [int(i) for i in self.active_indices(P.side) if i != c]
            nearest = min(mates, key=lambda i: float(np.linalg.norm(self.pos[i] - self.pos[c])))
            act = Action("pass", self.pos[nearest].copy(), nearest, "ground", False)
        if act.kind == "pass" and act.target is not None:
            self._kick_pass(c, act, restart)
        elif act.kind == "shot":
            self._shoot(c)
        elif act.kind == "carry" and act.target is not None:
            P.carry_target = act.target
            P.decide_at = self.t + self.p.carry_redecide * float(self.rng.uniform(0.7, 1.3))
        elif act.kind == "dribble":
            self._take_on(c, act)
        elif act.kind == "clear":
            self._clear(c, act)
        else:  # hold
            P.carry_target = None
            P.decide_at = self.t + float(self.rng.uniform(0.3, 0.8))

    def _engage(self, c: int) -> None:
        """Pressure events and tackle attempts from defenders near the carrier."""
        P = self.pl[c]
        opp = self.active_indices(1 - P.side)
        if len(opp) == 0:
            return
        rel = self.pos[c] - self.pos[opp]
        d = np.linalg.norm(rel, axis=1)
        for k in np.where(d < self.p.pressure_radius)[0]:
            j = int(opp[k])
            key = (j, self.carrier_spell)
            if key in self._pressured or (-1, self.carrier_spell) in self._pressured:
                continue
            closing = float(self.vel[j] @ (rel[k] / max(d[k], 1e-6)))
            if closing > 2.2 or d[k] < 1.3:
                self._pressured.add((-1, self.carrier_spell))
                edge = min(self.pos[c][1], PITCH_WIDTH - self.pos[c][1])
                if edge < 4.0 and self.rng.random() < self.p.touchline_squeeze:
                    out_y = 0.0 if self.pos[c][1] < PITCH_WIDTH / 2 else PITCH_WIDTH
                    self.emit(
                        EventType.PRESSURE,
                        side=1 - P.side,
                        player=j,
                        loc=self.pos[j],
                        target_player_id=P.player_id,
                        distance_m=round(float(d[k]), 2),
                        closing_speed_kmh=round(max(closing, 0.0) * 3.6, 1),
                    )
                    self._ball_out(
                        P.side,
                        np.array([self.pos[c][0], out_y]),
                        "bottom_touchline" if out_y == 0 else "top_touchline",
                    )
                    return
                self._pressured.add(key)
                self.emit(
                    EventType.PRESSURE,
                    side=1 - P.side,
                    player=j,
                    loc=self.pos[j],
                    target_player_id=P.player_id,
                    distance_m=round(float(d[k]), 2),
                    closing_speed_kmh=round(max(closing, 0.0) * 3.6, 1),
                )
                P.decide_at = min(P.decide_at, self.t + float(self.rng.uniform(0.15, 0.45)))
        k = int(np.argmin(d))
        j = int(opp[k])
        if (
            d[k] < self.p.press_engage_dist
            and self.t >= self.recover_until[j]
            and self.pl[j].slot.role != "GK"
        ):
            D = self.pl[j]
            team = self.teams[D.side]
            # A defender standing in the carrier's path forces a duel: you cannot dribble through people.
            move = self.vel[c]
            speed_c = float(np.hypot(*move))
            dribbling = P.carry_target is not None and speed_c > 2.0
            in_path = dribbling and float(move @ (-rel[k])) / max(speed_c * d[k], 1e-6) > 0.5
            key = (j, -self.carrier_spell)
            if in_path and key not in self._pressured:
                self._pressured.add(key)
                if self.rng.random() < self.p.path_duel_prob:
                    self._tackle(j, c)
                    return
            hazard = self.p.tackle_hazard_base + self.p.tackle_hazard_aggr * D.a.aggression / 100.0
            hazard *= 0.75 + 0.3 * team.tactics.pressing
            if self.rng.random() < hazard * DT:
                self._tackle(j, c)

    def _tackle(self, j: int, c: int) -> None:
        p = self.p
        D, A = self.pl[j].a, self.pl[c].a
        D_ps = self.pl[j]
        p_win = sigmoid(
            (
                D.tackling * 0.6
                + D.strength * 0.2
                + D.positioning * 0.2
                - (A.dribbling * 0.55 + A.strength * 0.2 + A.composure * 0.25)
            )
            / 16.0
            + 0.15
            + p.tackle_win_base
        )
        p_foul = p.foul_base + p.foul_aggression * (D.aggression / 100.0) - (D.tackling - 60) / 450.0
        p_foul += 0.05 * (1 - self.energy[j])
        if self.in_own_box(D_ps.side, self.pos[c]):
            p_foul *= getattr(p, "box_foul_factor", 1.0)  # older saved states predate this setting
        r = self.rng.random()
        loc = (self.pos[j] + self.pos[c]) / 2
        if r < p_foul:
            self._foul(j, c, "tackle")
            return
        won = r < p_foul + (1 - p_foul) * p_win
        attacker_side = self.pl[c].side
        edge = min(self.pos[c][1], PITCH_WIDTH - self.pos[c][1])
        if edge < 10.0 and self.rng.random() < 0.45:
            self.emit(
                EventType.TACKLE,
                side=D_ps.side,
                player=j,
                loc=loc,
                outcome="won" if won else "lost",
                opponent_id=self.pl[c].player_id,
                truth={"p_win": round(p_win, 3)},
                ball_out=True,
            )
            last = D_ps.side if won or self.rng.random() < 0.5 else attacker_side
            out_y = 0.0 if self.pos[c][1] < PITCH_WIDTH / 2 else PITCH_WIDTH
            self._ball_out(
                last, np.array([self.pos[c][0], out_y]), "bottom_touchline" if out_y == 0 else "top_touchline"
            )
            return
        if won:
            self._release(c)
            self.emit(
                EventType.TACKLE,
                side=D_ps.side,
                player=j,
                loc=loc,
                outcome="won",
                opponent_id=self.pl[c].player_id,
                truth={"p_win": round(p_win, 3)},
            )
            self.recover_until[c] = self.t + 0.6
            if self.rng.random() < 0.55:
                self._give_ball(j, "tackle")
            else:
                self.last_touch_side = D_ps.side
                self._make_loose(self.pos[c].copy(), self._scramble_velocity(self.pos[c], 3, 8))
        else:
            self.emit(
                EventType.TACKLE,
                side=D_ps.side,
                player=j,
                loc=loc,
                outcome="lost",
                opponent_id=self.pl[c].player_id,
                truth={"p_win": round(p_win, 3)},
            )
            self.recover_until[j] = self.t + 0.6
            self.last_touch_side = attacker_side
            if self.rng.random() < 0.3:  # even a failed challenge often knocks the ball loose
                self.last_touch_side = D_ps.side
                self._make_loose(self.pos[c].copy(), self._scramble_velocity(self.pos[c], 2, 6))

    def _take_on(self, c: int, act: Action) -> None:
        j = act.opponent
        P = self.pl[c]
        if j is None or not self.active[j]:
            P.carry_target = act.target
            P.decide_at = self.t + 0.6
            return
        D = self.pl[j]
        p_win = act.p_est
        success = self.rng.random() < p_win
        self.emit(
            EventType.DRIBBLE,
            side=P.side,
            player=c,
            loc=self.pos[c],
            outcome="success" if success else "fail",
            opponent_id=D.player_id,
            truth={"p_success": round(p_win, 3)},
        )
        if success:
            self.recover_until[j] = self.t + 0.8
            P.carry_target = act.target
            P.decide_at = self.t + float(self.rng.uniform(0.6, 1.1))
            return
        # Failed take-on: defender tackles or fouls.
        p_foul = 0.16 + 0.1 * D.a.aggression / 100
        if self.in_own_box(self.pl[j].side, self.pos[c]):
            p_foul *= getattr(self.p, "box_foul_factor", 1.0)
        if self.rng.random() < p_foul:
            self._foul(j, c, "challenge")
            return
        self._release(c)
        self.emit(
            EventType.TACKLE,
            side=D.side,
            player=j,
            loc=self.pos[j],
            outcome="won",
            opponent_id=P.player_id,
            truth={"p_win": round(1 - p_win, 3)},
        )
        self.recover_until[c] = self.t + 0.7
        if self.rng.random() < 0.6:
            self._give_ball(j, "tackle")
        else:
            self.last_touch_side = D.side
            self._make_loose(self.pos[c].copy(), self._scramble_velocity(self.pos[c], 3, 8))

    def _clear(self, c: int, act: Action) -> None:
        P = self.pl[c]
        start = self.pos[c].copy()
        sigma = 3.0 + 3.0 * (1 - P.a.passing / 100)
        end = act.target + self.rng.normal(0, sigma, 2)
        self._release(c)
        crossing = boundary_crossing(start, end)
        shown_end = crossing[0] if crossing else end
        self.emit(
            EventType.CLEARANCE,
            side=P.side,
            player=c,
            loc=start,
            end_location=_loc(shown_end),
            body_part="foot",
            height="high",
        )
        self._launch_aerial(c, start, end, "clearance", None, pending=None)

    # ------------------------------------------------------------------------------------
    # Passing
    # ------------------------------------------------------------------------------------
    def _kick_pass(self, c: int, act: Action, restart: str | None) -> None:
        p = self.p
        P = self.pl[c]
        a = P.a
        start = self.pos[c].copy()
        intended = np.array(act.target, dtype=float)
        dist_int = float(np.linalg.norm(intended - start))
        level, nearest = self.pressure_level(c)
        kind = act.pass_kind
        if act.receiver is None and not act.aerial:
            # Ground ball into a zone (e.g. a cut-back): aim it at the nearest teammate there.
            mates = [int(i) for i in self.active_indices(P.side) if i != c]
            if mates:
                dm = np.linalg.norm(self.pos[mates] - act.target, axis=1)
                if dm.min() < 7.0:
                    act.receiver = mates[int(np.argmin(dm))]
        if kind in ("cross", "corner", "free_kick_cross"):
            skill = a.crossing
        elif kind in ("long", "switch", "through"):
            skill = a.passing * 0.7 + a.vision * 0.3
        else:
            skill = a.passing
        sigma = p.pass_noise_base + p.pass_noise_per_m * dist_int * (1.6 - skill / 100.0)
        sigma *= (1 + p.pass_noise_pressure * level) * (1 + 0.35 * (1 - self.energy[c]))
        if act.aerial:
            sigma *= p.aerial_noise_mult
        if restart == "throw_in":
            sigma *= 0.6
        end = intended + self.rng.normal(0, sigma, 2)
        receiver = act.receiver if (act.receiver is not None and self.active[act.receiver]) else None

        ev_int = evaluate_pass(self, c, receiver, start, intended, act.aerial, kind, arrival=act.arrival)
        crossing = boundary_crossing(start, end)
        path_end = crossing[0] if crossing else end
        ev = evaluate_pass(
            self,
            c,
            receiver if not crossing else None,
            start,
            path_end,
            act.aerial,
            kind,
            arrival=act.arrival,
        )

        d = self.teams[P.side].direction
        vec = path_end - start
        prog = float(to_tf(path_end, d)[0] - to_tf(start, d)[0])
        height = "ground" if not act.aerial else ("high" if ev["peak"] > 2.5 else "low")
        pending = self._event(
            EventType.PASS,
            side=P.side,
            player=c,
            loc=start,
            recipient_id=self.pl[receiver].player_id if receiver is not None else None,
            recipient_name=self.pl[receiver].profile.name if receiver is not None else None,
            end_location=_loc(path_end),
            length_m=round(float(np.linalg.norm(path_end - start)), 2),
            angle_deg=round(math.degrees(math.atan2(vec[1], vec[0])), 1),
            progression_m=round(prog, 2),
            height=height,
            pass_kind=kind if not restart or kind in CROSSY else _restart_pass_kind(restart, kind),
            restart_type=restart,
            release_speed_kmh=round(ev["v0"] * 3.6, 1),
            under_pressure=nearest < 3.0,
            nearest_opponent_m=round(nearest, 2),
            outcome=None,
            truth={
                "p_success_intended": round(ev_int["p"], 3),
                "p_success_executed": round(ev["p"], 3),
                "execution_error_m": round(float(np.linalg.norm(end - intended)), 2),
            },
        )
        self._release(c)
        self.pl[c].received_from = None
        rng = self.rng

        # Interception along the path?
        pint, opps, tb, ot = ev["pint"], ev["opps"], ev["tb"], ev["ot"]
        order = np.argsort(np.min(ot - tb[None, :], axis=1)) if len(opps) else []
        interceptor, hit_k = None, None
        for k in order:
            if rng.random() < pint[k]:
                interceptor = int(opps[k])
                adv = tb - ot[k] - p.intercept_margin
                ok = np.where(adv > -p.intercept_temp)[0]
                hit_k = int(ok[0]) if len(ok) else int(np.argmax(adv))
                break
        if interceptor is not None:
            pt = ev["pts"][hit_k]
            outcome = {"type": "intercepted", "idx": interceptor, "deflect": rng.random() < 0.55}
            self._launch(c, start, pt, kind, ev, receiver, outcome, pending, t_total=float(tb[hit_k]))
            return
        if crossing:
            outcome = {"type": "out", "line": crossing[1], "point": crossing[0]}
            self._launch(c, start, path_end, kind, ev, receiver, outcome, pending)
            return
        if receiver is None:
            outcome = {"type": "aerial" if act.aerial else "loose"}
            self._launch(c, start, end, kind, ev, None, outcome, pending)
            return
        if act.extra.get("offside") or self.is_offside(receiver, P.side):
            outcome = {"type": "offside", "idx": receiver}
            self._launch(c, start, end, kind, ev, receiver, outcome, pending)
            return
        if act.aerial:
            outcome = {"type": "aerial", "idx": receiver}
        elif rng.random() >= ev["p_reach"]:
            outcome = {"type": "loose"}
        elif rng.random() >= ev["p_ctrl"]:
            outcome = {"type": "miscontrol", "idx": receiver}
        else:
            outcome = {"type": "received", "idx": receiver}
        self._launch(c, start, end, kind, ev, receiver, outcome, pending)

    def _launch(
        self,
        c: int,
        start: np.ndarray,
        end: np.ndarray,
        kind: str,
        ev: dict,
        receiver: int | None,
        outcome: dict,
        pending: dict | None,
        t_total: float | None = None,
    ) -> None:
        dist = float(np.linalg.norm(end - start))
        if ev["decel"] > 0:
            from .probability import ground_ball_time

            total = float(ground_ball_time(ev["v0"], ev["decel"], np.array([dist]))[0])
        else:
            total = dist / max(ev["vh"], 1e-3)
        if t_total is not None:
            total = t_total
        self.flight = Flight(
            kind=kind,
            start=start.copy(),
            end=np.array(end, dtype=float),
            t0=self.t,
            t1=self.t + max(total, 0.15),
            peak=ev["peak"],
            v0=ev["v0"],
            decel=ev["decel"],
            side=self.pl[c].side,
            kicker=c,
            receiver=receiver,
            outcome=outcome,
        )
        self.flight.pending = pending  # type: ignore[attr-defined]
        self.ball_state = BallState.FLIGHT
        self.carrier = None
        self.last_touch_side = self.pl[c].side

    def _launch_aerial(
        self,
        c: int,
        start: np.ndarray,
        end: np.ndarray,
        kind: str,
        receiver: int | None,
        pending: dict | None,
    ) -> None:
        from .probability import aerial_profile

        crossing = boundary_crossing(start, end)
        dist = float(np.linalg.norm(end - start))
        vh, peak = aerial_profile(dist, kind)
        ev = {"decel": 0.0, "vh": vh, "v0": vh, "peak": peak}
        if crossing:
            outcome = {"type": "out", "line": crossing[1], "point": crossing[0]}
            self._launch(c, start, crossing[0], kind, ev, None, outcome, pending)
        else:
            self._launch(c, start, end, kind, ev, receiver, {"type": "aerial"}, pending)

    def _flight_tick(self) -> None:
        f = self.flight
        if f is None:
            return
        if self.t + 1e-9 >= f.t1:
            self._resolve_flight()

    def _finish_pass_event(self, f: Flight, outcome: str, **extra) -> dict | None:
        pending = getattr(f, "pending", None)
        if pending is None:
            return None
        pending["details"]["outcome"] = outcome
        pending["details"].update(extra)
        f.pending = None  # type: ignore[attr-defined]
        return self._publish(pending)

    def _resolve_flight(self) -> None:
        f = self.flight
        assert f is not None
        self.ball_pos = f.end.copy()
        self.ball_z = 0.0
        if f.kind == "shot":
            self._resolve_shot(f)
            return
        o = f.outcome
        typ = o["type"]
        side = f.side
        if typ == "received":
            r = o["idx"]
            self._finish_pass_event(f, "complete")
            self.poss_passes += 1
            self.last_completed = {"passer": f.kicker, "receiver": r, "t": self.t, "kind": f.kind}
            self.pos[r] = 0.6 * self.pos[r] + 0.4 * f.end
            self._give_ball(r, "pass")
            self.pl[r].received_from = f.kicker
        elif typ == "miscontrol":
            r = o["idx"]
            self._finish_pass_event(f, "complete")
            self.emit(EventType.MISCONTROL, side=side, player=r, loc=f.end)
            ang = self.rng.uniform(-1.2, 1.2) + math.atan2(*(f.end - f.start)[::-1])
            self.last_touch_side = side
            self._make_loose(f.end, np.array([math.cos(ang), math.sin(ang)]) * self.rng.uniform(2.5, 6))
        elif typ == "intercepted":
            j = o["idx"]
            self._finish_pass_event(f, "intercepted", intercepted_by=self.pl[j].player_id)
            if o["deflect"]:
                self.emit(
                    EventType.BLOCK,
                    side=self.pl[j].side,
                    player=j,
                    loc=f.end,
                    blocked="pass",
                    passer_id=self.pl[f.kicker].player_id,
                )
            else:
                self.emit(EventType.INTERCEPTION, side=self.pl[j].side, player=j, loc=f.end)
            self.last_touch_side = self.pl[j].side
            if f.kind in CROSSY and self.in_own_box(self.pl[j].side, f.end) and self.rng.random() < 0.35:
                d_j = self.teams[self.pl[j].side].direction
                gx = 0.0 if d_j == 1 else PITCH_LENGTH
                self._ball_out(
                    self.pl[j].side,
                    np.array([gx, f.end[1]]),
                    "left_goal_line" if gx == 0 else "right_goal_line",
                )
            elif min(f.end[1], PITCH_WIDTH - f.end[1]) < 8.0 and self.rng.random() < 0.4:
                out_y = 0.0 if f.end[1] < PITCH_WIDTH / 2 else PITCH_WIDTH
                self._ball_out(
                    self.pl[j].side,
                    np.array([f.end[0], out_y]),
                    "bottom_touchline" if out_y == 0 else "top_touchline",
                )
            elif o["deflect"]:
                self._make_loose(f.end, self._scramble_velocity(f.end, 4, 10))
            else:
                self.pos[j] = 0.5 * self.pos[j] + 0.5 * f.end
                self._give_ball(j, "interception")
        elif typ == "out":
            self._finish_pass_event(f, "out")
            self._ball_out(side, o["point"], o["line"])
        elif typ == "offside":
            r = o["idx"]
            self._finish_pass_event(f, "offside")
            self.emit(EventType.OFFSIDE, side=side, player=r, loc=self.pos[r])
            self.stoppage[self.period - 1] += 15
            self._set_restart("offside", 1 - side, self.pos[r].copy(), self.p.restart_free_kick)
        elif typ == "loose":
            self._finish_pass_event(f, "incomplete")
            direction = f.end - f.start
            n = np.linalg.norm(direction)
            v = direction / n * max(f.speed(self.t), 2.0) * 0.6 if n > 1e-6 else np.zeros(2)
            self.last_touch_side = side
            self._make_loose(f.end, v)
        elif typ == "aerial":
            self._resolve_aerial(f)
        else:  # pragma: no cover
            raise RuntimeError(f"unknown outcome {typ}")

    def _resolve_aerial(self, f: Flight) -> None:
        L = f.end
        att, dfd = f.side, 1 - f.side
        rng = self.rng
        p = self.p
        crossy = f.kind in CROSSY
        in_box = self.in_own_box(dfd, L)
        gk = self.goalkeeper(dfd)
        if gk is not None and in_box:
            gd = float(np.linalg.norm(self.pos[gk] - L))
            base = p.corner_keeper_claim if f.kind == "corner" else p.cross_keeper_claim
            p_claim = base * clamp(1.7 - gd / 7.0, 0.0, 1.7) * (0.7 + self.pl[gk].a.goalkeeping / 260)
            if f.kind not in CROSSY:
                p_claim = clamp(p_claim * 2.2, 0, 0.9)
            if rng.random() < p_claim:
                self._finish_pass_event(f, "incomplete")
                self.pos[gk] = 0.5 * self.pos[gk] + 0.5 * L
                self.emit(EventType.KEEPER_CLAIM, side=dfd, player=gk, loc=L, kind="catch")
                self._give_ball(gk, "keeper_claim")
                self.keeper_holding = True
                self.pl[gk].decide_at = self.t + float(rng.uniform(*p.restart_keeper))
                return

        def score(i: int, attacking: bool) -> float:
            a = self.pl[i].a
            dd = float(np.linalg.norm(self.pos[i] - L))
            pos_term = a.positioning if not attacking else (a.positioning + a.finishing) / 2
            return a.heading * 0.55 + a.strength * 0.25 + pos_term * 0.2 - 7.0 * dd

        att_c = [
            int(i)
            for i in self.active_indices(att)
            if i != f.kicker and np.linalg.norm(self.pos[i] - L) < 4.5
        ]
        def_c = [
            int(i) for i in self.active_indices(dfd) if i != gk and np.linalg.norm(self.pos[i] - L) < 4.5
        ]
        best_a = max(att_c, key=lambda i: score(i, True)) if att_c else None
        best_d = max(def_c, key=lambda i: score(i, False)) if def_c else None
        if best_a is None and best_d is None:
            self._finish_pass_event(f, "incomplete")
            self.last_touch_side = att
            vec = f.end - f.start
            n = np.linalg.norm(vec)
            self._make_loose(L, vec / n * 3.0 if n > 1e-6 else np.zeros(2))
            return
        if best_d is None:
            winner = best_a
        elif best_a is None:
            winner = best_d
        else:
            p_att = sigmoid((score(best_a, True) - score(best_d, False)) / 14.0 - (0.35 if crossy else 0.0))
            winner = best_a if rng.random() < p_att else best_d
            loser = best_d if winner == best_a else best_a
            contested = float(np.linalg.norm(self.pos[best_a] - self.pos[best_d])) < 3.0 or crossy
            if contested:
                self.emit(
                    EventType.AERIAL_DUEL,
                    side=self.pl[winner].side,
                    player=winner,
                    loc=L,
                    opponent_id=self.pl[loser].player_id,
                    outcome="won",
                    truth={"p_attacker_win": round(p_att, 3)},
                )
        assert winner is not None
        self.pos[winner] = 0.45 * self.pos[winner] + 0.55 * L
        if self.pl[winner].side == att:
            self._finish_pass_event(f, "complete", completed_by=self.pl[winner].player_id)
            self.last_completed = {"passer": f.kicker, "receiver": winner, "t": self.t, "kind": f.kind}
            self.poss_passes += 1
            if crossy and self.in_own_box(dfd, L):
                r = rng.random()
                header_p = 0.62 if f.kind == "corner" else 0.55
                if r < header_p:
                    self._give_ball(winner, "header")
                    self._shoot(winner, header=True, kind="open_play")
                elif r < header_p + 0.2:
                    self._give_ball(winner, "pass")
                else:
                    ang = rng.uniform(0, 2 * math.pi)
                    self.last_touch_side = att
                    self._make_loose(L, np.array([math.cos(ang), math.sin(ang)]) * rng.uniform(3, 7))
            elif min(L[1], PITCH_WIDTH - L[1]) < 10 and rng.random() < 0.3:
                self.last_touch_side = att
                self._make_loose(L, self._scramble_velocity(L, 6, 10))
            elif rng.random() < 0.68:
                self._give_ball(winner, "pass")
            else:
                vec = f.end - f.start
                n = np.linalg.norm(vec)
                self.last_touch_side = att
                self._make_loose(L, (vec / n if n > 1e-6 else np.zeros(2)) * rng.uniform(5, 10))
        else:
            self._finish_pass_event(f, "incomplete")
            if in_box or (crossy and to_tf(L, self.teams[dfd].direction)[0] < 35):
                d = self.teams[dfd].direction
                out_tf = to_tf(L, d) + np.array([rng.uniform(14, 30), rng.uniform(-14, 14)])
                out_abs = np.clip(to_tf(out_tf, d), [0, 0], [PITCH_LENGTH, PITCH_WIDTH])
                self.emit(
                    EventType.CLEARANCE,
                    side=dfd,
                    player=winner,
                    loc=L,
                    end_location=_loc(out_abs),
                    body_part="head",
                    height="high",
                )
                self.last_touch_side = dfd
                if rng.random() < p.clearance_out_corner:
                    gx = 0.0 if d == 1 else PITCH_LENGTH
                    self._ball_out(
                        dfd, np.array([gx, L[1]]), "left_goal_line" if gx == 0 else "right_goal_line"
                    )
                else:
                    vec = to_tf(out_tf, d) - L
                    n = np.linalg.norm(vec)
                    self._make_loose(L, vec / n * rng.uniform(9, 15) if n > 1e-6 else np.zeros(2))
            elif rng.random() < 0.6:
                self._give_ball(winner, "aerial")
            else:
                ang = rng.uniform(0, 2 * math.pi)
                self.last_touch_side = dfd
                self._make_loose(L, np.array([math.cos(ang), math.sin(ang)]) * rng.uniform(4, 9))

    # ------------------------------------------------------------------------------------
    # Shooting
    # ------------------------------------------------------------------------------------
    def _shoot(self, c: int, header: bool = False, kind: str = "open_play") -> None:
        p = self.p
        rng = self.rng
        P = self.pl[c]
        side = P.side
        d = self.teams[side].direction
        sm = shot_model(self, c, header=header, kind=kind)
        start = self.pos[c].copy()
        a = P.a
        gk = sm["goalkeeper"]

        blocked_by = None
        if kind != "penalty" and sm["blockers"] and rng.random() < sm["p_block"]:
            blocked_by = min(sm["blockers"], key=lambda j: float(np.linalg.norm(self.pos[j] - start)))
            outcome = "blocked"
        elif rng.random() < sm["p_unblocked"]:
            outcome = "goal"
        else:
            keeper = self.pl[gk].a.goalkeeping if gk is not None else 40
            r = rng.random()
            on_t = p.on_target_base * (1.0 + (a.finishing - 70) / 160) * (1.15 if kind == "penalty" else 1.0)
            on_t *= 1.0 + (keeper - 75) / 300
            if header:
                on_t *= 0.9
            if r < p.post_share:
                outcome = "post"
            elif r < p.post_share + on_t:
                outcome = "saved"
            else:
                outcome = "off_target"

        if header:
            speed_kmh = 38 + 0.22 * a.heading + rng.normal(0, 5)
        elif kind == "penalty":
            speed_kmh = 82 + 0.3 * a.shot_power + rng.normal(0, 6)
        else:
            speed_kmh = 58 + 0.46 * a.shot_power + rng.normal(0, 7) + (6 if sm["distance"] > 20 else 0)
        speed_kmh = clamp(speed_kmh, 32, 132)

        if outcome == "goal":
            y_off = rng.choice([-1, 1]) * rng.uniform(0.4, GOAL_HALF_WIDTH - 0.25)
            z = rng.uniform(0.1, 2.3)
        elif outcome == "saved":
            y_off, z = clamp(rng.normal(0, 1.7), -3.4, 3.4), rng.uniform(0.1, 2.2)
        elif outcome == "post":
            y_off, z = rng.choice([-1, 1]) * GOAL_HALF_WIDTH, rng.uniform(0.2, 2.44)
        elif outcome == "off_target":
            if rng.random() < 0.55:
                y_off, z = rng.choice([-1, 1]) * rng.uniform(3.9, 9.0), rng.uniform(0.1, 2.0)
            else:
                y_off, z = rng.uniform(-4.5, 4.5), rng.uniform(2.6, 5.5)
        else:
            y_off, z = 0.0, 0.0
        goal_pt = goal_mouth_point(d, y_off)
        end = self.pos[blocked_by].copy() if blocked_by is not None else goal_pt

        _level, nearest = sm["pressure"], sm["nearest"]
        details = dict(
            shot_kind=kind,
            body_part="head" if header else ("left_foot" if P.profile.foot == "left" else "right_foot"),
            outcome=outcome,
            on_target=outcome in ("goal", "saved"),
            end_location={
                "x": round(float(end[0]), 2),
                "y": round(float(end[1]), 2),
                "z": round(float(z), 2),
            },
            goal_mouth={"y_offset_m": round(float(y_off), 2), "z_m": round(float(z), 2)},
            distance_m=round(sm["distance"], 2),
            angle_deg=round(math.degrees(sm["angle"]), 1),
            speed_kmh=round(float(speed_kmh), 1),
            under_pressure=nearest < 3.0,
            defenders_in_lane=sm["n_lane"],
            goalkeeper_id=self.pl[gk].player_id if gk is not None else None,
            truth={
                "xg": round(sm["xg"], 4),
                "p_block": round(sm["p_block"], 3),
                "p_goal_if_unblocked": round(sm["p_unblocked"], 4),
                "big_chance": sm["big_chance"],
            },
        )
        pending = self._event(EventType.SHOT, side=side, player=c, loc=start, **details)
        self._release(c)
        dist = float(np.linalg.norm(end - start))
        total = max(dist / (speed_kmh / 3.6 * 0.94), 0.12)
        self.flight = Flight(
            kind="shot",
            start=start,
            end=np.array(end, dtype=float),
            t0=self.t,
            t1=self.t + total,
            peak=max(0.3, z * 0.7) if not header else 0.0,
            v0=speed_kmh / 3.6,
            decel=0.0,
            side=side,
            kicker=c,
            receiver=None,
            outcome={
                "type": outcome,
                "gk": gk,
                "blocker": blocked_by,
                "xg": sm["xg"],
                "kind": kind,
                "header": header,
                "big_chance": sm["big_chance"],
            },
        )
        self.flight.pending = pending  # type: ignore[attr-defined]
        self.ball_state = BallState.FLIGHT
        self.last_touch_side = side
        self.stoppage[self.period - 1] += 4

    def _resolve_shot(self, f: Flight) -> None:
        o = f.outcome
        rng = self.rng
        p = self.p
        side = f.side
        shooter = f.kicker
        pending = getattr(f, "pending", None)
        if pending is not None:
            f.pending = None  # type: ignore[attr-defined]
            self._publish(pending)
        typ = o["type"]
        gk = o["gk"]
        d = self.teams[side].direction
        if typ == "goal":
            self._goal(shooter, o, pending)
        elif typ == "saved" and gk is not None:
            hold = rng.random() < p.save_hold_share * (0.7 + self.pl[gk].a.goalkeeping / 250)
            parry_corner = (not hold) and rng.random() < p.save_corner_share / (1 - p.save_hold_share)
            self.emit(
                EventType.SAVE,
                side=1 - side,
                player=gk,
                loc=f.end,
                kind="catch" if hold else "parry",
                shot_speed_kmh=round(f.v0 * 3.6, 1),
                shooter_id=self.pl[shooter].player_id,
            )
            self.teams[1 - side].confidence += 0.02
            if hold:
                self.pos[gk] = 0.5 * self.pos[gk] + 0.5 * f.end
                self._give_ball(gk, "save")
                self.keeper_holding = True
                self.pl[gk].decide_at = self.t + float(rng.uniform(*p.restart_keeper))
            elif parry_corner:
                gx = PITCH_LENGTH if d == 1 else 0.0
                self._ball_out(
                    1 - side,
                    np.array([gx, f.end[1] + rng.uniform(-6, 6)]),
                    "right_goal_line" if gx > 1 else "left_goal_line",
                )
            else:
                self.last_touch_side = 1 - side
                v_tf = np.array([-rng.uniform(5, 14), rng.uniform(-9, 9)])
                v = v_tf * (1 if d == 1 else -1)
                v = v / max(float(np.linalg.norm(v)), 1e-6) * rng.uniform(6, 11)
                self._make_loose(self.pos[gk] + v / np.linalg.norm(v) * 1.2, v)
        elif typ == "blocked":
            b = o["blocker"]
            self.emit(
                EventType.BLOCK,
                side=1 - side,
                player=b,
                loc=f.end,
                blocked="shot",
                shooter_id=self.pl[shooter].player_id,
            )
            self.last_touch_side = 1 - side
            if rng.random() < p.block_corner_share:
                gx = PITCH_LENGTH if d == 1 else 0.0
                yy = clamp(f.end[1] + rng.uniform(-10, 10), 1, PITCH_WIDTH - 1)
                self._ball_out(
                    1 - side, np.array([gx, yy]), "right_goal_line" if gx > 1 else "left_goal_line"
                )
            else:
                ang = rng.uniform(0, 2 * math.pi)
                self._make_loose(f.end, np.array([math.cos(ang), math.sin(ang)]) * rng.uniform(4, 10))
        elif typ == "post":
            self.last_touch_side = side
            v_tf = np.array([-rng.uniform(5, 12), rng.uniform(-6, 6)])
            v = v_tf * (1 if d == 1 else -1)
            self._make_loose(f.end + v / np.linalg.norm(v) * 0.6, v)
        else:  # off target
            gx = PITCH_LENGTH if d == 1 else 0.0
            self._ball_out(
                side,
                np.array([gx, clamp(f.end[1], 0, PITCH_WIDTH)]),
                "right_goal_line" if gx > 1 else "left_goal_line",
            )

    def _goal(self, scorer: int, o: dict, shot_event: dict | None) -> None:
        P = self.pl[scorer]
        side = P.side
        team, opp = self.teams[side], self.teams[1 - side]
        team.score += 1
        assist = None
        lc = self.last_completed
        if lc and lc["receiver"] == scorer and self.t - lc["t"] < 15 and self.pl[lc["passer"]].side == side:
            assist = lc["passer"]
        team.confidence = clamp(team.confidence + 0.15, -1, 1)
        opp.confidence = clamp(opp.confidence - 0.1, -1, 1)
        self.stoppage[self.period - 1] += 45
        clk = clock_strings(self.period, self.t)
        rec = {
            "scorer_id": P.player_id,
            "scorer_name": P.profile.name,
            "assist_id": self.pl[assist].player_id if assist is not None else None,
            "assist_name": self.pl[assist].profile.name if assist is not None else None,
            "shot_kind": o["kind"],
            "body_part": "head" if o["header"] else "foot",
            "xg": round(o["xg"], 4),
            "big_chance": o["big_chance"],
            "shot_event_id": shot_event["event_id"] if shot_event else None,
            "new_score": {"home": self.teams[0].score, "away": self.teams[1].score},
            "minute_display": clk["minute_display"],
        }
        self.emit(EventType.GOAL, side=side, player=scorer, loc=self.pos[scorer], **rec)
        self.goals.append({"team_id": team.team_id, **rec})
        self._set_restart("kick_off", 1 - side, np.array(CENTER), self.p.restart_goal)

    # ------------------------------------------------------------------------------------
    # Fouls, cards, loose balls, restarts
    # ------------------------------------------------------------------------------------
    def _foul(self, j: int, c: int, kind: str, forced_card: str | None = None) -> None:
        p = self.p
        rng = self.rng
        D, V = self.pl[j], self.pl[c]
        loc = self.pos[c].copy()
        victim_side = V.side
        d = self.teams[victim_side].direction
        v_tf = to_tf(loc, d)
        speed = float(np.linalg.norm(self.vel[c]))
        tactical = v_tf[0] > 45 and speed > 4.2 and self.possession_side == victim_side
        in_box = self.in_own_box(D.side, loc)
        if self.carrier == c:
            self._release(c)
        self.emit(
            EventType.FOUL,
            side=D.side,
            player=j,
            loc=loc,
            victim_id=V.player_id,
            victim_name=V.profile.name,
            kind="tactical" if tactical else kind,
            penalty=in_box,
        )
        self.stoppage[self.period - 1] += 20
        card = forced_card
        if card is None:
            p_y = (
                p.yellow_base
                + (p.yellow_tactical_bonus if tactical else 0.0)
                + max(0, D.a.aggression - 60) / 260
            )
            p_y += 0.12 if in_box else 0.0
            r = rng.random()
            if r < p.straight_red * (1 + D.a.aggression / 80):
                card = "red"
            elif r < p_y:
                card = "yellow"
        if card:
            self._card(
                j,
                card,
                "tactical foul"
                if tactical
                else ("denied a goal-scoring chance" if card == "red" else "reckless challenge"),
            )
        self.recover_until[j] = self.t + 1.0
        self.teams[victim_side].lost_ball_at = -99.0
        if in_box:
            spot_tf = np.array([PITCH_LENGTH - PENALTY_SPOT_DIST, PITCH_WIDTH / 2])
            self.stoppage[self.period - 1] += 40
            self._set_restart("penalty", victim_side, to_tf(spot_tf, d), p.restart_penalty)
        else:
            self._set_restart("free_kick", victim_side, loc, p.restart_free_kick)

    def _card(self, j: int, card: str, reason: str) -> None:
        P = self.pl[j]
        color = card
        if card == "yellow":
            P.yellow += 1
            if P.yellow >= 2:
                color = "second_yellow"
        self.emit(EventType.CARD, side=P.side, player=j, loc=self.pos[j], card=color, reason=reason)
        self.stoppage[self.period - 1] += 20
        if color in ("red", "second_yellow"):
            self._send_off(j)

    def _send_off(self, j: int) -> None:
        P = self.pl[j]
        team = self.teams[P.side]
        was_gk = P.slot.role == "GK"
        self.sync_stats(j)
        P.on_pitch = False
        P.sent_off = True
        P.minutes_to = match_seconds(self.period, self.t)
        self.active[j] = False
        self._active_cache.clear()
        self.pos[j] = np.array([PITCH_LENGTH / 2, -4.0])
        self.vel[j] = 0.0
        team.confidence = clamp(team.confidence - 0.2, -1, 1)
        if was_gk:
            # Nearest-rated outfielder goes in goal.
            cands = [self.pl[i] for i in self.active_indices(P.side)]
            if cands:
                new_gk = min(cands, key=lambda q: q.a.pace)
                self.set_slot(new_gk.idx, team.slots[0])
        base = team.tactics
        self._change_tactics(
            team,
            base.adjusted(line_height=base.line_height - 0.1, pressing=base.pressing - 0.1),
            team.escalation,
            "down to ten men",
            "Reorganising after the red card.",
        )

    def _loose_tick(self) -> None:
        speed = float(np.linalg.norm(self.ball_vel))
        if speed > 0:
            dec = min(speed, (1.4 + 0.25 * speed) * DT)
            self.ball_vel *= (speed - dec) / speed
        self.ball_pos = self.ball_pos + self.ball_vel * DT
        if not in_pitch(*self.ball_pos):
            prev = self.ball_pos - self.ball_vel * DT
            cr = boundary_crossing(prev, self.ball_pos)
            point, line = (
                cr
                if cr
                else (np.clip(self.ball_pos, [0, 0], [PITCH_LENGTH, PITCH_WIDTH]), "bottom_touchline")
            )
            self._ball_out(self.last_touch_side if self.last_touch_side is not None else 0, point, line)
            return
        idx = np.where(self.active)[0]
        d = np.linalg.norm(self.pos[idx] - self.ball_pos, axis=1)
        near = [int(idx[k]) for k in np.where(d < 1.1)[0] if self.t >= self.recover_until[int(idx[k])]]
        if not near:
            return
        sides = {self.pl[i].side for i in near}
        if len(sides) > 1:
            weights = np.array(
                [
                    self.pl[i].a.strength
                    + self.pl[i].a.positioning
                    + self.pl[i].a.first_touch
                    + (25.0 if self.in_own_box(self.pl[i].side, self.ball_pos) else 0.0)
                    for i in near
                ],
                dtype=float,
            )
            weights = np.exp((weights - weights.max()) / 25.0)
            winner = int(self.rng.choice(near, p=weights / weights.sum()))
        else:
            winner = min(near, key=lambda i: float(np.linalg.norm(self.pos[i] - self.ball_pos)))
        P = self.pl[winner]
        self.emit(
            EventType.BALL_RECOVERY, side=P.side, player=winner, loc=self.ball_pos, contested=len(sides) > 1
        )
        self._give_ball(winner, "recovery")

    def _ball_out(self, last_side: int, point: np.ndarray, line: str) -> None:
        point = np.clip(np.asarray(point, dtype=float), [0, 0], [PITCH_LENGTH, PITCH_WIDTH])
        if self.carrier is not None:
            self._release(self.carrier)
        restart_side = 1 - last_side
        if "touchline" in line:
            kind = "throw_in"
            loc = point.copy()
            delay = self.p.restart_throw_in
        else:
            defending = None
            for s in (0, 1):
                own_goal_x = 0.0 if self.teams[s].direction == 1 else PITCH_LENGTH
                if abs(point[0] - own_goal_x) < 1.0:
                    defending = s
            if defending is None:
                defending = restart_side
            if last_side == defending:
                kind, restart_side = "corner", 1 - defending
                cy = 0.5 if point[1] < PITCH_WIDTH / 2 else PITCH_WIDTH - 0.5
                loc = np.array([point[0], cy])
                delay = self.p.restart_corner
            else:
                kind, restart_side = "goal_kick", defending
                gx = 5.5 if self.teams[defending].direction == 1 else PITCH_LENGTH - 5.5
                loc = np.array([gx, PITCH_WIDTH / 2 + self.rng.uniform(-5, 5)])
                delay = self.p.restart_goal_kick
        self.emit(
            EventType.BALL_OUT,
            side=last_side,
            loc=point,
            line=line,
            last_touch_team_id=self._team_id(last_side),
            restart=kind,
            restart_team_id=self._team_id(restart_side),
        )
        self._set_restart(kind, restart_side, loc, delay)

    # ------------------------------------------------------------------------------------
    # Substitutions and scenario incidents
    # ------------------------------------------------------------------------------------
    def _apply_pending_subs(self) -> None:
        if self.restart is None or self.restart["kind"] == "penalty":
            return
        for team in self.teams:
            plan = self._sub_plan[team.side]
            injuries = [x for x in self._pending_injury if x["side"] == team.side]
            due = [w for w in plan if w[0] <= match_seconds(self.period, self.t)]
            if not due and not injuries:
                continue
            if team.subs_used >= 5:
                self._sub_plan[team.side] = []
                self._pending_injury = [x for x in self._pending_injury if x["side"] != team.side]
                continue
            made = 0
            for inj in injuries:
                self._pending_injury.remove(inj)
                out = inj.get("idx")
                if out is not None and self.active[out]:
                    made += self._substitute(team, out, "injury")
            if due:
                window = due[0]
                plan.remove(window)
                n = window[1]
                chasing = team.score < self.teams[1 - team.side].score
                for _ in range(n):
                    out = self._pick_sub_out(team, chasing)
                    if out is None:
                        break
                    made += self._substitute(team, out, "tactical" if chasing else "fatigue")
            if made:
                team.sub_windows += 1
                self.restart["resume"] += 12.0

    def _pick_sub_out(self, team: TeamState, chasing: bool) -> int | None:
        cands = [
            ps
            for ps in team.players
            if ps.on_pitch
            and ps.slot.role != "GK"
            and ps.idx not in getattr(self, "protected", ())
            and match_seconds(self.period, self.t) - ps.minutes_from > 600
        ]
        if not cands:
            return None
        if chasing:
            defensive = [ps for ps in cands if ps.slot.role in ("DM", "FB", "CB")]
            if defensive and self.rng.random() < 0.5:
                return min(
                    defensive, key=lambda ps: self.energy[ps.idx] + (0.2 if ps.slot.role == "CB" else 0)
                ).idx
        return min(cands, key=lambda ps: self.energy[ps.idx] + self.rng.uniform(0, 0.08)).idx

    def _substitute(self, team: TeamState, out_idx: int, reason: str) -> int:
        if team.subs_used >= 5 or not team.bench:
            return 0
        out = self.pl[out_idx]
        self.sync_stats(out_idx)
        chasing = team.score < self.teams[1 - team.side].score
        role = out.slot.role
        prefs = {
            "CB": ["CB", "DM", "FB"],
            "FB": ["FB", "W", "CB"],
            "DM": ["DM", "CM", "CB"],
            "CM": ["CM", "DM", "AM"],
            "AM": ["AM", "CM", "W"],
            "W": ["W", "AM", "ST"],
            "ST": ["ST", "W", "AM"],
        }
        order = prefs.get(role, [role])
        if chasing and reason == "tactical":
            order = ["ST", "W", "AM"] + order
        bench = [b for b in team.bench if b.role != "GK"]
        if not bench:
            return 0
        incoming: PlayerProfile = min(bench, key=lambda b: order.index(b.role) if b.role in order else 9)
        team.bench.remove(incoming)
        team.subs_used += 1
        new_ps = PlayerState(profile=incoming, side=team.side, idx=out_idx, slot=out.slot)
        new_ps.minutes_from = match_seconds(self.period, self.t)
        out.on_pitch = False
        out.minutes_to = match_seconds(self.period, self.t)
        team.used_players.append(out)
        team.players[out_idx - team.side * 11] = new_ps
        self._install(new_ps)
        self.vel[out_idx] = 0.0
        self.stoppage[self.period - 1] += 30
        self.emit(
            EventType.SUBSTITUTION,
            side=team.side,
            player=out_idx,
            loc=None,
            player_out_id=out.player_id,
            player_out_name=out.profile.name,
            player_in_id=incoming.player_id,
            player_in_name=incoming.name,
            position=out.slot.label,
            reason=reason,
            energy_out=round(float(out.energy), 2),
        )
        return 1

    def _scenario_tick(self) -> None:
        if self.scenario is None:
            return
        minute = self.match_minute()
        for k, inc in enumerate(self.scenario.incidents):
            if k in self._incidents_done or minute < inc.minute:
                continue
            if self._apply_incident(inc):
                self._incidents_done.add(k)

    def _apply_incident(self, inc) -> bool:
        side = 0 if inc.team == "home" else 1
        team = self.teams[side]
        kind = inc.kind
        if inc.condition:
            diff = team.score - self.teams[1 - side].score
            state = "leading" if diff > 0 else ("trailing" if diff < 0 else "level")
            if state != inc.condition:
                return True  # condition not met at the scheduled minute: skip the incident
        if kind == "tactics":
            self._change_tactics(
                team,
                team.tactics.adjusted(**inc.changes),
                team.escalation,
                inc.reason or "manager instruction",
                inc.description or "Tactical adjustment.",
            )
            return True
        if kind == "momentum":
            team.confidence = clamp(team.confidence + inc.value, -1, 1)
            return True
        if kind == "injury":
            if inc.player_id:
                idx = next(
                    (ps.idx for ps in team.players if ps.player_id == inc.player_id and ps.on_pitch), None
                )
            else:
                cands = [ps for ps in team.players if ps.on_pitch and ps.slot.role != "GK"]
                idx = cands[int(self.rng.integers(len(cands)))].idx if cands else None
            if idx is not None:
                self._pending_injury.append({"side": side, "idx": idx})
            return True
        if kind in ("red_card", "yellow_card", "penalty"):
            c = self.carrier
            if c is None or self.ball_state != BallState.HELD:
                return False
            if kind == "penalty":
                # Team `side` wins a penalty: their carrier must be in the opponent box with a defender close.
                if self.pl[c].side != side or not self.in_own_box(1 - side, self.pos[c]):
                    return False
                opp = self.active_indices(1 - side)
                dd = np.linalg.norm(self.pos[opp] - self.pos[c], axis=1)
                if dd.min() > 3.0:
                    return False
                self._foul(int(opp[int(np.argmin(dd))]), c, "challenge")
                return True
            if self.pl[c].side == side:
                return False
            mine = self.active_indices(side)
            if inc.player_id:
                cand = [i for i in mine if self.pl[i].player_id == inc.player_id]
                if not cand:
                    return True
                j = int(cand[0])
            else:
                dd = np.linalg.norm(self.pos[mine] - self.pos[c], axis=1)
                j = int(mine[int(np.argmin(dd))])
            if np.linalg.norm(self.pos[j] - self.pos[c]) > 3.5:
                return False
            self._foul(j, c, "challenge", forced_card="red" if kind == "red_card" else "yellow")
            return True
        return True

    # ------------------------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------------------------
    def _compute_targets(self) -> None:
        if self.ball_state == BallState.DEAD:
            tactics.setup_targets(self)
            self.shape_smooth = self.target.copy()
            return
        poss = self.possession_side
        ball = self.ball_pos
        for side in (0, 1):
            tactics.shape_targets(self, side, ball, in_possession=(side == poss))
        # The block shifts with a lag: players do not track every ball movement instantly.
        if self.shape_smooth is None:
            self.shape_smooth = self.target.copy()
        else:
            alpha = np.full(N_PLAYERS, min(1.0, 2 * DT / self.p.shape_lag))
            for side in (0, 1):
                if side != poss:
                    bx = to_tf(ball, self.teams[side].direction)[0]
                    if bx < 40.0:  # danger: defenders react quickly
                        alpha[side * 11 : (side + 1) * 11] = min(1.0, 2 * DT / 0.45)
            self.shape_smooth += (self.target - self.shape_smooth) * alpha[:, None]
            self.target[:] = self.shape_smooth
        for side in (0, 1):
            if side != poss:
                tactics.apply_marking(self, side)
        if self.ball_state == BallState.HELD and self.carrier is not None:
            cs = self.pl[self.carrier].side
            tactics.apply_support_and_runs(self, cs)
            if not self.keeper_holding:
                tactics.apply_pressing(self, 1 - cs)
            P = self.pl[self.carrier]
            if P.carry_target is not None:
                self.target[self.carrier] = P.carry_target
                _, nearest = self.pressure_level(self.carrier)
                self.urg[self.carrier] = 2 if nearest > 8.0 else 1
            else:
                self.target[self.carrier] = self.pos[self.carrier] + self.vel[self.carrier] * 0.2
                self.urg[self.carrier] = 0
        elif self.ball_state == BallState.FLIGHT and self.flight is not None:
            self._flight_targets(self.flight)
        elif self.ball_state == BallState.LOOSE:
            self._chase_targets()

    def _flight_targets(self, f: Flight) -> None:
        o = f.outcome
        if f.kind == "shot":
            gk = o.get("gk")
            if gk is not None and self.active[gk]:
                self.target[gk] = 0.7 * self.pos[gk] + 0.3 * f.end
                self.urg[gk] = 3
            return
        idx = o.get("idx")
        if idx is not None and self.active[idx]:
            self.target[idx] = f.end
            self.urg[idx] = 3 if f.kind == "through" or o["type"] == "intercepted" else 2
        if o["type"] in ("aerial", "loose") or f.kind in CROSSY or o["type"] == "out":
            for side in (0, 1):
                act = self.active_indices(side)
                if len(act) == 0:
                    continue
                d = np.linalg.norm(self.pos[act] - f.end, axis=1)
                for k in np.argsort(d)[:2]:
                    i = int(act[k])
                    if d[k] < 22 and self.pl[i].slot.role != "GK":
                        self.target[i] = f.end
                        self.urg[i] = 2 if d[k] < 15 else 1

    def _chase_targets(self) -> None:
        proj = self.ball_pos + self.ball_vel * 0.6
        for side in (0, 1):
            act = self.active_indices(side)
            if len(act) == 0:
                continue
            d = np.linalg.norm(self.pos[act] - proj, axis=1)
            order = np.argsort(d)
            # In their own box, defenders swarm a loose ball.
            n_chase = 3 if self.in_own_box(side, proj) else 2
            for rank, k in enumerate(order[:n_chase]):
                i = int(act[k])
                if rank == 1 and d[k] > 12:
                    break
                if self.pl[i].slot.role == "GK" and not self.in_own_box(side, proj):
                    continue
                self.target[i] = proj
                self.urg[i] = 3 if d[k] < 8 else 2

    def _move(self) -> None:
        """Advance every active player one tick (vectorised).

        Urgency: 0 = pace set by distance to target, 1 = at least running pace,
        2 = hard running (``fast_run_factor`` x top speed), 3 = flat-out sprint.
        """
        p = self.p
        act = self.active
        delta = self.target - self.pos
        dist = np.hypot(delta[:, 0], delta[:, 1])
        top = self.top_eff
        fast = top * p.fast_run_factor
        # Distance-driven pace: walk/jog close to target, run further away, hard run when far.
        base = np.minimum(dist * 0.55 + 0.5, p.jog_speed)
        base = np.where(
            dist > 3.0, p.jog_speed + (p.run_speed - p.jog_speed) * np.minimum((dist - 3.0) / 7.0, 1.0), base
        )
        base = np.where(
            dist > 10.0, p.run_speed + (fast - p.run_speed) * np.minimum((dist - 10.0) / 14.0, 1.0), base
        )
        base[dist < p.shape_deadzone] = 0.0
        urg = self.urg
        speed = base
        speed = np.where(urg == 1, np.maximum(base, p.run_speed), speed)
        speed = np.where(urg == 2, np.maximum(base, fast), speed)
        speed = np.where(urg == 3, top, speed)
        speed = np.minimum(np.minimum(speed, top), dist * (0.8 / DT))
        if self.carrier is not None and self.ball_state == BallState.HELD:
            c = self.carrier
            speed[c] = min(speed[c], top[c] * p.dribble_speed_factor)
        speed[self.recover_until > self.t] *= 0.7
        inv = 1.0 / np.maximum(dist, 1e-6)
        dvx = delta[:, 0] * inv * speed - self.vel[:, 0]
        dvy = delta[:, 1] * inv * speed - self.vel[:, 1]
        dvn = np.hypot(dvx, dvy)
        cur = np.hypot(self.vel[:, 0], self.vel[:, 1])
        maxdv = self.accel * DT * np.where(speed < cur, 1.7, 1.0)
        scale = np.minimum(1.0, maxdv / np.maximum(dvn, 1e-9))
        vel = self.vel
        vel[:, 0] += dvx * scale
        vel[:, 1] += dvy * scale
        vel[~act] = 0.0
        self.pos += vel * DT
        np.clip(self.pos[:, 0], -2.0, PITCH_LENGTH + 2.0, out=self.pos[:, 0])
        np.clip(self.pos[:, 1], -2.0, PITCH_WIDTH + 2.0, out=self.pos[:, 1])

        sp = np.hypot(vel[:, 0], vel[:, 1])
        step = sp * DT
        self.s_dist += step
        self.s_hsr += step * (sp > 5.5)
        above = sp > 7.0
        self.s_spr += step * above
        np.maximum(self.s_maxsp, sp, out=self.s_maxsp)
        self.s_above = (self.s_above + DT) * above
        new_sprint = above & (self.s_above >= p.sprint_min_duration) & ~self.s_counted
        self.s_nspr += new_sprint
        self.s_counted = (self.s_counted | new_sprint) & above
        cost = np.where(
            above,
            p.energy_cost_per_m_sprint,
            np.where(sp > 4.0, p.energy_cost_per_m_run, np.where(sp > 2.0, p.energy_cost_per_m_jog, 0.0)),
        )
        self.energy -= step * cost * self.stam_mult
        self.energy += (sp < 2.0) * (p.energy_recovery_per_s * DT)
        np.clip(self.energy, 0.3, 1.0, out=self.energy)
        if self.carrier is not None:
            P = self.pl[self.carrier]
            P.carry_max_speed = max(P.carry_max_speed, float(sp[self.carrier]))

    def _sync_ball(self) -> None:
        if self.ball_state == BallState.HELD and self.carrier is not None:
            c = self.carrier
            v = self.vel[c]
            n = np.linalg.norm(v)
            off = v / n * 0.45 if n > 0.3 else np.zeros(2)
            self.ball_pos = self.pos[c] + off
            self.ball_z = 0.0
        elif self.ball_state == BallState.FLIGHT and self.flight is not None:
            x, y, z = self.flight.position(self.t + DT)
            self.ball_pos = np.array([x, y])
            self.ball_z = z

    def _record_frame(self) -> None:
        self.frame_no += 1
        players = []
        sp = np.linalg.norm(self.vel, axis=1)
        for i in np.where(self.active)[0]:
            ps = self.pl[i]
            players.append(
                [
                    ps.player_id,
                    round(float(self.pos[i, 0]), 2),
                    round(float(self.pos[i, 1]), 2),
                    round(float(sp[i]), 2),
                ]
            )
        if self.ball_state == BallState.FLIGHT and self.flight is not None:
            bspeed = self.flight.speed(self.t)
        elif self.ball_state == BallState.LOOSE:
            bspeed = float(np.linalg.norm(self.ball_vel))
        elif self.ball_state == BallState.HELD and self.carrier is not None:
            bspeed = float(sp[self.carrier])
        else:
            bspeed = 0.0
        clk = clock_strings(self.period, self.t)
        self._buffer.append(
            (
                "frame",
                {
                    "frame": self.frame_no,
                    "period": self.period,
                    "t": round(self.t, 2),
                    "clock": clk["clock"],
                    "ball": {
                        "x": round(float(self.ball_pos[0]), 2),
                        "y": round(float(self.ball_pos[1]), 2),
                        "z": round(float(self.ball_z), 2),
                        "speed_ms": round(bspeed, 2),
                        "state": self.ball_state.name.lower(),
                        "owner_id": self.pl[self.carrier].player_id if self.carrier is not None else None,
                    },
                    "possession_team_id": self._team_id(self.possession_side)
                    if self.ball_state != BallState.DEAD
                    else None,
                    "players": players,
                },
            )
        )


def _native(obj):
    """Recursively convert numpy scalars so every payload is plain, JSON-serialisable Python."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _loc(p: np.ndarray) -> dict:
    return {"x": round(float(p[0]), 2), "y": round(float(p[1]), 2)}


def _restart_pass_kind(restart: str, kind: str) -> str:
    mapping = {
        "throw_in": "throw_in",
        "goal_kick": "goal_kick",
        "corner": "corner",
        "kick_off": "kick_off",
        "keeper": "keeper_distribution",
        "free_kick": "free_kick",
    }
    return mapping.get(restart, kind)
