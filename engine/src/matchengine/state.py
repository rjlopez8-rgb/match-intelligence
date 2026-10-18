"""Mutable match state used by the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from .formations import Slot
from .models import ClubProfile, PlayerProfile, Tactics


class BallState(IntEnum):
    HELD = 0
    FLIGHT = 1
    LOOSE = 2
    DEAD = 3


@dataclass
class Flight:
    """A ball travelling from a kick/header to a predetermined resolution."""

    kind: str  # pass | cross | corner | long | clearance | goal_kick | shot | throw_in
    start: np.ndarray
    end: np.ndarray
    t0: float
    t1: float
    peak: float  # 0 for ground balls
    v0: float  # release speed, m/s
    decel: float  # ground deceleration (0 for aerial)
    side: int
    kicker: int
    receiver: int | None
    outcome: dict
    origin_seq: int | None = None

    def position(self, now: float) -> tuple[float, float, float]:
        total = max(self.t1 - self.t0, 1e-6)
        tau = min(max(now - self.t0, 0.0), total)
        vec = self.end - self.start
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            return float(self.start[0]), float(self.start[1]), 0.0
        if self.decel > 0:
            s = self.v0 * tau - 0.5 * self.decel * tau * tau
            frac = min(max(s / dist, 0.0), 1.0)
            z = 0.0
        else:
            frac = tau / total
            z = 4.0 * self.peak * frac * (1.0 - frac) if self.peak > 0 else 0.0
        p = self.start + vec * frac
        return float(p[0]), float(p[1]), float(z)

    def speed(self, now: float) -> float:
        tau = max(now - self.t0, 0.0)
        if self.decel > 0:
            return max(self.v0 - self.decel * tau, 0.0)
        return self.v0


@dataclass
class PlayerState:
    profile: PlayerProfile
    side: int
    idx: int  # global array index (0-10 home, 11-21 away)
    slot: Slot
    energy: float = 1.0
    on_pitch: bool = True
    yellow: int = 0
    sent_off: bool = False
    decide_at: float = 0.0
    carry_target: np.ndarray | None = None
    carry_start: np.ndarray | None = None
    carry_t0: float = 0.0
    carry_max_speed: float = 0.0
    run_target_tf: np.ndarray | None = None
    run_until: float = 0.0
    run_kind: str = ""
    recover_until: float = 0.0
    received_from: int | None = None
    minutes_from: float = 0.0  # match second when the player came on
    minutes_to: float | None = None
    # Physical accumulators (metres, m/s)
    distance: float = 0.0
    hsr_distance: float = 0.0  # > 5.5 m/s (19.8 km/h)
    sprint_distance: float = 0.0  # > 7.0 m/s (25.2 km/h)
    sprints: int = 0
    in_sprint: bool = False
    max_speed: float = 0.0

    @property
    def player_id(self) -> str:
        return self.profile.player_id

    @property
    def a(self):  # attribute shortcut
        return self.profile.attributes


@dataclass
class TeamState:
    side: int  # 0 home, 1 away
    club: ClubProfile
    tactics: Tactics
    slots: list[Slot]
    players: list[PlayerState]  # index k -> global idx side*11 + k
    bench: list[PlayerProfile]
    direction: int = 1  # +1 attacking towards x=105
    score: int = 0
    confidence: float = 0.0
    subs_used: int = 0
    sub_windows: int = 0
    escalation: int = 0  # 0 normal, 1 chasing, 2 all-out, -1 protecting
    used_players: list[PlayerState] = field(default_factory=list)
    lost_ball_at: float = -99.0
    next_run_at: float = 0.0

    @property
    def team_id(self) -> str:
        return self.club.club_id

    def active(self) -> list[PlayerState]:
        return [p for p in self.players if p.on_pitch]
