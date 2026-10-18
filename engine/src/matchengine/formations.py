"""Formation templates.

Each slot has a depth ``u`` (0 = defensive line, 1 = forward line) and a lateral
``y`` in metres, expressed in the team's attacking frame (attacking towards
x = 105, so small ``y`` is the right flank).
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Role


@dataclass(frozen=True)
class Slot:
    label: str
    role: Role
    u: float
    y: float


GK_SLOT = Slot("GK", "GK", -1.0, 34.0)

FORMATIONS: dict[str, tuple[Slot, ...]] = {
    "4-3-3": (
        GK_SLOT,
        Slot("RB", "FB", 0.04, 7.5),
        Slot("RCB", "CB", 0.0, 24.5),
        Slot("LCB", "CB", 0.0, 43.5),
        Slot("LB", "FB", 0.04, 60.5),
        Slot("DM", "DM", 0.28, 34.0),
        Slot("RCM", "CM", 0.47, 22.0),
        Slot("LCM", "CM", 0.47, 46.0),
        Slot("RW", "W", 0.92, 8.5),
        Slot("ST", "ST", 1.0, 34.0),
        Slot("LW", "W", 0.92, 59.5),
    ),
    "4-2-3-1": (
        GK_SLOT,
        Slot("RB", "FB", 0.04, 7.5),
        Slot("RCB", "CB", 0.0, 24.5),
        Slot("LCB", "CB", 0.0, 43.5),
        Slot("LB", "FB", 0.04, 60.5),
        Slot("RDM", "DM", 0.3, 26.0),
        Slot("LDM", "DM", 0.3, 42.0),
        Slot("RW", "W", 0.74, 10.0),
        Slot("AM", "AM", 0.72, 34.0),
        Slot("LW", "W", 0.74, 58.0),
        Slot("ST", "ST", 1.0, 34.0),
    ),
    "4-4-2": (
        GK_SLOT,
        Slot("RB", "FB", 0.03, 7.5),
        Slot("RCB", "CB", 0.0, 25.0),
        Slot("LCB", "CB", 0.0, 43.0),
        Slot("LB", "FB", 0.03, 60.5),
        Slot("RM", "W", 0.5, 9.0),
        Slot("RCM", "CM", 0.45, 26.5),
        Slot("LCM", "CM", 0.45, 41.5),
        Slot("LM", "W", 0.5, 59.0),
        Slot("RS", "ST", 1.0, 28.0),
        Slot("LS", "ST", 1.0, 40.0),
    ),
    "3-5-2": (
        GK_SLOT,
        Slot("RCB", "CB", 0.0, 20.0),
        Slot("CB", "CB", -0.03, 34.0),
        Slot("LCB", "CB", 0.0, 48.0),
        Slot("RWB", "FB", 0.45, 6.0),
        Slot("RCM", "CM", 0.47, 23.5),
        Slot("DM", "DM", 0.3, 34.0),
        Slot("LCM", "CM", 0.47, 44.5),
        Slot("LWB", "FB", 0.45, 62.0),
        Slot("RS", "ST", 1.0, 28.5),
        Slot("LS", "ST", 1.0, 39.5),
    ),
    "3-4-3": (
        GK_SLOT,
        Slot("RCB", "CB", 0.0, 20.0),
        Slot("CB", "CB", -0.03, 34.0),
        Slot("LCB", "CB", 0.0, 48.0),
        Slot("RWB", "FB", 0.45, 6.0),
        Slot("RCM", "CM", 0.4, 26.5),
        Slot("LCM", "CM", 0.4, 41.5),
        Slot("LWB", "FB", 0.45, 62.0),
        Slot("RW", "W", 0.9, 12.0),
        Slot("ST", "ST", 1.0, 34.0),
        Slot("LW", "W", 0.9, 56.0),
    ),
}


def slots_for(formation: str) -> tuple[Slot, ...]:
    try:
        return FORMATIONS[formation]
    except KeyError as exc:
        raise ValueError(f"Unknown formation {formation!r}; choose from {sorted(FORMATIONS)}") from exc
