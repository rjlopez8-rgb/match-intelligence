"""Static domain models: players, clubs, tactics and match configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal

Role = Literal["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]

ATTRIBUTE_NAMES: tuple[str, ...] = (
    "pace",
    "acceleration",
    "stamina",
    "strength",
    "passing",
    "vision",
    "first_touch",
    "dribbling",
    "crossing",
    "finishing",
    "shot_power",
    "heading",
    "tackling",
    "positioning",
    "aggression",
    "composure",
    "goalkeeping",
)


@dataclass(frozen=True)
class Attributes:
    """Player ratings on a 1-99 scale."""

    pace: int = 65
    acceleration: int = 65
    stamina: int = 70
    strength: int = 65
    passing: int = 65
    vision: int = 60
    first_touch: int = 65
    dribbling: int = 60
    crossing: int = 55
    finishing: int = 55
    shot_power: int = 65
    heading: int = 60
    tackling: int = 55
    positioning: int = 60
    aggression: int = 55
    composure: int = 62
    goalkeeping: int = 20

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PlayerProfile:
    player_id: str
    name: str
    short_name: str
    shirt: int
    nationality: str
    age: int
    role: Role
    preferred_position: str
    foot: Literal["left", "right"]
    attributes: Attributes
    bio: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["attributes"] = self.attributes.as_dict()
        return d


@dataclass(frozen=True)
class Tactics:
    """Team tactical identity. All values are 0-1."""

    pressing: float = 0.5  # how high and how hard the team presses
    line_height: float = 0.5  # defensive line depth when out of possession
    width: float = 0.5  # how wide the team spreads in possession
    tempo: float = 0.5  # speed of decisions and ball circulation
    directness: float = 0.4  # preference for long, vertical passes
    risk: float = 0.5  # appetite for risky progressive passes
    crossing: float = 0.5  # preference for crossing from wide areas
    style: str = "balanced"
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def adjusted(self, **changes: float) -> Tactics:
        clamped = {k: min(1.0, max(0.0, v)) if isinstance(v, float) else v for k, v in changes.items()}
        return replace(self, **clamped)


@dataclass(frozen=True)
class ClubProfile:
    club_id: str
    name: str
    short_name: str
    code: str
    city: str
    stadium: str
    capacity: int
    founded: int
    colors: dict[str, str]
    manager: str
    formation: str
    tactics: Tactics
    squad: tuple[PlayerProfile, ...]
    starting_xi: tuple[str, ...]  # player_ids ordered to match formation slots
    bench: tuple[str, ...]
    identity: str = ""

    def player(self, player_id: str) -> PlayerProfile:
        for p in self.squad:
            if p.player_id == player_id:
                return p
        raise KeyError(player_id)

    def to_dict(self) -> dict:
        return {
            "club_id": self.club_id,
            "name": self.name,
            "short_name": self.short_name,
            "code": self.code,
            "city": self.city,
            "stadium": self.stadium,
            "capacity": self.capacity,
            "founded": self.founded,
            "colors": dict(self.colors),
            "manager": self.manager,
            "formation": self.formation,
            "tactics": self.tactics.to_dict(),
            "identity": self.identity,
            "squad": [p.to_dict() for p in self.squad],
            "starting_xi": list(self.starting_xi),
            "bench": list(self.bench),
        }


@dataclass(frozen=True)
class MatchConfig:
    home: ClubProfile
    away: ClubProfile
    seed: int = 7
    match_id: str | None = None
    competition: str = "Meridian League"
    season: str = "2026/27"
    matchday: int = 1
    kickoff_utc: str = "2026-10-24T14:00:00Z"
    referee: str = "M. Whitlock"
    weather: dict = field(
        default_factory=lambda: {"condition": "overcast", "temperature_c": 12, "wind_kph": 14}
    )
    attendance: int | None = None

    def resolved_match_id(self) -> str:
        return self.match_id or f"{self.home.code}-{self.away.code}-{self.seed}".lower()
