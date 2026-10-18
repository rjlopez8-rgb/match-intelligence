"""The Meridian League: a fully fictional league of eight clubs.

Every club, player, manager and stadium is invented. Squads are generated
deterministically, so the same league is produced on every machine. The two
showcase clubs (Riverside Athletic and Kingsport United) have hand-written star
players with backstories that the narrative agents can use for storylines.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

from .formations import slots_for
from .models import Attributes, ClubProfile, PlayerProfile, Role, Tactics

# Mean ratings per role for a club of quality 0 (mid-table). Tier offsets are added.
ROLE_TEMPLATES: dict[str, dict[str, int]] = {
    "GK": dict(
        pace=45,
        acceleration=50,
        stamina=55,
        strength=70,
        passing=58,
        vision=50,
        first_touch=55,
        dribbling=30,
        crossing=25,
        finishing=15,
        shot_power=60,
        heading=40,
        tackling=25,
        positioning=70,
        aggression=40,
        composure=68,
        goalkeeping=76,
    ),
    "CB": dict(
        pace=64,
        acceleration=60,
        stamina=72,
        strength=78,
        passing=64,
        vision=52,
        first_touch=62,
        dribbling=50,
        crossing=40,
        finishing=35,
        shot_power=62,
        heading=76,
        tackling=76,
        positioning=75,
        aggression=66,
        composure=68,
        goalkeeping=10,
    ),
    "FB": dict(
        pace=77,
        acceleration=76,
        stamina=80,
        strength=64,
        passing=68,
        vision=60,
        first_touch=68,
        dribbling=66,
        crossing=70,
        finishing=42,
        shot_power=60,
        heading=58,
        tackling=70,
        positioning=68,
        aggression=60,
        composure=64,
        goalkeeping=10,
    ),
    "DM": dict(
        pace=64,
        acceleration=64,
        stamina=80,
        strength=72,
        passing=74,
        vision=68,
        first_touch=72,
        dribbling=64,
        crossing=55,
        finishing=50,
        shot_power=68,
        heading=64,
        tackling=74,
        positioning=74,
        aggression=66,
        composure=72,
        goalkeeping=10,
    ),
    "CM": dict(
        pace=68,
        acceleration=68,
        stamina=80,
        strength=66,
        passing=76,
        vision=73,
        first_touch=75,
        dribbling=70,
        crossing=62,
        finishing=60,
        shot_power=70,
        heading=56,
        tackling=64,
        positioning=66,
        aggression=58,
        composure=72,
        goalkeeping=10,
    ),
    "AM": dict(
        pace=72,
        acceleration=74,
        stamina=72,
        strength=58,
        passing=78,
        vision=79,
        first_touch=79,
        dribbling=78,
        crossing=68,
        finishing=70,
        shot_power=70,
        heading=50,
        tackling=45,
        positioning=52,
        aggression=48,
        composure=74,
        goalkeeping=10,
    ),
    "W": dict(
        pace=83,
        acceleration=83,
        stamina=74,
        strength=58,
        passing=70,
        vision=68,
        first_touch=75,
        dribbling=79,
        crossing=72,
        finishing=68,
        shot_power=68,
        heading=50,
        tackling=42,
        positioning=48,
        aggression=50,
        composure=68,
        goalkeeping=10,
    ),
    "ST": dict(
        pace=77,
        acceleration=76,
        stamina=72,
        strength=72,
        passing=64,
        vision=62,
        first_touch=74,
        dribbling=72,
        crossing=52,
        finishing=78,
        shot_power=76,
        heading=72,
        tackling=36,
        positioning=45,
        aggression=58,
        composure=74,
        goalkeeping=10,
    ),
}

NAME_POOLS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "England": (
        (
            "Callum",
            "Harvey",
            "Jude",
            "Reece",
            "Owen",
            "Elliot",
            "Kieran",
            "Danny",
            "Josh",
            "Toby",
            "Lewis",
            "Sam",
        ),
        (
            "Ashworth",
            "Pennington",
            "Holloway",
            "Brierley",
            "Thornbury",
            "Kettley",
            "Marsden",
            "Haddon",
            "Whitcombe",
            "Lindley",
            "Croft",
            "Bramall",
            "Fairclough",
            "Ellery",
            "Stannard",
            "Pollitt",
        ),
    ),
    "Spain": (
        ("Álvaro", "Iker", "Unai", "Rubén", "Adrián", "Mikel", "Sergi", "Jaume", "Aitor", "Borja"),
        (
            "Larrabe",
            "Etxeberri",
            "Montalbán",
            "Quirós",
            "Bellido",
            "Sanromán",
            "Arregui",
            "Carbajosa",
            "Olabarri",
            "Escudero",
            "Valdivieso",
            "Garmendia",
        ),
    ),
    "Portugal": (
        ("Tiago", "Diogo", "Rui", "Gonçalo", "Nuno", "Vasco", "Henrique", "Afonso"),
        (
            "Mourato",
            "Esteves",
            "Calado",
            "Pimenta",
            "Valadares",
            "Antunes",
            "Barroso",
            "Quintela",
            "Seabra",
            "Lourenço",
        ),
    ),
    "Brazil": (
        ("Caio", "Thiago", "Murilo", "Vinícius", "Davi", "Igor", "Renan", "Otávio"),
        (
            "Barcellos",
            "Teixeira",
            "Amarante",
            "Fontoura",
            "Lacerda",
            "Prudente",
            "Cavalcanti",
            "Moreira",
            "Siqueira",
            "Bittencourt",
        ),
    ),
    "France": (
        ("Théo", "Mathis", "Enzo", "Loïc", "Yanis", "Baptiste", "Maxence", "Rayan"),
        (
            "Delaunay",
            "Ferrand",
            "Marchetti",
            "Royer",
            "Vasseur",
            "Lemaire",
            "Charpentier",
            "Morvan",
            "Guillot",
            "Besnard",
        ),
    ),
    "Netherlands": (
        ("Daan", "Sem", "Joep", "Bram", "Thijs", "Niels", "Stijn", "Wout"),
        (
            "van Dorsten",
            "Kuipers",
            "de Wilde",
            "Brouwer",
            "Veltman",
            "Schuurman",
            "Hoogland",
            "Verbeek",
            "van Rijn",
            "Dekkers",
        ),
    ),
    "Nigeria": (
        ("Chidi", "Emeka", "Tunde", "Obinna", "Femi", "Kelechi", "Ikenna", "Segun"),
        (
            "Adebayo",
            "Nwosu",
            "Okonkwo",
            "Eze",
            "Balogun",
            "Ogunleye",
            "Uzodinma",
            "Akinola",
            "Ihejirika",
            "Olatunji",
        ),
    ),
    "Ghana": (
        ("Kofi", "Yaw", "Kojo", "Nana", "Ebo", "Fiifi", "Kwabena", "Kwesi"),
        ("Asante", "Boakye", "Owusu", "Darko", "Ofori", "Amankwah", "Agyemang", "Frimpong", "Opoku", "Badu"),
    ),
    "Senegal": (
        ("Moussa", "Cheikh", "Pape", "Lamine", "Ousmane", "Abdou", "Mamadou", "Saliou"),
        ("Ndiaye", "Faye", "Gueye", "Diallo", "Cissé", "Seck", "Mbengue", "Thiam", "Sow", "Kane"),
    ),
    "Norway": (
        ("Sindre", "Eirik", "Henrik", "Jonas", "Magnus", "Torstein", "Vegard", "Even"),
        (
            "Brekke",
            "Solheim",
            "Aasen",
            "Lunde",
            "Haugland",
            "Rygg",
            "Fjeldstad",
            "Nygaard",
            "Skogen",
            "Vatne",
        ),
    ),
    "Japan": (
        ("Haruto", "Sota", "Ren", "Kaito", "Daiki", "Riku", "Yuma", "Kenta"),
        (
            "Nakahara",
            "Morishita",
            "Takeuchi",
            "Ishikawa",
            "Fujimura",
            "Okada",
            "Hirano",
            "Sakamoto",
            "Kubota",
            "Miyazawa",
        ),
    ),
    "Korea": (
        ("Min-jun", "Seo-jun", "Ji-ho", "Do-yun", "Hyun-woo", "Jae-won", "Tae-yang"),
        ("Bae", "Seong", "Hwang", "Yoon", "Kwon", "Jeon", "Moon", "Ryu"),
    ),
    "Mexico": (
        ("Emiliano", "Santiago", "Diego", "Rodrigo", "Mateo", "Ulises", "Gael"),
        ("Villaseñor", "Arreola", "Cárdenas", "Ibarra", "Montes", "Zavala", "Treviño", "Olvera"),
    ),
    "USA": (
        ("Tyler", "Brandon", "Cole", "Mason", "Jalen", "Grady", "Wyatt"),
        ("Whitaker", "Calloway", "Pruitt", "Hendrix", "Langford", "Sutter", "Kimbrough", "Ransom"),
    ),
    "Ireland": (
        ("Cian", "Darragh", "Oisín", "Ronan", "Seán", "Fionn", "Tadhg"),
        ("Mulcahy", "Brennan", "Fitzgerald", "Kinsella", "Doherty", "Gilmartin", "Ó Riain", "Hennessy"),
    ),
    "Croatia": (
        ("Luka", "Ivan", "Marin", "Domagoj", "Toma", "Ante"),
        ("Barišić", "Vuković", "Jurić", "Perković", "Šimunović", "Kovačić", "Babić", "Radošević"),
    ),
}


@dataclass(frozen=True)
class StarPlayer:
    slot: str
    name: str
    short_name: str
    shirt: int
    nationality: str
    age: int
    foot: str
    bio: str
    boosts: dict[str, int]


@dataclass(frozen=True)
class ClubSpec:
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
    quality: int  # rating offset vs. league average
    nationality_mix: tuple[str, ...]
    identity: str
    stars: tuple[StarPlayer, ...] = ()


CLUB_SPECS: tuple[ClubSpec, ...] = (
    ClubSpec(
        club_id="riverside-athletic",
        name="Riverside Athletic",
        short_name="Riverside",
        code="RIV",
        city="Riverside",
        stadium="The Wharf Ground",
        capacity=41200,
        founded=1889,
        colors={"primary": "#C8102E", "secondary": "#FFFFFF", "text": "#FFFFFF"},
        manager="Elena Marsh",
        formation="4-3-3",
        tactics=Tactics(
            pressing=0.82,
            line_height=0.74,
            width=0.68,
            tempo=0.66,
            directness=0.28,
            risk=0.55,
            crossing=0.42,
            style="high-press possession",
            description="Win it back high, keep it short, overload the half-spaces.",
        ),
        quality=4,
        nationality_mix=("England", "Ghana", "Spain", "Netherlands", "Norway", "France", "Croatia"),
        identity="A pressing machine built by Elena Marsh: suffocating out of possession, patient with the ball.",
        stars=(
            StarPlayer(
                "ST",
                "Kwame Mensah",
                "Mensah",
                9,
                "Ghana",
                27,
                "right",
                "Club captain and last season's top scorer, but without a goal in his last four league matches.",
                {"finishing": 12, "composure": 8, "heading": 6, "pace": 4},
            ),
            StarPlayer(
                "RCM",
                "Tomás Okafor",
                "Okafor",
                8,
                "Spain",
                25,
                "right",
                "The metronome. Leads the league in through balls; Spanish-born to a Nigerian father.",
                {"passing": 13, "vision": 14, "first_touch": 8, "composure": 8},
            ),
            StarPlayer(
                "LW",
                "Lucas Varga",
                "Varga",
                11,
                "Croatia",
                20,
                "left",
                "Academy graduate in his first full season; the quickest player in the league.",
                {"pace": 12, "acceleration": 12, "dribbling": 9, "composure": -6},
            ),
            StarPlayer(
                "GK",
                "Erik Halvorsen",
                "Halvorsen",
                1,
                "Norway",
                30,
                "right",
                "Sweeper-keeper who starts attacks with his feet.",
                {"goalkeeping": 7, "passing": 10},
            ),
        ),
    ),
    ClubSpec(
        club_id="kingsport-united",
        name="Kingsport United",
        short_name="Kingsport",
        code="KIN",
        city="Kingsport",
        stadium="Harborside Park",
        capacity=38600,
        founded=1902,
        colors={"primary": "#0B2D6B", "secondary": "#F2C230", "text": "#FFFFFF"},
        manager="Viktor Hale",
        formation="4-2-3-1",
        tactics=Tactics(
            pressing=0.36,
            line_height=0.3,
            width=0.55,
            tempo=0.62,
            directness=0.66,
            risk=0.6,
            crossing=0.58,
            style="low-block counter-attack",
            description="Sit deep, stay compact, then break fast through the wide players.",
        ),
        quality=3,
        nationality_mix=("England", "Portugal", "Senegal", "Japan", "Ireland", "Brazil", "Nigeria"),
        identity="Viktor Hale's side concede territory on purpose and kill teams on the break.",
        stars=(
            StarPlayer(
                "ST",
                "Rafael Duarte",
                "Duarte",
                9,
                "Portugal",
                31,
                "left",
                "Veteran poacher who spent five seasons at Riverside before a bitter exit two summers ago.",
                {"finishing": 13, "composure": 12, "heading": 6, "pace": -4},
            ),
            StarPlayer(
                "AM",
                "Yuto Arakawa",
                "Arakawa",
                10,
                "Japan",
                24,
                "left",
                "Creative hub; dangerous from free kicks and the left half-space.",
                {"passing": 9, "vision": 11, "dribbling": 9, "shot_power": 6},
            ),
            StarPlayer(
                "RCB",
                "Marcus Bell",
                "Bell",
                5,
                "England",
                29,
                "right",
                "Captain and organiser. Wins everything in the air, but walks a disciplinary tightrope.",
                {"heading": 10, "tackling": 8, "aggression": 18, "strength": 8, "pace": -6},
            ),
            StarPlayer(
                "RW",
                "Ibrahima Sarr",
                "Sarr",
                7,
                "Senegal",
                23,
                "right",
                "Explosive on the counter; the outlet for every Kingsport break.",
                {"pace": 11, "acceleration": 10, "dribbling": 6},
            ),
        ),
    ),
    ClubSpec(
        club_id="ashvale-rovers",
        name="Ashvale Rovers",
        short_name="Ashvale",
        code="ASH",
        city="Ashvale",
        stadium="Mill Lane",
        capacity=27400,
        founded=1878,
        colors={"primary": "#1E7B3C", "secondary": "#FFFFFF", "text": "#FFFFFF"},
        manager="Declan Furey",
        formation="4-4-2",
        tactics=Tactics(
            pressing=0.5,
            line_height=0.45,
            width=0.72,
            tempo=0.55,
            directness=0.6,
            risk=0.45,
            crossing=0.82,
            style="wing play and crosses",
            description="Get it wide, get it in the box.",
        ),
        quality=-2,
        nationality_mix=("England", "Ireland", "Norway", "USA", "Netherlands"),
        identity="Old-school and proud of it: two strikers, two wingers, crosses all day.",
    ),
    ClubSpec(
        club_id="northgate-city",
        name="Northgate City",
        short_name="Northgate",
        code="NGC",
        city="Northgate",
        stadium="Beacon Stadium",
        capacity=52300,
        founded=1894,
        colors={"primary": "#6CABDD", "secondary": "#1C2C5B", "text": "#1C2C5B"},
        manager="Sofia Brandt",
        formation="4-3-3",
        tactics=Tactics(
            pressing=0.66,
            line_height=0.68,
            width=0.7,
            tempo=0.5,
            directness=0.2,
            risk=0.4,
            crossing=0.35,
            style="positional possession",
            description="Dominate the ball and wait for the opening.",
        ),
        quality=6,
        nationality_mix=("Spain", "Brazil", "Netherlands", "France", "Portugal", "Korea"),
        identity="The league's richest club; expected to dominate possession every week.",
    ),
    ClubSpec(
        club_id="westmere-town",
        name="Westmere Town",
        short_name="Westmere",
        code="WES",
        city="Westmere",
        stadium="The Quarry",
        capacity=22100,
        founded=1911,
        colors={"primary": "#7A263A", "secondary": "#7CC0E8", "text": "#FFFFFF"},
        manager="Gareth Pryce",
        formation="3-5-2",
        tactics=Tactics(
            pressing=0.42,
            line_height=0.38,
            width=0.62,
            tempo=0.48,
            directness=0.52,
            risk=0.42,
            crossing=0.6,
            style="compact back three",
            description="Hard to break down; wing-backs supply the strikers.",
        ),
        quality=-3,
        nationality_mix=("England", "Ireland", "Mexico", "Croatia", "Senegal"),
        identity="Newly promoted and fighting for every point.",
    ),
    ClubSpec(
        club_id="crowmoor-wanderers",
        name="Crowmoor Wanderers",
        short_name="Crowmoor",
        code="CRW",
        city="Crowmoor",
        stadium="Heath Road",
        capacity=31800,
        founded=1885,
        colors={"primary": "#FDB913", "secondary": "#231F20", "text": "#231F20"},
        manager="Ines Calderón",
        formation="3-4-3",
        tactics=Tactics(
            pressing=0.72,
            line_height=0.62,
            width=0.66,
            tempo=0.74,
            directness=0.48,
            risk=0.66,
            crossing=0.5,
            style="chaotic gegenpress",
            description="Fast, vertical and relentless; matches get wild.",
        ),
        quality=0,
        nationality_mix=("Spain", "Mexico", "Portugal", "Brazil", "Ghana", "Japan"),
        identity="The league's entertainers: they score lots and concede lots.",
    ),
    ClubSpec(
        club_id="saltmarsh-fc",
        name="Saltmarsh FC",
        short_name="Saltmarsh",
        code="SAL",
        city="Saltmarsh",
        stadium="Estuary Road",
        capacity=19600,
        founded=1920,
        colors={"primary": "#E35205", "secondary": "#003B5C", "text": "#FFFFFF"},
        manager="Pieter Smit",
        formation="4-4-2",
        tactics=Tactics(
            pressing=0.3,
            line_height=0.25,
            width=0.5,
            tempo=0.4,
            directness=0.7,
            risk=0.35,
            crossing=0.6,
            style="deep block, long balls",
            description="Defend the box, go long to the big man.",
        ),
        quality=-5,
        nationality_mix=("England", "Netherlands", "Norway", "Ireland", "Nigeria"),
        identity="Survival specialists who rely on set pieces.",
    ),
    ClubSpec(
        club_id="elmhurst-albion",
        name="Elmhurst Albion",
        short_name="Elmhurst",
        code="ELM",
        city="Elmhurst",
        stadium="The Orchard",
        capacity=33900,
        founded=1899,
        colors={"primary": "#5B2C83", "secondary": "#E6E6E6", "text": "#FFFFFF"},
        manager="Tomasz Wrona",
        formation="4-2-3-1",
        tactics=Tactics(
            pressing=0.55,
            line_height=0.52,
            width=0.6,
            tempo=0.58,
            directness=0.42,
            risk=0.52,
            crossing=0.5,
            style="balanced",
            description="Flexible and well-drilled.",
        ),
        quality=1,
        nationality_mix=("France", "Croatia", "Korea", "USA", "Ghana", "England"),
        identity="Quietly consistent; nobody enjoys playing them.",
    ),
)

BENCH_ROLES: tuple[tuple[str, Role], ...] = (
    ("GK", "GK"),
    ("CB", "CB"),
    ("RB", "FB"),
    ("CM", "CM"),
    ("DM", "DM"),
    ("AM", "AM"),
    ("RW", "W"),
    ("LW", "W"),
    ("ST", "ST"),
)


def _rating(rng: random.Random, mean: float, sd: float = 5.5) -> int:
    return int(max(25, min(96, round(rng.gauss(mean, sd)))))


def _make_attributes(
    rng: random.Random, role: str, quality: int, boosts: dict[str, int] | None = None, bench: bool = False
) -> Attributes:
    template = ROLE_TEMPLATES[role]
    offset = quality - (3 if bench else 0)
    values = {k: _rating(rng, v + offset) for k, v in template.items()}
    for k, v in (boosts or {}).items():
        values[k] = int(max(25, min(97, values[k] + v)))
    if role != "GK":
        values["goalkeeping"] = rng.randint(5, 15)
    return Attributes(**values)


def _unique_name(
    rng: random.Random, nationality: str, league_used: set[str], club_used: set[str]
) -> tuple[str, str]:
    """Pick a name whose full form is unique in the league and whose first and last names are unique in the club."""
    firsts, lasts = NAME_POOLS[nationality]
    club_first = {u.split(" ", 1)[0] for u in club_used}
    club_last = {u.split(" ", 1)[-1] for u in club_used}
    for _ in range(400):
        first, last = rng.choice(firsts), rng.choice(lasts)
        full = f"{first} {last}"
        if full in league_used or last in club_last or first in club_first:
            continue
        league_used.add(full)
        club_used.add(full)
        return full, last
    for first in firsts:  # relax the first-name rule before giving up
        for last in lasts:
            full = f"{first} {last}"
            if full not in league_used and last not in club_last:
                league_used.add(full)
                club_used.add(full)
                return full, last
    raise RuntimeError(f"name pool for {nationality} exhausted")


def _pick_name(
    rng: random.Random, spec: ClubSpec, league_used: set[str], club_used: set[str]
) -> tuple[str, str, str]:
    order = list(spec.nationality_mix)
    first = rng.choice(order)
    order.remove(first)
    rng.shuffle(order)
    for nat in [first, *order]:
        try:
            name, short = _unique_name(rng, nat, league_used, club_used)
            return nat, name, short
        except RuntimeError:
            continue
    raise RuntimeError(f"could not name a player for {spec.name}")


def build_club(spec: ClubSpec, seed: int = 2026, league_names: set[str] | None = None) -> ClubProfile:
    """Generate a club. ``league_names`` (shared across clubs) keeps every name unique league-wide."""
    rng = random.Random(f"{spec.club_id}:{seed}")
    slots = slots_for(spec.formation)
    stars = {s.slot: s for s in spec.stars}
    used_names: set[str] = league_names if league_names is not None else set()
    used_names.update(s.name for s in spec.stars)
    club_names: set[str] = {s.name for s in spec.stars}
    used_shirts: set[int] = {s.shirt for s in spec.stars}
    squad: list[PlayerProfile] = []
    xi: list[str] = []

    def next_shirt(preferred: int) -> int:
        n = preferred
        while n in used_shirts:
            n += 1 if n < 40 else -38
        used_shirts.add(n)
        return n

    default_shirts = {
        "GK": 1,
        "RB": 2,
        "LB": 3,
        "RCB": 5,
        "LCB": 6,
        "CB": 4,
        "DM": 4,
        "RDM": 4,
        "LDM": 16,
        "RCM": 8,
        "LCM": 14,
        "CM": 8,
        "AM": 10,
        "RW": 7,
        "LW": 11,
        "RM": 7,
        "LM": 11,
        "ST": 9,
        "RS": 9,
        "LS": 19,
        "RWB": 2,
        "LWB": 3,
    }

    for i, slot in enumerate(slots):
        pid = f"{spec.code.lower()}-{i + 1:02d}"
        if slot.label in stars:
            s = stars[slot.label]
            attrs = _make_attributes(rng, slot.role, spec.quality, s.boosts)
            player = PlayerProfile(
                pid,
                s.name,
                s.short_name,
                s.shirt,
                s.nationality,
                s.age,
                slot.role,
                slot.label,
                s.foot,
                attrs,
                s.bio,
            )  # type: ignore[arg-type]
        else:
            nat, name, short = _pick_name(rng, spec, used_names, club_names)
            foot = (
                "left"
                if slot.label.startswith("L") and rng.random() < 0.75
                else ("left" if rng.random() < 0.2 else "right")
            )
            attrs = _make_attributes(rng, slot.role, spec.quality)
            player = PlayerProfile(
                pid,
                name,
                short,
                next_shirt(default_shirts.get(slot.label, 20)),
                nat,
                rng.randint(20, 33),
                slot.role,
                slot.label,
                foot,
                attrs,
            )  # type: ignore[arg-type]
        squad.append(player)
        xi.append(pid)

    bench: list[str] = []
    for j, (label, role) in enumerate(BENCH_ROLES):
        pid = f"{spec.code.lower()}-{len(slots) + j + 1:02d}"
        nat, name, short = _pick_name(rng, spec, used_names, club_names)
        attrs = _make_attributes(rng, role, spec.quality, bench=True)
        foot = "left" if rng.random() < 0.25 else "right"
        squad.append(
            PlayerProfile(
                pid, name, short, next_shirt(12 + j * 3), nat, rng.randint(18, 34), role, label, foot, attrs
            )
        )  # type: ignore[arg-type]
        bench.append(pid)

    return ClubProfile(
        club_id=spec.club_id,
        name=spec.name,
        short_name=spec.short_name,
        code=spec.code,
        city=spec.city,
        stadium=spec.stadium,
        capacity=spec.capacity,
        founded=spec.founded,
        colors=spec.colors,
        manager=spec.manager,
        formation=spec.formation,
        tactics=spec.tactics,
        squad=tuple(squad),
        starting_xi=tuple(xi),
        bench=tuple(bench),
        identity=spec.identity,
    )


@lru_cache(maxsize=4)
def load_league(seed: int = 2026) -> dict[str, ClubProfile]:
    """Return all clubs keyed by club code (e.g. ``"RIV"``). Every player name is unique."""
    names: set[str] = {s.name for spec in CLUB_SPECS for s in spec.stars}
    return {spec.code: build_club(spec, seed, names) for spec in CLUB_SPECS}


def get_club(code_or_id: str, seed: int = 2026) -> ClubProfile:
    league = load_league(seed)
    key = code_or_id.upper()
    if key in league:
        return league[key]
    for club in league.values():
        if club.club_id == code_or_id.lower():
            return club
    raise KeyError(f"Unknown club {code_or_id!r}. Options: {', '.join(league)}")
