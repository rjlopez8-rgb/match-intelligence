"""Agent-facing queries over a running :class:`MatchAnalyzer`.

Each public method of :class:`MatchQueries` is designed to become one agent tool
(MCP or function calling): small JSON answers, names resolved, numbers rounded,
and ``evidence`` event ids attached wherever a claim can be checked. Agents never
compute statistics themselves; they ask these questions and narrate the answers.

``TOOLS`` lists every query with a description and JSON-schema parameters so a
tool server can register them without restating anything.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .analyzer import MatchAnalyzer

TEAM_STAT_GROUPS: dict[str, list[tuple[str, str, str]]] = {
    # group: [(label, metric, key)]
    "attack": [
        ("Goals", "summary", "goals"),
        ("Expected goals (xG)", "shooting", "xg"),
        ("Shots", "shooting", "shots"),
        ("Shots on target", "shooting", "on_target"),
        ("Big chances", "shooting", "big_chances"),
        ("xG per shot", "shooting", "xg_per_shot"),
        ("Box entries", "possession", "box_entries"),
    ],
    "possession": [
        ("Possession", "possession", "possession"),
        ("Possession, last 10 min", "possession", "possession_last_10"),
        ("Field tilt", "possession", "field_tilt"),
        ("Final-third entries", "possession", "final_third_entries"),
        ("Avg passes per possession", "possession", "avg_passes_per_possession"),
        ("Longest passing sequence", "possession", "longest_sequence_passes"),
    ],
    "passing": [
        ("Passes", "passing", "passes"),
        ("Pass accuracy", "passing", "accuracy"),
        ("Progressive passes", "passing", "progressive"),
        ("Opponents bypassed by passes", "passing", "lines_broken"),
        ("Key passes", "passing", "key_passes"),
        ("Accuracy under pressure", "passing", "under_pressure_accuracy"),
        ("Avg pass difficulty (0-10)", "passing", "avg_difficulty"),
    ],
    "defending": [
        ("Pressures", "pressing", "pressures"),
        ("PPDA (lower = more intense press)", "pressing", "ppda"),
        ("High turnovers", "pressing", "high_turnovers"),
        ("Tackles won", "summary", "tackles_won"),
        ("Interceptions", "summary", "interceptions"),
        ("Blocks", "summary", "blocks"),
        ("Aerials won", "summary", "aerials_won"),
        ("Saves", "summary", "saves"),
    ],
    "physical": [
        ("Distance (km)", "physical", "distance_km"),
        ("High-speed running (km)", "physical", "high_speed_km"),
        ("Sprints", "physical", "sprints"),
    ],
    "discipline": [
        ("Fouls", "summary", "fouls"),
        ("Yellow cards", "summary", "yellow_cards"),
        ("Red cards", "summary", "red_cards"),
        ("Offsides", "summary", "offsides"),
        ("Corners", "summary", "corners"),
    ],
}

PLAYER_STATS: dict[str, tuple[str, str, str]] = {
    # stat: (label, metric, key) — metric's ``players`` dict
    "goals": ("Goals", "shooting", "goals"),
    "xg": ("Expected goals", "shooting", "xg"),
    "shots": ("Shots", "shooting", "shots"),
    "shots_on_target": ("Shots on target", "shooting", "on_target"),
    "passes": ("Passes", "passing", "passes"),
    "pass_accuracy": ("Pass accuracy", "passing", "accuracy"),
    "progressive_passes": ("Progressive passes", "passing", "progressive"),
    "lines_broken": ("Opponents bypassed by passes", "passing", "lines_broken"),
    "key_passes": ("Key passes", "passing", "key_passes"),
    "assists": ("Assists", "passing", "assists"),
    "pass_difficulty": ("Avg pass difficulty", "passing", "avg_difficulty"),
    "distance_km": ("Distance covered (km)", "physical", "distance_km"),
    "top_speed_kmh": ("Top speed (km/h)", "physical", "top_speed_kmh"),
    "sprints": ("Sprints", "physical", "sprints"),
    "high_speed_m": ("High-speed running (m)", "physical", "high_speed_m"),
}

SERIES = {
    "xg": ("shooting", "xg_timeline"),
    "possession": ("possession", "per_minute"),
    "momentum": ("momentum", "series"),
    "pressing": ("pressing", "pressure_pulse"),
    "control": ("control", "series"),
    "win_probability": ("win_probability", "series"),
}

EVENT_DETAIL_KEYS = (
    "outcome",
    "pass_kind",
    "height",
    "length_m",
    "progression_m",
    "under_pressure",
    "shot_kind",
    "body_part",
    "distance_m",
    "speed_kmh",
    "on_target",
    "card",
    "reason",
    "recipient_name",
    "max_speed_kmh",
    "scorer_name",
    "assist_name",
)
ANNOTATION_KEYS = (
    "xg",
    "difficulty",
    "p_complete",
    "lines_broken",
    "progressive",
    "key_pass_event_id",
    "win_prob_before",
    "win_prob_after",
    "moment_weight",
    "expected_swing_pts",
    "potential_swing_pts",
)


class QueryError(ValueError):
    """A question the data cannot answer (unknown player, moment, team...)."""


class MatchQueries:
    def __init__(self, analyzer: MatchAnalyzer):
        if analyzer.ctx is None:
            raise QueryError("the analyzer has not received a match yet")
        self.a = analyzer
        self.ctx = analyzer.ctx

    # -- resolution helpers -------------------------------------------------------------
    def _snap(self, metric: str) -> dict:
        return self.a.metric(metric).snapshot()

    def _as_of(self) -> dict:
        c = self.ctx
        return {
            "minute": c.minute_display,
            "period": c.period,
            "score": dict(c.score),
            "finished": c.finished,
        }

    def team(self, ref: str | None) -> str | None:
        """Resolve a team id, code, side ("home"/"away") or (partial) name."""
        if ref is None:
            return None
        r = str(ref).strip().lower()
        c = self.ctx
        if r in ("home", c.home_id):
            return c.home_id
        if r in ("away", c.away_id):
            return c.away_id
        for tid, t in c.teams.items():
            if r in (t["code"].lower(), t["short_name"].lower(), t["name"].lower()) or r in t["name"].lower():
                return tid
        raise QueryError(
            f"unknown team '{ref}'; use '{c.teams[c.home_id]['name']}' or '{c.teams[c.away_id]['name']}'"
        )

    def player(self, ref: str) -> str:
        """Resolve a player id, shirt name or (fuzzy) full name."""
        c = self.ctx
        r = str(ref).strip().lower()
        if r in c.players:
            return r
        names = {pid: p.name.lower() for pid, p in c.players.items()}
        exact = [
            pid for pid, n in names.items() if r in (n, n.split()[-1], c.players[pid].short_name.lower())
        ]
        if len(exact) == 1:
            return exact[0]
        partial = [pid for pid, n in names.items() if r in n]
        if len(partial) == 1:
            return partial[0]
        aliases: dict[str, set[str]] = {}
        for pid, n in names.items():
            for alias in (n, n.split()[-1], c.players[pid].short_name.lower()):
                aliases.setdefault(alias, set()).add(pid)
        for alias in difflib.get_close_matches(r, list(aliases), n=3, cutoff=0.75):
            if len(aliases[alias]) == 1:
                return next(iter(aliases[alias]))
        hint = ", ".join(c.players[p].name for p in (partial or exact)[:5])
        raise QueryError(
            f"unknown or ambiguous player '{ref}'" + (f" (did you mean: {hint})" if hint else "")
        )

    def _sided(self, d):
        """Replace team ids (as keys or values) with "home"/"away", recursively."""
        names = {self.ctx.home_id: "home", self.ctx.away_id: "away"}
        if isinstance(d, dict):
            return {names.get(k, k): self._sided(v) for k, v in d.items()}
        if isinstance(d, list):
            return [self._sided(v) for v in d]
        return names.get(d, d) if isinstance(d, str) else d

    def _player_ref(self, pid: str) -> dict:
        p = self.ctx.players[pid]
        return {
            "player_id": pid,
            "name": p.name,
            "team": self.ctx.team_name(p.team_id),
            "position": p.position,
            "shirt": p.shirt,
        }

    # -- queries --------------------------------------------------------------------------
    def scoreboard(self) -> dict:
        """Score, clock and the headline numbers for both teams."""
        c = self.ctx
        shooting, poss = self._snap("shooting")["teams"], self._snap("possession")["teams"]
        teams = {}
        for tid in (c.home_id, c.away_id):
            teams[c.side(tid)] = {
                "team": c.team_name(tid),
                "goals": c.score[c.side(tid)],
                "xg": round(shooting[tid]["xg"], 2),
                "shots": shooting[tid]["shots"],
                "on_target": shooting[tid]["on_target"],
                "possession": poss[tid]["possession"],
                "red_cards": c.red_cards.get(tid, 0),
            }
        return {
            "as_of": self._as_of(),
            **teams,
            "win_probability": self._snap("win_probability")["current"],
            "momentum": self.a.metric("momentum").index(),
            "goals": [self._goal_line(g) for g in self._snap("summary")["goals"]],
        }

    def _goal_line(self, g: dict) -> dict:
        return {
            "minute": g["minute_display"],
            "team": self.ctx.team_name(g["team_id"]),
            "scorer": self.ctx.player_name(g["scorer_id"]),
            "assist": self.ctx.player_name(g["assist_id"]) if g.get("assist_id") else None,
            "score": g["score"],
            "evidence": [g["event_id"]],
        }

    def team_comparison(self, groups: list[str] | None = None) -> dict:
        """Side-by-side team stats, grouped (attack, possession, passing, defending, physical, discipline)."""
        c = self.ctx
        groups = groups or list(TEAM_STAT_GROUPS)
        unknown = [g for g in groups if g not in TEAM_STAT_GROUPS]
        if unknown:
            raise QueryError(f"unknown group(s) {unknown}; choose from {list(TEAM_STAT_GROUPS)}")
        snaps = {m: self._snap(m) for m in {m for g in groups for _, m, _ in TEAM_STAT_GROUPS[g]}}
        out: dict[str, Any] = {
            "as_of": self._as_of(),
            "home": c.team_name(c.home_id),
            "away": c.team_name(c.away_id),
        }
        for g in groups:
            rows = []
            for label, metric, key in TEAM_STAT_GROUPS[g]:
                teams = snaps[metric].get("teams")
                if not teams:  # e.g. physical stats without tracking frames
                    continue
                h, a = teams[c.home_id].get(key), teams[c.away_id].get(key)
                rows.append({"stat": label, "home": _round(h), "away": _round(a)})
            out[g] = rows
        return out

    def player_stats(self, player: str) -> dict:
        """Everything the stats layer knows about one player, plus their moments."""
        pid = self.player(player)
        stats: dict[str, Any] = {}
        for stat, (label, metric, key) in PLAYER_STATS.items():
            players = self._snap(metric).get("players", {})
            if pid in players and key in players[pid]:
                stats[stat] = {"label": label, "value": _round(players[pid][key])}
        gk = self._snap("shooting")["goalkeepers"].get(pid)
        moments = [self._moment_line(m) for m in self.a.moments if pid in m["player_ids"]]
        return {
            "as_of": self._as_of(),
            "player": self._player_ref(pid),
            "on_pitch": pid in self.ctx.on_pitch,
            "stats": stats,
            "goalkeeping": gk,
            "moments": moments[-8:],
        }

    def top_players(self, stat: str, team: str | None = None, limit: int = 5) -> dict:
        """Leaderboard for one player stat, optionally for one team."""
        if stat not in PLAYER_STATS:
            raise QueryError(f"unknown stat '{stat}'; choose from {list(PLAYER_STATS)}")
        label, metric, key = PLAYER_STATS[stat]
        tid = self.team(team)
        players = self._snap(metric).get("players", {})
        rows = [
            (pid, v[key])
            for pid, v in players.items()
            if key in v and (tid is None or self.ctx.team_of(pid) == tid)
        ]
        if stat == "pass_accuracy":  # avoid 1/1 = 100% leaders
            rows = [(p, v) for p, v in rows if players[p]["passes"] >= 15]
        rows.sort(key=lambda r: -r[1])
        return {
            "as_of": self._as_of(),
            "stat": label,
            "team": self.ctx.team_name(tid) if tid else "both",
            "leaders": [
                {**self._player_ref(p), "value": _round(v)} for p, v in rows[: max(1, min(limit, 20))]
            ],
        }

    def momentum(self, window_min: int = 10) -> dict:
        """Who is on top right now, how that has changed, and why (threat, possession, pressing)."""
        c = self.ctx
        mom = self._snap("momentum")
        series = mom["series"]
        now = c.match_minute_float()
        recent = [p for p in series if p["minute"] >= now - window_min]
        start = recent[0]["momentum"] if recent else 0.0
        idx = mom["momentum"]
        leader = c.home_id if idx > 0.1 else c.away_id if idx < -0.1 else None
        poss, press = self._snap("possession")["teams"], self._snap("pressing")["teams"]
        shifts = [
            self._moment_line(m)
            for m in self.a.moments
            if m["type"] in ("momentum_shift", "dominance", "pressing_surge")
        ][-4:]
        return {
            "as_of": self._as_of(),
            "index": idx,
            "scale": "+1 = home fully on top, -1 = away fully on top",
            "on_top": c.team_name(leader) if leader else "neither team",
            "trend": _trend(idx - start),
            "change_over_window": round(idx - start, 3),
            "window_min": window_min,
            "drivers": {
                c.side(t): {
                    "team": c.team_name(t),
                    "threat_level": mom["threat_level"][t],
                    "possession_last_10": poss[t]["possession_last_10"],
                    "pressures": press[t]["pressures"],
                }
                for t in (c.home_id, c.away_id)
            },
            "control": self._sided({k: v for k, v in self._snap("control").items() if k != "series"}),
            "recent_shifts": shifts,
        }

    def moments(
        self,
        types: list[str] | None = None,
        team: str | None = None,
        player: str | None = None,
        since_minute: float | None = None,
        min_importance: float = 0.0,
        limit: int = 10,
        order: str = "recent",
    ) -> dict:
        """Filtered list of key moments. ``order`` is "recent", "chronological" or "importance"."""
        tid = self.team(team)
        pid = self.player(player) if player else None
        rows = []
        for m in self.a.moments:
            minute = (m["period"] - 1) * 45 + m["t"] / 60
            if types and m["type"] not in types:
                continue
            if tid and m["team_id"] != tid:
                continue
            if pid and pid not in m["player_ids"]:
                continue
            if since_minute is not None and minute < since_minute:
                continue
            if m["importance"] < min_importance:
                continue
            rows.append(m)
        if order == "importance":
            rows.sort(key=lambda m: -m["importance"])
        elif order == "recent":
            rows = rows[::-1]
        return {
            "as_of": self._as_of(),
            "total_matching": len(rows),
            "moments": [self._moment_line(m) for m in rows[: max(1, min(limit, 50))]],
        }

    def _moment_line(self, m: dict) -> dict:
        return {
            "moment_id": m["moment_id"],
            "type": m["type"],
            "minute": m["minute_display"],
            "title": m["title"],
            "team": m["team_name"],
            "players": m["player_names"],
            "importance": m["importance"],
            "moment_weight": m["moment_weight"],
        }

    def explain_moment(self, moment_id: str) -> dict:
        """One moment with all its facts and the evidence events behind it (the receipts)."""
        m = next((x for x in self.a.moments if x["moment_id"] == moment_id), None)
        if m is None:
            raise QueryError(f"unknown moment '{moment_id}'")
        return {
            "as_of": self._as_of(),
            **{k: v for k, v in m.items() if k != "evidence"},
            "evidence": [self.event(eid) for eid in m["evidence"]],
            "moment_weight_meaning": "percentage points of win probability this moment moved (realised for "
            "goals and red cards; expected for missed chances)",
        }

    def event(self, event_id: str) -> dict:
        """One feed event in readable form, with the stats layer's annotations."""
        e = self.ctx.by_id.get(event_id)
        if e is None:
            raise QueryError(f"unknown event '{event_id}'")
        d = e.get("details") or {}
        return {
            "event_id": event_id,
            "minute": e["minute_display"],
            "type": e["type"],
            "team": self.ctx.team_name(e["team_id"]) if e["team_id"] else None,
            "player": self.ctx.player_name(e["player_id"]) if e["player_id"] else None,
            "location": e["location"],
            "details": {k: d[k] for k in EVENT_DETAIL_KEYS if d.get(k) is not None},
            "analysis": {k: v for k, v in self.ctx.ann.get(event_id, {}).items() if k in ANNOTATION_KEYS},
        }

    def win_probability(self) -> dict:
        """Current result probabilities, the trend, and the moments that moved them most."""
        wp = self._snap("win_probability")
        swings = sorted((m for m in self.a.moments if m["moment_weight"]), key=lambda m: -m["moment_weight"])[
            :5
        ]
        return {
            "as_of": self._as_of(),
            "current": wp["current"],
            "model": wp["model"],
            "teams": {
                "home": self.ctx.team_name(self.ctx.home_id),
                "away": self.ctx.team_name(self.ctx.away_id),
            },
            "every_5_min": [
                {k: p[k] for k in ("minute", "home", "draw", "away")}
                for p in wp["series"]
                if p["minute"] % 5 == 0
            ],
            "biggest_swings": [self._moment_line(m) for m in swings],
        }

    def pass_network(self, team: str) -> dict:
        """Average positions and most frequent passing links for one team."""
        tid = self.team(team)
        net = self._snap("passing")["networks"][tid]
        nodes = [
            {**self._player_ref(n["player_id"]), "x": n["x"], "y": n["y"], "passes": n["passes"]}
            for n in net["nodes"]
        ]
        edges = [
            {
                "from": self.ctx.player_name(e["from"]),
                "to": self.ctx.player_name(e["to"]),
                "count": e["count"],
            }
            for e in net["edges"]
        ]
        return {
            "as_of": self._as_of(),
            "team": self.ctx.team_name(tid),
            "orientation": net["orientation"],
            "nodes": nodes,
            "edges": edges[:25],
        }

    def timeline(self, series: str, since_minute: float | None = None) -> dict:
        """A per-minute series: xg, possession, momentum, pressing, control or win_probability."""
        if series not in SERIES:
            raise QueryError(f"unknown series '{series}'; choose from {list(SERIES)}")
        metric, key = SERIES[series]
        points = self._snap(metric)[key]
        if since_minute is not None:
            points = [p for p in points if p["minute"] >= since_minute]
        points = [self._sided({k: v for k, v in p.items() if v is not None}) for p in points]
        return {"as_of": self._as_of(), "series": series, "points": points}

    def call(self, tool: str, args: dict | None = None) -> dict:
        """Dispatch by tool name (what an MCP server or agent runtime calls)."""
        if tool not in TOOLS:
            raise QueryError(f"unknown tool '{tool}'")
        return getattr(self, tool)(**(args or {}))


def _round(v):
    return round(v, 3) if isinstance(v, float) else v


def _trend(delta: float) -> str:
    if delta > 0.15:
        return "swinging towards home"
    if delta < -0.15:
        return "swinging towards away"
    return "steady"


def _obj(props: dict, required: tuple = ()) -> dict:
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


_TEAM = {"type": "string", "description": "team name, code, or 'home'/'away'"}
TOOLS: dict[str, dict] = {
    "scoreboard": {
        "description": "Score, clock, headline numbers, win probability and goals.",
        "parameters": _obj({}),
    },
    "team_comparison": {
        "description": "Side-by-side team stats in groups.",
        "parameters": _obj({"groups": {"type": "array", "items": {"enum": list(TEAM_STAT_GROUPS)}}}),
    },
    "player_stats": {
        "description": "All stats and moments for one player.",
        "parameters": _obj({"player": {"type": "string", "description": "player name or id"}}, ("player",)),
    },
    "top_players": {
        "description": "Leaderboard for a player stat.",
        "parameters": _obj(
            {
                "stat": {"enum": list(PLAYER_STATS)},
                "team": _TEAM,
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            ("stat",),
        ),
    },
    "momentum": {
        "description": "Who is on top now, the trend, and what is driving it.",
        "parameters": _obj({"window_min": {"type": "integer", "minimum": 3, "maximum": 45}}),
    },
    "moments": {
        "description": "Filtered key moments (goals, chances, saves, swings, records...).",
        "parameters": _obj(
            {
                "types": {"type": "array", "items": {"type": "string"}},
                "team": _TEAM,
                "player": {"type": "string"},
                "since_minute": {"type": "number"},
                "min_importance": {"type": "number", "minimum": 0, "maximum": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "order": {"enum": ["recent", "chronological", "importance"]},
            }
        ),
    },
    "explain_moment": {
        "description": "Facts and evidence events behind one moment.",
        "parameters": _obj({"moment_id": {"type": "string"}}, ("moment_id",)),
    },
    "event": {
        "description": "One feed event with the stats layer's analysis.",
        "parameters": _obj({"event_id": {"type": "string"}}, ("event_id",)),
    },
    "win_probability": {
        "description": "Result probabilities, trend and biggest swings.",
        "parameters": _obj({}),
    },
    "pass_network": {
        "description": "Average positions and passing links for a team.",
        "parameters": _obj({"team": _TEAM}, ("team",)),
    },
    "timeline": {
        "description": "Per-minute series for charts.",
        "parameters": _obj(
            {"series": {"enum": list(SERIES)}, "since_minute": {"type": "number"}}, ("series",)
        ),
    },
}
