"""Box score: goals, cards, fouls, set pieces, substitutions, duels."""

from __future__ import annotations

from .base import Metric, ratio, register, team_dict


def _box():
    return {
        "goals": 0,
        "fouls": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "corners": 0,
        "offsides": 0,
        "free_kicks_won": 0,
        "substitutions": 0,
        "tackles": 0,
        "tackles_won": 0,
        "interceptions": 0,
        "clearances": 0,
        "blocks": 0,
        "aerials_won": 0,
        "dribbles": 0,
        "dribbles_completed": 0,
        "saves": 0,
        "ball_recoveries": 0,
    }


@register
class Summary(Metric):
    name = "summary"
    order = 70

    def __init__(self, analyzer):
        super().__init__(analyzer)
        self.teams = team_dict(self.ctx, _box)
        self.goals: list[dict] = []
        self.cards: list[dict] = []
        self.subs: list[dict] = []
        self.tactics: list[dict] = []

    def on_event(self, e: dict) -> None:
        t, tid, d = e["type"], e["team_id"], e["details"]
        if not tid:
            return
        b = self.teams[tid]
        if t == "goal":
            b["goals"] += 1
            self.goals.append(
                {
                    "minute_display": d.get("minute_display", e["minute_display"]),
                    "team_id": tid,
                    "scorer_id": d["scorer_id"],
                    "assist_id": d.get("assist_id"),
                    "event_id": e["event_id"],
                    "score": d.get("new_score"),
                }
            )
        elif t == "foul":
            b["fouls"] += 1
            self.teams[self.ctx.opponent(tid)]["free_kicks_won"] += 1
        elif t == "card":
            card = d.get("card")
            if card == "yellow":
                b["yellow_cards"] += 1
            else:
                b["red_cards"] += 1
                if card == "second_yellow":
                    b["yellow_cards"] += 1
            self.cards.append(
                {
                    "minute_display": e["minute_display"],
                    "team_id": tid,
                    "player_id": e["player_id"],
                    "card": card,
                    "event_id": e["event_id"],
                }
            )
        elif t == "pass" and d.get("restart_type") == "corner":
            b["corners"] += 1
        elif t == "offside":
            b["offsides"] += 1
        elif t == "substitution":
            b["substitutions"] += 1
            self.subs.append(
                {
                    "minute_display": e["minute_display"],
                    "team_id": tid,
                    "out": d["player_out_id"],
                    "in": d["player_in_id"],
                    "reason": d.get("reason"),
                }
            )
        elif t == "tackle":
            b["tackles"] += 1
            b["tackles_won"] += d.get("outcome") == "won"
        elif t == "interception":
            b["interceptions"] += 1
        elif t == "clearance":
            b["clearances"] += 1
        elif t == "block":
            b["blocks"] += 1
        elif t == "aerial_duel":
            b["aerials_won"] += 1
        elif t == "dribble":
            b["dribbles"] += 1
            b["dribbles_completed"] += d.get("outcome") == "success"
        elif t == "save":
            b["saves"] += 1
        elif t == "ball_recovery":
            b["ball_recoveries"] += 1
        elif t == "tactical_change":
            self.tactics.append(
                {
                    "minute_display": e["minute_display"],
                    "team_id": tid,
                    "reason": d.get("reason"),
                    "description": d.get("description"),
                }
            )

    def snapshot(self) -> dict:
        teams = {}
        for tid, b in self.teams.items():
            teams[tid] = {
                **b,
                "tackle_success": ratio(b["tackles_won"], b["tackles"]),
                "dribble_success": ratio(b["dribbles_completed"], b["dribbles"]),
            }
        return {
            "teams": teams,
            "goals": self.goals,
            "cards": self.cards,
            "substitutions": self.subs,
            "tactical_changes": self.tactics,
        }
