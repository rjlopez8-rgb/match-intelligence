"""Key-moment detection.

A *moment* is something worth talking about or showing on screen. Every moment
carries:

* ``facts``: the numbers behind it, all computed by the stats layer;
* ``evidence``: the event ids that prove it (the "receipts");
* ``moment_weight``: percentage points of win probability it moved, where that applies;
* ``importance`` (0-1): how prominently a broadcast or agent should treat it.

Titles are factual templates, not prose. Writing commentary is the agents' job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .analyzer import MatchAnalyzer

BASE_IMPORTANCE = {
    "goal": 0.9,
    "penalty_awarded": 0.75,
    "red_card": 0.8,
    "big_chance": 0.55,
    "great_save": 0.5,
    "momentum_shift": 0.5,
    "dominance": 0.4,
    "pressing_surge": 0.35,
    "long_sequence": 0.35,
    "hard_pass": 0.3,
    "tactical_change": 0.45,
    "speed_alert": 0.25,
    "top_speed": 0.35,
    "distance_milestone": 0.25,
    "high_turnover": 0.2,
    "chaos_spike": 0.3,
    "yellow_card": 0.2,
    "substitution": 0.15,
    "period_summary": 0.7,
}

STOPPAGES = {"foul", "ball_out", "offside", "goal", "period_start", "substitution", "kick_off"}
POSSESSION_EVIDENCE = {
    "pass",
    "carry",
    "dribble",
    "shot",
    "ball_recovery",
    "interception",
    "tackle",
    "aerial_duel",
}


class MomentDetector:
    def __init__(self, analyzer: MatchAnalyzer):
        self.a = analyzer
        self.ctx = analyzer.ctx
        self.moments: list[dict] = []

    # ---------------------------------------------------------------------------------
    def _add(
        self,
        kind: str,
        e_or_where: dict,
        title: str,
        team_id: str | None = None,
        player_ids=(),
        facts: dict | None = None,
        evidence=(),
        weight: float | None = None,
        bonus: float = 0.0,
    ) -> dict:
        ctx = self.ctx
        period = e_or_where.get("period", ctx.period)
        t = e_or_where.get("t", ctx.t)
        clock = e_or_where.get("clock") or ctx.clock
        minute_display = e_or_where.get("minute_display") or ctx.minute_display
        importance = BASE_IMPORTANCE.get(kind, 0.2) + bonus
        if weight is not None:
            importance += min(0.35, weight / 100.0)
        m = {
            "moment_id": f"{ctx.match_id}-m{len(self.moments) + 1:03d}",
            "type": kind,
            "period": period,
            "t": round(t, 2),
            "clock": clock,
            "minute_display": minute_display,
            "team_id": team_id,
            "team_name": ctx.team_name(team_id) if team_id else None,
            "player_ids": [p for p in player_ids if p],
            "player_names": [ctx.player_name(p) for p in player_ids if p],
            "title": title,
            "facts": facts or {},
            "evidence": [x for x in evidence if x],
            "moment_weight": None if weight is None else round(weight, 1),
            "importance": round(min(1.0, importance), 3),
            "score": dict(ctx.score),
            "win_probability": self.a.metric("win_probability").snapshot()["current"],
            "momentum": self.a.metric("momentum").index(),
        }
        self.moments.append(m)
        self.a.new_moments.append(m)
        return m

    def _possession_trail(self, upto: dict, n: int = 6) -> list[str]:
        """On-ball events of the same possession leading up to ``upto``, starting no earlier than the
        last stoppage. A penalty's receipts are the foul and the kick, not the open play before it."""
        pid = (upto.get("possession") or {}).get("id")
        trail = []
        started = False
        for e in reversed(self.ctx.events):
            if not started:
                started = e["event_id"] == upto["event_id"]
                continue
            if e["type"] in STOPPAGES:
                if e["type"] == "foul":
                    trail.append(e["event_id"])
                break
            if (e.get("possession") or {}).get("id") != pid:
                if trail:
                    break
                continue
            if e["type"] in POSSESSION_EVIDENCE:
                trail.append(e["event_id"])
            if len(trail) >= n:
                break
        return list(reversed(trail))

    # ---------------------------------------------------------------------------------
    def on_event(self, e: dict) -> None:
        ctx = self.ctx
        t, tid, d = e["type"], e["team_id"], e["details"]
        ann = ctx.ann.get(e["event_id"], {})
        name = ctx.player_name(e["player_id"])
        team = ctx.team_name(tid)
        if t == "goal":
            shot = ctx.by_id.get(d.get("shot_event_id") or "")
            shot_ann = ctx.ann.get(shot["event_id"], {}) if shot else {}
            facts = {
                "scorer": d["scorer_name"],
                "assist": d.get("assist_name"),
                "xg": shot_ann.get("xg"),
                "shot_kind": d.get("shot_kind"),
                "body_part": d.get("body_part"),
                "distance_m": shot["details"]["distance_m"] if shot else None,
                "shot_speed_kmh": shot["details"]["speed_kmh"] if shot else None,
                "new_score": d.get("new_score"),
                "win_prob_before": ann.get("win_prob_before"),
                "win_prob_after": ann.get("win_prob_after"),
            }
            key_pass = shot_ann.get("key_pass_event_id")
            if key_pass:
                kp = ctx.ann.get(key_pass, {})
                facts["assist_pass_difficulty"] = kp.get("difficulty")
                facts["assist_pass_lines_broken"] = kp.get("lines_broken")
            score = d.get("new_score", ctx.score)
            title = f"GOAL: {name} ({team}) {d.get('minute_display', e['minute_display'])}, {score['home']}-{score['away']}"
            evidence = (
                self._possession_trail(shot) + [shot["event_id"], e["event_id"]] if shot else [e["event_id"]]
            )
            self._add(
                "goal",
                e,
                title,
                tid,
                [e["player_id"], d.get("assist_id")],
                facts,
                evidence,
                ann.get("moment_weight"),
            )
        elif t == "shot" and d.get("outcome") != "goal":
            xg = ann.get("xg", 0.0)
            if xg >= 0.28 and d.get("outcome") == "saved":
                gk = d.get("goalkeeper_id")
                facts = {
                    "xg": xg,
                    "shooter": name,
                    "goalkeeper": ctx.player_name(gk),
                    "shot_speed_kmh": d.get("speed_kmh"),
                    "distance_m": d.get("distance_m"),
                    "expected_swing_pts": ann.get("expected_swing_pts"),
                }
                self._add(
                    "great_save",
                    e,
                    f"Big save: {ctx.player_name(gk)} denies {name} (xG {xg:.2f})",
                    ctx.opponent(tid),
                    [gk, e["player_id"]],
                    facts,
                    self._possession_trail(e) + [e["event_id"]],
                    ann.get("moment_weight"),
                )
            elif d.get("shot_kind") == "penalty":
                self._add(
                    "big_chance",
                    e,
                    f"Penalty missed: {name} ({team})",
                    tid,
                    [e["player_id"]],
                    {
                        "xg": xg,
                        "outcome": d.get("outcome"),
                        "penalty": True,
                        "potential_swing_pts": ann.get("potential_swing_pts"),
                    },
                    self._possession_trail(e) + [e["event_id"]],
                    ann.get("moment_weight"),
                    bonus=0.1,
                )
            elif xg >= 0.28:
                facts = {
                    "xg": xg,
                    "outcome": d.get("outcome"),
                    "distance_m": d.get("distance_m"),
                    "body_part": d.get("body_part"),
                    "potential_swing_pts": ann.get("potential_swing_pts"),
                    "expected_swing_pts": ann.get("expected_swing_pts"),
                }
                self._add(
                    "big_chance",
                    e,
                    f"Big chance missed: {name} ({team}), xG {xg:.2f}",
                    tid,
                    [e["player_id"]],
                    facts,
                    self._possession_trail(e) + [e["event_id"]],
                    ann.get("moment_weight"),
                )
        elif t == "foul" and d.get("penalty"):
            self._add(
                "penalty_awarded",
                e,
                f"Penalty to {ctx.team_name(ctx.opponent(tid))}: {d.get('victim_name')} fouled",
                ctx.opponent(tid),
                [d.get("victim_id"), e["player_id"]],
                {"fouled": d.get("victim_name"), "fouled_by": name},
                [e["event_id"]],
            )
        elif t == "card":
            card = d.get("card")
            if card in ("red", "second_yellow"):
                self._add(
                    "red_card",
                    e,
                    f"Red card: {name} ({team})",
                    tid,
                    [e["player_id"]],
                    {
                        "card": card,
                        "reason": d.get("reason"),
                        "win_prob_before": ann.get("win_prob_before"),
                        "win_prob_after": ann.get("win_prob_after"),
                    },
                    [e["event_id"]],
                    ann.get("moment_weight"),
                )
            else:
                self._add(
                    "yellow_card",
                    e,
                    f"Yellow card: {name} ({team})",
                    tid,
                    [e["player_id"]],
                    {"reason": d.get("reason")},
                    [e["event_id"]],
                )
        elif t == "tactical_change":
            self._add(
                "tactical_change",
                e,
                f"Tactical change, {team}: {d.get('reason')}",
                tid,
                [],
                {
                    "description": d.get("description"),
                    "manager": d.get("manager"),
                    "before": d.get("before"),
                    "after": d.get("after"),
                },
                [e["event_id"]],
            )
        elif t == "substitution":
            self._add(
                "substitution",
                e,
                f"Substitution, {team}: {d['player_in_name']} on for {d['player_out_name']}",
                tid,
                [d["player_in_id"], d["player_out_id"]],
                {"reason": d.get("reason"), "position": d.get("position")},
                [e["event_id"]],
            )
        elif t == "period_end":
            self._period_summary(e)

    def on_signal(self, s: dict) -> None:
        ctx = self.ctx
        kind = s["kind"]
        where = {"period": s.get("period", ctx.period), "t": s.get("t", ctx.t)}
        tid = s.get("team_id")
        team = ctx.team_name(tid)
        if kind == "speed_alert":
            pid = s["player_id"]
            record = s.get("record")
            self._add(
                "top_speed" if record else "speed_alert",
                where,
                f"{'New top speed' if record else 'Speed'}: {ctx.player_name(pid)} {s['speed_kmh']} km/h",
                ctx.team_of(pid),
                [pid],
                {"speed_kmh": s["speed_kmh"], "match_record": bool(record)},
            )
        elif kind == "distance_milestone":
            pid = s["player_id"]
            self._add(
                "distance_milestone",
                where,
                f"{ctx.player_name(pid)} passes {s['km']} km covered",
                ctx.team_of(pid),
                [pid],
                {"distance_km": s["distance_km"], "rank": 1},
            )
        elif kind == "hard_pass":
            pid = s["player_id"]
            lines = s.get("lines_broken", 0)
            self._add(
                "hard_pass",
                where,
                f"Line-breaking pass: {ctx.player_name(pid)} (difficulty {s['difficulty']:.1f}"
                + (f", {lines} opponents bypassed)" if lines else ")"),
                tid,
                [pid],
                {"difficulty": s["difficulty"], "lines_broken": lines},
                [s["event_id"]],
            )
        elif kind == "long_sequence":
            self._add(
                "long_sequence",
                where,
                f"{team} string together {s['passes']} passes",
                tid,
                [],
                {"passes": s["passes"]},
                s.get("event_ids", [])[-10:],
            )
        elif kind == "dominance":
            self._add(
                "dominance",
                where,
                f"{team} dominate: {round(100 * s['share'])}% possession in the last "
                f"{s['window_min']} minutes",
                tid,
                [],
                {"possession_share": s["share"], "window_min": s["window_min"]},
            )
        elif kind == "pressing_surge":
            self._add(
                "pressing_surge",
                where,
                f"{team} turn up the press: {s['pressures']} pressures in {s['window_min']} minutes",
                tid,
                [],
                {"pressures": s["pressures"], "usual": s["usual"], "window_min": s["window_min"]},
            )
        elif kind == "momentum_shift":
            self._add(
                "momentum_shift",
                where,
                f"Momentum swings to {team}",
                tid,
                [],
                {"momentum_before": s["momentum_before"], "momentum_after": s["momentum_after"]},
            )
        elif kind == "chaos_spike":
            self._add(
                "chaos_spike",
                where,
                "The game breaks into chaos",
                None,
                [],
                {"chaos_index": s["chaos_index"]},
            )
        # high_turnover signals feed stats only; they are too frequent to be moments on their own.

    def _period_summary(self, e: dict) -> None:
        a = self.a
        ctx = self.ctx
        poss = a.metric("possession").shares()
        shooting = a.metric("shooting").teams
        passing = a.metric("passing").teams
        facts = {}
        for tid in (ctx.home_id, ctx.away_id):
            side = ctx.side(tid)
            facts[side] = {
                "team": ctx.team_name(tid),
                "possession": poss[tid],
                "shots": shooting[tid]["shots"],
                "on_target": shooting[tid]["on_target"],
                "xg": round(shooting[tid]["xg"], 2),
                "pass_accuracy": round(passing[tid]["completed"] / passing[tid]["passes"], 3)
                if passing[tid]["passes"]
                else 0.0,
            }
        label = "Half-time" if e["period"] == 1 else "Full-time"
        s = ctx.score
        self._add(
            "period_summary",
            e,
            f"{label}: {ctx.team_name(ctx.home_id)} {s['home']}-{s['away']} {ctx.team_name(ctx.away_id)}",
            None,
            [],
            facts,
            [e["event_id"]],
        )
