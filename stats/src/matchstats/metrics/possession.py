"""Possession, territory (field tilt and zone touches) and possession sequences."""

from __future__ import annotations

from collections import defaultdict

from ..pitch import in_box, zone_id
from .base import Metric, ratio, register, team_dict

ON_BALL = {"pass", "carry", "shot", "dribble", "ball_recovery", "interception", "clearance", "miscontrol"}
DOMINANCE_SHARE = 0.70
DOMINANCE_WINDOW_MIN = 10
LONG_SEQUENCE_PASSES = 25


@register
class Possession(Metric):
    name = "possession"
    order = 30

    def __init__(self, analyzer):
        super().__init__(analyzer)
        ids = (self.ctx.home_id, self.ctx.away_id)
        self.time_frames = dict.fromkeys(ids, 0.0)
        self.time_events = dict.fromkeys(ids, 0.0)
        self.per_minute = defaultdict(lambda: dict.fromkeys(ids, 0.0))
        self.zone_touches = team_dict(self.ctx, lambda: [0] * 18)
        self.final_third_passes = dict.fromkeys(ids, 0)
        self.entries = {tid: {"final_third": 0, "box": 0} for tid in ids}
        self._last_event = None
        self._last_frame_t = None
        self.sequences: list[dict] = []
        self._seq: dict | None = None
        self._dominance_signalled = dict.fromkeys(ids, -99.0)

    # -- time in possession --------------------------------------------------------------
    def _minute_key(self, period: int, t: float) -> int:
        return int(((period - 1) * 45.0) + t / 60.0)

    def on_frame(self, f: dict) -> None:
        if self._last_frame_t and self._last_frame_t[0] == f["period"]:
            dt = f["t"] - self._last_frame_t[1]
            tid = f.get("possession_team_id")
            if 0 < dt < 2 and tid and f["ball"]["state"] != "dead":
                self.time_frames[tid] += dt
                self.per_minute[self._minute_key(f["period"], f["t"])][tid] += dt
        self._last_frame_t = (f["period"], f["t"])

    def on_event(self, e: dict) -> None:
        ctx = self.ctx
        last = self._last_event
        if last and last["period"] == e["period"] and not ctx.has_frames:
            gap = e["t"] - last["t"]
            tid = last["possession"].get("team_id")
            if tid and 0 < gap < 20:
                self.time_events[tid] += gap
                self.per_minute[self._minute_key(e["period"], e["t"])][tid] += gap
        self._last_event = e

        tid = e["team_id"]
        if e["type"] in ON_BALL and tid and e["location"]:
            x, y = ctx.tf(e["location"], tid, e["period"])
            self.zone_touches[tid][zone_id(x, y)] += 1
            if e["type"] == "pass" and x >= 70:
                self.final_third_passes[tid] += 1
            if e["type"] in ("pass", "carry") and (
                e["type"] == "carry" or e["details"].get("outcome") == "complete"
            ):
                ex, ey = ctx.tf(e["details"]["end_location"], tid, e["period"])
                self.entries[tid]["final_third"] += x < 70 <= ex
                self.entries[tid]["box"] += (not in_box(x, y)) and in_box(ex, ey)
        self._track_sequence(e)
        if e["type"] in ("pass", "possession_change"):
            self._check_dominance(e)

    def _track_sequence(self, e: dict) -> None:
        poss = e.get("possession") or {}
        pid, tid = poss.get("id"), poss.get("team_id")
        if pid is None or tid is None:
            return
        if self._seq is None or self._seq["id"] != pid:
            if self._seq is not None:
                self.sequences.append(self._seq)
            self._seq = {
                "id": pid,
                "team_id": tid,
                "start": (e["period"], e["t"]),
                "end": (e["period"], e["t"]),
                "passes": 0,
                "reached_final_third": False,
                "reached_box": False,
                "shot": False,
                "event_ids": [],
                "signalled": False,
            }
        s = self._seq
        s["end"] = (e["period"], e["t"])
        if e["team_id"] == tid and e["type"] in ON_BALL:
            s["event_ids"].append(e["event_id"])
        if e["type"] == "pass" and e["team_id"] == tid and e["details"]["outcome"] == "complete":
            s["passes"] += 1
            ex, ey = self.ctx.tf(e["details"]["end_location"], tid, e["period"])
            s["reached_final_third"] |= ex >= 70
            s["reached_box"] |= in_box(ex, ey)
            if s["passes"] == LONG_SEQUENCE_PASSES and not s["signalled"]:
                s["signalled"] = True
                self.signal(
                    "long_sequence",
                    team_id=tid,
                    passes=s["passes"],
                    event_ids=list(s["event_ids"]),
                    period=e["period"],
                    t=e["t"],
                )
        if e["type"] == "shot" and e["team_id"] == tid:
            s["shot"] = True

    def _check_dominance(self, e: dict) -> None:
        minute = self._minute_key(e["period"], e["t"])
        if minute < 15:
            return
        window = [
            self.per_minute[m]
            for m in range(minute - DOMINANCE_WINDOW_MIN + 1, minute + 1)
            if m in self.per_minute
        ]
        totals = {tid: sum(w[tid] for w in window) for tid in (self.ctx.home_id, self.ctx.away_id)}
        total = sum(totals.values())
        if total < 240:  # need ~4 minutes of ball-in-play time in the window
            return
        for tid, v in totals.items():
            share = v / total
            if share >= DOMINANCE_SHARE and minute - self._dominance_signalled[tid] >= 15:
                self._dominance_signalled[tid] = minute
                self.signal(
                    "dominance",
                    team_id=tid,
                    share=round(share, 3),
                    window_min=DOMINANCE_WINDOW_MIN,
                    period=e["period"],
                    t=e["t"],
                )

    # -- views --------------------------------------------------------------------------
    def shares(self) -> dict:
        src = self.time_frames if self.ctx.has_frames else self.time_events
        total = sum(src.values())
        return {tid: ratio(v, total) for tid, v in src.items()}

    def rolling_share(self, minutes: int = 10) -> dict:
        now = int(self.ctx.match_minute_float())
        window = [self.per_minute[m] for m in range(now - minutes + 1, now + 1) if m in self.per_minute]
        totals = {tid: sum(w[tid] for w in window) for tid in (self.ctx.home_id, self.ctx.away_id)}
        total = sum(totals.values())
        return {tid: ratio(v, total) for tid, v in totals.items()}

    def snapshot(self) -> dict:
        seqs = self.sequences + ([self._seq] if self._seq else [])
        teams = {}
        tilt_total = sum(self.final_third_passes.values())
        for tid in (self.ctx.home_id, self.ctx.away_id):
            own = [s for s in seqs if s["team_id"] == tid]
            longest = max(own, key=lambda s: s["passes"], default=None)
            teams[tid] = {
                "possession": self.shares()[tid],
                "possession_last_10": self.rolling_share(10)[tid],
                "field_tilt": ratio(self.final_third_passes[tid], tilt_total),
                "final_third_entries": self.entries[tid]["final_third"],
                "box_entries": self.entries[tid]["box"],
                "possessions": len(own),
                "avg_passes_per_possession": round(sum(s["passes"] for s in own) / len(own), 2)
                if own
                else 0.0,
                "possessions_reaching_final_third": sum(s["reached_final_third"] for s in own),
                "possessions_reaching_box": sum(s["reached_box"] for s in own),
                "possessions_ending_in_shot": sum(s["shot"] for s in own),
                "longest_sequence_passes": longest["passes"] if longest else 0,
                "zone_touches": self.zone_touches[tid],
            }
        timeline = []
        for m in sorted(self.per_minute):
            v = self.per_minute[m]
            tot = sum(v.values())
            timeline.append({"minute": m, **{tid: ratio(x, tot) for tid, x in v.items()}})
        return {
            "teams": teams,
            "per_minute": timeline,
            "zone_grid": "6 columns (own goal -> opponent goal) x 3 rows; index = row * 6 + column",
        }
