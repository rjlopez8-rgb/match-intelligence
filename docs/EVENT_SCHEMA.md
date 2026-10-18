# Event and tracking schema (v1.0.0)

The data contract between the match engine (or, in production, an official data provider) and everything downstream. Machine-readable JSON Schemas are in [`schema/`](schema/); the Pydantic source of truth is `engine/src/matchengine/schema.py`, and tests validate every emitted event against it.

## Conventions

- **Coordinates:** metres on a 105 x 68 pitch. Origin is the bottom-left corner flag; `x` runs along the length, `y` along the width.
- **Direction of play:** `meta.json -> attacking_direction`. The home team attacks towards x = 105 in period 1 and towards x = 0 in period 2.
- **Timing:** `t` is seconds since the start of the period when the action *began*. `clock` is the broadcast clock (`"67:12"`, `"90:00+2:31"`). `minute_display` is the fan-facing label (`"68'"`, `"90+3'"`). `elapsed_s` is seconds since kick-off including half-time, used for real-time pacing.
- **Ordering:** events are published when their outcome is known, like a live feed, and `seq` is publication order. A pass event therefore arrives once the pass is received or lost.
- **Ground truth:** `details.truth` holds simulator internals (true pass success probability, true xG, duel probabilities). Use it to evaluate models. Never show it to fans or feed it to models as an input.

## Common envelope

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string | `"1.0.0"` |
| `event_id` | string | `<match_id>-<seq>`, unique |
| `match_id` | string | |
| `seq` | int | 1, 2, 3 … publication order |
| `type` | string | see below |
| `period` | 1 or 2 | |
| `t`, `minute`, `second`, `clock`, `minute_display`, `elapsed_s` | | timing (above) |
| `team_id`, `player_id`, `player_name` | string or null | actor |
| `location` | `{x, y}` or null | where the action started |
| `possession` | `{id, team_id, play_pattern}` | `play_pattern`: `regular_play`, `from_kick_off`, `from_throw_in`, `from_goal_kick`, `from_corner`, `from_free_kick`, `from_penalty` |
| `score` | `{home, away}` | score at publication |
| `details` | object | type-specific (below) |

## Event types

| Type | Key `details` fields |
|---|---|
| `match_start` | `home_team_id`, `away_team_id`, `kickoff_utc`, `competition`, `venue` |
| `period_start` / `period_end` | `period`, `score`, `duration_s` |
| `kick_off` | |
| `pass` | `recipient_id`, `end_location`, `length_m`, `angle_deg`, `progression_m`, `height` (ground/low/high), `pass_kind` (ground, back, through, long, switch, cross, cutback, corner, free_kick, free_kick_cross, throw_in, goal_kick, kick_off, keeper_distribution), `restart_type`, `release_speed_kmh`, `under_pressure`, `nearest_opponent_m`, `outcome` (complete, incomplete, intercepted, out, offside), `truth.p_success_intended`, `truth.p_success_executed`, `truth.execution_error_m` |
| `carry` | `end_location`, `distance_m`, `duration_s`, `progression_m`, `max_speed_kmh`, `under_pressure` |
| `dribble` | `outcome` (success/fail), `opponent_id`, `truth.p_success` |
| `pressure` | `target_player_id`, `distance_m`, `closing_speed_kmh` |
| `tackle` | `outcome` (won/lost), `opponent_id`, `ball_out`, `truth.p_win` |
| `interception` | |
| `block` | `blocked` (shot/pass), `shooter_id` or `passer_id` |
| `ball_recovery` | `contested` |
| `miscontrol` | |
| `clearance` | `end_location`, `body_part`, `height` |
| `aerial_duel` | `opponent_id`, `outcome` (won; the actor is the winner), `truth.p_attacker_win` |
| `shot` | `shot_kind` (open_play, free_kick, penalty), `body_part`, `outcome` (goal, saved, blocked, off_target, post), `on_target`, `end_location {x, y, z}`, `goal_mouth {y_offset_m, z_m}`, `distance_m`, `angle_deg`, `speed_kmh`, `under_pressure`, `defenders_in_lane`, `goalkeeper_id`, `truth.xg`, `truth.big_chance` |
| `save` | `kind` (catch/parry), `shot_speed_kmh`, `shooter_id` |
| `keeper_claim` | `kind` |
| `goal` | `scorer_id`, `assist_id`, `shot_kind`, `body_part`, `xg`, `big_chance`, `shot_event_id`, `new_score`, `minute_display` |
| `foul` | `victim_id`, `kind` (tackle, challenge, tactical), `penalty` |
| `card` | `card` (yellow, second_yellow, red), `reason` |
| `ball_out` | `line`, `last_touch_team_id`, `restart` (throw_in, goal_kick, corner), `restart_team_id` |
| `offside` | |
| `possession_change` | `from_team_id`, `how`, `previous_duration_s`, `previous_completed_passes` |
| `substitution` | `player_out_id`, `player_in_id`, `position`, `reason` (tactical, fatigue, injury), `energy_out` |
| `tactical_change` | `reason`, `description`, `manager`, `before`, `after` (pressing, line_height, tempo, risk, directness, crossing) |
| `added_time` | `minutes` |
| `match_end` | `final_score`, `winner` |

## Tracking frames

```json
{"frame": 1534, "period": 1, "t": 153.3, "clock": "02:33",
 "ball": {"x": 61.2, "y": 40.8, "z": 0.0, "speed_ms": 5.1, "state": "held", "owner_id": "riv-07"},
 "possession_team_id": "riverside-athletic",
 "players": [["riv-01", 18.4, 33.9, 0.6], ["riv-02", 47.0, 9.1, 3.2], ...]}
```

`state` is `held`, `flight`, `loose` or `dead`. Players are `[player_id, x, y, speed_m_per_s]`. The default is 10 Hz; `--frame-hz` lowers it.
