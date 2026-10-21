"""Physical constants and tunable engine parameters.

Every number that shapes match realism lives here, so calibration can adjust the
engine without touching simulation logic. Values were tuned against typical
top-flight averages (see ``matchengine.calibration``).
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Pitch geometry (metres). Origin is the bottom-left corner flag. ---
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_WIDTH = PITCH_WIDTH / 2
GOAL_HALF_WIDTH = 3.66
GOAL_HEIGHT = 2.44
PENALTY_BOX_DEPTH = 16.5
PENALTY_BOX_HALF_WIDTH = 20.16
SIX_YARD_DEPTH = 5.5
SIX_YARD_HALF_WIDTH = 9.16
PENALTY_SPOT_DIST = 11.0
CENTER = (PITCH_LENGTH / 2, PITCH_WIDTH / 2)

# --- Simulation timing ---
TICK_HZ = 10
DT = 1.0 / TICK_HZ
PERIOD_SECONDS = 45 * 60


@dataclass(frozen=True)
class EngineParams:
    """Tunable knobs. Defaults are the calibrated values."""

    # Movement
    top_speed_min: float = 7.4  # m/s for pace=1
    top_speed_max: float = 9.6  # m/s for pace=99
    accel_min: float = 3.2  # m/s^2
    accel_max: float = 5.2
    dribble_speed_factor: float = 0.7  # fraction of top speed when carrying
    jog_speed: float = 3.0
    walk_speed: float = 1.4
    run_speed: float = 5.2
    shape_deadzone: float = 0.6  # metres within which a player idles
    shape_lag: float = 1.3  # seconds; how quickly the team block follows the ball
    fast_run_factor: float = 0.86  # share of top speed for hard (non-sprint) running
    sprint_min_duration: float = 1.0  # seconds above 25.2 km/h to count a sprint

    # Decision timing (seconds)
    decision_delay_min: float = 0.35
    decision_delay_max: float = 1.9
    carry_redecide: float = 0.55
    carry_cost: float = 0.003
    hold_cost: float = 0.0062
    box_carry_cost: float = 0.016

    # Passing
    pass_noise_base: float = 0.4  # metres
    pass_noise_per_m: float = 0.034  # metres per metre of pass, scaled by skill
    pass_noise_pressure: float = 0.6  # multiplier added under full pressure
    aerial_noise_mult: float = 1.6
    aerial_pass_cost: float = 0.013
    intercept_temp: float = 0.26  # seconds; logistic temperature for races
    intercept_margin: float = 0.1  # seconds of head start a defender needs
    intercept_reaction: float = 0.75  # defenders react after the ball is struck
    reach_radius: float = 1.0  # metres a player can stretch to play the ball
    first_touch_base: float = 0.965
    ground_decel: float = 1.8  # m/s^2 rolling deceleration
    pass_arrival_speed: float = 9.0  # target arrival speed for ground passes

    # Decision utility
    decision_temperature: float = 0.0045
    turnover_cost: float = 1.15  # multiplier on opponent value after a loss
    possession_premium: float = 0.016  # value of simply keeping the ball
    shot_bias: float = 1.7  # multiplier on shot utility vs passing
    min_shot_xg: float = 0.008
    forward_bias: float = 0.0016  # small preference for progression
    back_pass_penalty: float = 0.0012
    return_pass_penalty: float = 0.004

    # Duels and defending
    press_engage_dist: float = 1.5
    press_jockey_dist: float = 1.7
    path_duel_prob: float = 0.55  # chance a defender in the dribbler's path engages
    touchline_squeeze: float = 0.45  # chance a pressed player hugging the touchline loses it out
    tackle_hazard_base: float = 0.12  # attempts per second when engaged
    tackle_hazard_aggr: float = 0.3
    tackle_win_base: float = 0.0  # logit offset
    foul_base: float = 0.42
    foul_aggression: float = 0.12
    # Defenders are far more careful inside their own box (a foul there is a penalty): real football has
    # roughly 0.3 penalties per match against about 22 fouls, so box challenges foul much less often.
    box_foul_factor: float = 0.09
    pressure_radius: float = 2.7
    dribble_attempt_bias: float = 0.0
    dribble_utility_scale: float = 0.8
    dribble_utility_cost: float = 0.0045

    # Cards (per foul)
    yellow_base: float = 0.13
    yellow_tactical_bonus: float = 0.25
    straight_red: float = 0.0035

    # Shots
    xg_intercept: float = -1.97
    xg_dist: float = -0.11
    xg_angle: float = 1.3
    xg_header: float = -0.95
    xg_pressure: float = -0.55
    shot_block_per_defender: float = 0.13
    on_target_base: float = 0.3  # of non-goal, non-blocked shots: share saved
    save_hold_share: float = 0.55
    save_corner_share: float = 0.38
    block_corner_share: float = 0.5
    post_share: float = 0.035

    # Set pieces and crosses
    corner_header_shot: float = 0.24
    corner_keeper_claim: float = 0.12
    corner_short: float = 0.12
    cross_bias: float = 0.03
    cross_header_shot: float = 0.15
    cross_control: float = 0.14
    cross_keeper_claim: float = 0.12
    cross_out_corner: float = 0.20
    clearance_out_corner: float = 0.32

    # Fatigue
    energy_cost_per_m_sprint: float = 0.00026
    energy_cost_per_m_run: float = 0.00010
    energy_cost_per_m_jog: float = 0.00004
    energy_recovery_per_s: float = 0.00008

    # Dead-ball durations (seconds): (min, max)
    restart_throw_in: tuple[float, float] = (15.0, 30.0)
    restart_goal_kick: tuple[float, float] = (24.0, 40.0)
    restart_corner: tuple[float, float] = (28.0, 46.0)
    restart_free_kick: tuple[float, float] = (24.0, 50.0)
    restart_goal: tuple[float, float] = (55.0, 85.0)
    restart_keeper: tuple[float, float] = (5.0, 11.0)
    restart_penalty: tuple[float, float] = (45.0, 70.0)
    restart_sub: tuple[float, float] = (22.0, 35.0)


DEFAULT_PARAMS = EngineParams()
