from conftest import GOAL, SAMPLE

from matchagents.checks import check_numbers, check_style, resolve_telestration
from matchagents.schemas import Commentary, Line, Shape, Telestration
from matchagents.session import MatchSession

EVIDENCE = [{"xg": 0.7385, "speed_kmh": 107.6, "score": {"home": 0, "away": 1}, "minute": "74'", "possession": 0.735}]


def lines(*texts):
    return Commentary(lines=[Line(text=t, facts_used=[]) for t in texts], headline="", caption="", question=None)


def test_supported_numbers_pass_including_rounding_and_percentages():
    c = lines("A 74% chance, struck at 108 km/h.", "Kingsport lead 0-1 after 74 minutes.", "xG 0.74, 73.5% possession")
    assert check_numbers(c, EVIDENCE) == []


def test_invented_numbers_are_caught():
    issues = check_numbers(lines("That was his 3rd goal at 112 km/h, now 2-1."), EVIDENCE)
    claims = {i["claim"] for i in issues}
    assert {"3", "112", "2-1"} <= claims


def test_other_languages_number_formats():
    assert check_numbers(lines("Un xG de 0,74 y un 74 % de probabilidad."), EVIDENCE) == []
    assert check_numbers(lines("Kingsport gana 0-1 con 108 km/h."), EVIDENCE) == []
    assert check_numbers(lines("Un xG de 0,91."), EVIDENCE)


def test_precision_is_respected():
    assert check_numbers(lines("xG of 0.74"), EVIDENCE) == []
    assert check_numbers(lines("xG of 0.75"), EVIDENCE)  # 0.7385 does not round to 0.75


def test_telestration_lands_where_players_were():
    s = MatchSession(SAMPLE)
    m = s.advance_to(GOAL)
    goal_event = s.ctx.by_id[m["evidence"][-1]]
    t = Telestration(
        freeze_event_id=goal_event["event_id"],
        shapes=[
            Shape(
                kind="ring",
                player="Duarte",
                event_id=None,
                to_player=None,
                to_event_id=None,
                label="Duarte",
                color="highlight",
            ),
            Shape(
                kind="ring",
                player="Nobody Atall",
                event_id=None,
                to_player=None,
                to_event_id=None,
                label=None,
                color="warning",
            ),
        ],
        caption="Duarte from the spot",
    )
    out = resolve_telestration(t, s, m)
    ring = out["shapes"][0]
    assert ring["player_id"] == "kin-11"
    assert 0 <= ring["x"] <= 105 and 0 <= ring["y"] <= 68
    assert len(out["dropped"]) == 1  # unknown players are dropped, never guessed


def test_pasted_data_and_robotic_numbers_never_air():
    bad = lines(
        "Win probability jumps from {'home': 0.4644, 'away': 0.2751} to {'home': 0.0815}.",
        "It was a penalty with xG 0.7385.",
        "The shot_speed_kmh was high.",
    )
    problems = [i["problem"] for i in check_style(bad)]
    assert any("pasted data" in p for p in problems)
    assert any("unrounded" in p for p in problems)
    assert {i["line"] for i in check_style(bad)} == {0, 1, 2}
    good = lines("Duarte sends the keeper the wrong way, and Kingsport's chances leap to 75%.")
    assert check_style(good) == []


def test_french_and_spanish_words_and_on_screen_text_are_checked():
    c = lines("Un xG de 0,74 et 74 pour cent de chances.", "Un 74 por ciento, a 108 km/h.")
    assert check_numbers(c, EVIDENCE) == []
    c = Commentary(lines=[Line(text="Arrêt décisif.", facts_used=[])], headline="BUT ! 0-1",
                   caption="La frappe à 108 km/h", question="Et si Duarte avait marqué à la 81e ?")
    issues = check_numbers(c, EVIDENCE)
    assert [i["line"] for i in issues] == ["question"] and issues[0]["claim"] == "81"
