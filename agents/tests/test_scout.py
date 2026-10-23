import json

from conftest import run

from matchagents import scout as S
from matchagents.llm import ScriptedChatClient

TRYOUT = {
    "team": "Riverside Athletic",
    "position": {"slot": 9, "label": "ST", "role": "ST"},
    "candidate": {"player_id": "wes-20", "name": "Ante Babić", "club": "Westmere Town", "role": "ST", "age": 28},
    "incumbent": {"player_id": "riv-10", "name": "Kwame Mensah", "club": "Riverside Athletic", "role": "ST", "age": 27},
    "with": {"futures": 12, "win": 0.42, "draw": 0.33, "loss": 0.25, "xpts": 1.62, "xpts_margin_95": 0.2,
             "xg_for": 1.41, "xg_against": 1.05, "per90": {"goals": 0.58, "xg": 0.61, "key_passes": 0.9}},
    "without": {"futures": 12, "win": 0.33, "draw": 0.42, "loss": 0.25, "xpts": 1.44, "xpts_margin_95": 0.2,
                "xg_for": 1.22, "xg_against": 1.07, "per90": {"goals": 0.41, "xg": 0.47, "key_passes": 0.7}},
    "diff": {"pairs": 12, "xpts": 0.18, "xpts_margin_95": 0.25, "xg_for": 0.19, "xg_against": -0.02, "clear": False},
}


def test_offline_note_is_true_and_cautious():
    note = run(S.Scout(offline_mode=True).note(TRYOUT))
    assert note["verdict"] == "shortlist"                # positive but inside the margin
    assert "0.18" in note["report"] and note["attempts"] == 1


def test_invented_numbers_fall_back_to_a_verified_note(monkeypatch):
    def liar(agent, messages, options):
        return json.dumps({"headline": "Babić would score 30 goals", "report": "He adds 0.9 points a game.", "verdict": "sign"})

    sc = S.Scout(offline_mode=True)
    sc.agent.client.responder = liar if isinstance(sc.agent.client, ScriptedChatClient) else liar
    note = run(sc.note(TRYOUT))
    assert note.get("fallback") and note["verdict"] == "shortlist"
