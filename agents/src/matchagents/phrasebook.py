"""The offline Commentator's phrasebook: rule-based lines in English, Spanish and French for every style and channel.

Only used by ``--offline`` runs and the tests, so the personalised broadcast can be built and checked for $0. The
real Commentator writes natively in each language; this just makes the free path behave like it: the same
structured output, every number taken from the Analyst's facts (so it passes the same fact-check), worded per
style (fan, tactician, analyst, academy), per channel (neutral or a club's own) and per language.
"""

from __future__ import annotations

import re

LANG_CODE = {"English": "en", "Spanish": "es", "French": "fr"}

FACT_PATTERNS = {
    "xg": re.compile(r"^The chance was worth ([\d.]+) xG"),
    "dist": re.compile(r"^The shot came from ([\d.]+) metres"),
    "speed": re.compile(r"^The ball flew at ([\d.]+) km/h"),
    "swing": re.compile(r"^The moment moved win probability by ([\d.]+) points"),
    "better": re.compile(r"^Better option: (.+), pass (\d+)%"),
    "right": re.compile(r"^Right call: best alternative only ([\d.]+) xG"),
    "poss": re.compile(r"^Possession (.+): (\d+)%"),
    "txg": re.compile(r"^Expected goals (.+): ([\d.]+)"),
}

# opening line: (fan neutral, own channel, other channel, neutral for the other styles)
OPEN = {
    "goal": {
        "en": (
            "GOAL! {p} scores for {team}!",
            "GOAL! {p}! Get in, {team}!",
            "{p} scores for {team}. That one hurts.",
            "Goal for {team}: {p} makes it {h}-{a}.",
        ),
        "es": (
            "¡GOL! ¡{p} marca para el {team}!",
            "¡GOOOL! ¡{p}! ¡Vamos, {team}!",
            "Marca {p} para el {team}. Duele.",
            "Gol del {team}: {p} pone el {h}-{a}.",
        ),
        "fr": (
            "BUT ! {p} marque pour {team} !",
            "BUT ! {p} ! Allez {team} !",
            "{p} marque pour {team}. Ça fait mal.",
            "But pour {team} : {p} fait {h}-{a}.",
        ),
    },
    "great_save": {
        "en": (
            "What a save from {p}! {q} can't believe it.",
            "{p}! What a save! He keeps us in it!",
            "Denied! {p} stops {q} again.",
            "{p} saves from {q}.",
        ),
        "es": (
            "¡Paradón de {p}! {q} no se lo cree.",
            "¡{p}! ¡Qué parada! ¡Nos mantiene vivos!",
            "¡Nada! {p} vuelve a frenar a {q}.",
            "{p} detiene el disparo de {q}.",
        ),
        "fr": (
            "Quel arrêt de {p} ! {q} n'en revient pas.",
            "{p} ! Quel arrêt ! Il nous garde en vie !",
            "Rien à faire ! {p} arrête encore {q}.",
            "{p} repousse la frappe de {q}.",
        ),
    },
    "penalty_awarded": {
        "en": (
            "Penalty to {team}! {p} goes down.",
            "Penalty! Yes! {p} is brought down!",
            "Oh no. Penalty to {team}.",
            "Penalty to {team}: {p} fouled by {q}.",
        ),
        "es": (
            "¡Penalti para el {team}! Cae {p}.",
            "¡Penalti! ¡Sí! ¡Derriban a {p}!",
            "Ay, no. Penalti para el {team}.",
            "Penalti para el {team}: falta de {q} sobre {p}.",
        ),
        "fr": (
            "Penalty pour {team} ! {p} est au sol.",
            "Penalty ! Oui ! {p} est fauché !",
            "Oh non. Penalty pour {team}.",
            "Penalty pour {team} : faute de {q} sur {p}.",
        ),
    },
    "summary": {
        "en": ("{label}: {home} {h}-{a} {away}.",) * 4,
        "es": ("{label}: {home} {h}-{a} {away}.",) * 4,
        "fr": ("{label} : {home} {h}-{a} {away}.",) * 4,
    },
}
OPEN["big_chance"] = OPEN["woodwork"] = OPEN["great_save"]
PERIOD = {"en": ("Half-time", "Full-time"), "es": ("Descanso", "Final"), "fr": ("Mi-temps", "Fin du match")}

HEADLINE = {
    "goal": {"en": "GOAL! {p} · {h}-{a}", "es": "¡GOL! {p} · {h}-{a}", "fr": "BUT ! {p} · {h}-{a}"},
    "great_save": {"en": "{p} denies {q}", "es": "{p} le niega el gol a {q}", "fr": "{p} prive {q} du but"},
    "penalty_awarded": {"en": "Penalty to {team}", "es": "Penalti para el {team}", "fr": "Penalty pour {team}"},
    "summary": {
        "en": "{label}: {home} {h}-{a} {away}",
        "es": "{label}: {home} {h}-{a} {away}",
        "fr": "{label} : {home} {h}-{a} {away}",
    },
}
CAPTION = {
    "goal": {"en": "The finish", "es": "La definición", "fr": "La finition"},
    "great_save": {"en": "The save", "es": "La parada", "fr": "L'arrêt"},
    "penalty_awarded": {"en": "The foul", "es": "La falta", "fr": "La faute"},
    "other": {"en": "The key moment", "es": "La jugada clave", "fr": "Le moment clé"},
}

STYLE = {
    "fan_speed_goal": {
        "en": "Struck at {speed} km/h: the keeper had no chance.",
        "es": "A {speed} km/h: el portero no tuvo opción.",
        "fr": "Frappée à {speed} km/h : le gardien n'avait aucune chance.",
    },
    "fan_speed_save": {
        "en": "{speed} km/h, and he still keeps it out!",
        "es": "¡A {speed} km/h y aun así la saca!",
        "fr": "À {speed} km/h, et il la sort quand même !",
    },
    "fan_swing": {
        "en": "That one swung the game by {swing} points.",
        "es": "Eso cambió el partido en {swing} puntos.",
        "fr": "Ce moment a fait basculer le match de {swing} points.",
    },
    "analyst_xg": {
        "en": "The chance was worth {xg} xG from {dist} metres.",
        "es": "La ocasión valía {xg} xG desde {dist} metros.",
        "fr": "L'occasion valait {xg} xG à {dist} mètres.",
    },
    "analyst_swing": {
        "en": "Win probability moved {swing} points.",
        "es": "La probabilidad de victoria se movió {swing} puntos.",
        "fr": "La probabilité de victoire a bougé de {swing} points.",
    },
    "analyst_summary": {
        "en": "Expected goals: {home} {xgh}, {away} {xga}.",
        "es": "Goles esperados: {home} {xgh}, {away} {xga}.",
        "fr": "Buts attendus : {home} {xgh}, {away} {xga}.",
    },
    "tactic_goal": {
        "en": "{team} broke the lines and found the space behind.",
        "es": "El {team} rompió líneas y encontró el espacio a la espalda.",
        "fr": "{team} a cassé les lignes et trouvé l'espace dans le dos.",
    },
    "tactic_save": {
        "en": "The block held just long enough for the keeper to set.",
        "es": "El bloque aguantó lo justo para que el portero se colocara.",
        "fr": "Le bloc a tenu juste assez pour que le gardien se place.",
    },
    "tactic_summary": {
        "en": "{team} had {poss} of the ball, but control is not chances.",
        "es": "El {team} tuvo el {poss} del balón, pero controlar no es crear.",
        "fr": "{team} a eu {poss} du ballon, mais contrôler n'est pas créer.",
    },
    "tactic_other": {
        "en": "Watch the distances between the lines here.",
        "es": "Fíjense en las distancias entre líneas.",
        "fr": "Regardez les distances entre les lignes.",
    },
    "academy_better": {
        "en": "Coaching point: {best} was free, and a pass there arrives {pc} of the time.",
        "es": "Para aprender: {best} estaba solo, y ese pase llega el {pc} de las veces.",
        "fr": "À retenir : {best} était seul, et cette passe arrive {pc} du temps.",
    },
    "academy_right": {
        "en": "Right call to shoot: the best pass would have led to only {x} xG.",
        "es": "Bien elegido: el mejor pase solo habría dado {x} xG.",
        "fr": "Bon choix de frapper : la meilleure passe n'aurait donné que {x} xG.",
    },
    "academy_other": {
        "en": "Watch his body shape before the ball arrives: that is the habit to practise.",
        "es": "Fíjense en su perfil antes de recibir: ese es el hábito que hay que practicar.",
        "fr": "Regardez son orientation avant de recevoir : c'est l'habitude à travailler.",
    },
}
WHAT_IF = {
    "en": "In the simulated futures where {it}, {team} win {a} of the time, against {b} as it happened.",
    "es": "En los futuros simulados en que {it}, el {team} gana el {a} de las veces, frente al {b} en la realidad.",
    "fr": "Dans les futurs simulés où {it}, {team} gagne {a} du temps, contre {b} dans la réalité.",
}
IT = {
    "goal": {"en": "it goes in", "es": "entra", "fr": "elle entre"},
    "saved": {"en": "it is kept out", "es": "no entra", "fr": "elle ne rentre pas"},
}
QUESTION = re.compile(r"What if (.+?) (had not scored|had scored) at (.+?)\?")
Q = {
    "es": ("¿Y si {who} hubiera marcado en el {min}?", "¿Y si {who} no hubiera marcado en el {min}?"),
    "fr": ("Et si {who} avait marqué à la {min} ?", "Et si {who} n'avait pas marqué à la {min} ?"),
}


def _num(v: float, lang: str, d: int = 2) -> str:
    s = f"{v:.{d}f}"
    return s.replace(".", ",") if lang in ("es", "fr") else s


def _pct(v: float, lang: str) -> str:
    n = round(v * 100) if v <= 1 else round(v)
    return f"{n}%" if lang == "en" else f"{n} %"


def commentary(p: dict) -> dict:
    prof = p.get("profile") or {}
    lang = LANG_CODE.get(prof.get("language", "English"), "en")
    style = prof.get("style", "fan")
    fav = prof.get("favourite_team")
    m = p.get("moment") or {}
    typ = m.get("type") or ""
    teams = p.get("teams") or {}
    names = m.get("player_names") or []
    team = m.get("team_name") or ""
    score = m.get("score") or {"home": 0, "away": 0}
    kind = "summary" if typ == "period_summary" else typ if typ in OPEN else None
    label = PERIOD[lang][0 if str(m.get("title", "")).startswith("Half") else 1]
    ctx = {
        "p": names[0] if names else team,
        "q": names[1] if len(names) > 1 else "",
        "team": team,
        "h": score.get("home", 0),
        "a": score.get("away", 0),
        "home": teams.get("home", ""),
        "away": teams.get("away", ""),
        "label": label,
    }

    facts: dict[str, tuple[int, tuple]] = {}
    for f in p.get("facts") or []:
        for key, rx in FACT_PATTERNS.items():
            mm = rx.match(f["statement"])
            if mm and key not in facts:
                facts[key] = (f["index"], mm.groups())
            elif mm and key in ("poss", "txg"):
                facts[f"{key}2"] = (f["index"], mm.groups())

    lines = []
    if kind:
        mine = bool(fav) and fav == team
        k = 3 if style != "fan" else 1 if mine else 2 if fav else 0
        lines.append({"text": OPEN[kind][lang][k].format(**ctx), "facts_used": []})
    else:
        lines.append({"text": p.get("headline") or m.get("title", ""), "facts_used": []})

    def say(key: str, used: list[int], **kw) -> None:
        lines.append({"text": STYLE[key][lang].format(**ctx, **kw), "facts_used": used})

    save = typ in ("great_save", "big_chance", "woodwork")
    if style == "fan":
        if "speed" in facts and kind in ("goal", "great_save", "big_chance", "woodwork"):
            i, (v,) = facts["speed"]
            say("fan_speed_save" if save else "fan_speed_goal", [i], speed=f"{float(v):.0f}")
        elif "swing" in facts:
            i, (v,) = facts["swing"]
            say("fan_swing", [i], swing=f"{float(v):.0f}")
    elif style == "analyst":
        if "xg" in facts and "dist" in facts:
            (i, (x,)), (j, (d,)) = facts["xg"], facts["dist"]
            say("analyst_xg", [i, j], xg=_num(float(x), lang), dist=f"{float(d):.0f}")
        elif "txg" in facts and "txg2" in facts:
            (i, (_, x1)), (j, (_, x2)) = facts["txg"], facts["txg2"]
            say("analyst_summary", [i, j], xgh=_num(float(x1), lang), xga=_num(float(x2), lang))
        if "swing" in facts:
            i, (v,) = facts["swing"]
            say("analyst_swing", [i], swing=f"{float(v):.0f}")
    elif style == "tactician":
        if typ == "goal":
            say("tactic_goal", [])
        elif save:
            say("tactic_save", [])
        elif "poss" in facts:
            i, (tname, v) = facts["poss"]
            lines.append(
                {
                    "text": STYLE["tactic_summary"][lang].format(team=tname, poss=_pct(int(v) / 100, lang)),
                    "facts_used": [i],
                }
            )
        else:
            say("tactic_other", [])
    else:  # academy
        if "better" in facts:
            i, (best, pc) = facts["better"]
            say("academy_better", [i], best=best, pc=_pct(int(pc) / 100, lang))
        elif "right" in facts:
            i, (x,) = facts["right"]
            say("academy_right", [i], x=_num(float(x), lang))
        else:
            say("academy_other", [])

    res = p.get("what_if_result") or None
    if res and res.get("with_change") and res.get("as_it_happened"):
        side = res.get("team_side") or ("home" if team == ctx["home"] else "away")
        a, b = res["with_change"].get(f"{side}_win"), res["as_it_happened"].get(f"{side}_win")
        if a is not None and b is not None:
            it = IT["goal" if res.get("changed_outcome") == "goal" else "saved"][lang]
            lines.append(
                {
                    "text": WHAT_IF[lang].format(it=it, team=teams.get(side, team), a=_pct(a, lang), b=_pct(b, lang)),
                    "facts_used": [],
                }
            )

    head_t = HEADLINE.get(kind or "", {}).get(lang)
    headline = head_t.format(**ctx) if head_t else (p.get("headline") or m.get("title", ""))
    caption = CAPTION.get(kind if kind in CAPTION else "other")[lang]
    question = p.get("what_if")
    if question and lang != "en":
        qm = QUESTION.match(question)
        if qm:
            who, verb, minute = qm.groups()
            question = Q[lang][1 if verb == "had not scored" else 0].format(who=who, min=minute)
    return {"lines": lines[:3], "headline": headline, "caption": caption, "question": question}
