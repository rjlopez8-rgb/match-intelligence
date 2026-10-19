# Match Intelligence (working title)

AI match intelligence for the Microsoft x Premier League hackathon: a multi-agent system that turns a live football data feed into **explainable, fact-checked, personalised** match narratives and on-screen overlays, so a data-hungry analyst and a first-time viewer each get the match made for them.

> Status: the synthetic match engine (Ingest) and the stats layer (Interpret) are complete. Agents, frontend and Azure deployment come next. See [`docs/STATUS.md`](docs/STATUS.md).

## Pipeline

```
[ Data feed ] -> [ Stats engine ] -> [ Agent team on Microsoft Foundry ] -> [ Personalisation ] -> [ Overlays / app ]
  synthetic now,   pass difficulty,   analyst, explainer, narrator,           analyst, casual,      corner widget,
  official feed    speeds, xG,        fact-checker, storyline memory,        newcomer, language,   timed overlay JSON,
  in production    momentum           overlay producer                        favourite club/player control room
```

| Stage (from the rules) | Component | Status |
|---|---|---|
| Ingest synthetic match events as they happen | [`engine/`](engine/): agent-based 10 Hz match simulator + live feed | Done |
| Interpret into stats, patterns, context | [`stats/`](stats/): live metrics, trained xG / pass-difficulty / win-probability models, key moments with evidence, 11 agent tools | Done |
| Explain why moments matter | `agents/` (Microsoft Foundry + Agent Framework) | Planned |
| Render on screen alongside the match | `frontend/` + overlay spec | Planned |
| Personalise for different fans | `agents/` personaliser + `frontend/` modes | Planned |

## Repository layout

```
engine/      synthetic match engine (Python package `matchengine`)
stats/       live statistics, models, key moments and agent tools (Python package `matchstats`)
data/sample/ one complete simulated match (events, 5 Hz tracking, metadata, summary)
docs/        event schema, JSON Schemas, engine calibration, model accuracy report, project status
```

## Quick start

```
cd engine
pip install -e ".[dev]"
matchengine league
matchengine simulate --home RIV --away KIN --seed 7 --out ../data/matches/riv-kin-7
matchengine stream --speed 10
pytest

cd ../stats
pip install -e ".[dev]"
matchstats analyze ../data/sample/riv-kin-7
matchstats live --home RIV --away KIN --seed 7
pytest
```

All data is synthetic. The clubs, players, managers and stadiums of the "Meridian League" are fictional, and no real match footage, data or trademarks are used.
