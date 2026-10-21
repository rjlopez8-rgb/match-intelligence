# Match Intelligence (working title)

AI match intelligence for the Microsoft x Premier League hackathon: a 3D virtual broadcast run by seven AI agents on **Microsoft Foundry**. They explain every key moment with fact-checked commentary, call the cameras, draw on the replay, and **settle the "what if" arguments by simulating the alternative futures**.

> Other tools tell you what happened. Ours shows you what could have happened, and proves it.

Status: engine, stats layer, seven-agent desk, ghost futures and the 3D broadcast all work end to end. Deployment and the demo video come next. See [`docs/STATUS.md`](docs/STATUS.md).

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
| Explain why moments matter | [`agents/`](agents/): seven agents as a Microsoft Agent Framework workflow on Foundry models, fact-checked, budget-guarded | Done (v1) |
| Simulate the alternatives | [`engine/`](engine/) ghost futures + Ghost Scout, driven by the Scenario agent | Done (v1) |
| Render on screen alongside the match | 3D broadcast (`stats/.../viewer/broadcast.html`): AI director, replays, telestration, ghost players, control room | Done (v1) |
| Personalise for different fans | Commentator styles (fan, tactician, analyst, academy), favourite team, any language | Done (v1) |

## Repository layout

```
engine/      synthetic match engine and ghost futures (Python package `matchengine`)
stats/       live statistics, models, key moments, agent tools and the 3D viewer (Python package `matchstats`)
agents/      the seven-agent broadcast desk on Microsoft Foundry (Python package `matchagents`)
data/sample/ one complete simulated match (events, 5 Hz tracking, metadata, summary), used by the tests
data/demo/   the demo match (Riverside 1-2 Kingsport, seed 1119) with its ghost checkpoints
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

cd ../agents
pip install -e ".[dev]"
matchagents match ../data/sample/riv-kin-7 --offline --out segments.json   # $0, rule-based agents
matchagents render ../data/sample/riv-kin-7 segments.json                   # 3D broadcast with the AI desk
pytest
```

All data is synthetic. The clubs, players, managers and stadiums of the "Meridian League" are fictional, and no real match footage, data or trademarks are used.
