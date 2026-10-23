# Match Intelligence (working title)

AI match intelligence for the Microsoft x Premier League hackathon: a 3D virtual broadcast run by AI agents on **Microsoft Foundry**. They explain every key moment with fact-checked commentary, call the cameras, draw on the replay, and **settle the "what if" arguments by simulating the alternative futures**.

**Your broadcast.** Every viewer gets their own cut of the same match. Pick a style (Fan, Analyst, Tactician, Academy), a language (English, Spanish, French), your club and your player: the live cameras, the replays, the graphics and the AI commentary all change. Commentary is written natively in each language and fact-checked in that language; a club's fans get their own channel; the Academy edition coaches the decision with the stats layer's options at the key moment; and player focus keeps your player in shot with his live stat line, alerts and every action marked on the timeline.

**One engine, two audiences.** For fans, the what-if engine replays a moment that could have gone the other way. For clubs, **Club mode** tries any player in the league in any position: the fixture is played many times with him and many times with the player who really starts there, and the recruitment board ranks who would actually make the team better.

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
| Club mode: try any player in any position | [`engine/`](engine/) `tryout.py` (paired full-match try-outs), `matchengine board` / `serve`, the Scout agent, the recruitment board in the 3D page | Done (v1) |
| Personalise for different fans | **Your broadcast**: 4 styles x club channels x 3 languages, each with its own Director cut and fact-checked commentary (`matchagents editions`); per-style live directing and graphics; club POV; player focus mode; the full page in EN/ES/FR | Done (v2) |

## Repository layout

```
engine/      synthetic match engine and ghost futures (Python package `matchengine`)
stats/       live statistics, models, key moments, agent tools and the 3D viewer (Python package `matchstats`)
agents/      the seven-agent broadcast desk on Microsoft Foundry (Python package `matchagents`)
data/sample/ one complete simulated match (events, 5 Hz tracking, metadata, summary), used by the tests
data/demo/   the demo match (Riverside 1-2 Kingsport, seed 1119) with its ghost checkpoints and Club mode try-outs
data/season/ the simulated 2026/27 season: 56 small match bundles (events + box score, no tracking) and season.json
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

### Your broadcast: every style, club channel and language

```
cd agents
matchagents editions ../data/demo/riv-kin-1119 --offline --out editions.json          # $0 preview (rule-based agents)
matchagents editions ../data/demo/riv-kin-1119 --futures 40 --out editions-live.json  # real models: asks first, shows the cost
matchagents render ../data/demo/riv-kin-1119 editions-live.json                       # open it and press Y
```

The 10 biggest moments x 18 editions (Fan neutral / home channel / away channel, Tactician, Analyst, Academy, each in
English, Spanish and French). Research and simulations run once per moment; each style's desk then writes and fact-checks
its own versions in parallel. The run is resumable: if the daily cap stops it, the same command tomorrow finishes the
rest and never pays twice for a finished edition. Links can preset the edition: `?style=academy&lang=fr&club=away&player=riv-10`.

### The season hub and the website

```
cd engine
matchengine season --out ../data/season --demo ../data/demo/riv-kin-1119   # 56 matches, ~30-60 min, no AI, $0

cd ../agents
matchagents site --season ../data/season --demo ../data/demo/riv-kin-1119 --segments editions-live.json --out ../site
# site/index.html: league table (actual and expected), points race, results by matchday, match reports with the
# win-probability story, player standings and pages, clubs; the demo match opens the 3D broadcast with the AI desk
```

### Club mode (player try-outs)

```
cd engine
# one try-out: Ante Babić (Westmere) up front for Riverside instead of Mensah, 20 matches each way
matchengine tryout --home RIV --away KIN --side home --position ST --player wes-20 --n 20 --cache ../data/demo/riv-kin-1119/tryouts
# the recruitment board: rank every candidate in every position and pre-simulate two per position (slow: hours)
matchengine board --home RIV --away KIN --n 12 --cache ../data/demo/riv-kin-1119/tryouts --out ../data/demo/riv-kin-1119/tryouts/board.json
# live try-outs for the 3D page (engine only, no model costs, daily compute cap)
matchengine serve --cache ../data/demo/riv-kin-1119/tryouts        # then open the page with ?api=http://127.0.0.1:8765

cd ../agents
matchagents scout ../data/demo/riv-kin-1119/tryouts/board.json     # the Scout agent's fact-checked notes (asks before spending)
matchagents render ../data/demo/riv-kin-1119 demo.json              # the page picks up the board automatically (key C)
```

All data is synthetic. The clubs, players, managers and stadiums of the "Meridian League" are fictional, and no real match footage, data or trademarks are used.
