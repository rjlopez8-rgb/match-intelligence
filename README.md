# Match Intelligence (working title)

AI match intelligence for the Microsoft x Premier League hackathon: a 3D virtual broadcast run by AI agents on **Microsoft Foundry**. They explain every key moment with fact-checked commentary, call the cameras, draw on the replay, and **settle the "what if" arguments by simulating the alternative futures**.

**Your broadcast.** Every viewer gets their own cut of the same match. Pick a style (Fan, Analyst, Tactician, Academy), a language (English, Spanish, French), your club and your player: the live cameras, the replays, the graphics and the AI commentary all change. Commentary is written natively in each language and fact-checked in that language; a club's fans get their own channel; the Academy edition coaches the decision with the stats layer's options at the key moment; and player focus keeps your player in shot with his live stat line, alerts and every action marked on the timeline.

**Commentary with receipts.** Click any line the AI says (or press R) and see where every fact in it came from: each number
matched to its source, the tool call and result field behind each fact, the match events it cites (one click jumps the 3D
view there), and the fact-check history, rejected drafts included. Analysts also get a shot map, an xG race and a pass
network that only ever use the match so far (key G).

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
| Explain why moments matter | [`agents/`](agents/): seven agents as a Microsoft Agent Framework workflow on Foundry models, fact-checked, budget-guarded; the Recap agent for every season match, with a simulated "what decided it?" | Done (v1) |
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
data/season/ the simulated 2026/27 season: 56 small match bundles (events + box score, no tracking) and season.json;
             `matchengine season --tracking` re-plays them with tracking for the 3D site
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
# 3D for every match: re-play each one on its seed with tracking (~5 MB a match, kept out of git in data/matches/).
# Every match is checked event for event against the committed season; any difference stops the run.
matchengine season --out ../data/season --demo ../data/demo/riv-kin-1119 --tracking ../data/matches/season   # ~15-30 min

cd ../agents
matchagents site --season ../data/season --demo ../data/demo/riv-kin-1119 --segments editions-live.json --tracking ../data/matches/season --out ../site
cd ../site && python -m http.server 8000      # open http://localhost:8000
```

`site/index.html` is the hub: league table (actual and expected), points race, results by matchday, match reports with
the win-probability story, player standings and pages, clubs. Every match has **3D highlights** (its goals, saves and
chances back to back, each clip named by a wipe), and each matchday's **match of the week** (the one whose result swung
most) can be watched in full. The demo match is MD7's pick and carries the AI desk and Club mode. All of it plays in one
shared page, `watch.html?m=<match id>&v=hl|full`, which loads `matches/<id>.hl.json` or `.full.json`; the whole site is
about 100 MB, under the 250 MB Static Web Apps limit. Serve the folder over HTTP (as above) rather than opening the
file directly, because the page fetches its match.

### Match recaps: what happened, and what decided it

```
cd agents
matchagents recaps ../data/season --demo ../data/demo/riv-kin-1119 --research-only   # $0: the what-ifs (any machine)
matchagents recaps ../data/season --out recaps-live.json                              # the Recap agent: asks first, ~$0.45
matchagents site ... --recaps recaps-live.json                                         # the hub shows them
```

For every season match the stats layer finds the turning point (the goal or chance that moved win probability most).
The engine froze each match the instant that shot was struck (`data/season/turning_forks.pkl.gz`; freezing needs an
exact replay, so it was done on Linux, where the season was generated), and plays it to full time 40 times as it
happened and 40 times with the shot flipped, on any machine. The Recap agent writes a headline, a short recap and what
decided it; code checks every number, scoreline and player name against the evidence, then the Fact-checker reviews
every claim; one rewrite, else a code-written recap. Without `--recaps` the hub shows the code-written ones.

### Match Intelligence as an MCP server (GitHub Copilot)

```
pip install -e "./stats[mcp]"
matchstats mcp                     # stdio; VS Code starts it from .vscode/mcp.json
```

19 tools: the season (`list_matches`, `league_table`, `season_player`, `what_decided_it`), the stats layer's own query
tools for any match as of any minute, and Club mode (`tryout_board`, `tryout_result`, `run_tryout` on the engine).
In VS Code, Copilot agent mode uses them directly; `/scout` and `/matchday-report` are ready-made prompts. Setup and a
demo script: [`docs/COPILOT_DEMO.md`](docs/COPILOT_DEMO.md).

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
