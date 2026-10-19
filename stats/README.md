# matchstats: the Interpret stage

`matchstats` turns the live match feed into numbers an AI agent can trust. It reads events and tracking frames as they arrive, keeps 9 families of metrics up to date, and detects the **key moments** worth talking about. It then answers agents' questions through 11 small, typed queries.

Agents never calculate a statistic themselves. They ask `matchstats`, and every answer carries the **event ids that prove it**. That is what lets the commentary show its receipts and lets the fact-checker agent verify every claim.

```mermaid
flowchart LR
    F["Live feed<br/>events + 5 Hz tracking"] --> C["MatchContext<br/>score, clock, line-ups,<br/>attacking direction"]
    C --> M["Metric plug-ins<br/>physical · shooting · passing<br/>possession · pressing · momentum<br/>control · summary · win probability"]
    W[("Trained models<br/>xG · pass difficulty<br/>win probability")] --> M
    M -- signals --> D["Moment detector<br/>facts + evidence +<br/>Moment Weight"]
    M --> Q["Query layer<br/>11 agent tools"]
    D --> Q
    Q --> A["Agents<br/>(Microsoft Foundry)"]
```

## What it guarantees

| Guarantee | How it is enforced |
|---|---|
| **No cheating with simulator truth.** Models only see what a real data feed would contain. | A test strips every `details.truth` field from the feed and requires a byte-identical snapshot and moment list. |
| **Every claim has receipts.** | Every moment lists its evidence event ids. A test checks that each one exists and happened before the moment. A penalty's receipts start at the foul, not the open play before it. |
| **Live and replay give the same answers.** | Moments streamed item by item must equal a batch run over the same feed (tested). |
| **Published accuracy.** | Models are evaluated on matches they never saw. See [Model accuracy](#model-accuracy). |
| **Stable contracts for agents and UI.** | Pydantic models for moments and snapshots, exported as JSON Schema; every tool has a JSON-schema parameter spec. |
| **Fast enough for live use.** | A full match (≈32,000 feed items) is analysed in about 1.5 s on a laptop, roughly 4,000× real time. |

## Quick start

```bash
cd stats
pip install -e ../engine -e ".[dev]"

matchstats analyze ../data/sample/riv-kin-7              # full-time report
matchstats live --home RIV --away KIN --seed 7           # moments as they happen
matchstats replay ../data/sample/riv-kin-7               # 2D replay page you can open in a browser
matchstats ask ../data/sample/riv-kin-7 explain_moment --args '{"moment_id": "riv-kin-7-m071"}'
matchstats tools                                         # the agent tool catalogue
pytest
```

```python
from matchstats import MatchAnalyzer, replay_bundle
from matchstats.queries import MatchQueries

analyzer = MatchAnalyzer()
for kind, item in replay_bundle("../data/sample/riv-kin-7"):
    for moment in analyzer.process(kind, item):  # new moments, the instant they are detected
        print(moment["minute_display"], moment["title"])

q = MatchQueries(analyzer)
q.momentum()  # who is on top, the trend, and what is driving it
q.player_stats("Duarte")  # names are resolved fuzzily, so typos are fine
```

## Metrics

Each metric is a plug-in that updates on every event (and frame), so a snapshot is always current.

| Plug-in | What it measures |
|---|---|
| `physical` | Distance, high-speed running (> 19.8 km/h), sprints (> 25.2 km/h for 1 s+) and top speed per player, from tracking frames. Raises speed alerts (33+ km/h) and distance milestones. |
| `shooting` | **Expected goals** from our trained model; shots, big chances, goalkeeper goals prevented, an xG timeline, and the fastest shot. |
| `passing` | **Pass difficulty (0–10)** from our completion model, plus progressive passes, opponents bypassed (lines broken, from tracking), key passes, assists, crosses, through balls, accuracy under pressure, and **pass networks**. |
| `possession` | Possession share by ball-in-play time, field tilt, final-third and box entries, zone touches (6×3 grid), and possession sequences (longest, average passes, ending in a shot). |
| `pressing` | Pressures by third, **PPDA**, high turnovers and a pressing pulse. |
| `momentum` | Decayed attacking threat (4-minute half-life) folded into one index from −1 (away on top) to +1 (home on top). |
| `control` | A **chaos index (0–100)** from turnovers, duels and scrambles, scaled to levels learned from simulated matches; team control, tempo and stoppages. |
| `summary` | The box score: goals, cards, fouls, corners, tackles, aerials, dribbles, saves and substitutions. |
| `win_probability` | Live home/draw/away probabilities and **Moment Weight** for each goal, red card and big chance. |

### Moment Weight

Moment Weight is the number of percentage points of win probability a moment moved: it puts a single number on how much a moment changed the match.

- **Goals and red cards** use the realised swing. Duarte's 74' penalty moved Kingsport's chances by 48 points.
- **Missed chances** use the expected swing: the swing a goal would have caused, multiplied by the chance's xG. A sitter missed at 0-0 in the 85th minute outweighs a speculative shot at 4-0.

## Key moments

Moments are factual, not prose. Writing the commentary is the agents' job. Each moment carries `facts`, `evidence`, `moment_weight` and an `importance` (0–1) for prioritising overlays.

| Driven by events | Driven by metric signals |
|---|---|
| goal, big chance (incl. missed penalty), great save, penalty awarded, red / yellow card, tactical change, substitution, half-time / full-time summary | speed alert, new top speed, distance milestone, line-breaking pass, long passing sequence (25+), possession dominance, pressing surge, momentum shift, chaos spike |

## Agent tools

`MatchQueries` exposes one method per tool; `matchstats.queries.TOOLS` holds descriptions and JSON-schema parameters, ready to register with an MCP server or Foundry function calling.

| Tool | Answers |
|---|---|
| `scoreboard` | Score, clock, headline numbers, win probability, goals |
| `team_comparison` | Side-by-side stats in groups: attack, possession, passing, defending, physical, discipline |
| `player_stats` | Everything about one player, plus their moments |
| `top_players` | Leaderboard for any player stat, optionally for one team |
| `momentum` | Who is on top, the trend over a window, and what is driving it |
| `moments` | Filtered moments by type, team, player, minute or importance |
| `explain_moment` | One moment's facts plus its evidence events in readable form |
| `event` | One feed event with the stats layer's analysis attached |
| `win_probability` | Current probabilities, trend and the biggest swings |
| `pass_network` | Average positions and most frequent passing links |
| `timeline` | Per-minute series for charts (xG, possession, momentum, pressing, control, win probability) |

## Model accuracy

Trained on 135 simulated matches and tested on 33 matches the models never saw. The simulator knows the true probability behind every shot and pass, so unlike real-world models these can be checked against the truth. Full report: [`docs/MODEL_REPORT.md`](../docs/MODEL_REPORT.md).

![Model accuracy on unseen matches](../docs/MODEL_REPORT.png)

| Model | Used for | Result on unseen matches |
|---|---|---|
| **Expected goals** | chance quality, big chances, Moment Weight | 0.994 correlation with true chance quality; 115 goals predicted vs. 116 scored over 739 shots |
| **Pass completion** | pass difficulty (0–10), hardest passes | predicted 84.9% completion vs. 84.9% observed over 27,217 passes; AUC 0.77 vs. 0.76 for the engine's own decision-time estimate |
| **Win probability** | live odds, Moment Weight | 37% lower error (Brier) than guessing league averages; 90% of results called correctly from the 85th minute |

All three models are small, explainable logistic models written in NumPy, so there is no GPU or cloud cost. Their weights ship as JSON inside the package. Retrain with:

```bash
python scripts/generate_training_data.py --matches 168 --out ../data/training   # ~30 min on 2 cores
matchstats train ../data/training --report ../docs/MODEL_REPORT.md               # ~10 s
```

## Layout

```
src/matchstats/
  analyzer.py     MatchAnalyzer: the single entry point
  context.py      incremental match state, attacking-direction normalisation
  feed.py         replay a recorded bundle or follow a live simulated match
  metrics/        one plug-in per metric family (ordered registry)
  moments.py      key-moment detection with facts and evidence
  queries.py      agent-facing tools
  schema.py       Pydantic contracts and JSON Schema export
  models/         features, NumPy logistic/softmax, trained weights
  training.py     dataset building, training, evaluation, accuracy report
  replay.py       self-contained 2D replay page (viewer/replay.html)
  cli.py          `matchstats` command
tests/            truth isolation, evidence integrity, live = replay, schemas, queries, models
```
