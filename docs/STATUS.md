# Project status and handover notes

Read this first when picking the project up in a new chat.

## Deadlines (Central time)
- Registration closes **Tue Oct 20, 2:00 PM**.
- Submission closes **Wed Oct 28, 1:59 AM** (Tue Oct 27, 11:59 PM Pacific). Internal target: submit Mon Oct 26.
- The live demo must stay up until Nov 10.

## Target
- **Category:** Best Multi-Agent System (primary), Best Use of Microsoft Foundry (secondary). Grand Prize eligible.
- **Pitch:** "AI match commentary you can trust, that remembers the story, for every kind of fan."
- **Signature features:** commentary with receipts (every claim links to the events behind it), a fact-checker agent with published accuracy numbers, storyline memory, the Moment Weight metric (win-probability swing), analyst vs. casual vs. newcomer modes, multi-language output, and a control-room view of agents handing off. Stretch: audio-description mode.

## Done (Oct 18)
- `engine/`: agent-based 10 Hz simulator, fictional 8-club league, events + tracking, scenarios, seed search, live feed, schema, CLI, 20 passing tests, calibration report (15/22 metrics in real-world range, the rest within a few percent).
- `data/sample/riv-kin-7`: Riverside 0-1 Kingsport (Duarte 74' against his former club), 68% Riverside possession.
- `stats/` (package `matchstats`): 9 metric plug-ins, moment detector (18 types, each with facts, evidence ids, Moment Weight and importance), 11 agent query tools with JSON-schema specs, Pydantic contracts, CLI (`analyze`, `live`, `ask`, `tools`, `train`, `schema`), trained models with an accuracy report (`docs/MODEL_REPORT.md`), 21 passing tests including truth isolation. Analyses a full match in ~1.5 s.
- Known engine gaps: scoring slightly high (1.75 vs ≤1.65 goals per team), slightly few crosses, tackles and corners. Tune in `engine/src/matchengine/params.py` and check with `matchengine calibrate --matches 24`.

## Next (in order)
0. Find the demo-match seed (slow, so run it in the background, ~15 s per candidate): `cd engine && matchengine find-seed scenarios/showcase.json --start 18 --tries 150`. Seeds 1-17 don't match. If it takes too long, relax the storyline in `scenarios/showcase.json` (e.g. drop `scorer`). Then record it with `matchengine simulate` and check it with `matchstats analyze`.
1. Azure smoke test: create a Foundry project, deploy a small model (Standard deployment), make one call from a laptop with the key in an environment variable. Set budget alerts first.
2. `agents/`: wrap `matchstats.queries.TOOLS` as MCP tools; analyst, explainer, narrator, fact-checker, personaliser and overlay agents (Agent Framework on Foundry). The fact-checker verifies claims with `explain_moment` / `event`.
3. `frontend/`: 2D pitch replay from tracking frames, corner widget, mode switcher, control room.
4. Deploy (Container Apps + Static Web Apps), then README, architecture diagram, video.

## Stats layer notes
- Entry point: `MatchAnalyzer().process(kind, item)` returns new moments; `MatchQueries(analyzer).call(tool, args)` answers agent questions.
- Models never read `details.truth`; `tests/test_analyzer.py::test_models_never_see_ground_truth` enforces it.
- Retrain: `python stats/scripts/generate_training_data.py --matches 168 --out data/training`, then `matchstats train data/training --report docs/MODEL_REPORT.md`. Changing a feature list in `models/features.py` requires retraining (loading refuses mismatched weights).

## Key IDs
- Showcase fixture: `RIV` (Riverside Athletic, 4-3-3 high press) v `KIN` (Kingsport United, 4-2-3-1 low block).
- Stars: `riv-10` Kwame Mensah (ST, goal drought), `riv-07` Tomás Okafor (playmaker), `riv-11` Lucas Varga (fast winger), `kin-11` Rafael Duarte (ex-Riverside striker), `kin-09` Yuto Arakawa (AM), `kin-03` Marcus Bell (captain, card-prone), `kin-08` Ibrahima Sarr (pace).

## Direction change (Oct 19, 2 AM): 3D virtual broadcast
- Agreed focus: **3D broadcast view (option 3)** + **Multiverse what-if simulations (1)** + **telestrator (2)** + small **season memory (5)**. Pitch: "Other tools tell you what happened. Ours shows you what could have happened, and proves it."
- Agents trimmed to ~4: Director (camera choices), Analyst (runs multiverse sims), Commentator (incl. fan modes), Fact-checker.
- **Player look rule from Ryan: clean over "realistic". No janky humans.** The 3D test builds two variants side by side: (A) clean capsule figures (rounded cylinder + head, number on back, no limbs), (B) smooth mannequin with limbs. Default to A unless B is clearly clean.
- Forking a match mid-game works: deepcopy the engine, reseed `rng`, run `_step()` to the end (~6 s per 20-min future in the cloud sandbox). Needs a proper `fork()` API + a faster mode.
- **3D broadcast v1 done** (`matchstats replay BUNDLE --3d`, template `stats/src/matchstats/viewer/broadcast.html`, three.js 0.170.0 from jsDelivr):
  floodlit stadium with crowd that jumps on goals, LED boards, nets; clean and mannequin player styles (kick, save and celebration animations from `touches`);
  7 cameras (broadcast, behind goal, tactical, skycam, player, pitch-side, free drag); rule-based auto director with visible reasons
  (attack building, in the box, set piece, shape check); goal replays from 3 angles with wipe + letterbox; lower-third graphics; win-probability timeline; stats panel.
- Next: Director/Analyst/Commentator/Fact-checker agents; engine `fork()` for multiverse ghosts; telestrator on the tactical camera; product name.

## Done (Oct 19-20): agents, ghosts, AI desk in 3D
- Azure: pay-as-you-go with $200 credit (expires Nov 18), budgets `hackathon` ($25) and `hackathon-guard` ($150), Foundry project `rjlopez8-5433` (westus3). `gpt-5-mini-1` deployed at 30K TPM. GPT-6 Luna/Sol have 0 quota on this subscription: quota increase requested. Env vars on Ryan's laptop: AZURE_OPENAI_ENDPOINT (resource host only), AZURE_OPENAI_API_KEY, MATCH_DAILY_CAP_USD=5.
- `agents/` (package `matchagents`): seven agents (Producer, Analyst, Telestrator, Scenario agent, Director, Commentator, Fact-checker) as a Microsoft Agent Framework workflow with fan-out/fan-in and a fact-check revision loop. Strict Pydantic outputs; stats tools with a receipts ledger; code number check (multi-language decimals) + style check + model review; budget middleware refuses calls before the daily cap. Styles: fan, tactician, analyst, academy. First live run: goal segment for $0.02, approved first time.
- `engine/ghost.py`: shot checkpoints shipped with the match (`data/sample/riv-kin-7/ghost_checkpoints.pkl.gz`, exact replay differs across machines), paired futures (`what_if_shot`) with margin of error, Ghost Scout (`scout_shot`: the engine's shot model re-run for the same chance with every attacker in the league). 40 futures take ~17 s on Ryan's laptop.
- 3D AI desk: `matchagents render BUNDLE segments.json` embeds segments; on each covered moment the broadcast plays live reaction -> Director's replay shots -> telestration freeze -> ghost future (translucent players, results card) or Ghost Scout table, with fact-checked captions and a control-room panel of the seven agents (key D).

## Done (Oct 20): ghost timeline + demo match
- Engine: penalties were ~5x too frequent (defenders fouled in the box as freely as anywhere). `box_foul_factor` (params.py, 0.09) makes box challenges far more careful; fouls ~21/match and goals ~2/match unchanged. Old checkpoints keep old behaviour (getattr default).
- Demo match: `data/demo/riv-kin-1119` (seed 1119, new engine). Duarte 1', Mensah 48', 85' Mensah chance, **82' Fairclough saves Varga's free kick at 1-1** (moment `riv-kin-1119-m048`), Gilmartin penalty 90+6', 1-2. What-if result (24 futures): Riverside win 92% if Mensah scores vs 4% as it happened.
- Ghost futures now record each future's game state once a minute (fan of win-probability curves) and, for a featured future, clips of the changed moment and every later goal (`what_if_shot` -> `featured`, `curves`).
- 3D ghost timeline: split (gold vs cyan ball paths) -> cut into the ghost timeline (real player models driven by the simulated clip, cyan grade, ghost scoreboard) -> fast-forward clock -> that timeline's goal -> rewind -> verdict card with the fan of futures. Segments with a what-if keep 2 replay angles so a moment stays under ~40 s.

## Done (Oct 20, evening): 3D polish pass
- Demo moment corrected: `riv-kin-1119-m048` is **82' Fairclough saves Varga's 24 m free kick** at 1-1 (caught). Great-save titles now say "denies" and name free kicks (the old "tips away" was wrong for catches).
- Ghost goals: when a shot the keeper reached is forced in, the flight's end point moves to the corner away from him (`ghost._beat_keeper`, same placement in every future), so the gold and cyan paths visibly part.
- Viewer data: touches and shots carry fractional frame times (kicks line up to 0.01 s), shot kind, foot and save kind (catch/parry); throw-ins and headers have their own animations.
- 3D ball: every release to the next touch is a "flight": the ball leaves the boot at the kick time and arrives at the receiver's feet (measured: 0.45-0.49 m from the player at both ends for 871 passes), lofted balls follow a smooth spline, throw-ins leave from overhead, keepers carry caught balls in their gloves and drop them to kick, dead balls re-spotted for restarts vanish and reappear instead of sliding across the pitch.
- Shots: free kicks clear the wall and curl (left foot curls right), saves end where the keeper meets the ball, parries deflect towards where the real ball went.
- Keepers: face the ball, crouch into a ready stance near their box, shuffle sideways, hop into the set as the striker shoots, then dive (push off, fly with the body rolling about the hips and arms reaching, contact, land, lie, get up, walk back). Wrong-footed on central goals, full stretch on near misses, overhead catches and claims.
- Split now shows two angles (beside the shot, then behind the goal), a ball travelling each path, burst rings where each ended and labels with the outcome ("As it happened · saved", "Ghost timeline · goal").
- Stoppages (fouls, subs, restarts; the 62' foul + sub was 53 s of standing) play at 8x with a "Stoppage · skipping ahead" chip; standing players breathe and shift weight.
- Fixed: console errors at the end of auto replays (reading a step past the last), and auto/AI-desk replays pointing at the wrong goal on saves (the moment belongs to the keeper's team; replays now aim at the goal the shot went at).

## Next
1. Live-model segments for the demo moment with 200 futures: `matchagents cover ../data/demo/riv-kin-1119 --moment riv-kin-1119-m048 --futures 200 --out demo.json`, then `matchagents render`. Tune prompts.
2. 3D polish pass, then push everything to GitHub.
3. Per-agent model evaluation harness (GPT-6 once quota lands); fact-checker on a different model family.
4. Voice (Azure AI Speech), language picker in the viewer, season memory.
5. Deploy (Container Apps + Static Web Apps), README, architecture diagram, video.
