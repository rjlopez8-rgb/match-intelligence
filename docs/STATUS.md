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
- Known engine gaps: scoring slightly high (1.75 vs ≤1.65 goals per team), slightly few crosses, tackles and corners. Tune in `engine/src/matchengine/params.py` and check with `matchengine calibrate --matches 24`.

## Next (in order)
0. Find the demo-match seed (slow, so run it in the background, ~15 s per candidate): `cd engine && matchengine find-seed scenarios/showcase.json --start 18 --tries 150`. Seeds 1-17 don't match. If it takes too long, relax the storyline in `scenarios/showcase.json` (e.g. drop `scorer`).
1. Azure smoke test: create a Foundry project, deploy a small model, make one call from a laptop. Set a budget alert.
2. `stats/`: stream consumer that computes pass difficulty, speeds and distance thresholds, xG, possession and momentum, and Moment Weight from a win-probability model trained on simulated matches. Expose these as MCP tools.
3. `agents/`: analyst, explainer, narrator, fact-checker, personaliser and overlay agents (Agent Framework on Foundry).
4. `frontend/`: 2D pitch replay from tracking frames, corner widget, mode switcher, control room.
5. Deploy (Container Apps + Static Web Apps), then README, architecture diagram, video.

## Key IDs
- Showcase fixture: `RIV` (Riverside Athletic, 4-3-3 high press) v `KIN` (Kingsport United, 4-2-3-1 low block).
- Stars: `riv-10` Kwame Mensah (ST, goal drought), `riv-07` Tomás Okafor (playmaker), `riv-11` Lucas Varga (fast winger), `kin-11` Rafael Duarte (ex-Riverside striker), `kin-09` Yuto Arakawa (AM), `kin-03` Marcus Bell (captain, card-prone), `kin-08` Ibrahima Sarr (pace).
