---
agent: 'agent'
description: 'Write a matchday report from the Match Intelligence tools into docs/reports/'
argument-hint: 'matchday number, e.g. 7'
---
Write the report for matchday ${input:matchday:7} of the Meridian League using the Match Intelligence MCP tools only
for facts and numbers.

1. `list_matches` for the matchday, then `league_table` (actual and expected).
2. For each match: `scoreboard`, `what_decided_it`, and `moments` with order "importance" and limit 3.
3. Create `docs/reports/matchday-${input:matchday:7}.md`: a headline, one paragraph per match (score, scorers, xG,
   the turning point and what the simulations say it changed), and the table after the matchday.
4. Every number must come from a tool result. Do not invent quotes. All clubs and players are fictional.
