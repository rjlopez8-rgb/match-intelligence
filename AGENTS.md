# Kairos: notes for coding agents

AI match intelligence for a simulated football league (the Meridian League, 8 fictional clubs, 56 matches).
Python 3.11+, three packages installed in editable mode from one `.venv`:

- `engine/` (`matchengine`): the 10 Hz match simulator, ghost futures, Club mode try-outs.
- `stats/` (`matchstats`): stats layer, trained models, key moments, query tools, the 3D viewer, the season hub,
  and the MCP server (`matchstats mcp`).
- `agents/` (`matchagents`): the nine agents on Microsoft Foundry (the Agent Framework broadcast desk, Recap, Scout), the website build.

## The Kairos MCP server
`.vscode/mcp.json` starts it (`python -m matchstats.cli mcp`). Use its tools for any question about matches,
players, the table or recruitment instead of reading data files: `list_matches`, `league_table`, `season_player`,
`what_decided_it`, the per-match stats tools (each takes `match_id` and an optional `minute` to answer as of then),
and Club mode (`tryout_board`, `tryout_result`, `run_tryout`). Quote only numbers the tools return.

## Rules
- Never spend money on models without asking: live `matchagents` commands print the cost and ask first. Tests and
  `--offline` runs are free. Never commit API keys (they live in environment variables).
- The committed season is the truth: never re-simulate it. 3D data and turning points are built on Linux.
- Run `pytest` in `engine/`, `stats/` and `agents/` after changes.
