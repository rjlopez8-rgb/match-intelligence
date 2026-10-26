# Kairos in GitHub Copilot (MCP)

The stats layer, the season and Club mode are an MCP server (`matchstats mcp`). In VS Code, GitHub Copilot's agent
mode picks it up from `.vscode/mcp.json` and can answer football questions, run player try-outs on the match engine
and write files from the results. No AI keys are needed for the server: the numbers come from the stats layer and the
engine; Copilot's own model only plans the tool calls and writes the answer.

## Setup (once, about 5 minutes)

1. Install the server's dependency in the project's `.venv` (from the repo root):
   ```powershell
   .venv\Scripts\Activate.ps1
   pip install -e "./stats[mcp]"
   ```
2. Open the repo folder in VS Code (File > Open Folder) with GitHub Copilot signed in.
3. Trust the folder when VS Code asks (servers in `.vscode/mcp.json` cannot start in Restricted Mode). Start the
   server: Command Palette > **MCP: List Servers** > kairos > **Start Server** (or Chat > Open
   Customizations > MCP Servers > ... > Start Server). VS Code also starts new workspace servers by itself on your first
   chat message. If it fails, the same menu's **Show Output** shows why (usually the Python path in `.vscode/mcp.json`,
   or `mcp` not installed). On macOS or Linux, change `command` to `${workspaceFolder}/.venv/bin/python`.
4. Open Chat, switch to **Agent**, and open the tools picker: a `kairos` group with the 19 tools should be listed.
   `AGENTS.md` tells Copilot about the project and to quote only numbers the tools return.

Check it outside VS Code: `matchstats mcp --help`, and `pytest stats/tests/test_mcp.py`.

## Tools (19)

| Group | Tools |
|---|---|
| Season | `list_matches`, `league_table` (actual or expected), `season_player`, `what_decided_it` |
| One match, as of any minute | `scoreboard`, `team_comparison`, `player_stats`, `top_players`, `momentum`, `moments`, `explain_moment`, `event`, `options`, `win_probability`, `pass_network`, `timeline` |
| Club mode | `tryout_board` (with `max_age`), `tryout_result`, `run_tryout` (engine only, $0, capped at 120 simulated matches per server run) |

The match tools take `match_id` and an optional `minute`: the match is replayed through the stats layer up to that
minute, so "who was on top at 80'?" is answered with what was known at 80', never with the result.

## Demo script (about 90 seconds of video)

Run these in Agent mode, in this order. The answers below are what the tools return for this repo's data.

1. **The table that lies.** *"Who is top of the Meridian League, and who should be on chance quality?"*
   Copilot calls `league_table` twice. Kingsport lead with 31 points but only 24.73 expected; on expected points
   Crowmoor (23.69) are above Riverside (22.99), who are second in the real table.

2. **No future knowledge.** *"In riv-kin-1119, who was on top at the 80th minute, and did it hold?"*
   `momentum` with `minute: 80` says Kingsport, swinging their way, at 1-1; then `scoreboard` at full time shows the
   90+6' penalty that won it 1-2. Point out that the first answer could not see the ending.

3. **What decided it.** *"What decided Elmhurst v Kingsport on matchday 4?"*
   `what_decided_it`: Duarte's 9th-minute goal. The match engine froze the match at that shot and played it out 20
   times each way: as it happened Kingsport won 80% of the futures and Elmhurst 5%; with the shot saved, Elmhurst
   win 40% and Kingsport 35% (margin of error ±22 points).

4. **Recruitment, live.** Type `/scout` and answer *"Riverside striker, 23 or younger"* (the prompt file in
   `.github/prompts/scout.prompt.md`). Copilot calls `tryout_board` with `max_age: 23`: Oisín Mulcahy (Ashvale, 23)
   is pre-simulated at +0.29 expected points per match over Kwame Mensah, but with a ±0.38 margin, so the Scout
   agent's verdict is "shortlist: too close to call". Copilot then runs `run_tryout` for Théo Guillot (Northgate,
   22) on the engine (6 matches each way, under a minute on a laptop) and recommends with the margins.

5. **From data to code.** Type `/matchday-report` with `7`. Copilot gathers the matchday with the tools and creates
   `docs/reports/matchday-7.md`, every number from a tool result. Show the file and the tool calls in the chat.

Tip for the recording: expand the tool calls in the chat so judges see the MCP requests and the JSON coming back.

## Other clients
The same server works with any MCP client over stdio (`python -m matchstats.cli mcp`) or streamable HTTP
(`matchstats mcp --http --port 8766`), for example Claude Desktop or a Microsoft Agent Framework agent.
