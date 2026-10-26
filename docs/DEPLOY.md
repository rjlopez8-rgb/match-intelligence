# Deploying Kairos

![Architecture](img/architecture.svg)

The Azure and GitHub resource names below keep the repo's name (`match-intelligence`): renaming deployed resources would break them.

What runs where:

| Piece | Where | Cost |
|---|---|---|
| Website: season hub, recaps, the 3D broadcast for 56 matches (~98 MB, 74 files) | Azure Static Web Apps, **Free** plan (250 MB per environment, 15,000 files, 100 GB bandwidth a month) | $0 |
| Club mode's live try-outs (match engine only, no AI) | Azure Container Apps, Consumption, scale to zero, 1 replica | $0 within the monthly free grant |
| AI desk, recaps, Scout | Microsoft Foundry (`gpt-5-mini`), run from a laptop; the results are committed JSON, so the site never calls a model | Only when someone runs a paid `matchagents` command (it asks first) |
| Builds | GitHub Actions: CI on every push, the site build on Linux, the try-out image to GitHub's container registry | $0 (public repo) |

The site build re-plays every season match with tracking on Linux (Ubuntu 24.04, numpy 2.5.3, the platform the season
was generated on) and checks each one event for event against the committed season; any difference stops the build.

## One-time setup (Ryan: subscription owner and repo admin, ~30 minutes)

Run in PowerShell with the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), signed in to the
hackathon subscription. Every command says what it costs.

```powershell
az login
az account show --query name -o tsv                                   # the hackathon subscription? ($0)
az group create -n match-intelligence -l westus3                      # resource group ($0)
```

### 1. The website (Static Web Apps, Free)

```powershell
az staticwebapp create -n match-intelligence -g match-intelligence -l westus2 --sku Free        # $0
az staticwebapp secrets list -n match-intelligence -g match-intelligence --query properties.apiKey -o tsv   # the deploy token
az staticwebapp show -n match-intelligence -g match-intelligence --query defaultHostname -o tsv            # the site's address
```

In GitHub: the repo > **Settings > Secrets and variables > Actions > New repository secret**:
`AZURE_STATIC_WEB_APPS_API_TOKEN` = the token. Never paste the token anywhere else; it only lives in that secret.

Then **Actions > Site > Run workflow**. The first run takes about 20 minutes (most of it re-playing the season;
later runs reuse the cache and take about 5). Open `https://<the site's address>`.

### 2. Live try-outs (Container Apps)

The **Try-out server image** workflow builds the image on every push that touches the engine. After its first run:
GitHub > your profile > **Packages > match-intelligence-tryouts > Package settings > Change visibility > Public**
(public images are free to pull; the image holds only the engine and simulated data).

```powershell
az extension add --name containerapp --upgrade                                                  # $0
az provider register --namespace Microsoft.App                                                  # $0
az containerapp env create -n mi-env -g match-intelligence -l westus3 --logs-destination none    # $0 (no Log Analytics)
az containerapp create -n mi-tryouts -g match-intelligence --environment mi-env `
  --image ghcr.io/rjlopez8-rgb/match-intelligence-tryouts:latest `
  --target-port 8765 --ingress external --min-replicas 0 --max-replicas 1 --cpu 1.0 --memory 2.0Gi `
  --env-vars MATCH_ALLOWED_ORIGIN=https://<the site's address> MATCH_TRYOUT_DAILY_MATCHES=150    # $0 at rest
az containerapp show -n mi-tryouts -g match-intelligence --query properties.configuration.ingress.fqdn -o tsv
```

In GitHub: **Settings > Secrets and variables > Actions > Variables > New repository variable**:
`TRYOUT_API_URL` = `https://<that fqdn>`. Run the **Site** workflow again so the 3D page's Club mode uses it.
Check: `https://<fqdn>/health` answers `{"ok": true, ...}` (the first request after a quiet spell takes ~10-20 s
while a replica starts).

To roll out a new image: `az containerapp update -n mi-tryouts -g match-intelligence --image ghcr.io/rjlopez8-rgb/match-intelligence-tryouts:<commit sha>`.

### Why it should stay at $0

- **Static Web Apps Free**: no charges; the site is ~98 MB of the 250 MB allowed.
- **Container Apps**: the monthly free grant is 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests per
  subscription. Scaled to zero, nothing is charged. A simulated match takes ~30 s of 1 vCPU, so the daily cap of 150
  matches is at most 4,500 vCPU-s and 9,000 GiB-s a day: about 135,000 and 270,000 in a month even if it were used up
  every day, inside the grant.
- One try-out runs at a time, at most 24 matches each way, 1 replica.
- **Caveat:** the daily counter lives in the container, so it starts again when the app scales to zero and back.
  The budget alerts on the subscription (`hackathon` $25, `hackathon-guard` $150) are the backstop. To switch live
  try-outs off at once: `az containerapp update -n mi-tryouts -g match-intelligence --set-env-vars MATCH_TRYOUT_DAILY_MATCHES=0`
  (the server then refuses new simulations with a message; pre-simulated players keep working, they are in the site).
  Or remove public access entirely: `az containerapp ingress disable -n mi-tryouts -g match-intelligence`.

## Keeping the demo up until Nov 10

- Nothing on the site calls a model, so it costs nothing to leave up. Model spend happens only when someone runs a
  paid `matchagents` command on a laptop.
- After committing new AI results (`editions-live.json`, `recaps-live.json`), push: the Site workflow rebuilds and
  redeploys with them.
- If the Site workflow ever stops with "the re-played match differs", the build machine no longer reproduces the
  season (a new numpy or runner image). Pin the runner (`ubuntu-24.04` is already pinned) and numpy (2.5.3), or ask
  for help before re-simulating anything.

Sources: [Static Web Apps quotas](https://learn.microsoft.com/azure/static-web-apps/quotas),
[Container Apps billing](https://learn.microsoft.com/azure/container-apps/billing).
