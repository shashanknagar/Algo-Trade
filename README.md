# Algo-Trade: Phase 0 Data Capture

Personal India algo-trading research platform. This is **Phase 0 only**: the
capture-and-persist loop that gets market data flowing into a local, durable
TimescaleDB, running unattended every trading day. No modeling/signal code lives here
yet -- see the full plan at `C:\Users\shash\.claude\plans\i-want-to-create-giggly-blanket.md`
for the complete architecture and roadmap (Phases 1-5).

> **On the `growwapi` SDK**: its published docs, its bundled README, and its actual
> 1.5.0 source code all disagree with each other in places (e.g. the docs describe a
> `subscribe_live_data` method that does not exist in the installed package, and a
> TOTP-only auth flow when the SDK actually supports two key types). The code in this
> repo was written and corrected against the **actual installed source** at
> `<site-packages>/growwapi/groww/{client,feed,constants}.py`, not the docs. If Groww
> ships a new SDK version, diff those files again before trusting this code unchanged.

## Your account's real API access (confirmed empirically, 2026-09-04)

Groww's public docs describe one paid plan (₹499+taxes/month, bundling everything).
This account's actual entitlements, found by calling every read-only method directly
(`scripts/probe_api_capabilities.py`) rather than trusting docs:

| Works for free | Needs the paid plan ("Access forbidden" today) |
|---|---|
| `get_user_profile`, `get_holdings_for_user`, `get_positions_for_user`, `get_available_margin_details`, `get_order_list` | `get_historical_candles` (daily **and** intraday) |
| `get_all_instruments`, `get_instrument_by_groww_symbol` | `get_quote`, `get_ltp`, `get_ohlc` (REST snapshots) |
| **Live WebSocket index feed** (`subscribe_index_value`/`get_index_value`) -- confirmed real NIFTY/BANKNIFTY/SENSEX ticks flowing through the actual `capture_intraday.py` code into Postgres | `get_expiries` (options chain) |

Net effect: **live intraday capture works today at zero extra cost. Historical
daily-bar data does not** -- `backfill_daily.py`/`capture_eod.py` can't use Groww for
that. See the NSE archive fallback below.

## What this does

- **`algotrade.backfill_daily`** -- one-time (re-runnable) pull of full daily-bar
  history since 2020 for the watchlist. NIFTY/BANKNIFTY come from NSE's free public
  index archive (`nse_index_archive.py`); anything else falls back to Groww's
  historical API (currently blocked on this account -- logged as a gap, not fatal).
- **`algotrade.capture_eod`** -- runs daily after market close, pulls/reconciles the
  last few days of daily bars via the same source split (so a missed prior run
  self-heals, and so does a not-yet-published NSE archive file for "today").
- **`algotrade.capture_intraday`** -- runs daily from just before market open to close,
  streams the live WebSocket feed (works today) and writes ticks continuously.

All writes are idempotent upserts keyed on `(instrument, time)`, and every run logs to
both `logs/<job>.log` and a `capture_runs` row in Postgres, so a missed or failed day is
visible instead of silent.

### On SENSEX specifically

NIFTY and BANKNIFTY have a confirmed working free historical source (NSE's archive).
**SENSEX does not** -- BSE has no documented/discoverable free historical index
archive (several plausible API endpoints were tried and all failed). SENSEX's *live*
feed still works fine via Groww; only its historical daily-bar backfill is unresolved.
Options if this matters: (a) find BSE's actual endpoint (would need inspecting real
browser network traffic on bseindia.com, which wasn't possible from here), (b) drop
SENSEX from the watchlist in favor of NSE-only indices, or (c) upgrade Groww's plan,
which unlocks `get_historical_candles` for any instrument including SENSEX.

## Prerequisites

1. **Docker Desktop** -- installed and running. If Docker's engine ever won't start,
   check `wsl --status`: this machine needed the Windows "Virtual Machine Platform"
   and "Windows Subsystem for Linux" optional features enabled (`dism.exe /online
   /enable-feature /featurename:VirtualMachinePlatform /all /norestart`, same for
   `Microsoft-Windows-Subsystem-Linux`, then a real restart) before WSL2 -- and
   therefore Docker's backend -- would start. Note: `Win32_Processor.
   VirtualizationFirmwareEnabled` via WMI/PowerShell is known-unreliable and may
   report `False` even when VT-x is genuinely on (Task Manager's Performance tab is
   the trustworthy check for that specifically).
2. **Python 3.11+** -- already installed (3.13 detected).
3. **A Groww API key/secret pair**, from the Groww Cloud API Keys page. Two possible
   types, both handled automatically by `groww_client.py`:
   - **TOTP-type**: a short `api_key` plus a base32 `secret` (rotating code generated
     via `pyotp`).
   - **Approval-type**: a long-lived JWT `api_key` (starts `eyJ...`) plus a `secret`
     passed straight through with no TOTP. This is what this account actually has.

## Setup

### 1. Set credentials as Windows environment variables

**Not a `.env` file** -- per the plan, credentials must come from the OS environment so
they're never accidentally committed or logged.

```powershell
setx GROWW_API_KEY "your-api-key"
setx GROWW_API_SECRET "your-secret"
setx POSTGRES_PASSWORD "algotrade_local_dev_password"
```

`POSTGRES_PASSWORD` isn't a credential to look up anywhere -- it's a placeholder
already baked into `docker-compose.yml`'s default for this local-only DB. Use the
value above verbatim unless you've deliberately changed the compose file.

`setx` only takes effect in **new** processes -- a plain new terminal window is enough
if you're running these scripts yourself; if an assistant/session is running them on
your behalf, that process needs to restart too (reloading a VS Code window, for
example) before it can see the new values. If it still can't after a restart, the
values may have been set as **machine-level** (System) variables (via the GUI's
"System variables" pane) rather than user-level -- check both:
`HKCU:\Environment` and `HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment`.

### 2. Start the database

```powershell
docker compose up -d
```

This runs TimescaleDB + pgvector in one container (`algotrade-timescaledb`), with a
4GB/4-core resource cap sized for this machine's 16GB RAM. First pull is a large image
(~900MB) and can take several minutes.

### 3. Install the Python package and apply the schema

```powershell
pip install -e .
python scripts/apply_migrations.py
```

### 4. Verify everything is wired up

```powershell
python scripts/check_env.py
```

Confirms env vars are set, Postgres is reachable, and Groww authentication actually
works end-to-end. Then check exactly what your account can access:

```powershell
python scripts/probe_api_capabilities.py
```

### 5. Run the one-time historical backfill

```powershell
python -m algotrade.backfill_daily
```

Pulls ~6 years of daily bars for NIFTY/BANKNIFTY from NSE's archive (~1600 requests,
politely rate-limited at ~3/sec -- takes several minutes; safe to re-run, it resumes
from the last date already in the DB rather than re-fetching everything). Check
`logs/daily_backfill.log` afterward for any unresolved watchlist entries or gaps.

### 6. Register the daily unattended jobs

```powershell
# In an elevated PowerShell prompt:
.\scripts\register_task_scheduler.ps1
```

This creates two Windows Task Scheduler tasks (`AlgoTrade-IntradayCapture`,
`AlgoTrade-EODCapture`) configured to run whether you're logged on or not, survive
sleep/wake, and retry on failure -- see that script's header comment for details. It
will prompt for your Windows password (required by Task Scheduler to run unattended).

## Verifying a full day worked

Per the plan's Phase 0 verification criterion: after a full trading day, check:

```powershell
docker exec -it algotrade-timescaledb psql -U algotrade -d algotrade -c "SELECT job_name, started_at, status, rows_written, gaps_detected FROM capture_runs ORDER BY started_at DESC LIMIT 10;"
```

Both jobs should show `status = success` with `rows_written > 0`. `gaps_detected > 0`
on `eod_capture`/`daily_backfill` is currently *expected* for SENSEX (see above) --
check `logs/eod_capture.log` / `logs/daily_backfill.log` to confirm a gap is that known
SENSEX limitation and not something new.

## Project layout

```
config/instruments.yaml   Phase 0 watchlist (NIFTY 50, BANK NIFTY, SENSEX)
db/migrations/            SQL schema, applied by scripts/apply_migrations.py
src/algotrade/            The package: config, db, groww_client, instruments,
                           historical, nse_index_archive, capture_eod,
                           capture_intraday, backfill_daily
scripts/                  One-off/setup scripts (migrations, env check, API
                           capability probe, task scheduler, logic validation)
data/                     Downloaded instrument master cache (gitignored, regenerable)
logs/                     Per-job log files (gitignored)
```
