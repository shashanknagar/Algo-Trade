"""EOD job: for each watchlist instrument, pulls the trailing few days of daily candles
and upserts them.

Source per instrument (same split as backfill_daily.py):
  - NIFTY / BANKNIFTY: NSE's free public index archive.
  - Anything else: Groww's historical API (currently "Access forbidden" on this
    account's plan -- logged as a gap, not fatal).

Runs shortly after market close (see scripts/register_task_scheduler.ps1). Pulling a
trailing window rather than just "today" means a missed prior EOD run gets backfilled
automatically the next time this runs -- the permanent daily record doesn't depend on
every single EOD run having succeeded. This also covers the case where today's NSE
archive file isn't published yet at 15:45 IST -- it gets picked up by tomorrow's run.
"""
from __future__ import annotations

from datetime import timedelta

from . import db
from . import instruments as instruments_mod
from . import nse_index_archive
from .config import load_config
from .groww_client import get_api_client
from .historical import coerce_time, fetch_daily_candles
from .logging_utils import capture_run, get_logger
from .market_calendar import is_trading_day, now_ist

JOB_NAME = "eod_capture"
TRAILING_WINDOW_DAYS = 5
DEFAULT_WATCHLIST_PATH = "config/instruments.yaml"


def _eod_from_nse(entry: dict, start, end, logger) -> list[dict]:
    candles = nse_index_archive.fetch_range(entry["trading_symbol"], start, end)
    return [
        {
            "time": c["time"],
            "trading_symbol": entry["trading_symbol"],
            "exchange": entry["exchange"],
            "segment": entry["segment"],
            "interval": "1day",
            "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"],
            "volume": c["volume"], "oi": c["oi"],
            "source": "nse_index_archive",
        }
        for c in candles
    ]


def _eod_from_groww(groww, entry: dict, start, end, logger) -> list[dict] | None:
    if not entry.get("groww_symbol"):
        return None
    try:
        candles = fetch_daily_candles(
            groww,
            groww_symbol=entry["groww_symbol"],
            exchange=entry["exchange"],
            segment=entry["segment"],
            start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception:
        logger.exception("Failed fetching EOD candles for %s", entry["trading_symbol"])
        return None

    return [
        {
            "time": coerce_time(c["time"]),
            "trading_symbol": entry["trading_symbol"],
            "exchange": entry["exchange"],
            "segment": entry["segment"],
            "interval": "1day",
            "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"],
            "volume": c["volume"], "oi": c["oi"],
            "source": "groww_eod",
        }
        for c in candles
    ]


def run(watchlist_path: str = DEFAULT_WATCHLIST_PATH) -> None:
    logger = get_logger(JOB_NAME)
    if not is_trading_day():
        logger.info("Not a trading day; skipping EOD capture.")
        return

    config = load_config()
    master = instruments_mod.load_instrument_master()
    watchlist = instruments_mod.load_watchlist(watchlist_path)
    resolved, unresolved = instruments_mod.resolve_watchlist(watchlist, master)
    for entry in unresolved:
        logger.warning(
            "Unresolved watchlist entry (check config/instruments.yaml against the "
            "instrument master): %s", entry,
        )

    end = now_ist()
    start = end - timedelta(days=TRAILING_WINDOW_DAYS)
    groww = None

    with db.get_connection(config) as conn:
        with capture_run(conn, JOB_NAME, logger) as stats:
            for entry in resolved:
                if entry["trading_symbol"] in nse_index_archive.INDEX_NAME_MAP:
                    rows = _eod_from_nse(entry, start, end, logger)
                else:
                    if groww is None:
                        groww = get_api_client(config)
                    rows = _eod_from_groww(groww, entry, start, end, logger)

                if rows is None:
                    stats["gaps_detected"] += 1
                    continue
                if not rows:
                    stats["gaps_detected"] += 1
                    logger.warning(
                        "No candles returned for %s in trailing window", entry["trading_symbol"]
                    )
                    continue
                stats["rows_written"] += db.upsert_ohlc_bars(conn, rows)


if __name__ == "__main__":
    run()
