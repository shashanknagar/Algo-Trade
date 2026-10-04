"""One-time (re-runnable) backfill: pulls full daily-bar history since 2020 for the
watchlist.

Source per instrument:
  - NIFTY / BANKNIFTY: NSE's free public index archive (nse_index_archive.py) --
    Groww's get_historical_candles was empirically confirmed "Access forbidden" on
    this account's plan (see README "Known unknowns"), so this is the real working
    path, not a fallback of convenience.
  - Anything else (e.g. SENSEX, a BSE index): Groww's historical API, which will
    currently fail the same way until either the account is upgraded or a free BSE
    index archive is found -- logged as a gap, not fatal to the run.

Resumable: for NSE-sourced symbols, already-backfilled dates are skipped on re-run
(checked against ohlc_bars) so re-running after adding a new watchlist entry doesn't
re-fetch years of NSE archive files that are already in the DB.

Not scheduled -- run manually once after setup, and again whenever a new instrument is
added to config/instruments.yaml:

    python -m algotrade.backfill_daily
"""
from __future__ import annotations

from datetime import datetime, timedelta

from . import db
from . import instruments as instruments_mod
from . import nse_index_archive
from .config import load_config
from .groww_client import get_api_client
from .historical import coerce_time, fetch_daily_candles
from .logging_utils import capture_run, get_logger
from .market_calendar import IST, now_ist

JOB_NAME = "daily_backfill"
BACKFILL_START = datetime(2020, 1, 1, tzinfo=IST)
DEFAULT_WATCHLIST_PATH = "config/instruments.yaml"


def _last_backfilled_date(conn, trading_symbol: str) -> datetime | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(time) FROM ohlc_bars WHERE trading_symbol = %s AND interval = '1day' "
            "AND source IN ('groww_backfill', 'nse_index_archive')",
            (trading_symbol,),
        )
        (result,) = cur.fetchone()
    return result


def _backfill_from_nse(conn, entry: dict, end: datetime, logger) -> int:
    trading_symbol = entry["trading_symbol"]
    last = _last_backfilled_date(conn, trading_symbol)
    start = (last + timedelta(days=1)) if last else BACKFILL_START
    if start.date() > end.date():
        logger.info("%s: already up to date (last=%s), nothing to fetch", trading_symbol, last)
        return 0

    logger.info("Backfilling %s from NSE archive: %s to %s", trading_symbol, start.date(), end.date())
    candles = nse_index_archive.fetch_range(trading_symbol, start, end)
    rows = [
        {
            "time": c["time"],
            "trading_symbol": trading_symbol,
            "exchange": entry["exchange"],
            "segment": entry["segment"],
            "interval": "1day",
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c["volume"],
            "oi": c["oi"],
            "source": "nse_index_archive",
        }
        for c in candles
    ]
    written = db.upsert_ohlc_bars(conn, rows)
    logger.info("%s: %d daily bars written from NSE archive", trading_symbol, written)
    return written


def _backfill_from_groww(conn, groww, entry: dict, end: datetime, logger) -> int:
    trading_symbol = entry["trading_symbol"]
    if not entry.get("groww_symbol"):
        logger.warning("%s: no groww_symbol resolved, skipping Groww fallback", trading_symbol)
        return 0
    logger.info("Backfilling %s from Groww: %s to %s", trading_symbol, BACKFILL_START.date(), end.date())
    try:
        candles = fetch_daily_candles(
            groww,
            groww_symbol=entry["groww_symbol"],
            exchange=entry["exchange"],
            segment=entry["segment"],
            start_time=BACKFILL_START.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception:
        logger.exception(
            "Groww backfill failed for %s (expected if this account's plan doesn't "
            "include historical data -- see README)", trading_symbol,
        )
        return -1  # signals "gap", distinct from 0 rows written cleanly

    rows = [
        {
            "time": coerce_time(c["time"]),
            "trading_symbol": trading_symbol,
            "exchange": entry["exchange"],
            "segment": entry["segment"],
            "interval": "1day",
            "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"],
            "volume": c["volume"], "oi": c["oi"],
            "source": "groww_backfill",
        }
        for c in candles
    ]
    written = db.upsert_ohlc_bars(conn, rows)
    logger.info("%s: %d daily bars written from Groww", trading_symbol, written)
    return written


def run(watchlist_path: str = DEFAULT_WATCHLIST_PATH) -> None:
    logger = get_logger(JOB_NAME)
    config = load_config()
    master = instruments_mod.load_instrument_master(force_refresh=True)
    watchlist = instruments_mod.load_watchlist(watchlist_path)
    resolved, unresolved = instruments_mod.resolve_watchlist(watchlist, master)
    for entry in unresolved:
        logger.warning("Unresolved watchlist entry: %s", entry)

    end = now_ist()
    groww = None  # only authenticate lazily if any entry actually needs the Groww path

    with db.get_connection(config) as conn:
        with capture_run(conn, JOB_NAME, logger) as stats:
            for entry in resolved:
                trading_symbol = entry["trading_symbol"]
                if trading_symbol in nse_index_archive.INDEX_NAME_MAP:
                    stats["rows_written"] += _backfill_from_nse(conn, entry, end, logger)
                else:
                    if groww is None:
                        groww = get_api_client(config)
                    written = _backfill_from_groww(conn, groww, entry, end, logger)
                    if written < 0:
                        stats["gaps_detected"] += 1
                    else:
                        stats["rows_written"] += written


if __name__ == "__main__":
    run()
