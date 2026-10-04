"""Intraday capture: connects to Groww's live WebSocket feed at market open and streams
ticks into TimescaleDB continuously until market close.

Meant to be launched once per trading day by a Windows Task Scheduler job (see
scripts/register_task_scheduler.ps1), a few minutes before market open.

The Phase 0 watchlist (config/instruments.yaml) is NIFTY 50 / BANK NIFTY / SENSEX --
indices, not tradable equities -- so this uses GrowwFeed.subscribe_index_value() /
get_index_value(), not the equity subscribe_ltp() path. Confirmed against the installed
growwapi 1.5.0 source (the published docs disagreed with the real code in several
places, so the actual package was treated as ground truth, not the docs).

Durability design: this job runs for ~6 hours, so it deliberately does NOT hold one
long-lived DB connection or transaction for the whole session (fragile against network
hiccups over that long). Instead, ticks are buffered in memory and flushed to a fresh
short-lived connection every few seconds. If a flush fails, the batch is put back at
the front of the buffer and retried on the next cycle instead of being dropped -- so a
transient DB/network blip loses no data, only delays it.
"""
from __future__ import annotations

import threading
import time as time_module
from datetime import datetime

from . import db
from . import instruments as instruments_mod
from .config import Config, load_config
from .groww_client import get_api_client, get_feed_client
from .logging_utils import get_logger
from .market_calendar import IST, MARKET_CLOSE, is_trading_day, now_ist

JOB_NAME = "intraday_capture"
FLUSH_INTERVAL_SECONDS = 5
DEFAULT_WATCHLIST_PATH = "config/instruments.yaml"

# First N raw index-value payloads are logged in full (not just the extracted LTP) so
# the actual field names can be confirmed/corrected against a real run -- the SDK's
# parsed response shape for this feed type isn't documented anywhere consulted so far.
RAW_PAYLOAD_LOG_LIMIT = 3
_raw_payload_logged = 0


class TickBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: list[dict] = []

    def add(self, row: dict) -> None:
        with self._lock:
            self._rows.append(row)

    def drain(self) -> list[dict]:
        with self._lock:
            rows, self._rows = self._rows, []
        return rows

    def requeue(self, rows: list[dict]) -> None:
        """Puts unflushed rows back at the front so they're retried before newer ticks."""
        with self._lock:
            self._rows = rows + self._rows


def _extract_ltp(payload) -> float | None:
    if isinstance(payload, dict):
        for key in ("ltp", "value", "last_price", "index_value", "close"):
            if key in payload and payload[key] is not None:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    return None
    elif isinstance(payload, (int, float)):
        return float(payload)
    return None


def _make_callback(buffer: TickBuffer, feed, entry: dict, logger):
    def on_data(*_args, **_kwargs) -> None:
        global _raw_payload_logged
        try:
            all_values = feed.get_index_value()
            payload = (
                all_values.get(entry["exchange"], {})
                .get(entry["segment"], {})
                .get(entry["exchange_token"])
            )
            if payload is None:
                return
            if _raw_payload_logged < RAW_PAYLOAD_LOG_LIMIT:
                logger.info("Raw index_value payload for %s: %r", entry["trading_symbol"], payload)
                _raw_payload_logged += 1
            buffer.add(
                {
                    "time": datetime.now(IST),
                    "trading_symbol": entry["trading_symbol"],
                    "exchange": entry["exchange"],
                    "segment": entry["segment"],
                    "ltp": _extract_ltp(payload),
                    "bid": None,
                    "ask": None,
                    "bid_qty": None,
                    "ask_qty": None,
                    "volume": None,
                    "oi": None,
                }
            )
        except Exception:
            logger.exception("Failed buffering tick for %s", entry["trading_symbol"])

    return on_data


def _flush(config: Config, rows: list[dict], logger) -> int:
    try:
        with db.get_connection(config) as conn:
            return db.upsert_ticks(conn, rows)
    except Exception:
        logger.exception(
            "Failed to flush %d ticks to DB -- will retry next cycle", len(rows)
        )
        return 0


def run(watchlist_path: str = DEFAULT_WATCHLIST_PATH) -> None:
    logger = get_logger(JOB_NAME)
    if not is_trading_day():
        logger.info("Not a trading day; skipping intraday capture.")
        return

    config = load_config()
    master = instruments_mod.load_instrument_master()
    watchlist = instruments_mod.load_watchlist(watchlist_path)
    resolved, unresolved = instruments_mod.resolve_watchlist(watchlist, master)
    for entry in unresolved:
        logger.warning(
            "Unresolved watchlist entry, skipped for intraday capture: %s", entry
        )
    resolved = [e for e in resolved if e.get("exchange_token")]
    if not resolved:
        logger.error("No watchlist instruments resolved; nothing to capture.")
        return

    with db.get_connection(config) as conn:
        run_id = db.start_capture_run(conn, JOB_NAME)

    total_written = 0
    gaps_detected = 0

    try:
        api_client = get_api_client(config)
        feed = get_feed_client(config, api_client=api_client)
        buffer = TickBuffer()

        instrument_list = [
            {
                "exchange": entry["exchange"],
                "segment": entry["segment"],
                "exchange_token": entry["exchange_token"],
            }
            for entry in resolved
        ]
        # One shared callback per instrument (subscribe_index_value takes the whole
        # list + a single callback -- there isn't a per-instrument callback hook, so
        # each instrument gets its own subscription call to keep the callback scoped).
        for entry, instrument in zip(resolved, instrument_list):
            feed.subscribe_index_value(
                [instrument], on_data_received=_make_callback(buffer, feed, entry, logger)
            )
            logger.info("Subscribed to live index feed for %s", entry["trading_symbol"])

        stop_at = now_ist().replace(
            hour=MARKET_CLOSE.hour, minute=MARKET_CLOSE.minute, second=0, microsecond=0
        )
        while now_ist() < stop_at:
            time_module.sleep(FLUSH_INTERVAL_SECONDS)
            rows = buffer.drain()
            if not rows:
                continue
            written = _flush(config, rows, logger)
            if written < len(rows):
                buffer.requeue(rows)
            else:
                total_written += written

        # Final drain after market close so nothing buffered is left behind.
        rows = buffer.drain()
        if rows:
            total_written += _flush(config, rows, logger)

        if total_written == 0:
            gaps_detected = 1
            logger.warning(
                "No ticks captured all day -- check subscriptions/feed connectivity."
            )

        with db.get_connection(config) as conn:
            db.finish_capture_run(conn, run_id, "success", total_written, gaps_detected)
        logger.info("Intraday capture finished: rows_written=%s", total_written)

    except Exception as exc:
        logger.exception("Intraday capture failed")
        with db.get_connection(config) as conn:
            db.finish_capture_run(
                conn, run_id, "failed", total_written, gaps_detected, str(exc)
            )
        raise


if __name__ == "__main__":
    run()
