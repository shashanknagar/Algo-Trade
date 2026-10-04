"""Probes which Groww API endpoints are actually callable under the current account's
subscription by calling each read-only method and recording success/failure with
Groww's actual error message. This is how we empirically answer "what does the free
plan actually include" -- Groww's public docs only describe the paid (Rs499/month)
plan, so this account's real entitlements are determined by testing, not by reading.

Only needs GROWW_API_KEY / GROWW_API_SECRET (not POSTGRES_* -- this never touches the
DB). Nothing here places orders, modifies account state, or has a "write" side effect
-- every call is a read. Run after setting the two env vars:

    python scripts/probe_api_capabilities.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from growwapi import GrowwAPI, GrowwFeed

from algotrade.groww_client import resolve_access_token


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"FAIL: required environment variable '{name}' is not set.")
        sys.exit(1)
    return value


def try_call(results: dict, label: str, fn):
    try:
        result = fn()
        preview = repr(result)
        if len(preview) > 250:
            preview = preview[:250] + " ...(truncated)"
        print(f"[OK]   {label}\n       -> {preview}")
        results[label] = True
    except Exception as exc:
        print(f"[FAIL] {label}\n       -> {type(exc).__name__}: {exc}")
        results[label] = False


def main() -> None:
    api_key = _require("GROWW_API_KEY")
    api_secret = _require("GROWW_API_SECRET")

    print("Authenticating...")
    token = resolve_access_token(api_key, api_secret)
    groww = GrowwAPI(token)
    print("Authenticated OK.\n")

    results: dict[str, bool] = {}
    end = datetime.now()
    start = end - timedelta(days=5)

    print("=== Account / portfolio (read-only) ===")
    try_call(results, "get_user_profile()", lambda: groww.get_user_profile(timeout=10))
    try_call(results, "get_holdings_for_user()", lambda: groww.get_holdings_for_user(timeout=10))
    try_call(results, "get_positions_for_user()", lambda: groww.get_positions_for_user(timeout=10))
    try_call(results, "get_available_margin_details()", lambda: groww.get_available_margin_details(timeout=10))
    try_call(results, "get_order_list()", lambda: groww.get_order_list(page=0, page_size=5, timeout=10))

    print("\n=== Instrument master (read-only) ===")
    try_call(results, "get_all_instruments()", lambda: groww.get_all_instruments())
    try_call(
        results, "get_instrument_by_groww_symbol('NSE-NIFTY')",
        lambda: groww.get_instrument_by_groww_symbol("NSE-NIFTY"),
    )

    print("\n=== Historical candles -- what Phase 0 actually needs ===")
    try_call(
        results, "get_historical_candles(NIFTY, 1day, last 5 days)",
        lambda: groww.get_historical_candles(
            exchange=GrowwAPI.EXCHANGE_NSE, segment=GrowwAPI.SEGMENT_CASH,
            groww_symbol="NSE-NIFTY",
            start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
            candle_interval="1day", timeout=15,
        ),
    )
    try_call(
        results, "get_historical_candles(NIFTY, 5minute, last 5 days)",
        lambda: groww.get_historical_candles(
            exchange=GrowwAPI.EXCHANGE_NSE, segment=GrowwAPI.SEGMENT_CASH,
            groww_symbol="NSE-NIFTY",
            start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
            candle_interval="5minute", timeout=15,
        ),
    )

    print("\n=== Live quote / LTP / OHLC (REST, not WebSocket) ===")
    try_call(
        results, "get_quote(NIFTY)",
        lambda: groww.get_quote(trading_symbol="NIFTY", exchange=GrowwAPI.EXCHANGE_NSE,
                                 segment=GrowwAPI.SEGMENT_CASH, timeout=10),
    )
    try_call(
        results, "get_ltp(['NSE_NIFTY'])",
        lambda: groww.get_ltp(exchange_trading_symbols=("NSE_NIFTY",),
                               segment=GrowwAPI.SEGMENT_CASH, timeout=10),
    )
    try_call(
        results, "get_ohlc(['NSE_NIFTY'])",
        lambda: groww.get_ohlc(exchange_trading_symbols=("NSE_NIFTY",),
                                segment=GrowwAPI.SEGMENT_CASH, timeout=10),
    )

    print("\n=== Options chain (read-only) ===")
    try_call(
        results, "get_expiries(NIFTY)",
        lambda: groww.get_expiries(exchange=GrowwAPI.EXCHANGE_NSE, underlying_symbol="NIFTY", timeout=10),
    )

    print("\n=== Live WebSocket feed (expected to fail per your free-plan note) ===")
    def _feed_probe():
        feed = GrowwFeed(groww)
        return feed.subscribe_index_value(
            [{"exchange": "NSE", "segment": "CASH", "exchange_token": "NIFTY"}],
        )
    try_call(results, "GrowwFeed.subscribe_index_value(NIFTY)", _feed_probe)

    print("\n" + "=" * 70)
    print("SUMMARY")
    ok_count = sum(1 for v in results.values() if v)
    print(f"{ok_count}/{len(results)} calls succeeded\n")
    for label, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}")


if __name__ == "__main__":
    main()
