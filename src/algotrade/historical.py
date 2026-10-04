"""Wraps Groww's historical-candle API.

Signature confirmed directly against the installed growwapi 1.5.0 source (the published
docs disagreed with the actual code, so the real package was treated as ground truth):

    GrowwAPI.get_historical_candles(exchange, segment, groww_symbol, start_time,
                                     end_time, candle_interval, timeout=None) -> dict

Note groww_symbol (not trading_symbol) and candle_interval as a string ("1day"), not
an integer minute count.

Important correction vs. the original plan: Groww's 3-month cap applies only to
sub-daily intervals (1min-4hr). Daily and weekly candles are full history (from 2020)
-- so Phase 0 does not need an NSE/BSE bhavcopy bootstrap after all; the daily
backfill/EOD jobs pull directly from this API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .market_calendar import IST


def fetch_daily_candles(
    groww,
    *,
    groww_symbol: str,
    exchange: str,
    segment: str,
    start_time: str,
    end_time: str,
) -> list[dict]:
    response = groww.get_historical_candles(
        exchange=exchange,
        segment=segment,
        groww_symbol=groww_symbol,
        start_time=start_time,
        end_time=end_time,
        candle_interval="1day",
    )
    return _normalize(response)


def _normalize(response: Any) -> list[dict]:
    """Response shape isn't fully confirmed beyond 'a dict with the candles in it
    somewhere' -- normalize defensively and let callers log the raw response the first
    time this returns something unexpected, rather than crashing the run."""
    candles = response
    if isinstance(response, dict):
        for key in ("candles", "candleData", "data"):
            if key in response:
                candles = response[key]
                break
    normalized: list[dict] = []
    for c in candles or []:
        if isinstance(c, dict):
            normalized.append(
                {
                    "time": c.get("timestamp") or c.get("time") or c.get("start_time"),
                    "open": c.get("open"),
                    "high": c.get("high"),
                    "low": c.get("low"),
                    "close": c.get("close"),
                    "volume": c.get("volume"),
                    "oi": c.get("oi"),
                }
            )
        else:
            # Positional form: [timestamp, open, high, low, close, volume, (oi)]
            ts, o, h, l, cl, *rest = c
            normalized.append(
                {
                    "time": ts,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": cl,
                    "volume": rest[0] if rest else None,
                    "oi": rest[1] if len(rest) > 1 else None,
                }
            )
    return normalized


def coerce_time(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 10**12 else value
        return datetime.fromtimestamp(seconds, tz=IST)
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
        except ValueError:
            return datetime.fromisoformat(value)
    raise ValueError(f"Unrecognized candle timestamp: {value!r}")
