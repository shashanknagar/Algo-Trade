"""Captures the full option chain for index underlyings, and persists it.

PURPOSE: accumulate the option dataset this project has never had. Groww's option
chain is a LIVE SNAPSHOT ONLY -- there is no historical chain API -- so the only way
to ever backtest OI/IV/Greeks features is to start recording them now. Every minute
captured is data that cannot be recovered later.

RESPONSE SHAPE IS NOT ASSUMED. The SDK's published docs have contradicted its actual
behaviour repeatedly in this project, so the normalizer below tries several plausible
shapes and stores the RAW payload of the first snapshot of each run in the `raw`
column. After the first real run, inspect that column and tighten the normalizer to
match reality rather than guesswork.

This module NEVER places an order -- see no_trade_guard.py, which structurally blocks
that on the API class itself.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

import psycopg2.extras

from .market_calendar import IST

# Candidate keys for locating the strike list inside the response.
_CHAIN_LIST_KEYS = ("option_chain", "optionChain", "data", "strikes", "chain", "records")
# Candidate keys for the call/put sub-objects within one strike row.
_CALL_KEYS = ("call", "CE", "ce", "callOption", "call_option")
_PUT_KEYS = ("put", "PE", "pe", "putOption", "put_option")


def _first_present(d: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _num(d: Any, *keys: str) -> float | None:
    """Pulls the first present numeric field from a set of candidate names."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return None


def _leg_to_row(leg: Any, strike: float, option_type: str, underlying: str,
                expiry: date | None, spot: float | None, ts: datetime) -> dict | None:
    if not isinstance(leg, dict):
        return None
    greeks = leg.get("greeks") if isinstance(leg.get("greeks"), dict) else leg
    return {
        "time": ts,
        "underlying": underlying,
        "expiry_date": expiry,
        "strike_price": strike,
        "option_type": option_type,
        "ltp": _num(leg, "ltp", "last_price", "lastPrice", "close", "price"),
        "oi": _num(leg, "oi", "open_interest", "openInterest"),
        "oi_change": _num(leg, "oi_change", "changeinOpenInterest", "oiChange", "change_in_oi"),
        "volume": _num(leg, "volume", "totalTradedVolume", "traded_volume", "ttv"),
        "iv": _num(leg, "iv", "implied_volatility", "impliedVolatility"),
        "delta": _num(greeks, "delta"),
        "gamma": _num(greeks, "gamma"),
        "theta": _num(greeks, "theta"),
        "vega": _num(greeks, "vega"),
        "rho": _num(greeks, "rho"),
        "bid": _num(leg, "bid", "bid_price", "bidPrice", "best_bid"),
        "ask": _num(leg, "ask", "ask_price", "askPrice", "best_ask", "offer"),
        "underlying_spot": spot,
    }


def normalize_chain(response: Any, underlying: str, expiry: date | None,
                    ts: datetime | None = None) -> list[dict]:
    """Best-effort flattening of an option-chain response into one row per (strike,
    option_type). Returns [] rather than raising if the shape is unrecognized -- the
    caller stores the raw payload so the shape can be corrected afterwards."""
    ts = ts or datetime.now(IST)
    if response is None:
        return []

    body: Any = response
    spot = None
    if isinstance(response, dict):
        spot = _num(response, "underlying_spot", "spot", "underlyingValue", "last_price", "ltp")
        found = _first_present(response, _CHAIN_LIST_KEYS)
        if found is not None:
            body = found
            # Some APIs nest one level deeper: {"records": {"data": [...]}}
            deeper = _first_present(body, _CHAIN_LIST_KEYS)
            if deeper is not None:
                body = deeper

    if not isinstance(body, list):
        return []

    rows: list[dict] = []
    for entry in body:
        if not isinstance(entry, dict):
            continue
        strike = _num(entry, "strike_price", "strikePrice", "strike")
        call = _first_present(entry, _CALL_KEYS)
        put = _first_present(entry, _PUT_KEYS)

        if call is None and put is None:
            # Flat shape: one row per contract, carrying its own option_type field.
            opt_type = (
                entry.get("option_type") or entry.get("optionType") or entry.get("instrument_type")
            )
            if strike is not None and opt_type:
                row = _leg_to_row(entry, strike, str(opt_type).upper()[:2],
                                  underlying, expiry, spot, ts)
                if row:
                    rows.append(row)
            continue

        if strike is None:
            continue
        for leg, opt_type in ((call, "CE"), (put, "PE")):
            row = _leg_to_row(leg, strike, opt_type, underlying, expiry, spot, ts) if leg else None
            if row and row["ltp"] is not None:
                rows.append(row)
    return rows


def upsert_snapshots(conn, rows: list[dict], raw_sample: Any = None) -> int:
    if not rows:
        return 0
    rows = [dict(r) for r in rows]
    for r in rows:
        r.setdefault("raw", None)
    if raw_sample is not None:
        rows[0]["raw"] = json.dumps(raw_sample, default=str)[:200000]

    sql = """
        INSERT INTO option_chain_snapshots
            (time, underlying, expiry_date, strike_price, option_type, ltp, oi, oi_change,
             volume, iv, delta, gamma, theta, vega, rho, bid, ask, underlying_spot, raw)
        VALUES
            (%(time)s, %(underlying)s, %(expiry_date)s, %(strike_price)s, %(option_type)s,
             %(ltp)s, %(oi)s, %(oi_change)s, %(volume)s, %(iv)s, %(delta)s, %(gamma)s,
             %(theta)s, %(vega)s, %(rho)s, %(bid)s, %(ask)s, %(underlying_spot)s, %(raw)s)
        ON CONFLICT (underlying, expiry_date, strike_price, option_type, time) DO UPDATE SET
            ltp = EXCLUDED.ltp, oi = EXCLUDED.oi, oi_change = EXCLUDED.oi_change,
            volume = EXCLUDED.volume, iv = EXCLUDED.iv, delta = EXCLUDED.delta,
            gamma = EXCLUDED.gamma, theta = EXCLUDED.theta, vega = EXCLUDED.vega,
            rho = EXCLUDED.rho, bid = EXCLUDED.bid, ask = EXCLUDED.ask,
            underlying_spot = EXCLUDED.underlying_spot
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    return len(rows)


def nearest_expiry(groww, exchange: str, underlying: str, logger=None) -> date | None:
    """Nearest non-expired expiry for an underlying. Shape-tolerant, for the same
    reason as the chain normalizer."""
    try:
        resp = groww.get_expiries(exchange=exchange, underlying_symbol=underlying, timeout=15)
    except Exception:
        if logger:
            logger.exception("get_expiries failed for %s", underlying)
        return None

    candidates: Any = resp
    if isinstance(resp, dict):
        for k in ("expiries", "expiry_dates", "expiryDates", "data"):
            if k in resp:
                candidates = resp[k]
                break
    if not isinstance(candidates, list):
        if logger:
            logger.warning("Unrecognized expiries shape for %s: %r", underlying, str(resp)[:300])
        return None

    today = datetime.now(IST).date()
    parsed = []
    for c in candidates:
        if isinstance(c, str):
            raw = c
        elif isinstance(c, dict):
            raw = c.get("expiry_date") or c.get("expiry")
        else:
            raw = None
        if not raw:
            continue
        try:
            d = datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if d >= today:
            parsed.append(d)
    return min(parsed) if parsed else None
