"""NSE full bhavcopy ingestion -- the whole equity universe, daily OHLCV, free.

WHY THIS EXISTS: the binding constraint on this entire project has been statistical
power, not ideas. After correcting for overlapping trades, six years on two correlated
indices yields only ~200 independent observations; detecting a realistic 3-4 point edge
needs roughly 2,800. No feature, model or strategy fixes that -- only breadth does.

This pulls ~2,000 stocks per day from the same free NSE archive already used for the
index data, which is roughly a 100x increase in setup occurrences. It is what makes
several previously-untestable ideas measurable: the volume-base breakout fired only 4
times on indices, accumulation only 16 -- because those are single-stock patterns being
asked of an average of fifty stocks.

Endpoint (verified working, ~3,500 rows/day):
    https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{DDMMYYYY}.csv

Only SERIES='EQ' rows are kept -- the regular rolling-settlement equity segment.
Everything else (bonds, ETFs, trades-to-trade) has different mechanics and would
contaminate cross-sectional tests.
"""
from __future__ import annotations

import csv
import io
import time
from datetime import date, timedelta

import psycopg2.extras
import requests

BHAV_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date}.csv"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}

_session = requests.Session()
_session.headers.update(_HEADERS)


def _num(value: str) -> float | None:
    v = (value or "").strip()
    if v in ("", "-"):
        return None
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return None


def fetch_day(day: date) -> list[dict] | None:
    """One day's full bhavcopy, EQ series only. Returns None (not an error) when the
    file doesn't exist -- weekend, holiday, or before the archive starts."""
    url = BHAV_URL.format(date=day.strftime("%d%m%Y"))
    response = _session.get(url, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    rows: list[dict] = []
    reader = csv.DictReader(io.StringIO(response.text))
    for raw in reader:
        # Column names in this file carry leading spaces -- normalize before use.
        r = { (k or "").strip(): v for k, v in raw.items() }
        if (r.get("SERIES") or "").strip() != "EQ":
            continue
        close = _num(r.get("CLOSE_PRICE"))
        if close is None:
            continue
        rows.append({
            "time": day,
            "trading_symbol": (r.get("SYMBOL") or "").strip(),
            "exchange": "NSE",
            "segment": "CASH",
            "interval": "1day",
            "open": _num(r.get("OPEN_PRICE")),
            "high": _num(r.get("HIGH_PRICE")),
            "low": _num(r.get("LOW_PRICE")),
            "close": close,
            "volume": _num(r.get("TTL_TRD_QNTY")),
            "oi": None,
            "source": "nse_bhavcopy",
        })
    return rows


def upsert(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO ohlc_bars
            (time, trading_symbol, exchange, segment, interval, open, high, low, close,
             volume, oi, source)
        VALUES
            (%(time)s, %(trading_symbol)s, %(exchange)s, %(segment)s, %(interval)s,
             %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(oi)s, %(source)s)
        ON CONFLICT (trading_symbol, segment, interval, time) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
            close = EXCLUDED.close, volume = EXCLUDED.volume, source = EXCLUDED.source,
            inserted_at = now()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=1000)
    conn.commit()
    return len(rows)


def _last_ingested(conn) -> date | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT MAX(time)::date FROM ohlc_bars WHERE source = 'nse_bhavcopy'"
        )
        (result,) = cur.fetchone()
    return result


def backfill(conn, start: date, end: date, polite_delay_seconds: float = 0.25,
             logger=None) -> int:
    """Resumable: restarts from the last ingested date rather than re-downloading."""
    last = _last_ingested(conn)
    if last and last >= start:
        start = last + timedelta(days=1)
        if logger:
            logger.info("Resuming equity backfill from %s", start)

    total = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
            try:
                rows = fetch_day(current)
            except Exception:
                if logger:
                    logger.exception("Fetch failed for %s -- continuing", current)
                rows = None
            if rows:
                written = upsert(conn, rows)
                total += written
                if logger and (current.day <= 2 or written == 0):
                    logger.info("%s: %d EQ rows (running total %d)", current, written, total)
            time.sleep(polite_delay_seconds)
        current += timedelta(days=1)
    return total
