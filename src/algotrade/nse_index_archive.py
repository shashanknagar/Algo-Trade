"""Free daily index OHLC from NSE's public archive -- the fallback historical-data
source for NIFTY 50 / NIFTY BANK now that empirical testing confirmed Groww's
get_historical_candles is paid-only on this account (see project memory /
README "Known unknowns" -- the original plan assumed Groww gave free full daily
history based on its docs; that assumption didn't hold for this account).

Endpoint confirmed working directly (not documented anywhere -- found by testing):
    https://nsearchives.nseindia.com/content/indices/ind_close_all_{DDMMYYYY}.csv

One CSV per calendar day, covering every NSE index. Requires a realistic browser
User-Agent header (NSE blocks bare/no-UA requests) but no cookies/session beyond that.
Returns 404 for weekends/holidays -- treated as "no data that day", not an error.
Confirmed available back to at least 2020-01-02.

Does NOT cover SENSEX (a BSE index) -- BSE has no equivalent documented/discoverable
free archive endpoint as of this writing (see README "Known unknowns"). SENSEX's live
feed still works fine via Groww; only its historical daily-bar backfill is unresolved.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import requests

from .market_calendar import IST

ARCHIVE_URL_TEMPLATE = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{date}.csv"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}

# Maps our watchlist's trading_symbol -> the exact "Index Name" string NSE uses in
# this archive. Extend this if more NSE indices are added to config/instruments.yaml.
INDEX_NAME_MAP = {
    "NIFTY": "Nifty 50",
    "BANKNIFTY": "Nifty Bank",
    # India VIX is NOT a tradeable instrument here -- it's stored as a FEATURE input
    # (the market's implied-volatility / fear gauge). It rides along in the same daily
    # archive file, so it costs nothing extra to collect. Deliberately kept out of
    # config/instruments.yaml so no signals are generated against it.
    "INDIAVIX": "India VIX",
}

_session = requests.Session()
_session.headers.update(_HEADERS)


def fetch_index_day(date: datetime) -> dict[str, dict] | None:
    """Downloads one day's archive and returns {index_name: {open,high,low,close,
    volume}} for every index in the file. Returns None (not an error) if the archive
    doesn't exist for this date -- weekend, holiday, or before NSE started publishing
    this format."""
    url = ARCHIVE_URL_TEMPLATE.format(date=date.strftime("%d%m%Y"))
    response = _session.get(url, timeout=20)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    rows: dict[str, dict] = {}
    lines = response.text.splitlines()
    for line in lines[1:]:  # skip header
        fields = line.split(",")
        if len(fields) < 10:
            continue
        name = fields[0].strip()
        try:
            rows[name] = {
                "open": float(fields[2]) if fields[2].strip() not in ("", "-") else None,
                "high": float(fields[3]) if fields[3].strip() not in ("", "-") else None,
                "low": float(fields[4]) if fields[4].strip() not in ("", "-") else None,
                "close": float(fields[5]) if fields[5].strip() not in ("", "-") else None,
                "volume": int(float(fields[8])) if fields[8].strip() not in ("", "-") else None,
            }
        except (ValueError, IndexError):
            continue
    return rows


def fetch_range(
    trading_symbol: str, start: datetime, end: datetime, polite_delay_seconds: float = 0.3
) -> list[dict]:
    """Fetches daily OHLC for one watchlist symbol across a date range (inclusive),
    skipping weekends. One HTTP request per calendar day -- polite_delay_seconds
    throttles requests so a multi-year backfill doesn't hammer NSE's servers."""
    index_name = INDEX_NAME_MAP.get(trading_symbol)
    if index_name is None:
        raise ValueError(f"No NSE index archive mapping for {trading_symbol!r}")

    results: list[dict] = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri only; NSE archive 404s on weekends anyway
            day_data = fetch_index_day(current)
            if day_data and index_name in day_data:
                row = day_data[index_name]
                if row["close"] is not None:
                    results.append({"time": current.replace(tzinfo=IST), **row, "oi": None})
            time.sleep(polite_delay_seconds)
        current += timedelta(days=1)
    return results
