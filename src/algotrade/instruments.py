"""Loads and caches Groww's public instrument master, and resolves the configured
watchlist (config/instruments.yaml) against it to get each instrument's exchange_token
(needed by the live feed) and groww_symbol.
"""
from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from typing import Iterable

import requests
import yaml

INSTRUMENTS_CSV_URL = "https://growwapi-assets.groww.in/instruments/instrument.csv"
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "instrument_master.csv"
CACHE_MAX_AGE_SECONDS = 24 * 3600


def _download_instrument_master() -> str:
    # requests (via certifi) rather than urllib.request -- this machine's Python
    # install can't verify TLS certs through urllib's default context (Windows Store
    # Python doesn't wire up the system cert store the way requests/certifi does).
    response = requests.get(INSTRUMENTS_CSV_URL, timeout=30)
    response.raise_for_status()
    return response.text


def load_instrument_master(force_refresh: bool = False) -> list[dict]:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not force_refresh and CACHE_PATH.exists():
        age_seconds = time.time() - CACHE_PATH.stat().st_mtime
        if age_seconds < CACHE_MAX_AGE_SECONDS:
            text = CACHE_PATH.read_text(encoding="utf-8")
            return list(csv.DictReader(io.StringIO(text)))
    text = _download_instrument_master()
    CACHE_PATH.write_text(text, encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def load_watchlist(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)["instruments"]


def resolve_watchlist(
    watchlist: Iterable[dict], master: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Matches each watchlist entry (exchange, segment, trading_symbol) against the
    instrument master. Unresolved entries are returned separately (not raised) so one
    bad symbol in the watchlist doesn't crash a whole capture run -- callers log them.
    """
    index = {
        (row["exchange"], row["segment"], row["trading_symbol"]): row for row in master
    }
    resolved: list[dict] = []
    unresolved: list[dict] = []
    for entry in watchlist:
        key = (entry["exchange"], entry["segment"], entry["trading_symbol"])
        match = index.get(key)
        if match:
            resolved.append(
                {
                    **entry,
                    "exchange_token": match.get("exchange_token"),
                    "groww_symbol": match.get("groww_symbol"),
                }
            )
        else:
            unresolved.append(entry)
    return resolved, unresolved
