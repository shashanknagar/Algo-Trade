"""Validates the parts of the system that don't need Postgres or real Groww
credentials: instrument-master resolution (a real, unauthenticated network call to
Groww's public instrument CSV) and the pure parsing/buffering logic in historical.py
and capture_intraday.py, exercised with fabricated data.

Not a permanent test suite -- a one-off validation script for the pre-Docker,
pre-credentials stage of setup. Safe to delete once real end-to-end validation
(scripts/check_env.py + a live capture run) is possible.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algotrade import instruments as instruments_mod
from algotrade.capture_intraday import TickBuffer, _extract_ltp
from algotrade.historical import _normalize, coerce_time

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


def validate_instrument_resolution() -> None:
    print("\n=== Instrument master resolution (real network call, no auth) ===")
    master = instruments_mod.load_instrument_master(force_refresh=True)
    check("Instrument master downloaded", len(master) > 0, f"got {len(master)} rows")
    if master:
        sample = master[0]
        for col in ("exchange", "segment", "trading_symbol", "exchange_token", "groww_symbol"):
            check(f"Master CSV has column '{col}'", col in sample, f"columns: {list(sample.keys())}")

    watchlist = instruments_mod.load_watchlist("config/instruments.yaml")
    resolved, unresolved = instruments_mod.resolve_watchlist(watchlist, master)
    print(f"Resolved: {len(resolved)}/{len(watchlist)}")
    for entry in resolved:
        print(f"  OK  {entry['label']:12s} -> exchange_token={entry['exchange_token']} groww_symbol={entry['groww_symbol']}")
    for entry in unresolved:
        print(f"  MISS {entry.get('label', entry)}: no match for "
              f"exchange={entry['exchange']} segment={entry['segment']} trading_symbol={entry['trading_symbol']}")
        # Help find the right spelling instead of just failing silently.
        candidates = [
            row for row in master
            if entry["trading_symbol"].upper() in (row.get("name") or "").upper()
            or entry["trading_symbol"].upper() in (row.get("trading_symbol") or "").upper()
        ][:5]
        if candidates:
            print("       possible matches in master CSV:")
            for c in candidates:
                print(f"         {c.get('trading_symbol')!r} / {c.get('groww_symbol')!r} "
                      f"(exchange={c.get('exchange')}, segment={c.get('segment')}, name={c.get('name')})")
    check("All watchlist entries resolved", len(unresolved) == 0,
          f"{len(unresolved)} unresolved -- see candidates above to fix config/instruments.yaml")


def validate_historical_normalize() -> None:
    print("\n=== historical.py: candle normalization (fabricated data) ===")
    dict_response = {
        "candles": [
            {"timestamp": "2026-08-31 00:00:00", "open": 100, "high": 105, "low": 99, "close": 104, "volume": 1000, "oi": None},
        ]
    }
    rows = _normalize(dict_response)
    check("Dict-shaped response normalizes to 1 row", len(rows) == 1, str(rows))
    if rows:
        check("Dict-shaped row has correct close", rows[0]["close"] == 104, str(rows[0]))

    positional_response = {"candles": [["2026-08-31 00:00:00", 100, 105, 99, 104, 1000]]}
    rows2 = _normalize(positional_response)
    check("Positional-shaped response normalizes to 1 row", len(rows2) == 1, str(rows2))
    if rows2:
        check("Positional-shaped row has correct close", rows2[0]["close"] == 104, str(rows2[0]))

    dt = coerce_time("2026-08-31 09:15:00")
    check("coerce_time parses 'YYYY-MM-DD HH:MM:SS'", dt.year == 2026 and dt.month == 8 and dt.day == 31, str(dt))

    dt_epoch = coerce_time(1798700100000)  # ms epoch
    check("coerce_time parses ms-epoch", dt_epoch.year >= 2026, str(dt_epoch))


def validate_intraday_parsing() -> None:
    print("\n=== capture_intraday.py: LTP extraction + buffer logic (fabricated data) ===")
    check("_extract_ltp reads 'ltp' key", _extract_ltp({"ltp": 24500.5}) == 24500.5)
    check("_extract_ltp reads 'value' key", _extract_ltp({"value": 24500.5}) == 24500.5)
    check("_extract_ltp reads 'last_price' key", _extract_ltp({"last_price": "24500.5"}) == 24500.5)
    check("_extract_ltp returns None for unrecognized dict", _extract_ltp({"foo": 1}) is None)
    check("_extract_ltp handles bare numeric payload", _extract_ltp(24500.5) == 24500.5)

    buf = TickBuffer()
    buf.add({"trading_symbol": "NIFTY", "ltp": 1})
    buf.add({"trading_symbol": "NIFTY", "ltp": 2})
    drained = buf.drain()
    check("TickBuffer drains all added rows", len(drained) == 2, str(drained))
    check("TickBuffer is empty after drain", buf.drain() == [])

    buf.requeue(drained)
    buf.add({"trading_symbol": "NIFTY", "ltp": 3})
    redrained = buf.drain()
    check(
        "TickBuffer.requeue puts failed rows back at the front (retry order preserved)",
        [r["ltp"] for r in redrained] == [1, 2, 3],
        str(redrained),
    )


def main() -> int:
    validate_instrument_resolution()
    validate_historical_normalize()
    validate_intraday_parsing()

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("All logic checks passed. (DB-integrated validation still needs Docker's engine up.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
