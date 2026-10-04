"""IST market-hours helpers.

NSE/BSE equity cash session: 09:15-15:30 IST, Monday-Friday, minus exchange holidays.
"""
from __future__ import annotations

from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

# TODO(Phase 0 data-quality step): replace with the real NSE/BSE trading-holiday
# calendar. Until then, a missed holiday just means a capture job runs harmlessly on a
# holiday and finds/writes nothing -- it does not corrupt or lose data.
KNOWN_HOLIDAYS: set[str] = set()


def now_ist() -> datetime:
    return datetime.now(IST)


def is_trading_day(dt: datetime | None = None) -> bool:
    dt = dt or now_ist()
    if dt.weekday() >= 5:  # Saturday/Sunday
        return False
    if dt.date().isoformat() in KNOWN_HOLIDAYS:
        return False
    return True


def is_market_open(dt: datetime | None = None) -> bool:
    dt = dt or now_ist()
    if not is_trading_day(dt):
        return False
    return MARKET_OPEN <= dt.time() <= MARKET_CLOSE
