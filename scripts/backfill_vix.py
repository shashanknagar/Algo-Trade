"""Backfills India VIX into ohlc_bars as a FEATURE series (not a tradeable
instrument).

India VIX is the market's own implied-volatility gauge -- i.e. actual forward-looking
sentiment, not another transformation of past price. It's free, has full history, and
already rides along inside the same daily NSE archive file we download for
NIFTY/BANKNIFTY, so collecting it costs nothing extra.

It's also the closest free proxy for the information a paid option-chain subscription
would provide: VIX minus realized volatility is a variance-risk-premium signal, which
is genuinely implied-vol information rather than a price-history derivative.

Stored with trading_symbol='INDIAVIX'. Deliberately absent from
config/instruments.yaml so no trade signals are ever generated against it.

Usage:
    python scripts/backfill_vix.py [--start 2020-01-01]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algotrade import db, nse_index_archive
from algotrade.config import load_postgres_config
from algotrade.logging_utils import get_logger
from algotrade.market_calendar import IST, now_ist

SYMBOL = "INDIAVIX"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default="2020-01-01")
    args = parser.parse_args()

    logger = get_logger("vix_backfill")
    config = load_postgres_config()
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=IST)
    end = now_ist()

    with db.get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MAX(time) FROM ohlc_bars WHERE trading_symbol = %s AND interval = '1day'",
                (SYMBOL,),
            )
            (last,) = cur.fetchone()
        if last:
            from datetime import timedelta
            start = last + timedelta(days=1)
            logger.info("Resuming India VIX backfill from %s", start.date())

        if start.date() > end.date():
            logger.info("India VIX already up to date.")
            return

        logger.info("Fetching India VIX from %s to %s", start.date(), end.date())
        candles = nse_index_archive.fetch_range(SYMBOL, start, end)

        rows = [
            {
                "time": c["time"],
                "trading_symbol": SYMBOL,
                "exchange": "NSE",
                "segment": "CASH",
                "interval": "1day",
                "open": c["open"], "high": c["high"], "low": c["low"], "close": c["close"],
                "volume": None,   # VIX has no volume
                "oi": None,
                "source": "nse_index_archive",
            }
            for c in candles
        ]
        written = db.upsert_ohlc_bars(conn, rows)
        logger.info("India VIX: %d daily bars written", written)


if __name__ == "__main__":
    main()
