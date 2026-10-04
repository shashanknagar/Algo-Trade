"""Backfills the NSE equity universe (all EQ-series stocks) into ohlc_bars.

This is the statistical-power fix: ~2,000 stocks/day instead of 2 indices. Run once;
it is resumable, so an interruption just means restarting it.

Usage:
    python scripts/backfill_equities.py [--start 2020-01-01]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algotrade import db, equity_bhavcopy
from algotrade.config import load_postgres_config
from algotrade.logging_utils import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    args = parser.parse_args()

    logger = get_logger("equity_backfill")
    config = load_postgres_config()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = date.today()

    logger.info("Equity backfill %s -> %s", start, end)
    with db.get_connection(config) as conn:
        total = equity_bhavcopy.backfill(conn, start, end, logger=logger)
    logger.info("Equity backfill finished: %d rows written", total)


if __name__ == "__main__":
    main()
