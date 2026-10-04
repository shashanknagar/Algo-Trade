"""Entry point for the live paper-trading loop.

NO ORDERS ARE PLACED. Order methods are structurally blocked (no_trade_guard.py);
this records recommendations to the paper_trades table and captures option chain
snapshots, nothing more.

Usage:
    python scripts/run_live_paper.py            # run until market close
    python scripts/run_live_paper.py --once     # single cycle (for testing wiring)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algotrade import live_paper_loop
from algotrade.config import load_config
from algotrade.groww_client import get_api_client
from algotrade.no_trade_guard import BLOCKED_METHODS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = parser.parse_args()

    print("=" * 72)
    print("PAPER TRADING ONLY -- no real orders can be placed.")
    print(f"Structurally blocked API methods: {', '.join(BLOCKED_METHODS)}")
    print("=" * 72)

    config = load_config()
    groww = get_api_client(config)
    live_paper_loop.run(config, groww, once=args.once)


if __name__ == "__main__":
    main()
