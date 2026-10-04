"""Day-one diagnostic for a paid Groww subscription.

Two unknowns decide whether the subscription actually pays off, and both are
answerable in minutes. Run this immediately after subscribing -- BEFORE building
anything on top of the paid data.

  Q1. Is there HISTORICAL option data (contract candles with OI)? If yes, OI/IV
      features can be backtested now. If no, they can only be accumulated forward,
      which is a ~1 year wait before anything is validatable.

  Q2. What is the real bid-ask spread on the options we'd actually trade? The VRP
      edge measured ~2 vol points per trade and dies above ~2.0-2.5 points of
      round-trip cost. If real spreads exceed that, the one validated edge in this
      project is not harvestable and the subscription does not pay for itself.

Also checks intraday candle depth (how far back 10-minute bars actually go), since
that determines how much intraday validation data exists on day one.

Nothing here places an order. Every call is a read.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from growwapi import GrowwAPI

from algotrade.groww_client import resolve_access_token


def require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        print(f"FAIL: {name} not set")
        sys.exit(1)
    return v


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def try_call(label, fn):
    try:
        r = fn()
        print(f"[OK]   {label}")
        return r
    except Exception as exc:
        print(f"[FAIL] {label}\n       {type(exc).__name__}: {exc}")
        return None


def main() -> None:
    groww = GrowwAPI(resolve_access_token(require("GROWW_API_KEY"), require("GROWW_API_SECRET")))
    print("Authenticated.")

    # ---- Q1: historical intraday depth on the INDEX -------------------------
    section("Q1a. How far back does INTRADAY (10-minute) history actually go?")
    now = datetime.now()
    for months in (1, 3, 6, 12, 24):
        start = now - timedelta(days=30 * months)
        r = try_call(
            f"10-minute candles, {months} month(s) back",
            lambda s=start: groww.get_historical_candles(
                exchange=GrowwAPI.EXCHANGE_NSE, segment=GrowwAPI.SEGMENT_CASH,
                groww_symbol="NSE-NIFTY",
                start_time=s.strftime("%Y-%m-%d %H:%M:%S"),
                end_time=(s + timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S"),
                candle_interval="10minute", timeout=20,
            ),
        )
        if r:
            candles = r.get("candles") if isinstance(r, dict) else r
            print(f"       -> {len(candles or [])} candles returned")

    # ---- Q1b: historical OPTION CONTRACT candles, and do they carry OI? ------
    section("Q1b. Historical OPTION-CONTRACT candles -- do they exist, and include OI?")
    print("If this works, OI/IV features can be BACKTESTED now instead of accumulated")
    print("forward for a year. This is the single highest-value unknown.\n")

    expiries = try_call(
        "get_expiries(NIFTY)",
        lambda: groww.get_expiries(exchange=GrowwAPI.EXCHANGE_NSE, underlying_symbol="NIFTY", timeout=15),
    )
    print(f"       expiries: {str(expiries)[:300]}")

    chain = try_call(
        "get_option_chain(NIFTY, nearest expiry)",
        lambda: groww.get_option_chain(
            exchange=GrowwAPI.EXCHANGE_NSE, underlying="NIFTY",
            expiry_date=(expiries or {}).get("expiries", [None])[0], timeout=20,
        ) if expiries else None,
    )
    if chain:
        print(f"       chain keys: {list(chain.keys()) if isinstance(chain, dict) else type(chain)}")
        print(f"       sample: {str(chain)[:600]}")

    # ---- Q2: the cost hurdle ------------------------------------------------
    section("Q2. REAL BID-ASK SPREAD vs the ~2 vol-point VRP hurdle")
    print("The VRP edge is ~2 vol points/trade gross. A 4-leg iron condor pays the")
    print("spread 4 times on entry and (unless held to expiry) 4 more on exit.\n")
    print("Look at the bid/ask on the ATM options in the chain output above and compute:")
    print("    spread_pct = (ask - bid) / mid  for each leg")
    print("If total round-trip cost exceeds ~2 vol points of premium, the edge is not")
    print("harvestable and the subscription does not pay for itself on VRP grounds.")

    section("VERDICT CHECKLIST")
    print("  [ ] Intraday history >= 3 months?      -> enables intraday validation")
    print("  [ ] Option candles include OI?          -> enables BACKTESTING OI features")
    print("  [ ] Round-trip spread < ~2 vol points?  -> VRP edge is harvestable")
    print("\nIf all three are NO, the subscription buys live data you cannot yet validate.")
    print("If the OI one is YES, it is worth it on that alone.")


if __name__ == "__main__":
    main()
