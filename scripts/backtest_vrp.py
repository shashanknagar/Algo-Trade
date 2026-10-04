"""Backtests variance-risk-premium harvesting variants, train vs held-out test.

Compares, in order of increasing risk control:
  1. Naked (uncapped) -- included specifically to show the tail that makes this
     dangerous, not as a recommendation.
  2. Defined-risk cap -- models a credit spread's maximum loss.
  3. + entry filter on the OBSERVABLE premium (VIX minus TRAILING realized vol).
  4. + crash-onset filter (skip when VIX is spiking).

All P&L is in vol points, with no costs deducted -- see vrp_strategy.py's header for
what that does and does not represent. The realistic capture after spreads, slippage
and margin will be a fraction of what is shown here.

Usage:
    python scripts/backtest_vrp.py [--cutoff YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from algotrade import db, ml_dataset, vrp_strategy
from algotrade.config import load_postgres_config
from algotrade.vrp_strategy import StrategyConfig

VARIANTS = [
    StrategyConfig(max_loss_vol_points=None, label="1. naked (uncapped) -- DANGEROUS"),
    StrategyConfig(max_loss_vol_points=5.0, label="2. defined-risk cap (-5 vol pts)"),
    StrategyConfig(max_loss_vol_points=5.0, min_observable_premium=0.0,
                   label="3. + only when IV > trailing RV"),
    StrategyConfig(max_loss_vol_points=5.0, min_observable_premium=0.0, max_vix_spike_5d=0.20,
                   label="4. + skip when VIX spiking >20% in 5d"),
]


def report(df: pd.DataFrame, period_label: str) -> None:
    print(f"\n{'=' * 78}\n{period_label}\n{'=' * 78}")
    rows = []
    for cfg in VARIANTS:
        trades = vrp_strategy.simulate(df, cfg)
        rows.append(vrp_strategy.summarize(trades, cfg))
    out = pd.DataFrame(rows)
    if out.empty or "n_trades" not in out:
        print("  no trades")
        return
    print(out.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=str, default="2024-01-01")
    args = parser.parse_args()
    cutoff = date.fromisoformat(args.cutoff)

    pd.set_option("display.width", 250)
    config = load_postgres_config()
    with db.get_connection(config) as conn:
        nifty = ml_dataset._load_full_ohlc(conn, "NIFTY")
        vix = ml_dataset._load_vix(conn)

    df = vrp_strategy.build_vol_frame(nifty, vix)
    print(f"Aligned NIFTY + India VIX: {len(df)} days, {df['date'].min()} to {df['date'].max()}")
    print(f"Holding period: {vrp_strategy.DEFAULT_HORIZON_DAYS} trading days, "
          f"non-overlapping (one position at a time)")
    print("P&L in VOL POINTS. No transaction costs, spreads or margin costs deducted.")

    report(df, "FULL SAMPLE")
    report(df[df["date"] < cutoff], f"TRAIN (before {cutoff})")
    report(df[df["date"] >= cutoff], f"TEST (on/after {cutoff}, held out)")

    # --- Cost sensitivity: the decisive practical question ----------------------
    # Everything above ignores bid-ask, slippage and margin cost, which in Indian
    # index options are material. Rather than guess a single number, this shows how
    # much round-trip cost the edge can absorb before it disappears.
    print(f"\n{'=' * 78}\nCOST SENSITIVITY (held-out test period, variant 3)\n{'=' * 78}")
    test_df = df[df["date"] >= cutoff]
    cfg = VARIANTS[2]
    trades = vrp_strategy.simulate(test_df, cfg)
    taken = trades[trades["entered"]]
    if not taken.empty:
        import math
        pnl = taken["pnl"]
        std, n = float(pnl.std()), len(pnl)
        ppy = 252 / cfg.horizon
        print(f"  gross mean P&L: {pnl.mean():.2f} vol pts/trade over {n} trades\n")
        print(f"  {'round-trip cost':>16} {'net mean':>10} {'net Sharpe':>12} {'still profitable?':>19}")
        for cost in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
            net = float(pnl.mean()) - cost
            sharpe = (net / std * math.sqrt(ppy)) if std > 0 else float("nan")
            print(f"  {cost:>13.1f} pts {net:>10.2f} {sharpe:>12.2f} {'yes' if net > 0 else 'NO':>19}")
        print("\n  The edge is roughly 2 vol points per trade. Whether real execution")
        print("  costs sit below that line is THE question -- and measuring it requires")
        print("  actual option quotes, i.e. the paid data.")

    print(
        "\nHOW TO READ THIS:\n"
        "  * Compare variant 1 vs 2: the cap is what turns a strategy with an\n"
        "    account-ending tail into a survivable one. That single change matters\n"
        "    more than any signal refinement.\n"
        "  * win_rate above ~70% is expected here and is NOT evidence of skill -- it\n"
        "    is the structural signature of premium selling. Judge this on\n"
        "    max_drawdown and sharpe_approx, never on win rate alone.\n"
        "  * Sample sizes are small (~12 non-overlapping periods per year). Treat\n"
        "    test-period figures as directional, not conclusive."
    )


if __name__ == "__main__":
    main()
