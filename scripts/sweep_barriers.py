"""Barrier-geometry sweep on NON-OVERLAPPING trades.

Two things make this different from every backtest number produced so far:

1. Trades are deduplicated to one open position per instrument at a time
   (trade_selection.py), so the sample counts are genuinely independent bets rather
   than the same market episode counted 5-10 times.

2. Risk:reward is held FIXED at 1.5 across every configuration. That matters: hit
   rate moves mechanically with R:R (a nearer target is hit more often while earning
   less), so sweeping R:R would just be the target-shrinking gaming we ruled out.
   Holding it fixed means the driftless random-walk baseline is the same 1/(1+1.5) =
   40% for every cell, and any cell meaningfully above 40% represents real
   directional edge rather than rearranged geometry.

What IS swept is the stop distance (in ATR multiples) and the time horizon -- i.e.
whether the price barrier and the time barrier are consistent with each other. The
current settings (2 ATR / 20 days) were picked by hand and never tested.

Sweep runs on the TRAIN period only; the held-out test period stays untouched.

Usage:
    python scripts/sweep_barriers.py [--cutoff YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from algotrade import db, ml_dataset, signal_factors, trade_selection, triple_barrier
from algotrade.config import load_postgres_config
from algotrade.signal_generator import RISK_REWARD_RATIO, average_true_range, combine_factors

MIN_TRAILING_BARS = 50
STOP_ATR_MULTIPLES = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
HORIZONS = (5, 10, 20, 40, 60)
DRIFTLESS_BASELINE = 1 / (1 + RISK_REWARD_RATIO)  # 40% -- same for every cell


def evaluate_config(candidates: pd.DataFrame, full: pd.DataFrame, stop_mult: float, horizon: int) -> pd.DataFrame:
    """Resolve every candidate under this barrier geometry, then keep only
    non-overlapping trades."""
    df = ml_dataset.resolve_candidates(candidates, full, stop_mult, horizon)
    if df.empty:
        return df
    return trade_selection.select_non_overlapping(df)


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "hit_rate": None, "timeout_pct": None, "expectancy_r": None, "edge": None}
    resolved = trades[trades["status"].isin(["target_hit", "stop_hit"])]
    hit_rate = float((resolved["status"] == "target_hit").mean()) if len(resolved) else None
    return {
        "n": len(trades),
        "n_resolved": len(resolved),
        "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
        "timeout_pct": round(float((trades["status"] == "timed_out").mean()), 3),
        "expectancy_r": round(float(trades["r_multiple"].mean()), 3),
        "edge": round(hit_rate - DRIFTLESS_BASELINE, 3) if hit_rate is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=str, default="2025-01-01")
    args = parser.parse_args()
    cutoff = date.fromisoformat(args.cutoff)

    pd.set_option("display.width", 220)
    config = load_postgres_config()

    all_candidates, fulls = [], {}
    with db.get_connection(config) as conn:
        for instrument in ("NIFTY", "BANKNIFTY"):
            print(f"Generating signal candidates for {instrument} (train period only)...")
            cands, full = ml_dataset.build_signal_candidates(conn, instrument, end_date=cutoff)
            print(f"  {len(cands)} directional candidates before {cutoff}")
            all_candidates.append(cands)
            fulls[instrument] = full

    candidates = pd.concat(all_candidates, ignore_index=True)

    print(f"\nSweeping stop distance x horizon at FIXED risk:reward {RISK_REWARD_RATIO}")
    print(f"Driftless random-walk baseline hit rate for every cell: {DRIFTLESS_BASELINE:.0%}")
    print("(trades are deduplicated to one open position per instrument -- these n's "
          "are independent bets, not overlapping ones)\n")

    results = []
    for stop_mult in STOP_ATR_MULTIPLES:
        for horizon in HORIZONS:
            parts = [
                evaluate_config(candidates[candidates["instrument"] == inst], fulls[inst], stop_mult, horizon)
                for inst in fulls
            ]
            parts = [p for p in parts if not p.empty]
            trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            summary = summarize(trades)
            summary.update({"stop_atr": stop_mult, "horizon": horizon})
            results.append(summary)
            print(f"  stop={stop_mult:>4} ATR  horizon={horizon:>3}d  ->  "
                  f"n={summary['n']:>4}  hit_rate={summary['hit_rate']}  "
                  f"timeout={summary['timeout_pct']}  expectancy={summary['expectancy_r']}  "
                  f"edge_vs_random={summary['edge']}")

    res = pd.DataFrame(results)

    print("\n\n=== HIT RATE (resolved trades only; 0.40 = no edge) ===")
    print(res.pivot(index="stop_atr", columns="horizon", values="hit_rate").to_string())
    print("\n=== INDEPENDENT TRADE COUNT ===")
    print(res.pivot(index="stop_atr", columns="horizon", values="n").to_string())
    print("\n=== EXPECTANCY (R per trade, incl. timeouts) ===")
    print(res.pivot(index="stop_atr", columns="horizon", values="expectancy_r").to_string())
    print("\n=== TIMEOUT FRACTION ===")
    print(res.pivot(index="stop_atr", columns="horizon", values="timeout_pct").to_string())

    print(
        "\nRead this as a surface, not a leaderboard: a single standout cell surrounded "
        "by mediocre neighbours is noise, whereas a smooth region of elevated values is "
        "a real relationship. Cells with small n are unreliable regardless of how good "
        "they look."
    )


if __name__ == "__main__":
    main()
