"""Confirms the barrier-geometry change ONCE on the held-out test period.

The train-period sweep (scripts/sweep_barriers.py) showed hit rate rising smoothly
and monotonically with horizon at a fixed 2 ATR stop, with the current 20-day setting
sitting exactly at the driftless random-walk baseline. The candidate replacement --
same 2 ATR stop, 60-day horizon -- was chosen from TRAIN data only. This script
checks it on test data exactly once, rather than searching the test set.

Both configurations hold risk:reward fixed at 1.5, so the no-edge baseline is 40% for
both and the comparison is apples-to-apples.

Usage:
    python scripts/validate_barrier_change.py [--cutoff YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from algotrade import db, ml_dataset, trade_selection
from algotrade.config import load_postgres_config
from algotrade.signal_generator import RISK_REWARD_RATIO

DRIFTLESS_BASELINE = 1 / (1 + RISK_REWARD_RATIO)
CONFIGS = {
    "current (2 ATR / 20d)": (2.0, 20),
    "candidate (2 ATR / 60d)": (2.0, 60),
}


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=str, default="2025-01-01")
    args = parser.parse_args()
    cutoff = date.fromisoformat(args.cutoff)

    config = load_postgres_config()
    candidates, fulls = [], {}
    with db.get_connection(config) as conn:
        for instrument in ("NIFTY", "BANKNIFTY"):
            cands, full = ml_dataset.build_signal_candidates(conn, instrument, start_date=cutoff)
            print(f"{instrument}: {len(cands)} directional candidates on/after {cutoff}")
            candidates.append(cands)
            fulls[instrument] = full
    candidates = pd.concat(candidates, ignore_index=True)

    print(f"\nHELD-OUT TEST PERIOD (on/after {cutoff}), non-overlapping trades")
    print(f"No-edge baseline for both configs: {DRIFTLESS_BASELINE:.0%}\n")

    for label, (stop_mult, horizon) in CONFIGS.items():
        parts = []
        for inst, full in fulls.items():
            sub = candidates[candidates["instrument"] == inst]
            df = ml_dataset.resolve_candidates(sub, full, stop_mult, horizon)
            if not df.empty:
                parts.append(trade_selection.select_non_overlapping(df))
        trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        if trades.empty:
            print(f"{label}: no trades")
            continue

        resolved = trades[trades["status"].isin(["target_hit", "stop_hit"])]
        hits = int((resolved["status"] == "target_hit").sum())
        n_res = len(resolved)
        hit_rate = hits / n_res if n_res else float("nan")
        lo, hi = wilson_ci(hits, n_res)

        print(f"{label}")
        print(f"  independent trades : {len(trades)}  (resolved {n_res}, "
              f"timed out {int((trades['status'] == 'timed_out').sum())})")
        print(f"  hit rate           : {hit_rate:.1%}  (95% CI {lo:.1%} - {hi:.1%})")
        print(f"  edge vs no-edge    : {hit_rate - DRIFTLESS_BASELINE:+.1%}"
              f"   {'(CI excludes baseline)' if lo > DRIFTLESS_BASELINE else '(CI includes baseline -- not distinguishable from no edge)'}")
        print(f"  expectancy         : {trades['r_multiple'].mean():+.3f} R per trade\n")

    print("Note: the test period is under two years, so independent-trade counts here "
          "are small by construction. Treat this as a consistency check on the train "
          "sweep's direction, not as standalone proof.")


if __name__ == "__main__":
    main()
