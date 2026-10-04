"""Market-state learning: bucket every historical signal by its market state, learn
the empirical hit rate per state from the TRAIN period only, and evaluate that
state-based gating on the held-out TEST period.

Also persists (a) every historical prediction + its realized outcome into
trade_signals, and (b) the learned state table into state_hit_rates -- so the
feedback loop has durable memory and live signal generation can read the same rates
this backtest validated.

Usage:
    python scripts/train_state_model.py [--cutoff YYYY-MM-DD] [--alpha 20] [--persist]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from algotrade import db, market_state, ml_dataset, ml_model, state_learning
from algotrade.config import load_postgres_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=str, default="2025-01-01")
    parser.add_argument("--alpha", type=float, default=state_learning.DEFAULT_ALPHA,
                        help="shrinkage strength toward the base rate (higher = more conservative)")
    parser.add_argument("--persist", action="store_true",
                        help="write predictions/outcomes and the learned state table to Postgres")
    args = parser.parse_args()
    cutoff = date.fromisoformat(args.cutoff)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 120)

    config = load_postgres_config()
    with db.get_connection(config) as conn:
        print("Building dataset from all available history...")
        df, counts = ml_dataset.build_dataset(conn)
        print(f"Signal-generation outcomes across all dates: {counts}")
        df = ml_model.drop_rows_with_missing_features(df)

        for dim_set_name, dims in market_state.STATE_DIMENSION_SETS.items():
            print(f"\n{'#' * 78}\n# STATE SET '{dim_set_name}': {' x '.join(dims)}\n{'#' * 78}")

            stated = market_state.add_state_column(df, dims)
            train_df, test_df = ml_model.train_test_split_by_date(stated, cutoff)
            print(f"Train: {len(train_df)} rows (before {cutoff})   "
                  f"Test: {len(test_df)} rows (on/after {cutoff}, held out)")

            table, base_rate = state_learning.build_state_table(train_df, alpha=args.alpha)
            print(f"Distinct states seen in training: {len(table)}   "
                  f"train base rate: {base_rate:.1%}   shrinkage alpha: {args.alpha}")

            well_sampled = table[table["n"] >= 30]
            print(f"\nStates with n>=30 training examples ({len(well_sampled)} of {len(table)}), "
                  f"best first -- raw_hit_rate is the naive count, shrunk_hit_rate is what's actually used:")
            print(well_sampled.head(15).to_string(index=False))

            scored = state_learning.apply_state_table(test_df, table, base_rate)
            unseen = (~scored["state_seen_in_train"]).sum()
            print(f"\nTest rows whose state was never seen in training (fall back to base rate): {unseen}")

            print("\nOUT-OF-SAMPLE evaluation, gating on the learned state hit rate:")
            print(state_learning.evaluate_state_gating(scored).to_string(index=False))

            print("\nSame, but only trusting states with >=30 training examples:")
            print(state_learning.evaluate_state_gating(scored, min_train_n=30).to_string(index=False))

            if args.persist:
                n_states = state_learning.persist_state_table(conn, table, dim_set_name, base_rate, args.alpha)
                print(f"\nPersisted {n_states} states to state_hit_rates (dimension_set='{dim_set_name}').")
                if dim_set_name == "coarse":
                    n_sig = ml_dataset.persist_dataset_to_trade_signals(conn, stated, dim_set_name)
                    print(f"Persisted {n_sig} predictions+outcomes to trade_signals.")

    print(
        "\nNOTE: a state's raw hit rate is only as trustworthy as its sample count. "
        "shrunk_hit_rate deliberately pulls thinly-sampled states back toward the base "
        "rate so the table can't turn 3 lucky occurrences into a fake edge."
    )


if __name__ == "__main__":
    main()
