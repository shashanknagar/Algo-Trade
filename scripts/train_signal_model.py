"""Builds the historical dataset, splits it chronologically into train/test,
trains the signal-quality classifiers, and reports out-of-sample performance
HONESTLY -- including if it never reaches the hit-rate you're hoping for.

This is deliberately not tuned to hit any specific number. If it can't reach 70% out
of sample, that's real information (see BREAKEVEN_HIT_RATE in ml_model.py for what
"profitable" actually requires at this plan's 1.5 risk:reward -- it's well below 70%).

Usage:
    python scripts/train_signal_model.py [--cutoff YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from algotrade import db, ml_dataset, ml_model
from algotrade.config import load_postgres_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutoff", type=str, default="2025-01-01",
        help="Train on signals before this date, test on signals on/after it (default 2025-01-01)",
    )
    args = parser.parse_args()
    cutoff = date.fromisoformat(args.cutoff)

    config = load_postgres_config()
    with db.get_connection(config) as conn:
        print("Building dataset from all available history (this walks every valid "
              "historical date -- may take a minute)...")
        df, counts = ml_dataset.build_dataset(conn)

    print(f"\nTotal signal-generation outcomes across all dates: {counts}")
    print(f"Labeled dataset (target_hit/stop_hit only, 'neutral'/'timed_out' excluded "
          f"from training labels): {len(df)} rows")

    if df.empty:
        print("No labeled rows -- nothing to train on.")
        return

    df = ml_model.drop_rows_with_missing_features(df)
    train_df, test_df = ml_model.train_test_split_by_date(df, cutoff)
    print(f"\nTrain: {len(train_df)} rows (before {cutoff})")
    print(f"Test:  {len(test_df)} rows (on/after {cutoff}, NEVER touched during training)")

    if len(train_df) < 30 or len(test_df) < 15:
        print(
            "\nWARNING: one of these splits is small enough that any result here is "
            "closer to anecdote than evidence. Reporting anyway, but treat the "
            "numbers below accordingly."
        )

    baseline_hit_rate = train_df["target_hit"].mean() if len(train_df) else None
    print(f"\nBaseline (rule-based v1, no ML gating) TRAIN hit-rate: "
          f"{baseline_hit_rate:.1%}" if baseline_hit_rate is not None else "n/a")
    test_baseline = test_df["target_hit"].mean() if len(test_df) else None
    print(f"Baseline (rule-based v1, no ML gating) TEST hit-rate (all trades, "
          f"threshold=0.0): {test_baseline:.1%}" if test_baseline is not None else "n/a")
    print(f"(Breakeven hit-rate at this plan's 1:1.5 risk:reward is "
          f"{ml_model.BREAKEVEN_HIT_RATE:.1%} -- above that is profitable in "
          f"expectancy even far below 70%.)")

    models = ml_model.train_models(train_df)
    for name, model in models.items():
        print(f"\n{'=' * 70}\n{name} -- OUT-OF-SAMPLE evaluation (test period only)\n{'=' * 70}")
        report = ml_model.evaluate(model, test_df)
        print(report.to_string(index=False))

    print(
        "\nReminder: this is one train/test split on a modest amount of history "
        "(not the plan's own >=300-signal, 3-regime bar for 'trustworthy'). A "
        "threshold with a good-looking hit-rate but very few surviving trades (small "
        "n_trades above) is not reliable -- prefer a threshold with both a good "
        "hit-rate AND a meaningful trade count."
    )


if __name__ == "__main__":
    main()
