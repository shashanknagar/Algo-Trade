"""Backtest harness: generates a signal strictly from data available as of a given
date, then walks FORWARD through the (already-collected) historical bars to see
which barrier -- target, stop, or timeout -- was actually hit first. This is the
before/after check you asked for: "pick a random date, predict, then see what really
happened."

Conservative convention when a single day's range touches both target and stop (can
happen on a gappy/volatile day): the stop is assumed hit first. This is a standard,
deliberately pessimistic backtesting convention, not an attempt to flatter results.

Usage:
    python -m algotrade.backtest_signal --instrument NIFTY --random 5
    python -m algotrade.backtest_signal --instrument BANKNIFTY --date 2024-03-15 --type positional
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import date

import pandas as pd

from . import db, signal_generator, triple_barrier
from .config import load_postgres_config

MIN_TRAILING_BARS = 50  # signal_generator's own minimum


@dataclass
class BacktestResult:
    signal: signal_generator.Signal
    outcome: str  # 'target_hit' | 'stop_hit' | 'timed_out' | 'no_trade'
    resolved_date: date | None
    resolved_price: float | None
    r_multiple: float | None
    days_to_resolve: int | None


def _load_ohlc_after(conn, instrument: str, as_of_date: date, max_days: int) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT time, open, high, low, close FROM ohlc_bars "
            "WHERE trading_symbol = %s AND interval = '1day' AND time::date > %s "
            "ORDER BY time ASC LIMIT %s",
            (instrument, as_of_date, max_days),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    return df


def walk_forward_outcome(
    conn, signal: signal_generator.Signal, max_horizon_days: int
) -> BacktestResult:
    if signal.direction == "neutral":
        return BacktestResult(signal, "no_trade", None, None, None, None)

    future = _load_ohlc_after(conn, signal.instrument, signal.as_of_date, max_horizon_days)
    outcome = triple_barrier.resolve(
        signal.direction, signal.entry_price, signal.target_price, signal.stop_loss_price,
        future, signal_generator.RISK_REWARD_RATIO,
    )
    return BacktestResult(
        signal, outcome.status, outcome.resolved_date, outcome.resolved_price,
        outcome.r_multiple, outcome.bars_to_resolve,
    )


def backtest_one(conn, instrument: str, as_of_date: date, signal_type: str, max_horizon_days: int | None = None) -> BacktestResult:
    signal = signal_generator.generate_signal(conn, instrument, as_of_date, signal_type)
    horizon = max_horizon_days or signal_generator.HORIZON_TRADING_DAYS[signal_type]
    return walk_forward_outcome(conn, signal, horizon)


def _get_valid_date_range(conn, instrument: str, signal_type: str) -> tuple[date, date] | None:
    """Earliest/latest as_of_date that has both enough trailing history to generate a
    signal and enough leading history to resolve one within its horizon."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT time::date FROM ohlc_bars WHERE trading_symbol = %s AND interval = '1day' "
            "ORDER BY time ASC", (instrument,),
        )
        all_dates = [r[0] for r in cur.fetchall()]
    if len(all_dates) < MIN_TRAILING_BARS + signal_generator.HORIZON_TRADING_DAYS[signal_type] + 1:
        return None
    earliest = all_dates[MIN_TRAILING_BARS]
    latest = all_dates[-(signal_generator.HORIZON_TRADING_DAYS[signal_type] + 1)]
    if earliest > latest:
        return None
    return earliest, latest


def random_backtest(conn, instrument: str, signal_type: str, n: int = 1, seed: int | None = None) -> list[BacktestResult]:
    rng = random.Random(seed)
    date_range = _get_valid_date_range(conn, instrument, signal_type)
    if date_range is None:
        raise ValueError(f"Not enough history for {instrument} to run a {signal_type} backtest.")
    earliest, latest = date_range

    with conn.cursor() as cur:
        cur.execute(
            "SELECT time::date FROM ohlc_bars WHERE trading_symbol = %s AND interval = '1day' "
            "AND time::date BETWEEN %s AND %s ORDER BY time ASC",
            (instrument, earliest, latest),
        )
        candidate_dates = [r[0] for r in cur.fetchall()]

    chosen = rng.sample(candidate_dates, min(n, len(candidate_dates)))
    return [backtest_one(conn, instrument, d, signal_type) for d in chosen]


def format_result(result: BacktestResult) -> str:
    s = result.signal
    lines = [
        f"--- {s.instrument} {s.signal_type} signal as of {s.as_of_date} ---",
        f"  Direction: {s.direction}  (confidence {s.confidence_score:.0f}/100)",
    ]
    for f in s.factor_breakdown:
        lines.append(f"    [{'ok' if f['available'] else 'n/a'}] {f['name']}: {f['explanation']}")
    if s.direction != "neutral":
        lines.append(f"  Entry: {s.entry_price:.1f}  Target: {s.target_price:.1f}  Stop: {s.stop_loss_price:.1f}")
    lines.append(f"  OUTCOME: {result.outcome}" + (
        f" on {result.resolved_date} at {result.resolved_price:.1f} "
        f"(R-multiple: {result.r_multiple:+.2f}, {result.days_to_resolve} trading day(s) later)"
        if result.resolved_date else ""
    ))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrument", default="NIFTY", choices=["NIFTY", "BANKNIFTY"])
    parser.add_argument("--type", dest="signal_type", default="positional", choices=["positional", "intraday"])
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD; omit for --random")
    parser.add_argument("--random", type=int, default=0, help="run N random historical backtests instead of --date")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_postgres_config()
    with db.get_connection(config) as conn:
        if args.random:
            results = random_backtest(conn, args.instrument, args.signal_type, n=args.random, seed=args.seed)
        else:
            as_of = date.fromisoformat(args.date) if args.date else date.today()
            results = [backtest_one(conn, args.instrument, as_of, args.signal_type)]

    outcomes = [r.outcome for r in results]
    for r in results:
        print(format_result(r))
        print()
    print(f"Ran {len(results)} backtest(s). Outcome counts: "
          + ", ".join(f"{o}={outcomes.count(o)}" for o in sorted(set(outcomes))))
    print(
        "\nReminder: this is a small sample, not a validated strategy. The plan's own "
        "bar for 'trustworthy' is >=300 resolved signals across 3+ volatility regimes "
        "(see i-want-to-create-giggly-blanket.md, Phase 4)."
    )


if __name__ == "__main__":
    main()
