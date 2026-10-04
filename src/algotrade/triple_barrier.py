"""Shared triple-barrier outcome resolution -- the ONE place this logic lives, used
by both backtest_signal.py (single-signal, DB-backed) and ml_dataset.py (bulk,
in-memory) so the two can never silently drift apart on how an outcome is decided.

Conservative convention when a single bar's range touches both target and stop (can
happen on a gappy/volatile day): the stop is assumed hit first. Deliberately
pessimistic, not tuned to flatter results.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class Outcome:
    status: str  # 'target_hit' | 'stop_hit' | 'timed_out' | 'insufficient_data'
    resolved_date: date | None
    resolved_price: float | None
    r_multiple: float | None
    bars_to_resolve: int | None


def resolve(
    direction: str, entry_price: float, target_price: float, stop_loss_price: float,
    future: pd.DataFrame, risk_reward_ratio: float, tie_breaker: str = "stop",
) -> Outcome:
    """future: OHLC rows strictly AFTER the signal date, in ascending time order,
    with 'high'/'low'/'close'/'time' columns. direction: 'bullish' or 'bearish'."""
    if future.empty:
        return Outcome("insufficient_data", None, None, None, None)

    risk = abs(entry_price - stop_loss_price)

    for i, row in enumerate(future.itertuples(index=False)):
        hit_target = row.high >= target_price if direction == "bullish" else row.low <= target_price
        hit_stop = row.low <= stop_loss_price if direction == "bullish" else row.high >= stop_loss_price
        if hit_stop and hit_target:
            # Both barriers inside one bar. A daily bar records the high and the low
            # but NOT their order, so which came first is genuinely unknowable here.
            # Default stays pessimistic; "target" exists only to bracket how much of a
            # result depends on this choice, which matters enormously once the barriers
            # are narrower than a typical daily range.
            if tie_breaker == "target":
                return Outcome("target_hit", row.time.date(), target_price, risk_reward_ratio, i + 1)
            return Outcome("stop_hit", row.time.date(), stop_loss_price, -1.0, i + 1)
        if hit_stop:
            return Outcome("stop_hit", row.time.date(), stop_loss_price, -1.0, i + 1)
        if hit_target:
            return Outcome("target_hit", row.time.date(), target_price, risk_reward_ratio, i + 1)

    last = future.iloc[-1]
    final_price = float(last["close"])
    raw_r = (
        (final_price - entry_price) / risk if direction == "bullish"
        else (entry_price - final_price) / risk
    )
    return Outcome("timed_out", last["time"].date(), final_price, round(raw_r, 2), len(future))
