"""Non-overlapping trade selection -- the fix for the single biggest measurement
distortion in this project's backtests.

Signals are generated every trading day, but a positional trade takes ~20 days to
resolve. So signals from Nov 6, 9, 10, 11 and 12 share almost their entire forward
window: if that stretch happened to go up, all five "win" together. Counting them as
five independent observations inflates apparent sample size ~5-10x and makes noisy
results look statistically solid. (This is exactly what turned 4 market episodes into
an apparently-impressive "23 trades, 65.2% hit rate".)

It's also just realistic: you hold one position per instrument at a time, not twenty
overlapping ones.

The greedy rule below -- walk signals in date order, take one only if the previously
taken trade has already resolved -- yields trades that are genuinely non-overlapping
and therefore much closer to statistically independent. Sample sizes drop sharply.
That drop is the honest number, not a loss.
"""
from __future__ import annotations

import pandas as pd


def select_non_overlapping(
    trades: pd.DataFrame,
    group_cols: tuple[str, ...] = ("instrument",),
    entry_index_col: str = "entry_index",
    resolve_index_col: str = "resolve_index",
) -> pd.DataFrame:
    """trades needs an integer entry_index (bar position of the signal) and
    resolve_index (bar position where it resolved). Returns only the trades that
    would actually have been taken holding one position at a time per group."""
    if trades.empty:
        return trades

    kept_parts = []
    for _, group in trades.groupby(list(group_cols), sort=False):
        group = group.sort_values(entry_index_col)
        free_from = -1
        keep_mask = []
        for row in group.itertuples(index=False):
            entry_i = getattr(row, entry_index_col)
            if entry_i > free_from:
                keep_mask.append(True)
                free_from = getattr(row, resolve_index_col)
            else:
                keep_mask.append(False)
        kept_parts.append(group[pd.Series(keep_mask, index=group.index)])

    return pd.concat(kept_parts).sort_values(entry_index_col).reset_index(drop=True)
