"""Price-action structures: swing points, trendlines, double tops/bottoms, wick pressure.

These are the chart-reading techniques a discretionary trader applies by eye --
drawing a line across recent swing highs and buying when price closes above it,
spotting two failed attempts at the same high and selling when the neckline gives
way, reading a long lower wick at support as buyers stepping in. This module makes
each one explicit and testable, which is the only way to learn whether they carry
information on this market rather than in folklore.

EVERY FUNCTION IS STRICTLY BACKWARD-LOOKING. A swing high can only be recognised
once enough bars have printed AFTER it, so the flag is shifted forward to the bar on
which the swing actually became knowable. Skipping that shift is the single most
common way chart-pattern backtests cheat, and it makes every pattern look prescient.

All functions take per-instrument series (already sorted by date) and return
series aligned to the input, NaN/False where a structure is not yet defined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SWING_N = 5                 # bars either side that a swing must dominate
TRENDLINE_LOOKBACK = 90     # swings older than this are not used to draw lines
DOUBLE_TOL = 0.02           # two highs within 2% count as a double top
DOUBLE_MIN_GAP = 10         # and must be at least this many bars apart
NEAR_LEVEL = 0.01           # "at" a level = within 1%
WICK_MIN = 0.5              # wick must be at least half the candle's range


# --------------------------------------------------------------------------- swings
def swing_flags(high: pd.Series, low: pd.Series, n: int = SWING_N) -> tuple[pd.Series, pd.Series]:
    """Boolean flags at the bar where a swing high / low BECOMES KNOWN (n bars after
    the swing itself), plus nothing else -- the swing's own price and position are
    recovered by the callers via shift(n)."""
    win = 2 * n + 1
    is_high = high == high.rolling(win, center=True).max()
    is_low = low == low.rolling(win, center=True).min()
    # A centered window at bar i uses bars i-n..i+n; it is only knowable at bar i+n.
    return is_high.shift(n, fill_value=False), is_low.shift(n, fill_value=False)


def _last_two(values: pd.Series, flags: pd.Series, n: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """For every bar, the price and bar-position of the two most recent CONFIRMED
    swings (v1/i1 older, v2/i2 newer). Positions are integer bar indices."""
    pos = pd.Series(np.arange(len(values)), index=values.index, dtype=float)
    sw_val = values.shift(n).where(flags)          # the swing's own price, at confirmation bar
    sw_pos = (pos - n).where(flags)                # the swing's own position
    v2 = sw_val.ffill()
    i2 = sw_pos.ffill()
    # Previous swing: shift within the swings-only series, then re-align and ffill.
    only = sw_val.dropna()
    only_pos = sw_pos.dropna()
    v1 = only.shift(1).reindex(values.index).ffill()
    i1 = only_pos.shift(1).reindex(values.index).ffill()
    return v1, i1, v2, i2


# ------------------------------------------------------------------------ trendlines
def trendline(high: pd.Series, low: pd.Series, kind: str, n: int = SWING_N,
              lookback: int = TRENDLINE_LOOKBACK) -> tuple[pd.Series, pd.Series]:
    """The line through the two most recent confirmed swing highs (kind='resistance')
    or swing lows ('support'), projected to each bar. Returns (level, slope_per_bar).
    Level is NaN when the older swing is too stale or the two swings coincide."""
    hi_f, lo_f = swing_flags(high, low, n)
    series, flags = (high, hi_f) if kind == "resistance" else (low, lo_f)
    v1, i1, v2, i2 = _last_two(series, flags, n)
    pos = pd.Series(np.arange(len(series)), index=series.index, dtype=float)
    gap = i2 - i1
    slope = (v2 - v1) / gap.replace(0, np.nan)
    level = v2 + slope * (pos - i2)
    stale = (pos - i1) > lookback
    level = level.mask(stale | gap.isna() | (gap <= 0))
    return level, slope


def trendline_break(close: pd.Series, level: pd.Series, direction: str) -> pd.Series:
    """First close through the line: above resistance ('up') or below support ('down').
    Requires the prior close to have been on the other side, so a bar that merely
    sits beyond an already-broken line does not re-trigger every day."""
    if direction == "up":
        return (close > level) & (close.shift(1) <= level.shift(1))
    return (close < level) & (close.shift(1) >= level.shift(1))


# ---------------------------------------------------------------- double top/bottom
def _neckline(series: pd.Series, conf_flags: pd.Series, span: pd.Series, n: int,
              how: str) -> pd.Series:
    """The extreme of `series` between the two swings, FROZEN at the bar where the
    newer swing was confirmed and held until the next confirmation. Recomputing it
    every bar would let the neckline drift with price after the pattern formed."""
    out = pd.Series(np.nan, index=series.index)
    conf_span = span.where(conf_flags)
    for k in conf_span.dropna().unique():
        k = int(k)
        if k < 1:
            continue
        roll = series.rolling(k).min() if how == "min" else series.rolling(k).max()
        # rolling ends at the newer swing (position i2); it is knowable n bars later.
        at_conf = roll.shift(n).where(conf_flags & (conf_span == k))
        out = out.where(at_conf.isna(), at_conf)
    return out.ffill()


def double_top(high: pd.Series, low: pd.Series, close: pd.Series, n: int = SWING_N,
               tol: float = DOUBLE_TOL, min_gap: int = DOUBLE_MIN_GAP) -> tuple[pd.Series, pd.Series]:
    """Two confirmed swing highs within `tol` of each other, at least `min_gap` bars
    apart, with a trough between them (the neckline). Returns (pattern_present,
    neckline_break) where the break is the first close below the neckline."""
    hi_f, lo_f = swing_flags(high, low, n)
    v1, i1, v2, i2 = _last_two(high, hi_f, n)
    similar = ((v2 - v1).abs() / v1 <= tol) & ((i2 - i1) >= min_gap)
    # Neckline: lowest low between the two peaks. Computed per bar from the current
    # pair, using a rolling min over the span between them.
    neck = _neckline(low, hi_f, i2 - i1, n, "min")
    present = similar.fillna(False)
    brk = present & (close < neck) & (close.shift(1) >= neck.shift(1))
    return present, brk.fillna(False)


def double_bottom(high: pd.Series, low: pd.Series, close: pd.Series, n: int = SWING_N,
                  tol: float = DOUBLE_TOL, min_gap: int = DOUBLE_MIN_GAP) -> tuple[pd.Series, pd.Series]:
    """Mirror of double_top: two similar swing lows, neckline = highest high between,
    break = first close above it."""
    hi_f, lo_f = swing_flags(high, low, n)
    v1, i1, v2, i2 = _last_two(low, lo_f, n)
    similar = ((v2 - v1).abs() / v1 <= tol) & ((i2 - i1) >= min_gap)
    neck = _neckline(high, lo_f, i2 - i1, n, "max")
    present = similar.fillna(False)
    brk = present & (close > neck) & (close.shift(1) <= neck.shift(1))
    return present, brk.fillna(False)


# ---------------------------------------------------------------------- wick pressure
def wick_ratios(open_: pd.Series, high: pd.Series, low: pd.Series,
                close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(lower_wick, upper_wick, close_position) each as a fraction of the bar's range."""
    rng = (high - low).replace(0, np.nan)
    lower = (np.minimum(open_, close) - low) / rng
    upper = (high - np.maximum(open_, close)) / rng
    close_pos = (close - low) / rng
    return lower, upper, close_pos


def buying_pressure_candle(open_, high, low, close, wick_min: float = WICK_MIN) -> pd.Series:
    """Long lower wick, close in the upper half: sellers pushed it down, buyers took it
    back. The 'hammer' idea without the strict textbook body rules."""
    lower, _, cpos = wick_ratios(open_, high, low, close)
    return (lower >= wick_min) & (cpos >= 0.5)


def selling_pressure_candle(open_, high, low, close, wick_min: float = WICK_MIN) -> pd.Series:
    """Long upper wick, close in the lower half: buyers pushed up, sellers took it back."""
    _, upper, cpos = wick_ratios(open_, high, low, close)
    return (upper >= wick_min) & (cpos <= 0.5)


def near(price: pd.Series, level: pd.Series, tol: float = NEAR_LEVEL) -> pd.Series:
    return ((price - level).abs() / level <= tol).fillna(False)


# ------------------------------------------------------------------------- strategies
def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """All structures and entry rules for one instrument. df must be sorted by date
    with open/high/low/close/volume. Returns df with boolean signal columns."""
    d = df.copy()
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    d["vol_ratio"] = v / v.rolling(20).mean()
    vol_ok = d["vol_ratio"] > 1.3

    res_level, res_slope = trendline(h, l, "resistance")
    sup_level, sup_slope = trendline(h, l, "support")
    d["res_level"], d["sup_level"] = res_level, sup_level

    # 1. Trendline breakout / breakdown, with and without volume.
    d["tl_break_up"] = trendline_break(c, res_level, "up")
    d["tl_break_down"] = trendline_break(c, sup_level, "down")
    # The classic version: breaking a DESCENDING resistance / ASCENDING support.
    d["tl_break_up_desc"] = d["tl_break_up"] & (res_slope < 0)
    d["tl_break_down_asc"] = d["tl_break_down"] & (sup_slope > 0)
    d["tl_break_up_vol"] = d["tl_break_up"] & vol_ok
    d["tl_break_down_vol"] = d["tl_break_down"] & vol_ok

    # 2. Double top breakdown / double bottom breakout.
    _, d["dt_break"] = double_top(h, l, c)
    _, d["db_break"] = double_bottom(h, l, c)
    d["dt_break_vol"] = d["dt_break"] & vol_ok
    d["db_break_vol"] = d["db_break"] & vol_ok

    # 3. Wick pressure AT a structural level (support/resistance line or 20d extreme).
    buy_c = buying_pressure_candle(o, h, l, c)
    sell_c = selling_pressure_candle(o, h, l, c)
    lo20, hi20 = l.rolling(20).min(), h.rolling(20).max()
    at_support = near(l, sup_level) | (l <= lo20 * (1 + NEAR_LEVEL))
    at_resist = near(h, res_level) | (h >= hi20 * (1 - NEAR_LEVEL))
    d["buy_wick_at_support"] = buy_c & at_support
    d["sell_wick_at_resist"] = sell_c & at_resist
    d["buy_wick_at_support_vol"] = d["buy_wick_at_support"] & vol_ok
    d["sell_wick_at_resist_vol"] = d["sell_wick_at_resist"] & vol_ok

    # 4. Confluence: structure break confirmed by the candle AND volume.
    d["confluence_long"] = (d["tl_break_up"] | d["db_break"]) & (c > o) & vol_ok
    d["confluence_short"] = (d["tl_break_down"] | d["dt_break"]) & (c < o) & vol_ok
    return d


LONG_SIGNALS = ["tl_break_up", "tl_break_up_desc", "tl_break_up_vol", "db_break", "db_break_vol",
                "buy_wick_at_support", "buy_wick_at_support_vol", "confluence_long"]
SHORT_SIGNALS = ["tl_break_down", "tl_break_down_asc", "tl_break_down_vol", "dt_break", "dt_break_vol",
                 "sell_wick_at_resist", "sell_wick_at_resist_vol", "confluence_short"]
