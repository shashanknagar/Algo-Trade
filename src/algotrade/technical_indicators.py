"""Vectorized (whole-series) technical indicators -- standard, well-known formulas,
used to enrich the ML classifier's feature set in ml_dataset.py. NOT used by the
rule-based direction picker in signal_generator.py, which stays as already validated
and unchanged.

Each function returns a pandas Series aligned to the input Series'/DataFrame's index.
By construction (rolling/ewm, which only look backward), the value at position i
depends only on rows <= i -- no lookahead, same guarantee as the rest of the pipeline.
"""
from __future__ import annotations

import math

import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index, standard Wilder smoothing. 0-100; >70 conventionally
    'overbought', <30 'oversold'."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Standard MACD (12/26/9). Returns the histogram (MACD line - signal line) --
    sign/magnitude is the commonly-used momentum-shift signal."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def bollinger_pct_b(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """%B: where price sits within its Bollinger Bands. 0 = at lower band, 1 = at
    upper band, >1/<0 = outside the bands."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return (close - lower) / (upper - lower).replace(0, float("nan"))


def average_directional_index(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ADX: trend STRENGTH (not direction) -- standard Wilder formula. >25
    conventionally 'trending', <20 'range-bound'."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr.replace(0, float("nan")))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr.replace(0, float("nan")))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    return dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def gap_pct(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Today's open vs. yesterday's close, as a percentage -- a well-known
    momentum/reversal signal (gap-and-go vs. gap-fill behavior)."""
    prev_close = close.shift(1)
    return (open_ - prev_close) / prev_close


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """ATR as a percentage of price -- volatility normalized for comparison across
    different price levels/instruments."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean() / close


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualized realized volatility as a whole series (decimal, e.g. 0.12 = 12%).
    Paired with India VIX (quoted in percentage points) this gives the variance risk
    premium -- see ml_dataset's vix_minus_realized."""
    log_returns = (close / close.shift(1)).apply(math.log)
    return log_returns.rolling(window).std() * math.sqrt(252)


def realized_vol_percentile(close: pd.Series, window: int = 20, lookback: int = 252) -> pd.Series:
    """Vectorized equivalent of signal_factors.volatility_regime_factor's percentile
    (same definition: annualized realized vol's percentile rank within its own
    trailing `lookback` history), computed for the whole series at once via pandas'
    rolling().rank(pct=True) instead of a slow per-date Python loop -- this is what
    made bulk dataset-building fast enough to actually iterate on."""
    return realized_vol(close, window).rolling(lookback).rank(pct=True)


def sma_ratio(close: pd.Series, short: int, long: int) -> pd.Series:
    """(SMA_short - SMA_long) / SMA_long -- a longer-timeframe trend confirmation,
    distinct from signal_factors.trend_factor's SMA20/50 (e.g. 50/100 here)."""
    sma_short = close.rolling(short).mean()
    sma_long = close.rolling(long).mean()
    return (sma_short - sma_long) / sma_long


def rate_of_change(close: pd.Series, window: int = 10) -> pd.Series:
    """N-day price rate of change -- a distinct momentum measure from the existing
    breakout/breakdown factor (which only looks at N-day high/low, not the return)."""
    return (close - close.shift(window)) / close.shift(window)


def days_left_in_month(dates: pd.Series) -> pd.Series:
    """Trading-day-agnostic proxy for proximity to monthly expiry (NIFTY/BANKNIFTY
    monthly options expire near month-end) -- calendar days remaining, not exact
    expiry-day logic (which varies and wasn't confirmed with certainty)."""
    return dates.apply(lambda d: (pd.Timestamp(d).days_in_month - d.day))


def overnight_return(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Return earned while the market was SHUT: open[t] / close[t-1] - 1.

    Measured on this project's own data, the index premium is earned almost entirely
    overnight (NIFTY +24.8%/yr, t=5.47, sign consistent across both sample halves)
    while the intraday session is negative (-12.9%/yr, t=-2.34). This matches a large
    literature (Cooper/Cliff/Gulen 2008 and later replications). It is the strongest
    directional result found anywhere in this project.
    """
    return open_ / close.shift(1) - 1


def intraday_return(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Return earned while the market was OPEN: close[t] / open[t] - 1. Structurally
    negative on these indices -- a strategy that round-trips within the session is
    fighting a documented headwind before it predicts anything."""
    return close / open_ - 1


def is_turn_of_month(dates: pd.Series) -> pd.Series:
    """Last trading day of a month plus the first three of the next.

    Measured here at ~+19 bps/day versus ~+1 bp/day for the rest of the month, on BOTH
    indices and in BOTH sample halves (t=2.2 to 2.7). The effect is weaker in the late
    period, which is the usual fate of a published anomaly, but the sign holds.
    """
    df = pd.DataFrame({"date": dates.values})
    df["ym"] = df["date"].apply(lambda d: (d.year, d.month))
    rank = df.groupby("ym").cumcount()
    from_end = df.groupby("ym")["date"].transform("size") - 1 - rank
    return pd.Series(((rank <= 2) | (from_end == 0)).values, index=dates.index)


def vol_target_scale(close: pd.Series, window: int = 20, cap: float = 3.0) -> pd.Series:
    """Position multiplier that sizes inversely to recent realized volatility.

    Improves RISK-ADJUSTED return without requiring any better prediction -- which is
    the point, since prediction is the part that does not work here. Measured on this
    data, vol-targeted long-only reached Sharpe 1.11 (NIFTY) / 0.88 (BANKNIFTY) versus
    0.74 / 0.70 unscaled buy-and-hold, replicating in both halves.
    """
    rvol = close.pct_change().rolling(window).std()
    return (rvol.median() / rvol).clip(0, cap)


def is_monthly_expiry(dates: pd.Series) -> pd.Series:
    """Last Thursday of the calendar month -- the long-standing NSE monthly expiry.

    Measured (with corrected session dates) at -12.2 bps vs baseline on NIFTY and
    -7.7 on BANKNIFTY, with the sign holding in BOTH sample halves on BOTH indices.
    That is a higher bar than most effects tested in this project have cleared.
    NIFTY also shows consistently elevated volatility on this day (+2.8 bps, both
    halves); BANKNIFTY's volatility effect does not replicate.

    Note this is the MONTHLY expiry only. Weekly expiry weekdays have been changed by
    the exchange more than once and differ by index, so they are deliberately not
    hardcoded here.
    """
    df = pd.DataFrame({"date": dates.values})
    df["ym"] = df["date"].apply(lambda d: (d.year, d.month))
    df["is_thu"] = df["date"].apply(lambda d: d.weekday() == 3)
    last_thu = df[df["is_thu"]].groupby("ym")["date"].max()
    return pd.Series(df["date"].isin(set(last_thu.values)).values, index=dates.index)
