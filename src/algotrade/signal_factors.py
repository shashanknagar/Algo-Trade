"""Rule-based signal factors -- each a well-established, widely-known technique (not
an invented scheme): SMA trend-following, breakout/breakdown detection, standard
candlestick patterns, realized-volatility regime, and OI-buildup-style positioning
from participant data. No ML here -- there's no labeled outcome data yet to train on;
that's what backtest_signal.py and, later, the live trade_signals table start
producing.

Every function takes a slice of data ending at (and including) the row being
evaluated -- callers are responsible for not passing future rows (see
signal_generator.py's as-of-date loading). Each returns a FactorResult with a score
in [-1, +1] (bearish..bullish; 0 = neutral/no-opinion) and a human-readable
explanation, so a generated signal's factor_breakdown is genuinely inspectable, not a
black box.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FactorResult:
    name: str
    score: float  # -1..+1, bearish..bullish; 0 = neutral
    explanation: str
    available: bool = True
    extra: dict = field(default_factory=dict)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def trend_factor(ohlc: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> FactorResult:
    """Classic SMA crossover trend-following. Score scales with how far apart the
    two SMAs are, relative to price, so a strong trend scores closer to +-1 than a
    marginal one."""
    if len(ohlc) < long_window:
        return FactorResult("trend", 0.0, f"insufficient history ({len(ohlc)} < {long_window} bars)", available=False)

    close = ohlc["close"]
    sma_short = close.tail(short_window).mean()
    sma_long = close.tail(long_window).mean()
    last_close = close.iloc[-1]

    spread_pct = (sma_short - sma_long) / sma_long
    score = _clip(spread_pct * 20)  # ~5% spread -> full-strength score
    direction = "bullish" if score > 0 else "bearish" if score < 0 else "flat"
    explanation = (
        f"SMA{short_window}={sma_short:.1f} vs SMA{long_window}={sma_long:.1f} "
        f"({spread_pct:+.2%} spread) -> {direction} trend; last close {last_close:.1f}"
    )
    return FactorResult("trend", score, explanation)


def momentum_breakout_factor(ohlc: pd.DataFrame, lookback: int = 20) -> FactorResult:
    """Price vs. its own N-day rolling high/low -- a classic breakout/breakdown
    detector.

    NOTE: this used to multiply the score by the volume ratio, on the conventional
    assumption that a breakout on heavy volume is more meaningful. Measured against
    3,262 next-day returns (scripts/analyze_patterns.py), that assumption did not hold
    -- breakups on NORMAL volume outperformed breakups on high volume (+18.4 bps,
    t=3.39 vs +12.2 bps, t=1.01), and high-volume BREAKDOWNS were followed by strong
    UP moves (+53.6 bps, t=2.56), the opposite of a bearish reading.

    The volume amplification is therefore removed. It is NOT inverted: "high-volume
    breakdowns bounce" is a hypothesis generated from the same sample it was measured
    on, and acting on it without out-of-sample confirmation would be the overfitting
    this project keeps guarding against. Volume ratio is still reported in the
    explanation for context, and remains available to the ML layer as a raw feature
    so a model can learn the real relationship from data instead of from folklore.
    """
    if len(ohlc) < lookback + 1:
        return FactorResult("momentum_breakout", 0.0, f"insufficient history ({len(ohlc)} < {lookback + 1} bars)", available=False)

    window = ohlc.iloc[-(lookback + 1):-1]  # the N days BEFORE today, excluding today
    today = ohlc.iloc[-1]
    prior_high = window["high"].max()
    prior_low = window["low"].min()
    avg_volume = window["volume"].mean()
    volume_ratio = (today["volume"] / avg_volume) if avg_volume else 1.0

    if today["close"] > prior_high:
        score = _clip(min((today["close"] - prior_high) / prior_high * 50, 1.0))
        explanation = (
            f"close {today['close']:.1f} broke above prior {lookback}-day high "
            f"{prior_high:.1f} (volume {volume_ratio:.1f}x avg, not used for scoring) "
            f"-> bullish breakout"
        )
    elif today["close"] < prior_low:
        score = _clip(-min((prior_low - today["close"]) / prior_low * 50, 1.0))
        explanation = (
            f"close {today['close']:.1f} broke below prior {lookback}-day low "
            f"{prior_low:.1f} (volume {volume_ratio:.1f}x avg, not used for scoring) "
            f"-> bearish breakdown"
        )
    else:
        score = 0.0
        explanation = f"close {today['close']:.1f} within prior {lookback}-day range [{prior_low:.1f}, {prior_high:.1f}] -> no breakout"

    return FactorResult("momentum_breakout", score, explanation)


def candle_pattern_factor(ohlc: pd.DataFrame) -> FactorResult:
    """Detects a small set of well-known single/two-candle patterns: bullish/bearish
    engulfing, hammer, shooting star, doji.

    IMPORTANT: these no longer contribute to the rule-based direction (every result is
    returned with available=False, so combine_factors excludes it and renormalizes the
    remaining weights). Measured against 3,262 next-day returns
    (scripts/analyze_patterns.py), the textbook directions were not supported:

        bullish_engulfing  +17.1 bps vs baseline  (t=1.65, not significant)
        bearish_engulfing  +12.5 bps vs baseline  (t=2.23) <- UP, not down
        hammer              -8.3 bps vs baseline  (t=-0.25) <- down, not up
        shooting_star       -4.5 bps vs baseline  (t=0.02, nothing)
        doji                -4.7 bps vs baseline  (t=0.03, nothing)

    Aggregated, "bearish" patterns preceded UP moves more strongly than "bullish" ones
    did. This is consistent with the broader academic record on classical candlestick
    patterns. Rather than invert the signs -- which would be fitting to the very sample
    that produced these numbers -- the factor is simply removed from the directional
    vote. The detected pattern is still returned in `extra` so it stays visible in
    every signal's factor_breakdown and remains available to the ML layer, which can
    learn a real relationship from data if one exists.
    """
    if len(ohlc) < 2:
        return FactorResult("candle_pattern", 0.0, "insufficient history for candle pattern", available=False)

    prev, cur = ohlc.iloc[-2], ohlc.iloc[-1]
    body = abs(cur["close"] - cur["open"])
    full_range = cur["high"] - cur["low"]
    if full_range <= 0:
        return _candle_result("none", "zero-range candle, no pattern")

    upper_wick = cur["high"] - max(cur["close"], cur["open"])
    lower_wick = min(cur["close"], cur["open"]) - cur["low"]
    prev_body_bearish = prev["close"] < prev["open"]
    prev_body_bullish = prev["close"] > prev["open"]
    cur_body_bullish = cur["close"] > cur["open"]
    cur_body_bearish = cur["close"] < cur["open"]

    # Engulfing: current body fully engulfs the prior body, opposite direction.
    if prev_body_bearish and cur_body_bullish and cur["close"] > prev["open"] and cur["open"] < prev["close"]:
        return _candle_result("bullish_engulfing", "bullish engulfing (engulfs prior bearish candle)")
    if prev_body_bullish and cur_body_bearish and cur["open"] > prev["close"] and cur["close"] < prev["open"]:
        return _candle_result("bearish_engulfing", "bearish engulfing (engulfs prior bullish candle)")

    # Hammer: small body near the top of the range, long lower wick, little upper wick.
    if body > 0 and lower_wick >= 2 * body and upper_wick <= body * 0.5 and body / full_range < 0.35:
        return _candle_result("hammer", "hammer (long lower wick, small body near top)")

    # Shooting star: mirror of hammer.
    if body > 0 and upper_wick >= 2 * body and lower_wick <= body * 0.5 and body / full_range < 0.35:
        return _candle_result("shooting_star", "shooting star (long upper wick, small body near bottom)")

    # Doji: negligible body relative to range -> indecision.
    if body / full_range < 0.1:
        return _candle_result("doji", "doji (negligible body) -> indecision")

    return _candle_result("none", "no recognized candle pattern")


def _candle_result(pattern: str, description: str) -> FactorResult:
    """Always available=False and score 0: detected and recorded, but excluded from
    the directional vote (see candle_pattern_factor's docstring for the measurements
    behind that decision)."""
    return FactorResult(
        "candle_pattern", 0.0,
        f"{description} [detected but NOT scored -- textbook direction unsupported by measurement]",
        available=False,
        extra={"pattern": pattern},
    )


def realized_volatility(ohlc: pd.DataFrame, window: int = 20) -> float | None:
    """Annualized realized volatility from daily log returns -- reused by
    option_selection.py as the volatility PROXY for theoretical Greeks (not real
    market-implied volatility)."""
    if len(ohlc) < window + 1:
        return None
    closes = ohlc["close"].tail(window + 1).astype(float)
    log_returns = (closes / closes.shift(1)).apply(math.log).dropna()
    daily_std = log_returns.std()
    return float(daily_std * math.sqrt(252)) if daily_std == daily_std else None  # NaN check


def volatility_regime_factor(ohlc: pd.DataFrame, window: int = 20, lookback: int = 252) -> FactorResult:
    """Realized-vol percentile vs. its own trailing history. Informational only --
    does not contribute to direction, but flags a 'choppy'/high-vol regime that
    should dampen confidence in the trend/breakout factors (handled by the caller)."""
    if len(ohlc) < window + lookback:
        return FactorResult("volatility_regime", 0.0, f"insufficient history for {lookback}-day vol percentile", available=False)

    vols = []
    closes = ohlc["close"].tail(window + lookback)
    for i in range(window, len(closes)):
        sub = closes.iloc[i - window : i + 1]
        log_returns = (sub / sub.shift(1)).apply(math.log).dropna()
        vols.append(float(log_returns.std() * math.sqrt(252)))
    current_vol = vols[-1]
    percentile = sum(1 for v in vols if v <= current_vol) / len(vols)

    regime = "high" if percentile > 0.8 else "low" if percentile < 0.2 else "normal"
    return FactorResult(
        "volatility_regime", 0.0,
        f"current {window}-day realized vol {current_vol:.1%} is at the {percentile:.0%} percentile "
        f"of the trailing {lookback} days -> {regime} volatility regime",
        extra={"regime": regime, "percentile": percentile, "current_vol": current_vol},
    )


def oi_positioning_factor(oi_today: dict | None, oi_prev: dict | None) -> FactorResult | None:
    """Day-over-day change in FII+DII Option-Index Call-Long vs. Put-Long, from NSE's
    participant-wise OI report. Market-wide across ALL index derivatives combined --
    NOT separable per NIFTY-vs-BANKNIFTY, so this is a coarser, slower-moving
    positioning signal, not a precise per-instrument one. Returns None (not a
    zero-score) when data isn't available, so callers can omit it from the weighted
    combination entirely rather than silently treating "no data" as "neutral"."""
    if not oi_today or not oi_prev:
        return None

    def institutional_call_put_delta(day: dict) -> int:
        fii = day.get("FII", {})
        dii = day.get("DII", {})
        call_long = fii.get("option_index_call_long", 0) + dii.get("option_index_call_long", 0)
        put_long = fii.get("option_index_put_long", 0) + dii.get("option_index_put_long", 0)
        return call_long - put_long

    today_delta = institutional_call_put_delta(oi_today)
    prev_delta = institutional_call_put_delta(oi_prev)
    change = today_delta - prev_delta
    reference = max(abs(prev_delta), 1)
    score = _clip(change / reference)

    direction = "bullish (net Call-Long positioning increasing)" if score > 0 else \
        "bearish (net Put-Long positioning increasing)" if score < 0 else "flat"
    return FactorResult(
        "oi_positioning", score,
        f"FII+DII net (Call-Long - Put-Long) OI moved from {prev_delta:,} to {today_delta:,} "
        f"({change:+,} contracts) -> {direction}",
    )
