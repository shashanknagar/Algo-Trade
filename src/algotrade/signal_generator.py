"""Combines signal_factors.py's individual factors into one directional trade
signal, with entry/target/stop-loss on the underlying index (NOT an option premium --
see option_selection.py for the separate, unbacktested strike-suggestion layer).

Strictly as-of a given date: _load_ohlc_as_of only selects rows with time <= as_of_date,
so generate_signal(..., as_of_date=X) is guaranteed not to see anything that happened
after X -- this is what makes backtest_signal.py's forward-walk a real, lookahead-free
test rather than one that's quietly cheating.

Rule-based v1 (model_version='rule_v1') -- not ML. Risk:reward fixed at 1.5 (the
master plan's default). Direction requires the weighted composite score to clear a
deadzone; otherwise the signal is 'neutral' (no trade recommended), which is a valid,
common outcome, not a failure.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from . import signal_factors

RISK_REWARD_RATIO = 1.5  # matches the master plan's Trade Signal Lifecycle default
NEUTRAL_DEADZONE = 0.15  # |composite score| below this -> no_trade, not a weak guess

# Factor weights for the composite direction score. oi_positioning's weight is only
# applied when the factor is actually available (data exists for this date) --
# remaining weights are renormalized, not silently zeroed out.
FACTOR_WEIGHTS = {
    "trend": 0.35,
    "momentum_breakout": 0.30,
    "oi_positioning": 0.20,
    # candle_pattern is deliberately absent: it now always returns available=False
    # (its textbook directions failed measurement -- see signal_factors), so
    # combine_factors excludes it and renormalizes the weights above.
}

# ATR multiples defining risk (the stop distance) per signal type -- positional needs
# more room than intraday, whose horizon is a single day.
ATR_STOP_MULTIPLE = {"positional": 2.0, "intraday": 0.5}
HORIZON_TRADING_DAYS = {"positional": 20, "intraday": 1}


@dataclass
class Signal:
    instrument: str
    as_of_date: date
    signal_type: str
    direction: str  # 'bullish' | 'bearish' | 'neutral'
    entry_price: float | None
    target_price: float | None
    stop_loss_price: float | None
    confidence_score: float
    factor_breakdown: list[dict] = field(default_factory=list)
    model_version: str = "rule_v1"


def _load_ohlc_as_of(conn, instrument: str, as_of_date: date) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT time, open, high, low, close, volume FROM ohlc_bars "
            "WHERE trading_symbol = %s AND interval = '1day' AND time::date <= %s "
            "ORDER BY time ASC",
            (instrument, as_of_date),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def _load_participant_oi_pair(conn, as_of_date: date) -> tuple[dict | None, dict | None]:
    """Returns ({client_type: {...}}, {client_type: {...}}) for the two most recent
    participant_oi_daily snapshot dates <= as_of_date -- (today, previous). Either or
    both may be None if no data exists that far back (e.g. before 2023, or backfill
    hasn't been run)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT date FROM participant_oi_daily WHERE date <= %s "
            "ORDER BY date DESC LIMIT 2",
            (as_of_date,),
        )
        dates = [r[0] for r in cur.fetchall()]

    def _load_for(d):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT client_type, option_index_call_long, option_index_put_long, "
                "option_index_call_short, option_index_put_short "
                "FROM participant_oi_daily WHERE date = %s",
                (d,),
            )
            return {
                r[0]: {
                    "option_index_call_long": r[1],
                    "option_index_put_long": r[2],
                    "option_index_call_short": r[3],
                    "option_index_put_short": r[4],
                }
                for r in cur.fetchall()
            }

    today = _load_for(dates[0]) if len(dates) >= 1 else None
    prev = _load_for(dates[1]) if len(dates) >= 2 else None
    return today, prev


def average_true_range(ohlc: pd.DataFrame, window: int = 14) -> float | None:
    if len(ohlc) < window + 1:
        return None
    highs, lows, closes = ohlc["high"], ohlc["low"], ohlc["close"]
    prev_close = closes.shift(1)
    true_range = pd.concat(
        [highs - lows, (highs - prev_close).abs(), (lows - prev_close).abs()], axis=1
    ).max(axis=1)
    return float(true_range.tail(window).mean())


def combine_factors(factors: list[signal_factors.FactorResult]) -> tuple[float, str, float]:
    """Weighted composite over whichever factors are available (missing/unavailable
    factors are excluded, not treated as zero -- weights renormalize over what's
    actually present)."""
    weighted_sum = 0.0
    weight_total = 0.0
    for f in factors:
        if not f.available:
            continue
        w = FACTOR_WEIGHTS.get(f.name, 0.0)
        weighted_sum += f.score * w
        weight_total += w

    composite = weighted_sum / weight_total if weight_total > 0 else 0.0
    if composite > NEUTRAL_DEADZONE:
        direction = "bullish"
    elif composite < -NEUTRAL_DEADZONE:
        direction = "bearish"
    else:
        direction = "neutral"
    confidence = min(abs(composite) / 1.0, 1.0) * 100
    return composite, direction, confidence


@dataclass(frozen=True)
class FixedBarriers:
    """Barriers expressed in absolute INDEX POINTS rather than ATR multiples.

    CAVEAT, because it changes how results must be read: a fixed point distance is
    non-stationary. 50 points was 0.55% of NIFTY in 2020 (index ~9,000) and is 0.20%
    today (~25,000), and on BANKNIFTY it is 0.09%. So one setting means very different
    trades in different years and on different instruments, and a hit rate measured
    across the whole sample is a blend of several distinct strategies rather than one.
    ATR scaling exists precisely to hold that difficulty constant.

    Use it when you want a specific rupee outcome per trade; prefer ATR when you want
    a comparable measurement across time.
    """
    target_points: float
    stop_points: float

    @property
    def risk_reward(self) -> float:
        return self.target_points / self.stop_points


def generate_signal(conn, instrument: str, as_of_date: date, signal_type: str,
                    barriers: "FixedBarriers | None" = None) -> Signal:
    if signal_type not in ("positional", "intraday"):
        raise ValueError(f"signal_type must be 'positional' or 'intraday', got {signal_type!r}")

    ohlc = _load_ohlc_as_of(conn, instrument, as_of_date)
    if len(ohlc) < 50:
        raise ValueError(
            f"Not enough history for {instrument} as of {as_of_date} "
            f"({len(ohlc)} bars, need >= 50) -- pick a later as_of_date."
        )

    trend = signal_factors.trend_factor(ohlc)
    momentum = signal_factors.momentum_breakout_factor(ohlc)
    candle = signal_factors.candle_pattern_factor(ohlc)
    vol_regime = signal_factors.volatility_regime_factor(ohlc)

    oi_today, oi_prev = _load_participant_oi_pair(conn, as_of_date)
    oi = signal_factors.oi_positioning_factor(oi_today, oi_prev)

    directional_factors = [trend, momentum, candle] + ([oi] if oi is not None else [])
    composite, direction, confidence = combine_factors(directional_factors)

    all_factors_for_breakdown = directional_factors + [vol_regime]
    if oi is None:
        all_factors_for_breakdown.append(
            signal_factors.FactorResult("oi_positioning", 0.0, "no participant-OI data available for this date", available=False)
        )

    entry_price = float(ohlc["close"].iloc[-1])
    atr = average_true_range(ohlc)

    entry = target = stop = None
    if direction != "neutral" and (atr or barriers):
        entry = entry_price
        if barriers is not None:
            up, down = barriers.target_points, barriers.stop_points
            if direction == "bullish":
                target, stop = entry + up, entry - down
            else:
                target, stop = entry - up, entry + down
        else:
            risk = atr * ATR_STOP_MULTIPLE[signal_type]
            if direction == "bullish":
                stop = entry - risk
                target = entry + risk * RISK_REWARD_RATIO
            else:
                stop = entry + risk
                target = entry - risk * RISK_REWARD_RATIO

    return Signal(
        instrument=instrument,
        as_of_date=as_of_date,
        signal_type=signal_type,
        direction=direction,
        entry_price=entry,
        target_price=target,
        stop_loss_price=stop,
        confidence_score=round(confidence, 1),
        factor_breakdown=[
            {"name": f.name, "score": f.score, "explanation": f.explanation, "available": f.available, **f.extra}
            for f in all_factors_for_breakdown
        ],
    )


def persist_signal(conn, signal: Signal) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO trade_signals
                (as_of_date, instrument, signal_type, direction, entry_price,
                 target_price, stop_loss_price, confidence_score, factor_breakdown,
                 model_version, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s)
            RETURNING id
            """,
            (
                signal.as_of_date, signal.instrument, signal.signal_type, signal.direction,
                signal.entry_price, signal.target_price, signal.stop_loss_price,
                signal.confidence_score, json.dumps(signal.factor_breakdown),
                signal.model_version,
                "no_trade" if signal.direction == "neutral" else "open",
            ),
        )
        (signal_id,) = cur.fetchone()
    conn.commit()
    return signal_id
