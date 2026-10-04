"""Builds a labeled dataset for training the signal-quality classifier: for every
valid historical date (across NIFTY/BANKNIFTY x positional/intraday), computes the
same rule-based factor scores and entry/target/stop as signal_generator.py (strictly
as-of that date -- hist = full.iloc[:i+1] never includes row i+1 onward, so this is
the same no-lookahead guarantee) PLUS a set of standard technical indicators
(technical_indicators.py, precomputed vectorized per-instrument so this stays fast)
purely as extra ML features, and resolves the real outcome via the exact same
triple_barrier.resolve() used by backtest_signal.py -- one shared implementation, so
training and backtesting can never silently disagree on what counts as a win.

Rows where the rule-based composite says "neutral" (no trade) are excluded -- there's
no entry/target/stop to label. Rows that "timed_out" (neither barrier hit within the
horizon) are excluded from the classifier's training labels too (ambiguous target;
consistent with the master plan's own hit-rate definition, which also excludes
timeouts from its denominator) -- but their counts are reported separately by the
caller for honesty about overall signal-generation behavior, not hidden.
"""
from __future__ import annotations

import bisect
from datetime import date

import pandas as pd

from . import signal_factors, technical_indicators, triple_barrier
from .backtest_signal import MIN_TRAILING_BARS
from .market_calendar import IST
from .signal_generator import (
    ATR_STOP_MULTIPLE,
    HORIZON_TRADING_DAYS,
    RISK_REWARD_RATIO,
    average_true_range,
    combine_factors,
)


def _load_full_ohlc(conn, instrument: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT time, open, high, low, close, volume FROM ohlc_bars "
            "WHERE trading_symbol = %s AND interval = '1day' ORDER BY time ASC",
            (instrument,),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    # Convert to IST before taking the calendar date. Index bars are stored as
    # midnight IST, which Postgres holds as 18:30 UTC the PREVIOUS day -- calling
    # .date() on the UTC value shifted every session back by one day (which showed
    # up as 331 "Sunday" bars and no Fridays). Equity bhavcopy rows are stored at
    # midnight UTC, and this same conversion leaves those correct, so the two
    # sources line up on the real session date.
    df["date"] = df["time"].apply(lambda t: t.astimezone(IST).date())
    return df


def _load_vix(conn) -> pd.DataFrame:
    """India VIX daily closes -- the market's own implied-volatility / fear gauge,
    used as a FEATURE (never a signal target). Free, full history, and the closest
    stand-in available without a paid option-chain subscription for genuinely
    forward-looking (rather than price-derived) information."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT time, close FROM ohlc_bars WHERE trading_symbol = 'INDIAVIX' "
            "AND interval = '1day' ORDER BY time ASC"
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["date", "vix"])
    df = pd.DataFrame(rows, columns=["time", "vix"])
    df["vix"] = df["vix"].astype(float)
    # Convert to IST before taking the calendar date. Index bars are stored as
    # midnight IST, which Postgres holds as 18:30 UTC the PREVIOUS day -- calling
    # .date() on the UTC value shifted every session back by one day (which showed
    # up as 331 "Sunday" bars and no Fridays). Equity bhavcopy rows are stored at
    # midnight UTC, and this same conversion leaves those correct, so the two
    # sources line up on the real session date.
    df["date"] = df["time"].apply(lambda t: t.astimezone(IST).date())
    return df[["date", "vix"]]


def _precompute_indicators(full: pd.DataFrame, vix: pd.DataFrame | None = None) -> pd.DataFrame:
    """Whole-series, vectorized -- computed once per instrument rather than
    recomputed from scratch on every as-of-date (that per-date recomputation, for the
    252-day volatility percentile specifically, was the original bottleneck)."""
    ind = pd.DataFrame(index=full.index)
    ind["rsi_14"] = technical_indicators.rsi(full["close"])
    ind["macd_hist"] = technical_indicators.macd_histogram(full["close"])
    ind["bollinger_pct_b"] = technical_indicators.bollinger_pct_b(full["close"])
    ind["adx_14"] = technical_indicators.average_directional_index(full["high"], full["low"], full["close"])
    ind["gap_pct"] = technical_indicators.gap_pct(full["open"], full["close"])
    ind["atr_pct"] = technical_indicators.atr_pct(full["high"], full["low"], full["close"])
    ind["vol_percentile_fast"] = technical_indicators.realized_vol_percentile(full["close"])
    ind["days_left_in_month"] = technical_indicators.days_left_in_month(full["date"])
    ind["sma50_100_spread"] = technical_indicators.sma_ratio(full["close"], 50, 100)
    ind["roc_10"] = technical_indicators.rate_of_change(full["close"], 10)

    # Volume relative to its own 20-day average. Included because the volume x
    # breakout interaction replicated across both halves of the sample in the
    # opposite direction to the textbook assumption (high-volume breakdowns were
    # followed by UP moves; high-volume breakups underperformed). Rather than
    # hardcode that inverted rule, the raw ingredient is handed to the ML layer so a
    # model can learn the relationship from data.
    ind["volume_ratio"] = full["volume"] / full["volume"].rolling(20).mean()

    # Session decomposition. The index premium is earned overnight while the
    # intraday session is negative (NIFTY: +24.8%/yr vs -12.9%/yr, both signs
    # replicating across halves) -- the strongest directional result in this
    # project, so the model gets to see both components separately.
    ind["overnight_ret"] = technical_indicators.overnight_return(full["open"], full["close"])
    ind["intraday_ret"] = technical_indicators.intraday_return(full["open"], full["close"])
    ind["prev_overnight_ret"] = ind["overnight_ret"].shift(1)
    # NOTE: is_turn_of_month was originally adopted on a run that used SHIFTED
    # session dates (see the timezone bug fixed in _load_full_ohlc). Re-measured on
    # correct dates it no longer replicates on NIFTY (early +29.9, late -1.5 bps),
    # so it is kept only for reference and dropped from FEATURE_COLUMNS.
    ind["is_turn_of_month"] = technical_indicators.is_turn_of_month(full["date"]).astype(int)
    ind["is_monthly_expiry"] = technical_indicators.is_monthly_expiry(full["date"]).astype(int)
    ind["vol_target_scale"] = technical_indicators.vol_target_scale(full["close"])

    # India VIX features. vix_minus_realized is the variance risk premium: implied
    # volatility (VIX, quoted in percentage points) minus realized annualized
    # volatility (decimal). Positive means options are pricing more movement than has
    # actually occurred.
    vix_cols = ["vix_level", "vix_percentile", "vix_change_5d", "vix_minus_realized"]
    if vix is not None and not vix.empty:
        merged = full[["date"]].merge(vix, on="date", how="left")
        series = merged["vix"].ffill()
        series.index = full.index
        ind["vix_level"] = series
        ind["vix_percentile"] = series.rolling(252).rank(pct=True)
        ind["vix_change_5d"] = series / series.shift(5) - 1
        ind["vix_minus_realized"] = (series / 100.0) - technical_indicators.realized_vol(full["close"])
    else:
        for col in vix_cols:
            ind[col] = float("nan")
    return ind


def _participant_type_deltas(oi_today: dict | None, oi_prev: dict | None) -> dict[str, float]:
    """Per-client-type signed (Call-Long - Put-Long) OI change, normalized to [-1, 1]
    the same way signal_factors.oi_positioning_factor does its combined FII+DII
    figure -- but kept separate per participant type here (FII/DII/Pro/Client), which
    is genuinely more granular real information than the single combined score."""
    result = {}
    for client_type in ("FII", "DII", "Pro", "Client"):
        key = f"oi_delta_{client_type.lower()}"
        if not oi_today or not oi_prev or client_type not in oi_today or client_type not in oi_prev:
            result[key] = 0.0
            continue
        today_val = oi_today[client_type]["option_index_call_long"] - oi_today[client_type]["option_index_put_long"]
        prev_val = oi_prev[client_type]["option_index_call_long"] - oi_prev[client_type]["option_index_put_long"]
        change = today_val - prev_val
        reference = max(abs(prev_val), 1)
        result[key] = max(-1.0, min(1.0, change / reference))
    return result


def _load_oi_index(conn) -> dict[date, dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT date, client_type, option_index_call_long, option_index_put_long, "
            "option_index_call_short, option_index_put_short FROM participant_oi_daily "
            "ORDER BY date ASC"
        )
        rows = cur.fetchall()
    by_date: dict[date, dict] = {}
    for d, client_type, cl, pl, cs, ps in rows:
        by_date.setdefault(d, {})[client_type] = {
            "option_index_call_long": cl, "option_index_put_long": pl,
            "option_index_call_short": cs, "option_index_put_short": ps,
        }
    return by_date


def _oi_pair_for(oi_index: dict, sorted_dates: list[date], as_of: date) -> tuple[dict | None, dict | None]:
    idx = bisect.bisect_right(sorted_dates, as_of)
    if idx == 0:
        return None, None
    today = oi_index[sorted_dates[idx - 1]]
    prev = oi_index[sorted_dates[idx - 2]] if idx >= 2 else None
    return today, prev


def build_signal_candidates(
    conn, instrument: str, start_date: date | None = None, end_date: date | None = None,
    min_trailing_bars: int = MIN_TRAILING_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Directional signal candidates WITHOUT barriers attached: direction, entry
    price and ATR depend only on the factors, not on the stop/target geometry. Kept
    separate so barrier settings can be swept or changed without regenerating (or
    accidentally altering) the underlying signals.

    Returns (candidates, full_ohlc). end_date is exclusive.
    """
    full = _load_full_ohlc(conn, instrument)
    oi_index = _load_oi_index(conn)
    sorted_oi_dates = sorted(oi_index.keys())

    rows = []
    for i in range(min_trailing_bars, len(full) - 1):
        as_of = full["date"].iloc[i]
        if start_date is not None and as_of < start_date:
            continue
        if end_date is not None and as_of >= end_date:
            continue

        hist = full.iloc[: i + 1]
        trend = signal_factors.trend_factor(hist)
        momentum = signal_factors.momentum_breakout_factor(hist)
        candle = signal_factors.candle_pattern_factor(hist)
        oi_today, oi_prev = _oi_pair_for(oi_index, sorted_oi_dates, as_of)
        oi = signal_factors.oi_positioning_factor(oi_today, oi_prev)

        factors = [trend, momentum, candle] + ([oi] if oi is not None else [])
        _, direction, _ = combine_factors(factors)
        if direction == "neutral":
            continue

        atr = average_true_range(hist)
        if not atr:
            continue

        rows.append(
            {
                "instrument": instrument,
                "entry_index": i,
                "as_of_date": as_of,
                "direction": direction,
                "entry_price": float(hist["close"].iloc[-1]),
                "atr": atr,
            }
        )
    return pd.DataFrame(rows), full


def resolve_candidates(
    candidates: pd.DataFrame, full: pd.DataFrame, stop_atr_multiple: float, horizon: int,
) -> pd.DataFrame:
    """Applies a barrier geometry to candidates and resolves each outcome. Risk:reward
    stays fixed at RISK_REWARD_RATIO -- only the stop distance and horizon vary."""
    resolved = []
    for c in candidates.itertuples(index=False):
        risk = c.atr * stop_atr_multiple
        if c.direction == "bullish":
            stop = c.entry_price - risk
            target = c.entry_price + risk * RISK_REWARD_RATIO
        else:
            stop = c.entry_price + risk
            target = c.entry_price - risk * RISK_REWARD_RATIO

        future = full.iloc[c.entry_index + 1 : c.entry_index + 1 + horizon]
        outcome = triple_barrier.resolve(c.direction, c.entry_price, target, stop, future, RISK_REWARD_RATIO)
        if outcome.status == "insufficient_data":
            continue

        resolved.append(
            {
                "instrument": c.instrument,
                "entry_index": c.entry_index,
                "as_of_date": c.as_of_date,
                "direction": c.direction,
                "status": outcome.status,
                "r_multiple": outcome.r_multiple,
                # A trade occupies the position until it resolves; timeouts hold for
                # the full horizon.
                "resolve_index": c.entry_index + (outcome.bars_to_resolve or horizon),
            }
        )
    return pd.DataFrame(resolved)


def build_dataset(
    conn,
    instruments: tuple[str, ...] = ("NIFTY", "BANKNIFTY"),
    signal_types: tuple[str, ...] = ("positional", "intraday"),
) -> tuple[pd.DataFrame, dict]:
    """Returns (labeled_dataframe, summary_counts). summary_counts includes rows this
    dataframe deliberately excludes (neutral, timed_out) so that's visible, not lost."""
    oi_index = _load_oi_index(conn)
    sorted_oi_dates = sorted(oi_index.keys())
    vix = _load_vix(conn)

    records: list[dict] = []
    counts = {"neutral": 0, "target_hit": 0, "stop_hit": 0, "timed_out": 0, "insufficient_data": 0}

    for instrument in instruments:
        full = _load_full_ohlc(conn, instrument)
        indicators = _precompute_indicators(full, vix)
        n = len(full)

        for signal_type in signal_types:
            horizon = HORIZON_TRADING_DAYS[signal_type]
            atr_mult = ATR_STOP_MULTIPLE[signal_type]

            for i in range(MIN_TRAILING_BARS, n - horizon):
                hist = full.iloc[: i + 1]
                as_of_date = full["date"].iloc[i]

                trend = signal_factors.trend_factor(hist)
                momentum = signal_factors.momentum_breakout_factor(hist)
                candle = signal_factors.candle_pattern_factor(hist)
                oi_today, oi_prev = _oi_pair_for(oi_index, sorted_oi_dates, as_of_date)
                oi = signal_factors.oi_positioning_factor(oi_today, oi_prev)

                directional_factors = [trend, momentum, candle] + ([oi] if oi is not None else [])
                composite, direction, confidence = combine_factors(directional_factors)
                if direction == "neutral":
                    counts["neutral"] += 1
                    continue

                atr = average_true_range(hist)
                if not atr:
                    counts["insufficient_data"] += 1
                    continue

                entry_price = float(hist["close"].iloc[-1])
                risk = atr * atr_mult
                if direction == "bullish":
                    stop = entry_price - risk
                    target = entry_price + risk * RISK_REWARD_RATIO
                else:
                    stop = entry_price + risk
                    target = entry_price - risk * RISK_REWARD_RATIO

                future = full.iloc[i + 1 : i + 1 + horizon]
                outcome = triple_barrier.resolve(direction, entry_price, target, stop, future, RISK_REWARD_RATIO)
                counts[outcome.status] = counts.get(outcome.status, 0) + 1

                if outcome.status not in ("target_hit", "stop_hit"):
                    continue  # ambiguous (timed_out) or unresolvable -- not a training label

                row_ind = indicators.iloc[i]
                oi_deltas = _participant_type_deltas(oi_today, oi_prev)
                records.append(
                    {
                        "instrument": instrument,
                        "signal_type": signal_type,
                        "as_of_date": as_of_date,
                        "direction": direction,
                        "is_bullish": 1 if direction == "bullish" else 0,
                        "trend_score": trend.score,
                        "momentum_score": momentum.score,
                        # candle_pattern is recorded for inspection but is NOT an ML
                        # feature: its textbook directions failed measurement (see
                        # signal_factors.candle_pattern_factor), and adding it back as
                        # a feature chosen on the strength of that same measurement
                        # would be fitting to it.
                        "candle_pattern": candle.extra.get("pattern", "none"),
                        "oi_score": oi.score if oi is not None else 0.0,
                        "oi_available": oi is not None,
                        "composite_score": composite,
                        "rule_confidence": confidence,
                        "rsi_14": row_ind["rsi_14"],
                        "macd_hist": row_ind["macd_hist"],
                        "bollinger_pct_b": row_ind["bollinger_pct_b"],
                        "adx_14": row_ind["adx_14"],
                        "gap_pct": row_ind["gap_pct"],
                        "atr_pct": row_ind["atr_pct"],
                        "vol_percentile": row_ind["vol_percentile_fast"],
                        "days_left_in_month": row_ind["days_left_in_month"],
                        "sma50_100_spread": row_ind["sma50_100_spread"],
                        "roc_10": row_ind["roc_10"],
                        "volume_ratio": row_ind["volume_ratio"],
                        "prev_overnight_ret": row_ind["prev_overnight_ret"],
                        "intraday_ret": row_ind["intraday_ret"],
                        "is_turn_of_month": row_ind["is_turn_of_month"],
                        "is_monthly_expiry": row_ind["is_monthly_expiry"],
                        "vol_target_scale": row_ind["vol_target_scale"],
                        "vix_level": row_ind["vix_level"],
                        "vix_percentile": row_ind["vix_percentile"],
                        "vix_change_5d": row_ind["vix_change_5d"],
                        "vix_minus_realized": row_ind["vix_minus_realized"],
                        **oi_deltas,
                        "entry_price": entry_price,
                        "target_price": target,
                        "stop_loss_price": stop,
                        "outcome_status": outcome.status,
                        "resolved_date": outcome.resolved_date,
                        "bars_to_resolve": outcome.bars_to_resolve,
                        "target_hit": 1 if outcome.status == "target_hit" else 0,
                        "r_multiple": outcome.r_multiple,
                    }
                )

    df = pd.DataFrame.from_records(records)
    return df, counts


def persist_dataset_to_trade_signals(conn, df: pd.DataFrame, dimension_set: str | None = None) -> int:
    """Writes every historical prediction AND its realized outcome into
    trade_signals -- the durable record of "what we predicted / what actually
    happened", which is what lets the feedback loop rebuild its state table from the
    database rather than from a re-run of the whole backtest.

    Idempotent on (as_of_date, instrument, signal_type, model_version): re-running
    updates the existing row rather than piling up duplicates."""
    import json

    import psycopg2.extras

    rows = []
    for r in df.itertuples(index=False):
        factor_breakdown = {
            "trend_score": r.trend_score, "momentum_score": r.momentum_score,
            "candle_score": r.candle_score, "oi_score": r.oi_score,
            "rsi_14": r.rsi_14, "macd_hist": r.macd_hist, "adx_14": r.adx_14,
            "vol_percentile": r.vol_percentile,
        }
        rows.append(
            {
                "as_of_date": r.as_of_date,
                "instrument": r.instrument,
                "signal_type": r.signal_type,
                "direction": r.direction,
                "entry_price": r.entry_price,
                "target_price": r.target_price,
                "stop_loss_price": r.stop_loss_price,
                "confidence_score": r.rule_confidence,
                "factor_breakdown": json.dumps(factor_breakdown),
                "model_version": "rule_v1",
                "status": r.outcome_status,
                "resolved_date": r.resolved_date,
                "r_multiple": r.r_multiple,
                "market_state": getattr(r, "state", None),
                "state_dimension_set": dimension_set,
            }
        )

    sql = """
        INSERT INTO trade_signals
            (as_of_date, instrument, signal_type, direction, entry_price, target_price,
             stop_loss_price, confidence_score, factor_breakdown, model_version, status,
             resolved_date, r_multiple, market_state, state_dimension_set)
        VALUES
            (%(as_of_date)s, %(instrument)s, %(signal_type)s, %(direction)s, %(entry_price)s,
             %(target_price)s, %(stop_loss_price)s, %(confidence_score)s, %(factor_breakdown)s,
             %(model_version)s, %(status)s, %(resolved_date)s, %(r_multiple)s,
             %(market_state)s, %(state_dimension_set)s)
        ON CONFLICT (as_of_date, instrument, signal_type, model_version) DO UPDATE SET
            direction = EXCLUDED.direction, entry_price = EXCLUDED.entry_price,
            target_price = EXCLUDED.target_price, stop_loss_price = EXCLUDED.stop_loss_price,
            confidence_score = EXCLUDED.confidence_score,
            factor_breakdown = EXCLUDED.factor_breakdown, status = EXCLUDED.status,
            resolved_date = EXCLUDED.resolved_date, r_multiple = EXCLUDED.r_multiple,
            market_state = EXCLUDED.market_state,
            state_dimension_set = EXCLUDED.state_dimension_set
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    conn.commit()
    return len(rows)
