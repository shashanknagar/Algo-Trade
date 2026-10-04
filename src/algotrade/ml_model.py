"""Trains and evaluates a signal-quality classifier: given the rule-based direction
call (signal_generator.py) and its factor scores, predicts P(target hit before stop).
This operationalizes the master plan's "Gate" step -- only surface signals above a
confidence threshold -- with that threshold's real hit-rate measured out-of-sample,
not assumed.

Chronological train/test split ONLY -- never a random shuffle. Financial time series
are autocorrelated; shuffling would leak future information into training and produce
a fake, inflated in-sample-looking score. The test period is touched only for final
evaluation, never for picking features, weights, or thresholds.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

FEATURE_COLUMNS = [
    # candle_score was dropped: the factor no longer contributes a direction (its
    # textbook signs failed measurement), so the column was constant-zero -- a
    # dimension carrying no information.
    "trend_score", "momentum_score", "oi_score", "oi_available",
    "vol_percentile", "composite_score", "is_bullish",
    "rsi_14", "macd_hist", "bollinger_pct_b", "adx_14", "gap_pct", "atr_pct",
    "days_left_in_month", "sma50_100_spread", "roc_10",
    "oi_delta_fii", "oi_delta_dii", "oi_delta_pro", "oi_delta_client",
    # Volume: the volume x breakout interaction replicated across both sample halves
    # (inverted vs the textbook assumption), so the raw ratio is given to the model
    # to learn from rather than hardcoded as a rule.
    "volume_ratio",
    # India VIX: genuinely forward-looking (implied vol), not another price
    # transformation. vix_level and vix_change_5d are recorded in the dataset but
    # deliberately left out here as largely redundant with these two -- the sample is
    # too small to spend dimensions on near-duplicates.
    "vix_percentile", "vix_minus_realized",
    # Session decomposition + calendar effect: both replicated across sample
    # halves on both instruments (see scripts/analyze_candidate_strategies.py).
    "prev_overnight_ret", "vol_target_scale", "is_monthly_expiry",
    # Today's intraday session direction predicts TOMORROW'S overnight gap
    # (16.3 bps after an up session vs 3.8 after a down one, t=6.32, n=802,
    # replicating in both halves). Near-independent day to day, so this is the
    # best-powered signal found in this project.
    "intraday_ret",
]

# Breakeven hit-rate at the plan's 1.5 risk:reward: lose 1R on a loss, win 1.5R on a
# win -> breakeven at 1 / (1 + 1.5) = 40%. Anything reliably above this is genuinely
# profitable in expectancy terms even if nowhere near 70%.
BREAKEVEN_HIT_RATE = 1 / (1 + 1.5)


def drop_rows_with_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Early rows for each instrument lack a full 252-day volatility lookback (and
    similar warm-up periods for the other indicators), so some FEATURE_COLUMNS are
    NaN there. LogisticRegression can't accept NaN at all; dropping these rows
    (rather than imputing some arbitrary fill value) is the safer choice given we
    have plenty of data -- this only affects each instrument's earliest ~1 year."""
    before = len(df)
    cleaned = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    dropped = before - len(cleaned)
    if dropped:
        print(f"Dropped {dropped} rows with missing indicator warm-up data (kept {len(cleaned)}).")
    return cleaned


def train_test_split_by_date(df: pd.DataFrame, cutoff: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["as_of_date"] < cutoff].copy()
    test = df[df["as_of_date"] >= cutoff].copy()
    return train, test


def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    X["oi_available"] = X["oi_available"].astype(int)
    return X


def train_models(train_df: pd.DataFrame) -> dict:
    X = _prep_features(train_df)
    y = train_df["target_hit"]
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=42),
    }
    for model in models.values():
        model.fit(X, y)
    return models


def evaluate(
    model, test_df: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8),
) -> pd.DataFrame:
    X = _prep_features(test_df)
    proba = model.predict_proba(X)[:, 1]
    scored = test_df.copy()
    scored["predicted_proba"] = proba

    rows = []
    for t in thresholds:
        subset = scored[scored["predicted_proba"] >= t]
        n = len(subset)
        if n == 0:
            rows.append({"threshold": t, "n_trades": 0, "hit_rate": None, "expectancy_r": None})
            continue
        rows.append(
            {
                "threshold": t,
                "n_trades": n,
                "hit_rate": round(float(subset["target_hit"].mean()), 3),
                "expectancy_r": round(float(subset["r_multiple"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)
