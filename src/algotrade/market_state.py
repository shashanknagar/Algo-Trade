"""Discretizes a signal's market context into a compact, human-readable STATE
SIGNATURE, so that "what happened last time the market looked like this?" becomes a
literal lookup rather than an opaque model weight.

This is the case-based / empirical-hit-rate approach: bucket every historical signal
by its state, count how often the target was actually hit in that state, and use that
observed rate (shrunk toward the base rate -- see state_learning.py) as the
prediction for a new signal in the same state.

Deliberately LOW-CARDINALITY buckets. Every extra dimension multiplies the number of
states and divides the samples per state; a state with 3 historical occurrences
showing "100% hit rate" is noise, not knowledge. Bucket thresholds below are chosen
to be conventional and roughly balanced, and are documented rather than tuned against
the test set.
"""
from __future__ import annotations

import pandas as pd

# Which dimensions form the state key. "coarse" keeps buckets well-populated;
# "fine" adds breakout at the cost of ~3x more states.
STATE_DIMENSION_SETS = {
    "coarse": ("signal_type", "direction", "trend", "vol_regime"),
    "fine": ("signal_type", "direction", "trend", "vol_regime", "breakout"),
}


def _bucket_trend(trend_score: float) -> str:
    # trend_score is spread_pct * 20 clipped to [-1, 1]; 0.2 ~= a 1% SMA20/50 spread.
    if trend_score > 0.2:
        return "bull"
    if trend_score < -0.2:
        return "bear"
    return "flat"


def _bucket_vol(vol_percentile: float) -> str:
    if vol_percentile < 0.33:
        return "lowvol"
    if vol_percentile > 0.67:
        return "highvol"
    return "normvol"


def _bucket_breakout(momentum_score: float) -> str:
    if momentum_score > 0.05:
        return "breakup"
    if momentum_score < -0.05:
        return "breakdown"
    return "norange"


def _bucket_oi(oi_score: float, oi_available: bool) -> str:
    if not oi_available:
        return "oina"
    if oi_score > 0.05:
        return "oibull"
    if oi_score < -0.05:
        return "oibear"
    return "oiflat"


def _bucket_rsi(rsi_14: float) -> str:
    if rsi_14 < 35:
        return "oversold"
    if rsi_14 > 65:
        return "overbought"
    return "rsimid"


def state_components(row) -> dict[str, str]:
    """All available state dimensions for one signal row. Callers pick which subset
    forms the key (see STATE_DIMENSION_SETS)."""
    return {
        "signal_type": row["signal_type"],
        "direction": row["direction"],
        "trend": _bucket_trend(row["trend_score"]),
        "vol_regime": _bucket_vol(row["vol_percentile"]),
        "breakout": _bucket_breakout(row["momentum_score"]),
        "oi_bias": _bucket_oi(row["oi_score"], bool(row["oi_available"])),
        "rsi_zone": _bucket_rsi(row["rsi_14"]),
    }


def add_state_column(df: pd.DataFrame, dimensions: tuple[str, ...]) -> pd.DataFrame:
    """Adds a 'state' column: the '|'-joined values of the chosen dimensions."""
    out = df.copy()
    components = out.apply(lambda r: state_components(r), axis=1)
    for dim in set(d for dims in STATE_DIMENSION_SETS.values() for d in dims) | {"oi_bias", "rsi_zone"}:
        out[f"state_{dim}"] = components.apply(lambda c, d=dim: c[d])
    out["state"] = components.apply(lambda c: "|".join(str(c[d]) for d in dimensions))
    return out
