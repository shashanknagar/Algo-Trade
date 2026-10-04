"""Tests two strategy hypotheses that fall out of this project's own findings, both
fundamentally different from the directional trend-following model built so far.

WHY THESE TWO:
Feature-importance analysis showed volatility-related features carrying ~44% of the
model's total split gain while directional features stayed weak -- and the direction
picker has scored at or below a coin flip in every configuration tested. That is the
classic finance asymmetry: returns are near-unpredictable, volatility is forecastable.
We have been attacking the hard problem and ignoring the tractable one.

HYPOTHESIS A -- VARIANCE RISK PREMIUM (trade volatility, not direction)
Does India VIX (implied vol) systematically exceed the volatility that subsequently
occurs? If so, selling volatility has positive expected value and, importantly, a
naturally HIGH win rate -- which is the only honest route to a 70%+ hit rate. The
catch (quantified below) is negative skew: many small wins, occasional large losses.

HYPOTHESIS B -- MEAN REVERSION (we may be trading the wrong way round)
The rule-based model is trend-following (SMA crossover + breakout). But measurement
kept producing reversal signals: bearish engulfing preceded UP moves, high-volume
breakdowns preceded strong UP moves (+51 bps, replicated across both sample halves).
If Indian index returns mean-revert at short horizons, trend-following is structurally
backwards.

Both are checked on non-overlapping samples and split train/test, because overlapping
windows are what invalidated earlier results in this project.

Usage:
    python scripts/analyze_alternatives.py
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from algotrade import db, ml_dataset
from algotrade.config import load_postgres_config

VIX_HORIZON_DAYS = 21  # ~30 calendar days, matching India VIX's own horizon
SPLIT = date(2024, 1, 1)


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std() == 0:
        return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def load(conn) -> pd.DataFrame:
    nifty = ml_dataset._load_full_ohlc(conn, "NIFTY")
    vix = ml_dataset._load_vix(conn)
    df = nifty.merge(vix, on="date", how="inner").sort_values("date").reset_index(drop=True)

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    # Volatility that ACTUALLY occurred over the NEXT VIX_HORIZON_DAYS, annualized.
    # shift(-1) so the window starts strictly after today; rolling() then covers
    # log_ret[i+1 .. i+21]; the final shift realigns that window back onto day i.
    fwd_std = (
        df["log_ret"].shift(-1).rolling(VIX_HORIZON_DAYS).std().shift(-(VIX_HORIZON_DAYS - 1))
    )
    df["fwd_realized_vol"] = fwd_std * math.sqrt(252)
    df["implied_vol"] = df["vix"] / 100.0
    df["vrp"] = df["implied_vol"] - df["fwd_realized_vol"]
    return df


def hypothesis_a(df: pd.DataFrame) -> None:
    print("=" * 78)
    print("HYPOTHESIS A: VARIANCE RISK PREMIUM (is implied vol > subsequent realized?)")
    print("=" * 78)

    valid = df.dropna(subset=["vrp"])
    # Non-overlapping: one observation per VIX_HORIZON_DAYS, since forward windows
    # sampled daily share almost all their data.
    indep = valid.iloc[::VIX_HORIZON_DAYS]

    for label, sub in (("ALL (overlapping -- inflated n)", valid), ("NON-OVERLAPPING", indep)):
        pos = float((sub["vrp"] > 0).mean())
        print(f"\n{label}: n={len(sub)}")
        print(f"  mean implied vol      : {sub['implied_vol'].mean():.1%}")
        print(f"  mean realized vol      : {sub['fwd_realized_vol'].mean():.1%}")
        print(f"  mean premium (VRP)     : {sub['vrp'].mean():+.2%}   t={tstat(sub['vrp']):.2f}")
        print(f"  % of periods VRP > 0   : {pos:.1%}   <- this is the natural 'win rate'")

    print("\n--- Replication: does the premium hold in both halves? ---")
    for label, sub in (("early", indep[indep["date"] < SPLIT]), ("late", indep[indep["date"] >= SPLIT])):
        if len(sub) < 5:
            continue
        print(f"  {label:5s} n={len(sub):>3}  mean VRP={sub['vrp'].mean():+.2%}  "
              f"t={tstat(sub['vrp']):>5.2f}  win rate={float((sub['vrp'] > 0).mean()):.1%}")

    print("\n--- When is the premium largest? (by VIX level, non-overlapping) ---")
    indep = indep.copy()
    indep["vix_bucket"] = pd.qcut(indep["vix"], 4, labels=["low", "mid-low", "mid-high", "high"])
    for name, g in indep.groupby("vix_bucket", observed=True):
        print(f"  VIX {str(name):<9} n={len(g):>3}  avg VIX={g['vix'].mean():>5.1f}  "
              f"mean VRP={g['vrp'].mean():+.2%}  win rate={float((g['vrp'] > 0).mean()):.1%}")

    print("\n--- THE CATCH: shape of the losses (non-overlapping) ---")
    wins, losses = indep[indep["vrp"] > 0]["vrp"], indep[indep["vrp"] <= 0]["vrp"]
    if len(wins) and len(losses):
        print(f"  avg win  : {wins.mean():+.2%}   (n={len(wins)})")
        print(f"  avg loss : {losses.mean():+.2%}   (n={len(losses)})")
        print(f"  worst    : {indep['vrp'].min():+.2%}")
        print(f"  win/loss size ratio: {abs(wins.mean() / losses.mean()):.2f}x")
        print("  ^ A high win rate with a <1x size ratio is the classic premium-selling")
        print("    profile: wins often, loses big. Win rate ALONE would be misleading.")


def hypothesis_b(df: pd.DataFrame) -> None:
    print("\n\n" + "=" * 78)
    print("HYPOTHESIS B: MEAN REVERSION vs TREND (are we trading the wrong direction?)")
    print("=" * 78)

    d = df.copy()
    d["fwd_5d"] = d["close"].shift(-5) / d["close"] - 1
    d["past_5d"] = d["close"] / d["close"].shift(5) - 1

    print("\n--- Return autocorrelation (negative = mean-reverting, positive = trending) ---")
    for lag in (1, 2, 3, 5, 10):
        ac = d["log_ret"].autocorr(lag=lag)
        tag = "mean-reverting" if ac < -0.02 else "trending" if ac > 0.02 else "~none"
        print(f"  lag {lag:>2}d: autocorr={ac:+.4f}  {tag}")

    print("\n--- Forward 5d return by PAST 5d return quintile (non-overlapping, every 5th day) ---")
    print("    If mean-reverting, the worst past quintile should have the BEST forward return.")
    for label, sub in (("FULL", d), ("early", d[d["date"] < SPLIT]), ("late", d[d["date"] >= SPLIT])):
        s = sub.dropna(subset=["fwd_5d", "past_5d"]).iloc[::5].copy()
        if len(s) < 30:
            continue
        s["q"] = pd.qcut(s["past_5d"], 5, labels=["worst", "q2", "q3", "q4", "best"])
        parts = []
        for name, g in s.groupby("q", observed=True):
            parts.append(f"{name}={g['fwd_5d'].mean()*10000:+6.1f}")
        spread = (s[s["q"] == "worst"]["fwd_5d"].mean() - s[s["q"] == "best"]["fwd_5d"].mean()) * 10000
        print(f"  {label:5s} (n={len(s):>3}): " + "  ".join(parts) + f"   | worst-minus-best spread = {spread:+.1f} bps")

    print("\n  A POSITIVE worst-minus-best spread in BOTH halves = genuine mean reversion")
    print("  (buy weakness, sell strength). A negative spread = momentum. A sign flip = noise.")


def main() -> None:
    pd.set_option("display.width", 200)
    config = load_postgres_config()
    with db.get_connection(config) as conn:
        df = load(conn)
    print(f"NIFTY + India VIX aligned: {len(df)} days, {df['date'].min()} to {df['date'].max()}\n")
    hypothesis_a(df)
    hypothesis_b(df)


if __name__ == "__main__":
    main()
