"""Tests four specific, widely-used chart setups that are more precise than the
generic breakout factor already measured:

  1. RESISTANCE BREAKOUT -- breaking a level price has repeatedly FAILED at (multiple
     touches), not merely a rolling 20-day high. Prior rejection is what makes a level
     meaningful.
  2. VOLUME-BASE BREAKOUT -- contraction then expansion: a quiet, tight base (volume
     and range both compressed) that then breaks out on a volume surge. This is the
     volatility-contraction idea used by Minervini/O'Neil-style momentum traders.
  3. SELLING PRESSURE AT THE TOP (distribution) -- price near highs, heavy volume, but
     closing weak in its own range: buyers absorbed.
  4. BUYING PRESSURE AT THE BOTTOM (accumulation/capitulation) -- price near lows,
     heavy volume, closing strong in its range: sellers exhausted.

3 and 4 are Wyckoff distribution/accumulation logic. 4 is the closest formalization of
this project's strongest replicated finding (high-volume breakdowns preceded strong UP
moves, +24 then +51 bps across both sample halves).

Every setup is measured against the SAME baseline drift and split early/late, because
in this project the only findings that survived were the ones that kept their sign in
both halves.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from algotrade import db, ml_dataset
from algotrade.config import load_postgres_config

SPLIT = date(2024, 1, 1)


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std() == 0:
        return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["avg_vol_20"] = d["volume"].rolling(20).mean()
    d["avg_vol_50"] = d["volume"].rolling(50).mean()
    d["avg_vol_10"] = d["volume"].rolling(10).mean()
    d["vol_ratio"] = d["volume"] / d["avg_vol_20"]

    d["hi20"] = d["high"].rolling(20).max()
    d["lo20"] = d["low"].rolling(20).min()
    d["hi20_prev"] = d["hi20"].shift(1)
    d["lo20_prev"] = d["lo20"].shift(1)

    # Where in its own daily range did it close? 1 = at the high, 0 = at the low.
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["close_pos"] = (d["close"] - d["low"]) / rng

    # Touches: how many of the last 20 days poked within 1% of the 20d high/low --
    # a level rejected repeatedly is a real level, one touched once is not.
    near_hi = (d["high"] >= 0.99 * d["hi20"]).astype(int)
    near_lo = (d["low"] <= 1.01 * d["lo20"]).astype(int)
    d["touches_hi"] = near_hi.rolling(20).sum().shift(1)
    d["touches_lo"] = near_lo.rolling(20).sum().shift(1)

    # Base tightness: 10-day range as a fraction of price
    d["range10"] = (d["high"].rolling(10).max() - d["low"].rolling(10).min()) / d["close"]

    for h in (1, 5, 10):
        d[f"fwd_{h}d"] = d["close"].shift(-h) / d["close"] - 1
    return d


def setups(d: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "1. resistance breakout (>=3 prior touches, vol>1.2x)": (
            (d["close"] > d["hi20_prev"]) & (d["touches_hi"] >= 3) & (d["vol_ratio"] > 1.2)
        ),
        "1b. resistance breakout (>=3 touches, any volume)": (
            (d["close"] > d["hi20_prev"]) & (d["touches_hi"] >= 3)
        ),
        "2. volume-base breakout (quiet tight base -> surge)": (
            (d["avg_vol_10"] < 0.85 * d["avg_vol_50"]) & (d["range10"] < 0.05)
            & (d["close"] > d["hi20_prev"]) & (d["vol_ratio"] > 1.3)
        ),
        "3. selling pressure at top (distribution)": (
            (d["close"] >= 0.985 * d["hi20"]) & (d["vol_ratio"] > 1.3) & (d["close_pos"] < 0.35)
        ),
        "4. buying pressure at bottom (accumulation)": (
            (d["close"] <= 1.015 * d["lo20"]) & (d["vol_ratio"] > 1.3) & (d["close_pos"] > 0.65)
        ),
    }


def report(data: pd.DataFrame, horizon: str) -> None:
    print(f"\n{'=' * 100}")
    print(f"FORWARD {horizon.upper()} RETURN BY SETUP  (bps relative to that period's own baseline drift)")
    print(f"{'=' * 100}")
    print(f"{'setup':<52} {'n':>5} {'full':>9} {'t':>6} | {'early n':>7} {'early':>8} | {'late n':>6} {'late':>8} {'agree':>6}")

    early_all, late_all = data[data["date"] < SPLIT], data[data["date"] >= SPLIT]
    base_f = data[horizon].mean()
    base_e, base_l = early_all[horizon].mean(), late_all[horizon].mean()

    for name, mask in setups(data).items():
        sub = data[mask]
        e, l = sub[sub["date"] < SPLIT], sub[sub["date"] >= SPLIT]
        if len(sub) < 10:
            print(f"{name:<52} {len(sub):>5}  too few occurrences to measure")
            continue
        f_rel = (sub[horizon].mean() - base_f) * 10000
        e_rel = (e[horizon].mean() - base_e) * 10000 if len(e) >= 5 else float("nan")
        l_rel = (l[horizon].mean() - base_l) * 10000 if len(l) >= 5 else float("nan")
        agree = "YES" if (not np.isnan(e_rel) and not np.isnan(l_rel) and (e_rel > 0) == (l_rel > 0)) else "no"
        print(f"{name:<52} {len(sub):>5} {f_rel:>9.1f} {tstat(sub[horizon]):>6.2f} | "
              f"{len(e):>7} {e_rel:>8.1f} | {len(l):>6} {l_rel:>8.1f} {agree:>6}")


def main() -> None:
    pd.set_option("display.width", 220)
    config = load_postgres_config()
    frames = []
    with db.get_connection(config) as conn:
        for inst in ("NIFTY", "BANKNIFTY"):
            d = prepare(ml_dataset._load_full_ohlc(conn, inst))
            d["instrument"] = inst
            frames.append(d)
    data = pd.concat(frames, ignore_index=True).dropna(subset=["fwd_1d", "vol_ratio", "close_pos"])
    print(f"Observations: {len(data)} index-days ({data['date'].min()} to {data['date'].max()})")

    for h in ("fwd_1d", "fwd_5d", "fwd_10d"):
        report(data, h)

    print("\n'agree' = did the setup keep its SIGN in both halves? That column is the")
    print("whole test. In this project every finding that failed to keep its sign")
    print("across halves later evaporated, and every one that kept it held up.")
    print("\nNOTE ON SAMPLE SIZE: these are restrictive conditions on only 2 correlated")
    print("indices. Rare setups will show few occurrences -- that is a limit of the")
    print("instrument universe, not of the idea, and is exactly what expanding to")
    print("equities would fix.")


if __name__ == "__main__":
    main()
