"""Robustness pass on the three setups that survived date-clustered testing.

The first run said something surprising: two setups predict the OPPOSITE of what
the textbook (and the user) expected. Distribution-at-top is bearish as expected,
but "buying pressure at the bottom" -- high volume, strong close, at 20-day lows --
keeps FALLING, and a breakout from a quiet tight base FADES.

A surprising result is exactly the kind that deserves adversarial checking before
anyone acts on it, so this asks the three questions that would most plausibly
expose it as an artifact:

  1. LATE HALF ALONE. Every result here leans on the early period. If an effect
     only existed in 2020-2023 it is history, not a strategy.
  2. LIQUIDITY. If the edge lives only in the least liquid tercile it is a
     spread-capture illusion that dies on contact with a real order.
  3. COSTS. Indian round-trip is ~10-20 bps. Gross alpha smaller than that is not
     a trade, however clean the t-stat.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

import analyze_setups_equities as base

CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("features_cache.pkl")
SPLIT = base.SPLIT
COST_BPS = 15.0


def get_frame() -> pd.DataFrame:
    if CACHE.exists():
        print(f"Using cached features: {CACHE}")
        return pd.read_pickle(CACHE)
    from algotrade import db
    from algotrade.config import load_postgres_config
    with db.get_connection(load_postgres_config()) as conn:
        df = base.load(conn)
    print("Computing features...")
    df = base.add_features(df)
    df = df.dropna(subset=["fwd_1d", "vol_ratio", "close_pos"])
    df.to_pickle(CACHE)   # pickle, not parquet: no arrow engine here and this is scratch
    return df


def t_of(exc: pd.Series) -> float:
    exc = exc.dropna()
    if len(exc) < 20 or exc.std() == 0:
        return float("nan")
    return float(exc.mean() / (exc.std() / np.sqrt(len(exc))))


def daily_excess(sub: pd.DataFrame, data: pd.DataFrame, h: str) -> pd.Series:
    base_by_date = data.groupby("date")[h].mean()
    daily = sub.groupby("date")[h].mean()
    return (daily - base_by_date.reindex(daily.index)).dropna()


def main() -> None:
    df = get_frame()
    print(f"{len(df):,} stock-days, {df['sym'].nunique():,} symbols\n")

    # Liquidity terciles, computed once per symbol on its median turnover.
    med = df.groupby("sym")["turnover"].median()
    df["liq"] = df["sym"].map(pd.qcut(med, 3, labels=["low", "mid", "high"]))

    masks = base.setups(df)
    interesting = {
        "2. volume-base breakout": ("fwd_5d", masks["2. volume-base breakout (quiet tight base -> surge)"]),
        "3. selling pressure top": ("fwd_10d", masks["3. selling pressure at top (distribution)"]),
        "4. buying pressure bottom": ("fwd_10d", masks["4. buying pressure at bottom (accumulation)"]),
        "1. resistance breakout": ("fwd_1d", masks["1. resistance breakout (>=3 touches, vol>1.2x)"]),
    }

    print("=" * 100)
    print("CHECK 1 & 3: late half alone, and net of a 15 bps round trip")
    print("=" * 100)
    print(f"{'setup':<26} {'horiz':>7} {'full bps':>9} {'t':>6} | {'LATE bps':>9} "
          f"{'late t':>7} {'late n':>7} | {'net late':>9}")
    for name, (h, mask) in interesting.items():
        sub = df[mask.fillna(False)]
        exc = daily_excess(sub, df, h)
        late = exc[exc.index >= SPLIT]
        net = abs(late.mean() * 10000) - COST_BPS      # traded in its own direction
        print(f"{name:<26} {h:>7} {exc.mean()*10000:>9.1f} {t_of(exc):>6.2f} | "
              f"{late.mean()*10000:>9.1f} {t_of(late):>7.2f} {len(late):>7} | {net:>9.1f}")

    print("\n" + "=" * 100)
    print("CHECK 2: does the effect survive in liquid names, or only in the dregs?")
    print("=" * 100)
    print(f"{'setup':<26} {'horiz':>7} | " + " | ".join(f"{l:>18}" for l in ("low liq", "mid liq", "high liq")))
    for name, (h, mask) in interesting.items():
        cells = []
        for liq in ("low", "mid", "high"):
            sub = df[mask.fillna(False) & (df["liq"] == liq)]
            if len(sub) < 100:
                cells.append(f"{'too few':>18}")
                continue
            exc = daily_excess(sub, df[df["liq"] == liq], h)
            cells.append(f"{exc.mean()*10000:>10.1f} (t{t_of(exc):>5.1f})")
        print(f"{name:<26} {h:>7} | " + " | ".join(cells))

    print("\n" + "=" * 100)
    print("A finding must clear all three: late-half t past ~2, present in high")
    print("liquidity, and gross alpha bigger than the 15 bps it costs to capture.")
    print("=" * 100)


if __name__ == "__main__":
    main()
