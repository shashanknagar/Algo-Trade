"""Bridges the single-stock finding to the index model you actually trade.

WHAT SURVIVED: of everything tested on 1,594 stocks, exactly one cell cleared all
three bars -- stocks drifting to 20-day lows on NO volume underperform by 54.6 bps
over 10 days (t=-12.77), still -23.4 bps with t=-3.47 in the held-out late half, and
present in the most liquid tercile (t=-7.3). Nothing else was both stable and
tradeable net of costs.

But that is a SHORT signal on individual stocks, and you trade index options. Two
things follow, and this script tests the second:

  1. As an EXCLUSION FILTER it is free -- you are not paying a round trip to learn
     "don't be long this name", so the full effect is available.
  2. AGGREGATED it becomes market breadth: the fraction of the universe sitting in
     that state is a market-state variable with one value per day. Breadth
     deterioration is a classic regime indicator, and unlike the single-stock
     version it applies directly to NIFTY.

Tested here: does the breadth of quiet-new-lows predict NIFTY forward returns, and
-- more importantly given what this project has actually established -- does it
predict forward VOLATILITY, which is the thing that has proven predictable?
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from algotrade import db, ml_dataset
from algotrade.config import load_postgres_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_setups_equities import SPLIT

CACHE = Path(sys.argv[1])


def t_of(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) >= 20 and x.std() else float("nan")


def build_breadth(df: pd.DataFrame) -> pd.DataFrame:
    at_low = df["close"] <= 1.015 * df["lo20"]
    at_high = df["close"] >= 0.985 * df["hi20"]
    spike = df["vol_ratio"] > 1.3
    df = df.assign(quiet_low=(at_low & ~spike), at_high=at_high, at_low=at_low)
    b = df.groupby("date").agg(
        quiet_low_pct=("quiet_low", "mean"),
        at_low_pct=("at_low", "mean"),
        at_high_pct=("at_high", "mean"),
        n=("close", "size"),
    ).reset_index()
    b["net_hl"] = b["at_high_pct"] - b["at_low_pct"]
    return b[b["n"] >= 200]


def main() -> None:
    pd.set_option("display.width", 200)
    df = pd.read_pickle(CACHE)
    breadth = build_breadth(df)
    print(f"Breadth series: {len(breadth)} days, {breadth['date'].min()} to {breadth['date'].max()}")

    with db.get_connection(load_postgres_config()) as conn:
        nifty = ml_dataset._load_full_ohlc(conn, "NIFTY")[["date", "close", "high", "low"]]

    m = breadth.merge(nifty, on="date", how="inner").sort_values("date").reset_index(drop=True)
    print(f"Merged with NIFTY: {len(m)} days\n")

    m["ret_fwd_5"] = m["close"].shift(-5) / m["close"] - 1
    m["ret_fwd_10"] = m["close"].shift(-10) / m["close"] - 1
    m["logret"] = np.log(m["close"]).diff()
    # Forward realized vol -- the quantity this project HAS shown to be predictable.
    m["vol_fwd_10"] = m["logret"].shift(-10).rolling(10).std() * np.sqrt(252) * 100
    m["vol_trail_10"] = m["logret"].rolling(10).std() * np.sqrt(252) * 100

    for col in ("quiet_low_pct", "net_hl"):
        # Percentile vs its own trailing history: levels drift, ranks are comparable.
        m[f"{col}_pct"] = m[col].rolling(252, min_periods=120).rank(pct=True)

    print("=" * 100)
    print("Does breadth predict NIFTY? (quintiles of each breadth measure)")
    print("=" * 100)
    for signal in ("quiet_low_pct_pct", "net_hl_pct"):
        print(f"\n  {signal}")
        d = m.dropna(subset=[signal, "ret_fwd_10", "vol_fwd_10"]).copy()
        d["q"] = pd.qcut(d[signal], 5, labels=[1, 2, 3, 4, 5])
        print(f"    {'quintile':<10} {'n':>5} {'fwd10 ret bps':>14} {'fwd10 vol':>11} | "
              f"{'early ret':>10} {'late ret':>9} {'agree':>6}")
        for q, g in d.groupby("q", observed=True):
            e, l = g[g["date"] < SPLIT], g[g["date"] >= SPLIT]
            agree = ("YES" if len(e) >= 20 and len(l) >= 20
                     and (e["ret_fwd_10"].mean() > 0) == (l["ret_fwd_10"].mean() > 0) else "no")
            print(f"    Q{q:<9} {len(g):>5} {g['ret_fwd_10'].mean()*10000:>14.1f} "
                  f"{g['vol_fwd_10'].mean():>11.2f} | {e['ret_fwd_10'].mean()*10000:>10.1f} "
                  f"{l['ret_fwd_10'].mean()*10000:>9.1f} {agree:>6}")

    print("\n" + "=" * 100)
    print("Correlation with FORWARD 10d volatility (controlling for trailing vol)")
    print("=" * 100)
    d = m.dropna(subset=["quiet_low_pct", "net_hl", "vol_fwd_10", "vol_trail_10"])
    for col in ("quiet_low_pct", "net_hl"):
        raw = d[col].corr(d["vol_fwd_10"])
        # Residual correlation: does breadth say anything trailing vol does not?
        resid = d["vol_fwd_10"] - np.poly1d(np.polyfit(d["vol_trail_10"], d["vol_fwd_10"], 1))(d["vol_trail_10"])
        print(f"  {col:<18} raw corr {raw:>7.3f} | residual (beyond trailing vol) {d[col].corr(resid):>7.3f}")
    print(f"  {'vol_trail_10':<18} raw corr {d['vol_trail_10'].corr(d['vol_fwd_10']):>7.3f}   (the benchmark)")


if __name__ == "__main__":
    main()
