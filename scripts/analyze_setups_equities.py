"""Re-tests the four chart setups on the FULL EQUITY UNIVERSE instead of two indices.

WHY THIS RUN MATTERS: on NIFTY+BANKNIFTY, two of these setups were untestable --
the volume-base breakout fired 4 times and accumulation 16 times in six years. That
was never evidence against them; it was a sample-size failure. These are SINGLE-STOCK
patterns, and an index is an average of fifty stocks, which smooths away exactly the
tight bases and capitulation lows the patterns are defined by.

With ~3,400 symbols the same definitions should fire thousands of times, which is the
first time this project has had the statistical power to give an honest verdict.

LIQUIDITY FILTER: microcaps have stale prints, huge spreads and unreliable volume;
including them would swamp the sample with noise that no one could actually trade.
Only symbols with sufficient median daily turnover and history are kept.

Split early/late as always -- sign agreement across halves remains the filter.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from algotrade import db
from algotrade.config import load_postgres_config
from algotrade.market_calendar import IST

SPLIT = date(2024, 1, 1)
MIN_DAYS = 500
MIN_MEDIAN_TURNOVER = 5_000_000     # rupees/day -- excludes untradeable microcaps


def load(conn) -> pd.DataFrame:
    print("Loading equity universe...")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT trading_symbol, ohlc_bars.time, open, high, low, close, volume "
            "FROM ohlc_bars WHERE source = 'nse_bhavcopy' AND close > 0 "
            "ORDER BY trading_symbol, 2"
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["sym", "time", "open", "high", "low", "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["date"] = df["time"].apply(lambda t: t.astimezone(IST).date())
    df = df.drop(columns=["time"]).dropna(subset=["open", "high", "low", "close", "volume"])
    print(f"  {len(df):,} rows, {df['sym'].nunique():,} symbols")

    df["turnover"] = df["close"] * df["volume"]
    stats = df.groupby("sym").agg(days=("close", "size"), med_to=("turnover", "median"))
    keep = stats[(stats["days"] >= MIN_DAYS) & (stats["med_to"] >= MIN_MEDIAN_TURNOVER)].index
    df = df[df["sym"].isin(keep)].reset_index(drop=True)
    print(f"  after liquidity filter: {len(df):,} rows, {df['sym'].nunique():,} symbols")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("sym", sort=False)
    df["avg_vol_20"] = g["volume"].transform(lambda s: s.rolling(20).mean())
    df["avg_vol_50"] = g["volume"].transform(lambda s: s.rolling(50).mean())
    df["avg_vol_10"] = g["volume"].transform(lambda s: s.rolling(10).mean())
    df["vol_ratio"] = df["volume"] / df["avg_vol_20"]

    df["hi20_prev"] = g["high"].transform(lambda s: s.rolling(20).max().shift(1))
    df["lo20_prev"] = g["low"].transform(lambda s: s.rolling(20).min().shift(1))
    df["hi20"] = g["high"].transform(lambda s: s.rolling(20).max())
    df["lo20"] = g["low"].transform(lambda s: s.rolling(20).min())

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_pos"] = (df["close"] - df["low"]) / rng

    # How many of the last 20 days came within 1% of the running 20d high -- a level
    # touched repeatedly is what makes it "resistance" rather than just a high.
    near_hi = (df["high"] >= 0.99 * df["hi20"]).astype(float)
    df["touches_hi"] = near_hi.groupby(df["sym"], sort=False).transform(
        lambda s: s.rolling(20).sum().shift(1))

    hi10 = g["high"].transform(lambda s: s.rolling(10).max())
    lo10 = g["low"].transform(lambda s: s.rolling(10).min())
    df["range10"] = (hi10 - lo10) / df["close"]

    for h in (1, 5, 10):
        df[f"fwd_{h}d"] = g["close"].transform(lambda s, h=h: s.shift(-h) / s - 1)
    return df


def setups(d: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "1. resistance breakout (>=3 touches, vol>1.2x)": (
            (d["close"] > d["hi20_prev"]) & (d["touches_hi"] >= 3) & (d["vol_ratio"] > 1.2)),
        "2. volume-base breakout (quiet tight base -> surge)": (
            (d["avg_vol_10"] < 0.85 * d["avg_vol_50"]) & (d["range10"] < 0.10)
            & (d["close"] > d["hi20_prev"]) & (d["vol_ratio"] > 1.3)),
        "3. selling pressure at top (distribution)": (
            (d["close"] >= 0.985 * d["hi20"]) & (d["vol_ratio"] > 1.3) & (d["close_pos"] < 0.35)),
        "4. buying pressure at bottom (accumulation)": (
            (d["close"] <= 1.015 * d["lo20"]) & (d["vol_ratio"] > 1.3) & (d["close_pos"] > 0.65)),
    }


def clustered_t(sub: pd.DataFrame, horizon: str, baseline_by_date: pd.Series) -> tuple[float, int]:
    """t-stat on the setup's DAILY excess return, not on pooled stock-days.

    THIS IS THE WHOLE BALLGAME. Stocks on the same date share a market factor, so
    58,000 pooled stock-days behave far more like ~1,700 independent days. Pooling
    them ignores that correlation and inflates every t-stat by roughly sqrt(stocks
    per day) -- easily 5-10x. Averaging within each date first, then testing across
    dates, is the standard Fama-MacBeth fix and gives an honest n.
    """
    daily = sub.groupby("date")[horizon].mean()
    excess = (daily - baseline_by_date.reindex(daily.index)).dropna()
    if len(excess) < 20 or excess.std() == 0:
        return float("nan"), len(excess)
    return float(excess.mean() / (excess.std() / np.sqrt(len(excess)))), len(excess)


def report(data: pd.DataFrame, horizon: str) -> None:
    print(f"\n{'=' * 116}")
    print(f"FORWARD {horizon.upper()} EXCESS RETURN (bps vs same-day cross-sectional mean)")
    print(f"{'=' * 116}")
    print(f"{'setup':<52} {'obs':>8} {'days':>6} {'bps':>8} {'t':>7} | "
          f"{'early bps':>10} {'late bps':>9} {'agree':>6}")

    # The right benchmark is the market itself on that same day: a setup that fires
    # mostly on strong days would otherwise look predictive when it is just present.
    base_by_date = data.groupby("date")[horizon].mean()

    for name, mask in setups(data).items():
        sub = data[mask.fillna(False)]
        if len(sub) < 100:
            print(f"{name:<52} {len(sub):>8}  too few occurrences")
            continue
        t, ndays = clustered_t(sub, horizon, base_by_date)
        daily = sub.groupby("date")[horizon].mean()
        exc = (daily - base_by_date.reindex(daily.index)).dropna()
        e, l = exc[exc.index < SPLIT], exc[exc.index >= SPLIT]
        e_bps = e.mean() * 10000 if len(e) >= 20 else float("nan")
        l_bps = l.mean() * 10000 if len(l) >= 20 else float("nan")
        agree = "YES" if (not np.isnan(e_bps) and not np.isnan(l_bps)
                          and (e_bps > 0) == (l_bps > 0)) else "no"
        print(f"{name:<52} {len(sub):>8,} {ndays:>6} {exc.mean()*10000:>8.1f} {t:>7.2f} | "
              f"{e_bps:>10.1f} {l_bps:>9.1f} {agree:>6}")


def main() -> None:
    pd.set_option("display.width", 220)
    with db.get_connection(load_postgres_config()) as conn:
        df = load(conn)
    print("Computing per-symbol features (this takes a few minutes)...")
    df = add_features(df)
    df = df.dropna(subset=["fwd_1d", "vol_ratio", "close_pos"])
    print(f"Usable observations: {len(df):,} stock-days\n")

    for h in ("fwd_1d", "fwd_5d", "fwd_10d"):
        report(df, h)

    print("\n" + "=" * 116)
    print("'days' is the real sample size, not 'obs'. A t below ~2 with sign")
    print("agreement is suggestive; a t above 3 that agrees across both halves")
    print("is a finding. Indian equity round-trip cost is roughly 10-20 bps --")
    print("subtract it before calling any of these tradeable.")
    print("=" * 116)


if __name__ == "__main__":
    main()
