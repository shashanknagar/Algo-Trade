"""Measures whether the candle patterns and volume signals this project already uses
actually predict anything -- something never tested until now. They were included
because they're conventional, not because they were shown to work here.

Method: NEXT-DAY forward returns, grouped by pattern / volume bucket. One-day forward
returns from consecutive days are close to independent, so unlike the barrier
backtests (where daily-sampled 20-day trades overlap heavily) this has real
statistical power -- thousands of near-independent observations rather than ~200.

Reported per group: mean next-day return in basis points, a t-statistic against zero,
and the count. A t-stat beyond about +/-2 is the usual "unlikely to be chance"
threshold -- but see the multiple-comparisons note at the end, because testing a dozen
patterns guarantees roughly one crosses that line by luck alone.

Usage:
    python scripts/analyze_patterns.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from algotrade import db, ml_dataset
from algotrade.config import load_postgres_config


def classify_candles(df: pd.DataFrame) -> pd.Series:
    """Vectorized version of signal_factors.candle_pattern_factor's logic, so what's
    measured here is exactly what the model uses."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    po, pc = o.shift(1), c.shift(1)

    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([c, o], axis=1).max(axis=1)
    lower = pd.concat([c, o], axis=1).min(axis=1) - l

    prev_bear, prev_bull = pc < po, pc > po
    cur_bull, cur_bear = c > o, c < o

    out = pd.Series("none", index=df.index, dtype=object)
    # Order mirrors the original: engulfing first, then hammer/star, then doji.
    out[(body / rng < 0.1)] = "doji"
    out[(body > 0) & (upper >= 2 * body) & (lower <= 0.5 * body) & (body / rng < 0.35)] = "shooting_star"
    out[(body > 0) & (lower >= 2 * body) & (upper <= 0.5 * body) & (body / rng < 0.35)] = "hammer"
    out[prev_bull & cur_bear & (o > pc) & (c < po)] = "bearish_engulfing"
    out[prev_bear & cur_bull & (c > po) & (o < pc)] = "bullish_engulfing"
    return out


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std() == 0:
        return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def summarize(df: pd.DataFrame, group_col: str, ret_col: str = "fwd_1d") -> pd.DataFrame:
    rows = []
    overall = df[ret_col].mean()
    for name, g in df.groupby(group_col):
        rows.append(
            {
                group_col: name,
                "n": len(g),
                "mean_bps": round(float(g[ret_col].mean()) * 10000, 1),
                "vs_baseline_bps": round(float(g[ret_col].mean() - overall) * 10000, 1),
                "t_stat": round(tstat(g[ret_col]), 2),
                "win_rate": round(float((g[ret_col] > 0).mean()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_bps", ascending=False)


def main() -> None:
    pd.set_option("display.width", 200)
    config = load_postgres_config()

    frames = []
    with db.get_connection(config) as conn:
        for instrument in ("NIFTY", "BANKNIFTY"):
            df = ml_dataset._load_full_ohlc(conn, instrument)
            df["instrument"] = instrument
            df["pattern"] = classify_candles(df)
            df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
            df["fwd_1d"] = df["close"].shift(-1) / df["close"] - 1
            df["fwd_5d"] = df["close"].shift(-5) / df["close"] - 1
            # 20-day breakout state, matching momentum_breakout_factor's definition
            prior_high = df["high"].rolling(20).max().shift(1)
            prior_low = df["low"].rolling(20).min().shift(1)
            df["breakout"] = np.where(df["close"] > prior_high, "breakup",
                              np.where(df["close"] < prior_low, "breakdown", "norange"))
            frames.append(df)

    data = pd.concat(frames, ignore_index=True).dropna(subset=["fwd_1d", "vol_ratio"])
    print(f"Observations (near-independent next-day returns): {len(data)}")
    print(f"Overall mean next-day return: {data['fwd_1d'].mean()*10000:.1f} bps "
          f"(both indices drift up over this period)\n")

    print("=== 1. CANDLE PATTERNS -> next-day return ===")
    print("(vs_baseline_bps is the interesting column: return ABOVE the overall drift)")
    print(summarize(data, "pattern").to_string(index=False))

    print("\n=== 2. VOLUME -> next-day return ===")
    data["vol_bucket"] = pd.cut(
        data["vol_ratio"], [0, 0.7, 0.9, 1.1, 1.5, 100],
        labels=["very_low(<0.7x)", "low(0.7-0.9x)", "normal(0.9-1.1x)", "high(1.1-1.5x)", "very_high(>1.5x)"],
    )
    print(summarize(data, "vol_bucket").to_string(index=False))

    print("\n=== 3. DOES VOLUME CONFIRM BREAKOUTS? ===")
    print("(the model assumes a breakout on high volume is more meaningful -- testing that)")
    br = data[data["breakout"] != "norange"].copy()
    br["combo"] = br["breakout"] + " + " + np.where(br["vol_ratio"] > 1.1, "high_vol", "normal_vol")
    print(summarize(br, "combo").to_string(index=False))

    print("\n=== 4. BULLISH vs BEARISH PATTERNS: do they point the right way? ===")
    bull_pats = ["bullish_engulfing", "hammer"]
    bear_pats = ["bearish_engulfing", "shooting_star"]
    for label, pats in (("bullish patterns", bull_pats), ("bearish patterns", bear_pats)):
        sub = data[data["pattern"].isin(pats)]
        if len(sub):
            print(f"  {label:18s} n={len(sub):>5}  mean={sub['fwd_1d'].mean()*10000:>6.1f} bps  "
                  f"t={tstat(sub['fwd_1d']):>5.2f}  "
                  f"(bearish patterns should be NEGATIVE if they work)")

    print(
        "\nMULTIPLE COMPARISONS: ~15 groups are tested above. At the usual |t|>2 "
        "threshold, roughly one would clear it by chance even if nothing predicted "
        "anything. Treat a lone significant group with suspicion; a consistent "
        "direction across related groups is the meaningful signal."
    )

    # --- Does any of it replicate? ---------------------------------------------
    # The findings above come from the full sample. Splitting by date shows which
    # survive on data the first half never saw -- the only real check on whether
    # these are effects or artifacts.
    cutoff = pd.Timestamp("2024-01-01").date()
    early, late = data[data["date"] < cutoff], data[data["date"] >= cutoff]
    print(f"\n\n=== 5. REPLICATION CHECK: before vs after {cutoff} ===")
    print(f"(early n={len(early)}, late n={len(late)})")

    def compare(group_col: str, subset_filter=None) -> None:
        e = early if subset_filter is None else early[subset_filter(early)]
        l = late if subset_filter is None else late[subset_filter(late)]
        e_base, l_base = e["fwd_1d"].mean(), l["fwd_1d"].mean()
        print(f"\n  {group_col}:")
        print(f"    {'group':<26} {'early bps':>10} {'early t':>8} {'late bps':>10} {'late t':>8}  same sign?")
        for name in sorted(set(e[group_col].dropna().unique()) & set(l[group_col].dropna().unique())):
            eg, lg = e[e[group_col] == name], l[l[group_col] == name]
            if len(eg) < 15 or len(lg) < 15:
                continue
            e_rel = (eg["fwd_1d"].mean() - e_base) * 10000
            l_rel = (lg["fwd_1d"].mean() - l_base) * 10000
            agree = "YES" if (e_rel > 0) == (l_rel > 0) else "no"
            print(f"    {str(name):<26} {e_rel:>10.1f} {tstat(eg['fwd_1d']):>8.2f} "
                  f"{l_rel:>10.1f} {tstat(lg['fwd_1d']):>8.2f}  {agree}")

    compare("pattern")
    compare("vol_bucket")

    early["combo"] = early["breakout"] + " + " + np.where(early["vol_ratio"] > 1.1, "high_vol", "normal_vol")
    late["combo"] = late["breakout"] + " + " + np.where(late["vol_ratio"] > 1.1, "high_vol", "normal_vol")
    compare("combo", subset_filter=lambda d: d["breakout"] != "norange")

    print(
        "\n  'same sign?' is the column that matters. An effect that flips sign "
        "between halves is noise, however impressive its full-sample t-statistic."
    )


if __name__ == "__main__":
    main()
