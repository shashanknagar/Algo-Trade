"""Is the candle's close position informative, or is it all just 'volume spike at an extreme'?

Setups 3 and 4 were defined as opposites -- distribution (weak close at the highs)
vs accumulation (strong close at the lows) -- and were expected to predict opposite
directions. Both came out BEARISH. That is a strong hint that the close-position
half of each definition carries nothing, and the real driver is the part they share:
an unusual volume spike while the stock sits at a 20-day extreme.

If that hypothesis is right, splitting on close position should barely move forward
returns, while the volume spike alone should reproduce most of the effect. If it is
wrong, the strong-close and weak-close cells will separate clearly.

This matters practically: it decides whether candle anatomy earns a place in the
model, or whether one cruder feature captures everything real.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

CACHE = Path(sys.argv[1])
H = "fwd_10d"


def t_of(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) >= 20 and x.std() else float("nan")


def cell(df: pd.DataFrame, mask: pd.Series, label: str) -> None:
    sub = df[mask.fillna(False)]
    if len(sub) < 100:
        print(f"  {label:<44} {len(sub):>7}  too few")
        return
    base_by_date = df.groupby("date")[H].mean()
    daily = sub.groupby("date")[H].mean()
    exc = (daily - base_by_date.reindex(daily.index)).dropna()
    print(f"  {label:<44} {len(sub):>7,} {exc.mean()*10000:>9.1f} {t_of(exc):>7.2f}")


def main() -> None:
    df = pd.read_pickle(CACHE)
    at_high = df["close"] >= 0.985 * df["hi20"]
    at_low = df["close"] <= 1.015 * df["lo20"]
    spike = df["vol_ratio"] > 1.3
    strong = df["close_pos"] > 0.65
    weak = df["close_pos"] < 0.35

    print(f"Forward 10-day excess return, {len(df):,} stock-days\n")
    print(f"  {'cell':<44} {'n':>7} {'bps':>9} {'t':>7}")
    print("  " + "-" * 68)
    print("  AT 20-DAY HIGH")
    cell(df, at_high & spike & weak, "    + volume spike, WEAK close  (setup 3)")
    cell(df, at_high & spike & strong, "    + volume spike, STRONG close")
    cell(df, at_high & spike, "    + volume spike, close ignored")
    cell(df, at_high & ~spike, "    no volume spike")
    print("  AT 20-DAY LOW")
    cell(df, at_low & spike & strong, "    + volume spike, STRONG close (setup 4)")
    cell(df, at_low & spike & weak, "    + volume spike, WEAK close")
    cell(df, at_low & spike, "    + volume spike, close ignored")
    cell(df, at_low & ~spike, "    no volume spike")
    print("  ANYWHERE (not at an extreme)")
    cell(df, ~at_high & ~at_low & spike, "    volume spike only")

    print("\n  If 'close ignored' reproduces the setup rows, candle anatomy adds nothing")
    print("  and the volume spike at an extreme is the entire story.")


if __name__ == "__main__":
    main()
