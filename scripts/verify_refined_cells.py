"""Early/late stability of the four cells that actually separated.

The close-position split refuted my own "it's all just a volume spike" hypothesis:
at a 20-day high, the same spike is worth +39.9 bps with a strong close and -60.1
with a weak one. And the largest effect in the table was one nobody proposed --
drifting to new lows on NO volume.

Large t-stats are where this project has been fooled before, so these go through
the filter that has been the reliable discriminator all along: sign agreement across
an early/late split, late-half significance on its own, and survival in liquid names.
An effect that only existed in 2020-2023 is history, not a strategy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_setups_equities import SPLIT

CACHE = Path(sys.argv[1])
COST_BPS = 15.0


def t_of(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) >= 20 and x.std() else float("nan")


def excess(df: pd.DataFrame, mask: pd.Series, h: str, universe: pd.DataFrame | None = None) -> pd.Series:
    u = df if universe is None else universe
    base_by_date = u.groupby("date")[h].mean()
    daily = df[mask.fillna(False)].groupby("date")[h].mean()
    return (daily - base_by_date.reindex(daily.index)).dropna()


def main() -> None:
    df = pd.read_pickle(CACHE)
    at_high = df["close"] >= 0.985 * df["hi20"]
    at_low = df["close"] <= 1.015 * df["lo20"]
    spike = df["vol_ratio"] > 1.3

    cells = {
        "high + spike + STRONG close": at_high & spike & (df["close_pos"] > 0.65),
        "high + spike + weak close": at_high & spike & (df["close_pos"] < 0.35),
        "low, NO spike (quiet new lows)": at_low & ~spike,
        "low + spike + strong close": at_low & spike & (df["close_pos"] > 0.65),
    }

    med = df.groupby("sym")["turnover"].median()
    df["liq"] = df["sym"].map(pd.qcut(med, 3, labels=["low", "mid", "high"]))

    for h in ("fwd_5d", "fwd_10d"):
        print("=" * 104)
        print(f"{h.upper()} excess return -- stability")
        print("=" * 104)
        print(f"{'cell':<32} {'obs':>8} {'bps':>8} {'t':>7} | {'early':>8} {'late':>8} "
              f"{'late t':>7} {'agree':>6} | {'net late':>8}")
        for name, mask in cells.items():
            exc = excess(df, mask, h)
            e, l = exc[exc.index < SPLIT], exc[exc.index >= SPLIT]
            agree = "YES" if len(e) >= 20 and len(l) >= 20 and (e.mean() > 0) == (l.mean() > 0) else "no"
            net = abs(l.mean() * 10000) - COST_BPS
            print(f"{name:<32} {int(mask.sum()):>8,} {exc.mean()*10000:>8.1f} {t_of(exc):>7.2f} | "
                  f"{e.mean()*10000:>8.1f} {l.mean()*10000:>8.1f} {t_of(l):>7.2f} {agree:>6} | {net:>8.1f}")
        print()

    print("=" * 104)
    print("LIQUIDITY (fwd_10d) -- an edge only in the dregs is a spread illusion")
    print("=" * 104)
    print(f"{'cell':<32} | " + " | ".join(f"{x:>18}" for x in ("low liq", "mid liq", "high liq")))
    for name, mask in cells.items():
        out = []
        for liq in ("low", "mid", "high"):
            u = df[df["liq"] == liq]
            exc = excess(df, mask & (df["liq"] == liq), "fwd_10d", universe=u)
            out.append(f"{exc.mean()*10000:>10.1f} (t{t_of(exc):>5.1f})" if len(exc) >= 20 else f"{'too few':>18}")
        print(f"{name:<32} | " + " | ".join(out))


if __name__ == "__main__":
    main()
