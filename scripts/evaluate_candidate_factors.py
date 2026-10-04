"""Uniform bake-off of the candidate strategies, before any of them join the model.

You supplied ten strategies. Four of them (market-making, scalping, VWAP/TWAP,
smart order routing) are EXECUTION algorithms, not prediction algorithms -- they
decide how to fill an order, not which way the market goes, so they cannot raise a
hit rate and are out of scope here (and all four require order placement, which is
prohibited in this project). Three more are already implemented and already measured:
pairs trading (analyze_pairs_spread.py), breakout (signal_factors.momentum_breakout),
and ML (ml_model.py). That leaves the ones genuinely worth testing, plus one
hypothesis the factor autopsy raised.

WHY A BAKE-OFF RATHER THAN JUST ADDING THEM. The current model's problem is not too
few factors -- it is that its existing factors are wrong. The trend factor is
anti-monotonic: its most bearish readings precede the best forward returns. Adding
more unvalidated factors to a weighted vote dilutes whatever signal exists and buys
false confidence, which is the opposite of what you want. So every candidate faces
the same test the existing ones just failed:

  MONOTONICITY  -- do forward returns order correctly across the factor's buckets?
  REPLICATION   -- does that ordering hold in BOTH sample halves, not just overall?
  RELATIVE      -- measured against the same period's own mean, because the index
                   rose throughout and every raw bucket looks "positive" otherwise.

Only factors passing all three earn a place. The rest are reported honestly and left
out.
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from algotrade import db, ml_dataset, technical_indicators as ti
from algotrade.config import load_postgres_config

SPLIT = date(2024, 1, 1)
HORIZON = 20


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def build(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True).copy()
    c, h, l = d["close"], d["high"], d["low"]

    # (1) Trend-following -- your suggested 50/200 EMA, plus the 20/50 SMA already in use.
    d["ema_50_200"] = (ema(c, 50) - ema(c, 200)) / ema(c, 200)
    d["sma_20_50"] = (c.rolling(20).mean() - c.rolling(50).mean()) / c.rolling(50).mean()

    # (2) Mean reversion -- Bollinger %B (0 = lower band, 1 = upper band).
    d["bollinger_b"] = ti.bollinger_pct_b(c)
    # A z-score version, which is the same idea without the band clipping.
    d["z_20"] = (c - c.rolling(20).mean()) / c.rolling(20).std()

    # (3) Momentum systems.
    d["rsi_14"] = ti.rsi(c)
    d["macd_hist"] = ti.macd_histogram(c) / c        # scaled, so it is comparable over time
    d["adx_14"] = ti.average_directional_index(h, l, c)
    d["roc_10"] = ti.rate_of_change(c)

    # The autopsy hypothesis: trend read the OTHER way round.
    d["trend_inverted"] = -d["sma_20_50"]

    d["fwd"] = c.shift(-HORIZON) / c - 1
    return d


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std() / math.sqrt(len(x)))) if len(x) > 2 and x.std() else float("nan")


def evaluate(d: pd.DataFrame, col: str, label: str, n_buckets: int = 3) -> dict:
    sub = d[d[col].notna() & d["fwd"].notna()].copy()
    if len(sub) < 120:
        return {"factor": label, "verdict": f"insufficient data ({len(sub)})"}

    try:
        sub["b"] = pd.qcut(sub[col], n_buckets, labels=False, duplicates="drop")
    except ValueError:
        return {"factor": label, "verdict": "not bucketable"}

    # Relative to the same period's mean -- the index rose, so raw levels mislead.
    base_e = sub[sub["date"] < SPLIT]["fwd"].mean()
    base_l = sub[sub["date"] >= SPLIT]["fwd"].mean()
    sub["rel"] = np.where(sub["date"] < SPLIT, sub["fwd"] - base_e, sub["fwd"] - base_l)

    lo = sub[sub["b"] == 0]
    hi = sub[sub["b"] == sub["b"].max()]
    spread = (hi["rel"].mean() - lo["rel"].mean()) * 10000

    def half_spread(mask):
        a, b = lo[mask(lo)], hi[mask(hi)]
        if len(a) < 20 or len(b) < 20:
            return float("nan")
        return (b["rel"].mean() - a["rel"].mean()) * 10000

    e = half_spread(lambda x: x["date"] < SPLIT)
    la = half_spread(lambda x: x["date"] >= SPLIT)
    replicates = (not np.isnan(e) and not np.isnan(la) and (e > 0) == (la > 0))

    # Monotonic across ALL buckets, not just the endpoints.
    means = sub.groupby("b")["rel"].mean().values
    mono = bool(np.all(np.diff(means) > 0) or np.all(np.diff(means) < 0))

    t = tstat(pd.concat([hi["rel"], -lo["rel"]]))
    verdict = "PASS" if (replicates and mono and abs(t) > 2) else (
        "replicates, not monotonic" if replicates and not mono else
        "monotonic, does not replicate" if mono and not replicates else "fail")
    return {"factor": label, "n": len(sub), "spread_bps": round(spread, 1),
            "t": round(t, 2), "early": round(e, 1), "late": round(la, 1),
            "monotonic": "yes" if mono else "no",
            "replicates": "yes" if replicates else "no", "verdict": verdict}


CANDIDATES = [
    ("ema_50_200", "1. Trend: EMA 50/200 (yours)"),
    ("sma_20_50", "1b. Trend: SMA 20/50 (current)"),
    ("trend_inverted", "1c. Trend INVERTED (autopsy)"),
    ("bollinger_b", "2. Mean reversion: Bollinger %B"),
    ("z_20", "2b. Mean reversion: 20d z-score"),
    ("rsi_14", "3. Momentum: RSI(14)"),
    ("macd_hist", "3b. Momentum: MACD histogram"),
    ("roc_10", "3c. Momentum: 10d rate of change"),
    ("adx_14", "3d. Trend strength: ADX(14)"),
]


def main() -> None:
    pd.set_option("display.width", 220)
    with db.get_connection(load_postgres_config()) as conn:
        frames = {i: build(ml_dataset._load_full_ohlc(conn, i)) for i in ("NIFTY", "BANKNIFTY")}

    for inst, d in frames.items():
        print("=" * 118)
        print(f"{inst}  --  forward {HORIZON}-day return spread (top bucket minus bottom), "
              f"relative to period mean   [{len(d)} days]")
        print("=" * 118)
        rows = [evaluate(d, c, lab) for c, lab in CANDIDATES]
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    # Pooling both indices roughly doubles the sample. They are correlated, so this is
    # not double the independent information -- but a factor that works on only one of
    # two highly-correlated indices is almost certainly noise, so it is a useful check.
    both = pd.concat([d.assign(inst=i) for i, d in frames.items()], ignore_index=True)
    print("=" * 118)
    print("POOLED (both indices) -- a real factor should survive here too")
    print("=" * 118)
    print(pd.DataFrame([evaluate(both, c, lab) for c, lab in CANDIDATES]).to_string(index=False))

    print("\nPASS requires all three: monotonic across buckets, same-signed spread in")
    print("both halves, and |t| > 2. Anything else stays out of the model -- adding an")
    print("unvalidated factor to a weighted vote dilutes signal and manufactures")
    print("confidence, which is precisely the failure mode already diagnosed here.")


if __name__ == "__main__":
    main()
