"""Tests the daily-testable strategies from a broker marketplace listing.

WHAT THE LISTING SHOWED: five NIFTY strategies with a "win rate" and a "weekly
return" each -- Explosive Candle (61%, +45.5% weekly), RSI Magic (62%, +15.8%),
ORB (43%, +6.7%), Quickstrike (40%, +1.5%), Scalpmax Pro (40%, -17.7%).

WHAT THOSE NUMBERS ARE NOT. A weekly return is one week. +45% and -18% in the same
list is the variance of leveraged option buying, not a track record, and a win rate
with no risk:reward attached is not a statistic. None of them is evidence.

WHAT CAN AND CANNOT BE TESTED HERE. ORB, Quickstrike and Scalpmax are intraday
systems; the only intraday data held is 101 ticks from one day and the Groww
historical API is currently refusing the key. Testing them on daily bars would be
fabrication, so they are not tested. The two that have daily-bar forms are:

  EXPLOSIVE CANDLE -- a wide-range bar (range >= k x ATR) closing near its extreme,
  traded for continuation next day. Also the "rejection candle" variant from the
  same family: wait for the next bar to break the rejection bar's high/low.

  RSI MAGIC -- the name says nothing, so the standard RSI rule families are tested:
  classic 30/70 mean reversion, the 50-line momentum cross, Connors' RSI(2), and
  "oversold and turning up" confirmation.

Each rule enters at the NEXT OPEN and is resolved by the project's standard triple
barrier (stop 1xATR, target 1.5x stop, 20 days) on NIFTY and BANKNIFTY, with the
40% random-walk breakeven and an always-long control. Forward-return screens on the
equity universe supply power. Early/late split throughout.
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from algotrade import db, ml_dataset, technical_indicators as ti, trade_selection
from algotrade.config import load_postgres_config
from test_price_action import clustered

SPLIT = date(2024, 1, 1)
STOP_ATR, RR, HORIZON = 1.0, 1.5, 20
BREAKEVEN = 100 / (1 + RR)


def atr14(d):
    c, h, l = d["close"], d["high"], d["low"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def build(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True).copy()
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    d["atr"] = atr14(d)
    rng = (h - l).replace(0, np.nan)
    cpos = (c - l) / rng
    vol_ok = (v / v.rolling(20).mean()) > 1.3

    # ---- Explosive candle family ----------------------------------------
    wide = (h - l) >= 1.5 * d["atr"].shift(1)
    d["explosive_long"] = wide & (cpos >= 0.75) & (c > o)
    d["explosive_short"] = wide & (cpos <= 0.25) & (c < o)
    d["explosive_long_vol"] = d["explosive_long"] & vol_ok
    d["explosive_short_vol"] = d["explosive_short"] & vol_ok
    # Rejection candle, then break of its extreme on the NEXT bar (the "golden
    # rejection" rule): long wick, then confirmation.
    lower_w = (np.minimum(o, c) - l) / rng
    upper_w = (h - np.maximum(o, c)) / rng
    rej_bull = (lower_w >= 0.5)
    rej_bear = (upper_w >= 0.5)
    d["rejection_break_long"] = rej_bull.shift(1, fill_value=False) & (c > h.shift(1))
    d["rejection_break_short"] = rej_bear.shift(1, fill_value=False) & (c < l.shift(1))

    # ---- RSI family ------------------------------------------------------
    rsi14, rsi2 = ti.rsi(c, 14), ti.rsi(c, 2)
    d["rsi_oversold_long"] = rsi14 < 30
    d["rsi_overbought_short"] = rsi14 > 70
    d["rsi_cross50_long"] = (rsi14 > 50) & (rsi14.shift(1) <= 50)
    d["rsi_cross50_short"] = (rsi14 < 50) & (rsi14.shift(1) >= 50)
    d["rsi2_long"] = rsi2 < 10
    d["rsi2_short"] = rsi2 > 90
    d["rsi_turn_up_long"] = (rsi14.shift(1) < 30) & (rsi14 > rsi14.shift(1)) & (c > c.shift(1))
    d["rsi_turn_down_short"] = (rsi14.shift(1) > 70) & (rsi14 < rsi14.shift(1)) & (c < c.shift(1))
    for h_ in (5, 10, 20):
        d[f"fwd{h_}"] = c.shift(-h_) / c - 1
    return d


LONGS = ["explosive_long", "explosive_long_vol", "rejection_break_long", "rsi_oversold_long",
         "rsi_cross50_long", "rsi2_long", "rsi_turn_up_long"]
SHORTS = ["explosive_short", "explosive_short_vol", "rejection_break_short", "rsi_overbought_short",
          "rsi_cross50_short", "rsi2_short", "rsi_turn_down_short"]


def triple_barrier(d: pd.DataFrame, sig: str, side: int) -> pd.DataFrame:
    o, h, l, c, atr, dates = (d[k].values for k in ("open", "high", "low", "close", "atr", "date"))
    s = d[sig].values
    rows, i, n = [], 0, len(d)
    while i < n - 2:
        if not s[i] or np.isnan(atr[i]):
            i += 1
            continue
        e = i + 1
        entry, stop_pts = float(o[e]), STOP_ATR * float(atr[i])
        stop = entry - side * stop_pts
        target = entry + side * RR * stop_pts
        out, r, x = "timeout", None, min(e + HORIZON - 1, n - 1)
        for j in range(e, min(e + HORIZON, n)):
            hs = l[j] <= stop if side == 1 else h[j] >= stop
            ht = h[j] >= target if side == 1 else l[j] <= target
            if hs:
                out, r, x = "stop", -1.0, j
                break
            if ht:
                out, r, x = "target", RR, j
                break
        if r is None:
            r = side * (float(c[x]) - entry) / stop_pts
        rows.append({"date": dates[i], "outcome": out, "r": r, "entry_index": i, "resolve_index": x})
        i = x + 1
    return pd.DataFrame(rows)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    cen = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (cen - half) * 100, (cen + half) * 100


def index_strategy_table(frames: dict) -> None:
    print("=" * 118)
    print(f"INDEX STRATEGY TEST -- next-open entry, stop 1xATR, target 1.5x, {HORIZON}d; breakeven {BREAKEVEN:.0f}%")
    print("=" * 118)
    print(f"  {'rule':<24} {'side':<5} {'n':>4} {'hit%':>6} {'CI':>14} {'E[R]':>7} | {'early E[R]':>10} {'late E[R]':>10} "
          f"{'agree':>6} | {'always-long E[R]':>16} {'adds':>7}")
    for sig in LONGS + SHORTS:
        side = 1 if sig in LONGS else -1
        parts, ctrl = [], []
        for inst, d in frames.items():
            t = triple_barrier(d, sig, side).assign(instrument=inst)
            parts.append(t)
            ctrl.append(triple_barrier(d, sig, 1).assign(instrument=inst))
        t = pd.concat(parts, ignore_index=True)
        cl = pd.concat(ctrl, ignore_index=True)
        if t.empty or len(t) < 10:
            print(f"  {sig:<24} {'L' if side == 1 else 'S':<5} {len(t):>4}  too few")
            continue
        t = trade_selection.select_non_overlapping(t, group_cols=("instrument",))
        cl = trade_selection.select_non_overlapping(cl, group_cols=("instrument",))
        k = int((t["outcome"] == "target").sum())
        lo, hi = wilson(k, len(t))
        e, la = t[t["date"] < SPLIT]["r"], t[t["date"] >= SPLIT]["r"]
        agree = "YES" if len(e) >= 8 and len(la) >= 8 and (e.mean() > 0) == (la.mean() > 0) else "no"
        print(f"  {sig:<24} {'L' if side == 1 else 'S':<5} {len(t):>4} {k/len(t)*100:>6.1f} [{lo:>5.1f},{hi:>5.1f}] "
              f"{t['r'].mean():>+7.3f} | {e.mean() if len(e) else np.nan:>+10.3f} {la.mean() if len(la) else np.nan:>+10.3f} "
              f"{agree:>6} | {cl['r'].mean():>+16.3f} {t['r'].mean() - cl['r'].mean():>+7.3f}")
    print("\n  'adds' = E[R] of the rule minus E[R] of always-long on the same dates and barriers.")
    print("  A positive E[R] with a negative 'adds' is market drift wearing a strategy's name.\n")


def equity_screen(eq: pd.DataFrame) -> None:
    print("=" * 118)
    print("EQUITY UNIVERSE SCREEN -- forward 10-day, sign-aware, date-clustered (the powered test)")
    print("=" * 118)
    rows = []
    for sig in LONGS + SHORTS:
        side = 1 if sig in LONGS else -1
        u = eq[eq["fwd10"].notna()]
        r = clustered(u[u[sig].fillna(False)], u, "fwd10", side)
        t_, e_ = (r["t"] or 0), (r["edge_bps"] or 0)
        v = ("PASS" if r["agree"] == "YES" and abs(t_) > 2 and e_ > 0
             else "WRONG WAY" if r["agree"] == "YES" and abs(t_) > 2 and e_ < 0 else "")
        rows.append({"rule": sig, "side": "L" if side == 1 else "S", **r, "verdict": v})
    print(pd.DataFrame(rows).to_string(index=False))
    print()


def main() -> None:
    pd.set_option("display.width", 240)
    with db.get_connection(load_postgres_config()) as conn:
        frames = {i: build(ml_dataset._load_full_ohlc(conn, i)) for i in ("NIFTY", "BANKNIFTY")}
    index_strategy_table(frames)

    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if cache and cache.exists():
        eq = pd.read_pickle(cache)[["sym", "date", "open", "high", "low", "close", "volume"]]
        eq = eq.groupby("sym", group_keys=False).apply(build)
        equity_screen(eq)


if __name__ == "__main__":
    main()
