"""Donchian channel breakout + trend filter + ATR trailing stop, on REAL data.

ORIGIN: a document proposing this system, whose own backtest ran on synthetic
np.random.normal data with a fixed seed -- so its -4.17% return and -0.62 Sharpe
describe a random number sequence, not any market, and cannot support or refute
anything. The strategy itself (essentially Turtle Trading) is well established and
genuinely untested here, so it is worth measuring properly.

TWO COMPONENTS ARE NEW TO THIS PROJECT and are isolated so we learn which, if either,
contributes:

  A. REGIME FILTER. We currently take breakouts unconditionally. This only takes an
     upside breakout when price is above its 50-EMA (and the mirror for shorts). The
     idea is to avoid buying breakouts that fail inside downtrends.

  B. ATR TRAILING STOP. We currently exit at a fixed target. A trailing stop lets a
     winner run and is the standard trend-following exit -- it changes the shape of
     the return distribution (lower hit rate, larger average win), so hit rate alone
     is the WRONG yardstick and total return / Sharpe is used instead.

The source document also computed ATR and then never used it, and had no position
state at all -- its "signal" was a per-day condition rather than a trade held to an
exit. Both are implemented properly here, long AND short.

Every variant is split early/late; sign agreement across halves remains the filter.
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

SPLIT = date(2024, 1, 1)
CHANNEL, EMA_SPAN, ATR_WIN = 20, 50, 14
ATR_TRAIL_MULT = 3.0
MAX_HOLD = 60


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("date").reset_index(drop=True).copy()
    c, h, l = d["close"], d["high"], d["low"]
    d["ema"] = c.ewm(span=EMA_SPAN, adjust=False).mean()
    # shift(1): the channel must be built from bars STRICTLY BEFORE today, else the
    # breakout test compares today's close against a high that includes today.
    d["upper"] = h.shift(1).rolling(CHANNEL).max()
    d["lower"] = l.shift(1).rolling(CHANNEL).min()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(ATR_WIN).mean()
    return d.dropna(subset=["upper", "lower", "atr", "ema"]).reset_index(drop=True)


def backtest(d: pd.DataFrame, use_filter: bool, allow_short: bool,
             trailing: bool, fixed_rr: float = 1.5) -> pd.DataFrame:
    """One position at a time, so trades are non-overlapping by construction."""
    trades, i, n = [], 0, len(d)
    while i < n - 1:
        row = d.iloc[i]
        long_sig = row["close"] > row["upper"]
        short_sig = row["close"] < row["lower"] and allow_short
        if use_filter:
            long_sig = long_sig and row["close"] > row["ema"]
            short_sig = short_sig and row["close"] < row["ema"]
        if not (long_sig or short_sig):
            i += 1
            continue

        side = 1 if long_sig else -1
        entry = float(row["close"])
        atr0 = float(row["atr"])
        stop = entry - side * ATR_TRAIL_MULT * atr0
        target = entry + side * fixed_rr * ATR_TRAIL_MULT * atr0
        best = entry
        exit_i, exit_px, reason = None, None, "timeout"

        for j in range(i + 1, min(i + 1 + MAX_HOLD, n)):
            bar = d.iloc[j]
            hi, lo = float(bar["high"]), float(bar["low"])
            # Stop checked before target: the pessimistic convention used throughout.
            if (side == 1 and lo <= stop) or (side == -1 and hi >= stop):
                exit_i, exit_px, reason = j, stop, "stop"
                break
            if not trailing and ((side == 1 and hi >= target) or (side == -1 and lo <= target)):
                exit_i, exit_px, reason = j, target, "target"
                break
            if trailing:
                best = max(best, hi) if side == 1 else min(best, lo)
                stop = max(stop, best - ATR_TRAIL_MULT * float(bar["atr"])) if side == 1 \
                    else min(stop, best + ATR_TRAIL_MULT * float(bar["atr"]))
        if exit_i is None:
            exit_i = min(i + MAX_HOLD, n - 1)
            exit_px = float(d.iloc[exit_i]["close"])

        trades.append({
            "entry_date": row["date"], "exit_date": d.iloc[exit_i]["date"], "side": side,
            "ret": side * (exit_px / entry - 1), "days": exit_i - i, "reason": reason,
        })
        i = exit_i + 1
    return pd.DataFrame(trades)


def summarize(t: pd.DataFrame, label: str, years: float) -> dict:
    if t.empty or len(t) < 5:
        return {"variant": label, "n": len(t)}
    r = t["ret"]
    per_yr = len(t) / years if years else float("nan")
    return {
        "variant": label, "n": len(t),
        "win%": round(float((r > 0).mean()) * 100, 1),
        "mean_bps": round(float(r.mean()) * 10000, 1),
        "total%": round(float((1 + r).prod() - 1) * 100, 1),
        "avg_days": round(float(t["days"].mean()), 1),
        "sharpe": round(float(r.mean() / r.std() * math.sqrt(per_yr)), 2) if r.std() else None,
        "worst%": round(float(r.min()) * 100, 1),
    }


def main() -> None:
    pd.set_option("display.width", 220)
    with db.get_connection(load_postgres_config()) as conn:
        frames = {i: prepare(ml_dataset._load_full_ohlc(conn, i)) for i in ("NIFTY", "BANKNIFTY")}

    variants = [
        ("breakout only, long-only, fixed exit", False, False, False),
        ("+ EMA50 filter, long-only, fixed exit", True, False, False),
        ("+ EMA50 filter, LONG+SHORT, fixed exit", True, True, False),
        ("breakout only, LONG+SHORT, fixed exit", False, True, False),
        ("+ EMA50 filter, long-only, ATR TRAIL", True, False, True),
        ("+ EMA50 filter, LONG+SHORT, ATR TRAIL", True, True, True),
        ("breakout only, long-only, ATR TRAIL", False, False, True),
    ]

    for inst, d in frames.items():
        years = len(d) / 252
        bh = (d["close"].iloc[-1] / d["close"].iloc[0] - 1) * 100
        print("=" * 124)
        print(f"{inst}  [{len(d)} days, {years:.1f}y]   buy-and-hold over the same span: {bh:+.1f}%")
        print("=" * 124)
        rows = []
        for label, filt, shorts, trail in variants:
            t = backtest(d, filt, shorts, trail)
            rows.append(summarize(t, label, years))
            if not t.empty:
                e = t[t["entry_date"] < SPLIT]
                l = t[t["entry_date"] >= SPLIT]
                y_e = len(d[d["date"] < SPLIT]) / 252
                y_l = len(d[d["date"] >= SPLIT]) / 252
                for sub, tag, yy in ((e, "    early", y_e), (l, "    LATE ", y_l)):
                    if len(sub) >= 5:
                        rows.append(summarize(sub, tag, yy))
        print(pd.DataFrame(rows).to_string(index=False))
        print()

    print("=" * 124)
    print("Compare every variant against buy-and-hold on the SAME span, not against zero.")
    print("A long-only trend system in a rising market will show a profit that is mostly")
    print("beta; only beating buy-and-hold, or matching it at materially lower risk,")
    print("is evidence of anything. And check the LATE rows -- that is the held-out half.")
    print("=" * 124)


if __name__ == "__main__":
    main()
