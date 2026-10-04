"""The outright short lost. Does the HEDGED version -- short the stock, long the index
over the same window -- capture what the screens actually measured?

THE DISCONNECT, stated plainly. The cross-sectional screens found that after a
bearish CHoCH on volume a stock lags its peers by ~100 bps over ten days, t=4.4,
stable out of sample. The strategy backtest then shorted those stocks outright and
lost money at every barrier geometry. Both are correct. The screen measured RELATIVE
return; a short position pays ABSOLUTE return; and the market rose through the
whole test period. A stock that lagged the market by 100 bps still went up more
often than not, and the short paid for it -- then costs took the rest.

What the screens support is a market-neutral position: short the stock, long the
index, in equal rupee notional. That is executable with stock futures against a
NIFTY future. This measures that, on the same trades, entries and horizons as the
outright test, so the only thing that changes is the hedge.

Costs are doubled: two legs in, two legs out.
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

from algotrade import db, ml_dataset
from algotrade.config import load_postgres_config
from backtest_structure_short import NOTIONAL, SIGNALS, SPLIT, prepare, stats

COST_BPS_HEDGED = 30.0            # two legs each way
HORIZONS = [5, 10, 20]


def hedged_trades(eq: pd.DataFrame, nifty: pd.Series, signal_col: str, horizon: int,
                  shift_entry: bool = True) -> pd.DataFrame:
    """Enter at next open (or at the signal close, to quantify the gap cost), hold a
    fixed horizon, no barriers: this isolates the relative-return claim."""
    rows = []
    for sym, g in eq.groupby("sym", sort=False):
        g = g.reset_index(drop=True)
        sig = g[signal_col].values
        o, c, d = g["open"].values, g["close"].values, g["date"].values
        i, n = 0, len(g)
        while i < n - horizon - 2:
            if not sig[i]:
                i += 1
                continue
            e = i + 1 if shift_entry else i
            entry = float(o[e]) if shift_entry else float(c[e])
            x = min(e + horizon, n - 1)
            stock_ret = float(c[x]) / entry - 1
            d_entry, d_exit = d[e], d[x]
            if d_entry not in nifty.index or d_exit not in nifty.index:
                i = x + 1
                continue
            idx_ret = float(nifty[d_exit]) / float(nifty[d_entry]) - 1
            rows.append({
                "sym": sym, "entry_date": d_entry, "exit_date": d_exit,
                "stock_ret": stock_ret, "index_ret": idx_ret,
                "outright_Rs": -stock_ret * NOTIONAL - 15 / 1e4 * NOTIONAL,
                "hedged_Rs": (-stock_ret + idx_ret) * NOTIONAL - COST_BPS_HEDGED / 1e4 * NOTIONAL,
                # winner = the stock fell more than (or rose less than) the index
                "outcome": "target" if stock_ret < idx_ret else "stop",
                "r": (-stock_ret + idx_ret) * 100, "pnl_pts": -stock_ret * entry, "tie": False,
            })
            i = x + 1
    return pd.DataFrame(rows)


def summarize(t: pd.DataFrame, col: str) -> dict:
    if t.empty:
        return {"n": 0}
    daily = t.groupby("entry_date")[col].sum()
    ts = daily.mean() / (daily.std() / math.sqrt(len(daily))) if len(daily) > 2 and daily.std() else float("nan")
    win = (t[col] > 0).mean() * 100
    return {"n": len(t), "win%": round(win, 1), "Rs/trade": round(t[col].mean(), 0),
            "total_Rs": round(t[col].sum(), 0), "t(daily)": round(ts, 2)}


def main() -> None:
    pd.set_option("display.width", 220)
    eq = prepare(Path(sys.argv[1]))
    with db.get_connection(load_postgres_config()) as conn:
        nf = ml_dataset._load_full_ohlc(conn, "NIFTY")
    nifty = pd.Series(nf["close"].values, index=nf["date"].values)

    for hz in HORIZONS:
        print("=" * 128)
        print(f"HOLD {hz} DAYS, entry at NEXT OPEN, TEST period (2024+) -- Rs at Rs 1 lakh per leg")
        print("=" * 128)
        print(f"  {'signal':<46} {'n':>5} | {'OUTRIGHT short':^30} | {'HEDGED (short stock, long NIFTY)':^36}")
        print(f"  {'':<46} {'':>5} | {'win%':>6} {'Rs/trade':>9} {'t':>6} {'total':>8} | "
              f"{'win%':>6} {'Rs/trade':>9} {'t':>6} {'total':>10}")
        for col, label in SIGNALS.items():
            t = hedged_trades(eq, nifty, col, hz)
            te = t[t["entry_date"] >= SPLIT]
            a, b = summarize(te, "outright_Rs"), summarize(te, "hedged_Rs")
            print(f"  {label:<46} {a['n']:>5} | {a['win%']:>6} {a['Rs/trade']:>9} {a['t(daily)']:>6} {a['total_Rs']:>8,.0f} | "
                  f"{b['win%']:>6} {b['Rs/trade']:>9} {b['t(daily)']:>6} {b['total_Rs']:>10,.0f}")
        t = hedged_trades(eq, nifty, "any_short", hz)
        te = t[t["entry_date"] >= SPLIT]
        a, b = summarize(te, "outright_Rs"), summarize(te, "hedged_Rs")
        print(f"  {'ANY of the four (ensemble)':<46} {a['n']:>5} | {a['win%']:>6} {a['Rs/trade']:>9} {a['t(daily)']:>6} {a['total_Rs']:>8,.0f} | "
              f"{b['win%']:>6} {b['Rs/trade']:>9} {b['t(daily)']:>6} {b['total_Rs']:>10,.0f}")
        print()

    # How much of the edge is lost to the overnight gap between signal close and next open?
    print("=" * 128)
    print("GAP COST: hedged Rs/trade, 10-day hold, entering at the SIGNAL CLOSE vs the NEXT OPEN (test)")
    print("=" * 128)
    for col, label in SIGNALS.items():
        at_close = hedged_trades(eq, nifty, col, 10, shift_entry=False).query("entry_date >= @SPLIT")
        at_open = hedged_trades(eq, nifty, col, 10, shift_entry=True).query("entry_date >= @SPLIT")
        c_, o_ = summarize(at_close, "hedged_Rs"), summarize(at_open, "hedged_Rs")
        print(f"  {label:<46} close {c_['Rs/trade']:>8}  next-open {o_['Rs/trade']:>8}  "
              f"gap costs {c_['Rs/trade'] - o_['Rs/trade']:>+7.0f} Rs")
    print("\n  Entering at the signal close is not possible in practice; the next-open column")
    print("  is the real one. The difference is edge that lives in the overnight gap.")


if __name__ == "__main__":
    main()
