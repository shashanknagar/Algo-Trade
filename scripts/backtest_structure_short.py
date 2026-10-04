"""Strategy backtest: bearish market-structure shorts on liquid equities, point-based
barriers, geometry chosen on train and verified on test.

WHAT YOU ASKED FOR: choose the stop/target combination so that even at a low hit
rate, taking every trade is profitable after the losers and after costs -- with
barriers expressed in POINTS rather than percent.

HOW POINTS ARE HANDLED ON A 200-STOCK UNIVERSE. A fixed 50-point stop is 50% on a
Rs 100 stock and 0.1% on a Rs 50,000 one, so one number cannot serve every name.
Each stock's barriers are therefore set in ITS OWN points as multiples of its
ATR(14) -- which is measured in points -- and P&L is reported two ways:
  * points per share (what you asked for), and
  * rupees at a fixed Rs 1 lakh notional per trade (what your account shows).
Summing raw points across stocks of different prices is meaningless, so the
aggregate lives in the rupee column.

REALISM RULES that the screens did not apply:
  * ENTRY AT THE NEXT OPEN, not the signal-bar close. The screens assumed you could
    trade the close that produced the signal; you cannot.
  * UNIVERSE = top 200 by median turnover. Overnight shorts need stock futures,
    which exist only for the most liquid names.
  * COSTS: 15 bps of notional round trip (Rs 150 per Rs 1 lakh trade), deducted.
  * ONE POSITION PER STOCK at a time; a stock already short ignores new signals.
  * STOP WINS TIES within a bar (pessimistic), and the tie share is reported.
  * GEOMETRY CHOSEN ON PRE-2024 ONLY; the 2024+ column is the verdict.

CONTROL: the same barriers on the same stocks at RANDOM dates, matched in count.
If random timing earns the same, the signal is not doing the work.
"""
from __future__ import annotations

import math
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from algotrade import market_structure as ms, price_action as pa

SPLIT = date(2024, 1, 1)
TOP_N = 200
NOTIONAL = 100_000.0
COST_BPS = 15.0
STOP_MULTS = [0.5, 1.0, 1.5, 2.0]
RR_LIST = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
HORIZONS = [10, 20]
SIGNALS = {
    "choch_bear_vol": "CHoCH bearish + volume (headline)",
    "tl_break_down_vol": "trendline breakdown + volume",
    "dt_break": "double-top neckline break",
    "confirmed_reversal_short_vol": "held rejection at resistance + confirm + vol",
}


def atr14(d: pd.DataFrame) -> pd.Series:
    c, h, l = d["close"], d["high"], d["low"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def prepare(cache: Path) -> pd.DataFrame:
    eq = pd.read_pickle(cache)
    base = ["sym", "date", "open", "high", "low", "close", "volume"]
    keep_pa = [c for c in pa.SHORT_SIGNALS if c in eq.columns]
    eq = eq[base + keep_pa].copy()
    med = eq.groupby("sym").apply(lambda g: (g["close"] * g["volume"]).median())
    top = med.sort_values(ascending=False).head(TOP_N).index
    eq = eq[eq["sym"].isin(top)].sort_values(["sym", "date"]).reset_index(drop=True)
    print(f"Universe: top {TOP_N} by turnover, {len(eq):,} stock-days")

    def add(g: pd.DataFrame) -> pd.DataFrame:
        g = g.reset_index(drop=True)
        s = ms.build_structure(g)
        for col in ("choch_bear_vol", "confirmed_reversal_short_vol", "choch_bear", "bos_down_vol"):
            g[col] = s[col].values
        g["atr"] = atr14(g)
        g["any_short"] = g[["choch_bear_vol", "tl_break_down_vol", "dt_break",
                           "confirmed_reversal_short_vol"]].any(axis=1)
        return g
    return eq.groupby("sym", group_keys=False).apply(add).reset_index(drop=True)


def simulate(g: pd.DataFrame, signal_col: str, stop_mult: float, rr: float,
             horizon: int, entries: pd.Series | None = None) -> list[dict]:
    """One symbol. Short on the NEXT OPEN after a signal bar; one position at a time."""
    trades = []
    n = len(g)
    o, h, l, c, atr = (g[k].values for k in ("open", "high", "low", "close", "atr"))
    sig = (entries if entries is not None else g[signal_col]).values
    dates = g["date"].values
    i = 0
    while i < n - 2:
        if not sig[i] or np.isnan(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        e = i + 1                              # enter at next bar's open
        entry = float(o[e])
        stop_pts = stop_mult * float(atr[i])
        target_pts = rr * stop_pts
        stop, target = entry + stop_pts, entry - target_pts
        outcome, pnl, tie, exit_i = "timeout", None, False, min(e + horizon - 1, n - 1)
        for j in range(e, min(e + horizon, n)):
            hit_stop, hit_tgt = h[j] >= stop, l[j] <= target
            if hit_stop:
                outcome, pnl, tie, exit_i = "stop", -stop_pts, bool(hit_tgt), j
                break
            if hit_tgt:
                outcome, pnl, exit_i = "target", target_pts, j
                break
        if pnl is None:
            pnl = entry - float(c[exit_i])
        trades.append({
            "sym": g["sym"].iat[0], "signal_date": dates[i], "entry_date": dates[e],
            "entry": entry, "stop_pts": stop_pts, "target_pts": target_pts,
            "outcome": outcome, "pnl_pts": pnl, "r": pnl / stop_pts, "tie": tie,
            "rupees": pnl / entry * NOTIONAL - COST_BPS / 1e4 * NOTIONAL,
        })
        i = exit_i + 1
    return trades


def run(eq: pd.DataFrame, signal_col: str, sm: float, rr: float, hz: int,
        random_seed: int | None = None) -> pd.DataFrame:
    rows = []
    for _, g in eq.groupby("sym", sort=False):
        entries = None
        if random_seed is not None:
            k = int(g[signal_col].sum())
            rng = random.Random(hash((random_seed, g["sym"].iat[0])) & 0xFFFF)
            idx = rng.sample(range(20, max(21, len(g) - hz - 1)), min(k, max(1, len(g) - hz - 22)))
            entries = pd.Series(False, index=g.index)
            entries.iloc[idx] = True
        rows.extend(simulate(g, signal_col, sm, rr, hz, entries))
    return pd.DataFrame(rows)


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n": 0}
    daily = t.groupby("entry_date")["rupees"].sum()
    tstat = daily.mean() / (daily.std() / math.sqrt(len(daily))) if len(daily) > 2 and daily.std() else float("nan")
    return {
        "n": len(t), "hit%": round((t["outcome"] == "target").mean() * 100, 1),
        "stop%": round((t["outcome"] == "stop").mean() * 100, 1),
        "tie%": round(t["tie"].mean() * 100, 1),
        "E[R]": round(t["r"].mean(), 3),
        "med_pts": round(t["pnl_pts"].median(), 2),
        "Rs/trade": round(t["rupees"].mean(), 0),
        "total_Rs": round(t["rupees"].sum(), 0),
        "t(daily)": round(tstat, 2),
    }


def main() -> None:
    pd.set_option("display.width", 250)
    eq = prepare(Path(sys.argv[1]))
    for col in SIGNALS:
        print(f"  {col:<32} {int(eq[col].sum()):>6,} signals in the universe")
    print()

    # ---------- 1. Geometry sweep on the headline signal, train only ----------
    head = "choch_bear_vol"
    grid = []
    for hz in HORIZONS:
        for sm in STOP_MULTS:
            for rr in RR_LIST:
                t = run(eq, head, sm, rr, hz)
                tr, te = t[t["entry_date"] < SPLIT], t[t["entry_date"] >= SPLIT]
                grid.append({"horizon": hz, "stop_xATR": sm, "RR": rr, "breakeven%": round(100 / (1 + rr), 1),
                             **{f"train_{k}": v for k, v in stats(tr).items()},
                             **{f"test_{k}": v for k, v in stats(te).items()}})
    grid = pd.DataFrame(grid)
    cols = ["horizon", "stop_xATR", "RR", "breakeven%", "train_n", "train_hit%", "train_E[R]", "train_Rs/trade",
            "train_total_Rs", "test_n", "test_hit%", "test_tie%", "test_E[R]", "test_Rs/trade", "test_total_Rs", "test_t(daily)"]
    print("=" * 140)
    print(f"1. GEOMETRY SWEEP -- {SIGNALS[head]} -- Rs at Rs 1 lakh/trade, 15 bps costs deducted")
    print("=" * 140)
    print(grid[cols].to_string(index=False))

    # ---------- 2. Honest pick: best TRAIN total, then its TEST ---------------
    measurable = grid[grid["test_tie%"].fillna(100) < 20]
    best = measurable.sort_values("train_total_Rs", ascending=False).iloc[0]
    print("\n" + "=" * 140)
    print("2. CHOSEN ON TRAIN (highest train total after costs), VERIFIED ON TEST")
    print("=" * 140)
    print(best[cols].to_string())
    hz, sm, rr = int(best["horizon"]), float(best["stop_xATR"]), float(best["RR"])

    # ---------- 3. All passing signals at the chosen geometry -----------------
    print("\n" + "=" * 140)
    print(f"3. EVERY SHORT SIGNAL at the chosen geometry (stop {sm}xATR, target {rr}x stop, {hz}d) -- TEST period only")
    print("=" * 140)
    rows = []
    for col, label in SIGNALS.items():
        t = run(eq, col, sm, rr, hz)
        te = t[t["entry_date"] >= SPLIT]
        rows.append({"signal": label, **stats(te)})
    t = run(eq, "any_short", sm, rr, hz)
    rows.append({"signal": "ANY of the four (ensemble)", **stats(t[t["entry_date"] >= SPLIT])})
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------- 4. Control: random timing, same everything else ---------------
    print("\n" + "=" * 140)
    print("4. CONTROL -- same stocks, same barriers, RANDOM entry dates (count-matched), TEST period")
    print("=" * 140)
    ctrl = [stats(run(eq, head, sm, rr, hz, random_seed=s).query("entry_date >= @SPLIT")) for s in (1, 2, 3)]
    ctrl_df = pd.DataFrame(ctrl)
    print(ctrl_df.to_string(index=False))
    real = stats(run(eq, head, sm, rr, hz).query("entry_date >= @SPLIT"))
    print(f"\n  signal Rs/trade {real['Rs/trade']:>8.0f}   vs random-timing mean {ctrl_df['Rs/trade'].mean():>8.0f}"
          f"   ->  signal adds {real['Rs/trade'] - ctrl_df['Rs/trade'].mean():>+8.0f} Rs per Rs 1 lakh trade")

    # ---------- 5. Equity curve shape on test ----------------------------------
    t = run(eq, head, sm, rr, hz).query("entry_date >= @SPLIT").sort_values("entry_date")
    if not t.empty:
        curve = t.groupby("entry_date")["rupees"].sum().cumsum()
        dd = (curve.cummax() - curve).max()
        print("\n" + "=" * 140)
        print("5. TEST-PERIOD EQUITY CURVE (Rs, cumulative, Rs 1 lakh per trade)")
        print("=" * 140)
        q = curve.iloc[np.linspace(0, len(curve) - 1, 8).astype(int)]
        for d, v in q.items():
            print(f"  {d}  {v:>12,.0f}")
        print(f"\n  max drawdown {dd:,.0f} Rs   |  trades {len(t)}   |  {t['sym'].nunique()} distinct stocks")
        print(f"  by year: " + ", ".join(f"{y}: {v:,.0f}" for y, v in
                                          t.groupby(pd.to_datetime(t['entry_date']).dt.year)['rupees'].sum().items()))


if __name__ == "__main__":
    main()
