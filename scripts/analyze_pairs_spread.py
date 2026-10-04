"""NIFTY-BANKNIFTY spread (statistical arbitrage) test.

WHY THIS IS WORTH TESTING, when so much else has failed here:

  * MARKET NEUTRAL. Long one index, short the other. It does not require predicting
    market direction -- the thing this project has repeatedly shown it cannot do
    (35-46% vs a 40% no-edge baseline, in every configuration tried).
  * INDEPENDENT OBSERVATIONS. If the spread mean-reverts over days rather than weeks,
    a one-position-at-a-time rule produces many genuinely non-overlapping trades. Our
    binding constraint has been ~200 independent observations; this could multiply
    that, which is worth more than any feature we could add.
  * The two indices are economically linked (banks are a large NIFTY component), so
    there is a real mechanism for the spread to be tethered rather than drift freely.

METHOD:
  spread    = log(BANKNIFTY) - beta * log(NIFTY)
  z         = (spread - rolling mean) / rolling std      [both computed only from past]
  entry     = |z| > entry_z          (bet on reversion)
  exit      = |z| < exit_z, or max holding period
  P&L       = r_BANKNIFTY - beta * r_NIFTY while positioned, signed by direction

Two hedge ratios are compared: beta=1 (equal notional) and a rolling regression beta.
Positions are strictly one-at-a-time, so trades are non-overlapping by construction.

Costs matter DOUBLE here -- every entry and exit trades two legs.
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
LOOKBACK = 60          # rolling window for the z-score
MAX_HOLD = 20          # trading days
ENTRY_Z, EXIT_Z = 2.0, 0.5


def load(conn) -> pd.DataFrame:
    n = ml_dataset._load_full_ohlc(conn, "NIFTY")[["date", "close"]].rename(columns={"close": "nifty"})
    b = ml_dataset._load_full_ohlc(conn, "BANKNIFTY")[["date", "close"]].rename(columns={"close": "banknifty"})
    df = n.merge(b, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["ln_n"] = np.log(df["nifty"])
    df["ln_b"] = np.log(df["banknifty"])
    df["r_n"] = df["nifty"].pct_change()
    df["r_b"] = df["banknifty"].pct_change()
    return df


def rolling_beta(df: pd.DataFrame, window: int = 120) -> pd.Series:
    """Rolling OLS beta of BANKNIFTY returns on NIFTY returns, using past data only."""
    cov = df["r_b"].rolling(window).cov(df["r_n"])
    var = df["r_n"].rolling(window).var()
    return (cov / var).shift(1)   # shift so today's beta uses only data through yesterday


def half_life(spread: pd.Series) -> float:
    """AR(1) half-life of mean reversion: how many days for a deviation to decay by
    half. Short half-life = fast reversion = more independent trades."""
    s = spread.dropna()
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    idx = lag.index.intersection(delta.index)
    lag, delta = lag.loc[idx], delta.loc[idx]
    lam = np.polyfit(lag - lag.mean(), delta, 1)[0]
    if lam >= 0:
        return float("inf")
    return float(-math.log(2) / math.log(1 + lam))


def backtest(df: pd.DataFrame, beta_mode: str) -> pd.DataFrame:
    d = df.copy()
    d["beta"] = 1.0 if beta_mode == "equal" else rolling_beta(d)
    d["spread"] = d["ln_b"] - d["beta"] * d["ln_n"]
    d["mu"] = d["spread"].rolling(LOOKBACK).mean()
    d["sd"] = d["spread"].rolling(LOOKBACK).std()
    d["z"] = (d["spread"] - d["mu"]) / d["sd"]
    d = d.dropna(subset=["z", "beta"]).reset_index(drop=True)

    trades = []
    i = 1
    while i < len(d) - 1:
        z = d.at[i, "z"]
        if abs(z) < ENTRY_Z:
            i += 1
            continue
        # z > ENTRY: spread is rich -> short the spread (short BN, long N) -> side=-1
        side = -1 if z > 0 else 1
        entry_i, entry_z = i, z
        beta = d.at[i, "beta"]
        pnl = 0.0
        exit_i = None
        for j in range(i + 1, min(i + 1 + MAX_HOLD, len(d))):
            # P&L accrues from the day AFTER the signal
            pnl += side * (d.at[j, "r_b"] - beta * d.at[j, "r_n"])
            if abs(d.at[j, "z"]) < EXIT_Z:
                exit_i = j
                break
        if exit_i is None:
            exit_i = min(i + MAX_HOLD, len(d) - 1)
        trades.append({
            "entry_date": d.at[entry_i, "date"], "exit_date": d.at[exit_i, "date"],
            "side": side, "entry_z": entry_z, "days_held": exit_i - entry_i, "pnl": pnl,
        })
        i = exit_i + 1     # one position at a time -> trades never overlap
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, label: str, years: float) -> dict:
    if trades.empty:
        return {"variant": label, "n": 0}
    p = trades["pnl"]
    per_year = len(trades) / years
    return {
        "variant": label,
        "n": len(trades),
        "per_yr": round(per_year, 1),
        "win": round(float((p > 0).mean()), 3),
        "mean_bps": round(float(p.mean()) * 10000, 1),
        "total_pct": round(float(p.sum()) * 100, 1),
        "avg_days": round(float(trades["days_held"].mean()), 1),
        "sharpe": round(float(p.mean() / p.std() * math.sqrt(per_year)), 2) if p.std() > 0 else None,
        "worst_bps": round(float(p.min()) * 10000, 1),
    }


def main() -> None:
    pd.set_option("display.width", 220)
    with db.get_connection(load_postgres_config()) as conn:
        df = load(conn)
    years = len(df) / 252
    print(f"Data: {len(df)} paired days ({df['date'].min()} to {df['date'].max()}), {years:.1f} years\n")

    # ---- Is the spread even mean-reverting? --------------------------------
    print("=" * 92)
    print("STEP 1: is the spread mean-reverting at all?")
    print("=" * 92)
    train = df[df["date"] < SPLIT]
    for mode in ("equal", "rolling"):
        d = df.copy()
        d["beta"] = 1.0 if mode == "equal" else rolling_beta(d)
        d["spread"] = d["ln_b"] - d["beta"] * d["ln_n"]
        hl_all = half_life(d["spread"])
        hl_tr = half_life(d[d["date"] < SPLIT]["spread"])
        hl_te = half_life(d[d["date"] >= SPLIT]["spread"])
        print(f"  beta={mode:<8} half-life: full {hl_all:>6.1f}d | train {hl_tr:>6.1f}d | test {hl_te:>6.1f}d")
    print("\n  Short, stable half-life = genuine reversion. Infinite or wildly different")
    print("  across periods = the relationship is drifting, not reverting.")

    # ---- Strategy ----------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"STEP 2: z-score reversion strategy (entry |z|>{ENTRY_Z}, exit |z|<{EXIT_Z}, max {MAX_HOLD}d)")
    print("=" * 92)
    rows = []
    for mode in ("equal", "rolling"):
        t = backtest(df, mode)
        if t.empty:
            continue
        rows.append(summarize(t, f"full sample (beta={mode})", years))
        tr = t[t["entry_date"] < SPLIT]
        te = t[t["entry_date"] >= SPLIT]
        yrs_tr = len(df[df["date"] < SPLIT]) / 252
        yrs_te = len(df[df["date"] >= SPLIT]) / 252
        if not tr.empty:
            rows.append(summarize(tr, f"  train (beta={mode})", yrs_tr))
        if not te.empty:
            rows.append(summarize(te, f"  TEST (beta={mode})", yrs_te))
    print(pd.DataFrame(rows).to_string(index=False))

    # ---- Cost sensitivity: two legs, so double ------------------------------
    print("\n" + "=" * 92)
    print("STEP 3: cost sensitivity -- TWO legs per entry and exit")
    print("=" * 92)
    t = backtest(df, "equal")
    te = t[t["entry_date"] >= SPLIT]
    if not te.empty:
        p = te["pnl"]
        yrs_te = len(df[df["date"] >= SPLIT]) / 252
        per_year = len(te) / yrs_te
        print(f"  Held-out test: {len(te)} trades over {yrs_te:.1f}yrs = {per_year:.0f}/yr, "
              f"gross mean {p.mean()*10000:.1f} bps\n")
        print(f"  {'cost per leg':>14} {'round-trip':>12} {'net mean bps':>14} {'net Sharpe':>12}")
        for c in (0.0, 0.01, 0.02, 0.03, 0.05):
            rt = c * 4          # 2 legs in, 2 legs out
            net = p - rt / 100
            sh = net.mean() / net.std() * math.sqrt(per_year) if net.std() > 0 else float("nan")
            print(f"  {c:>13.2f}% {rt:>11.2f}% {net.mean()*10000:>14.1f} {sh:>12.2f}")


if __name__ == "__main__":
    main()
