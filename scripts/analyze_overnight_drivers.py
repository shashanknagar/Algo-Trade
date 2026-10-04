"""Tests whether US futures / cross-asset moves help with the ONE thing this project
has established matters: the overnight session.

WHY THIS TEST: NIFTY's entire risk premium is earned overnight (+24.8%/yr, t=5.47)
while the intraday session is negative (-12.9%/yr, t=-2.34), both replicating across
sample halves. India's overnight window is precisely when US markets trade -- so US
futures are a mechanistically plausible driver, not just another indicator.

THREE QUESTIONS, ONLY TWO OF WHICH MATTER:

  Q1. Do US futures explain NIFTY's overnight gap?
      Almost certainly yes -- but this is NOT tradeable. The gap has already happened
      by the time you can transact at the open. Explaining it earns nothing. Included
      only to size the relationship.

  Q2. Do US futures predict the INTRADAY session that follows (open -> close)?
      This IS tradeable: you see the US move and the Indian open, then trade the day.

  Q3. Can anything known at TODAY'S CLOSE predict whether holding overnight is
      favourable? This IS tradeable -- you decide at the close whether to carry a
      position through the gap. Given that the overnight window is where all the
      return lives, this is the highest-value question in the project.

Everything is split early/late. Sign agreement across halves is the filter that has
survived where t-statistics have not.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import yfinance as yf

from algotrade import db, ml_dataset
from algotrade.config import load_postgres_config

SPLIT = date(2024, 1, 1)
EXTERNAL = {
    "DOW_FUT": "YM=F",
    "SP500_FUT": "ES=F",
    "NASDAQ_FUT": "NQ=F",
    "USDINR": "USDINR=X",
    "CRUDE": "CL=F",
}


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std() == 0:
        return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def fetch_external(start: str) -> pd.DataFrame:
    frames = []
    for name, ticker in EXTERNAL.items():
        try:
            d = yf.download(ticker, start=start, progress=False, auto_adjust=True)
            if d.empty:
                print(f"  SKIP {name}: no data")
                continue
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            s = d["Close"].rename(name)
            frames.append(s)
            print(f"  OK {name}: {len(s)} rows")
        except Exception as exc:
            print(f"  SKIP {name}: {exc}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index).date
    out.index.name = "date"
    return out


def main() -> None:
    pd.set_option("display.width", 220)
    config = load_postgres_config()
    with db.get_connection(config) as conn:
        nifty = ml_dataset._load_full_ohlc(conn, "NIFTY")

    print("Fetching external drivers...")
    ext = fetch_external(str(nifty["date"].min()))
    if ext.empty:
        print("No external data -- aborting.")
        return

    df = nifty.merge(ext.reset_index(), on="date", how="left").sort_values("date").reset_index(drop=True)
    df["overnight"] = df["open"] / df["close"].shift(1) - 1
    df["intraday"] = df["close"] / df["open"] - 1
    # US futures move over the window between two Indian closes -- i.e. what happened
    # while India was shut. Forward-filled first so US holidays don't create false zeros.
    for name in EXTERNAL:
        if name in df.columns:
            df[name] = df[name].ffill()
            df[f"{name}_chg"] = df[name].pct_change()
    df = df.dropna(subset=["overnight", "intraday"])

    drivers = [f"{n}_chg" for n in EXTERNAL if f"{n}_chg" in df.columns]
    early, late = df[df["date"] < SPLIT], df[df["date"] >= SPLIT]

    # ---- Q1: explaining the gap (not tradeable) -----------------------------
    print("\n" + "=" * 92)
    print("Q1. Do US futures EXPLAIN the overnight gap?  (NOT tradeable -- gap is done by the open)")
    print("=" * 92)
    print(f"  {'driver':<16} {'corr w/ gap':>12} {'early':>9} {'late':>9}  agree")
    for d in drivers:
        c_all = df[d].corr(df["overnight"])
        c_e, c_l = early[d].corr(early["overnight"]), late[d].corr(late["overnight"])
        agree = "YES" if (c_e > 0) == (c_l > 0) else "no"
        print(f"  {d:<16} {c_all:>12.3f} {c_e:>9.3f} {c_l:>9.3f}  {agree}")

    # ---- Q2: predicting the tradeable intraday session ----------------------
    print("\n" + "=" * 92)
    print("Q2. Do US futures PREDICT the intraday session (open -> close)?  TRADEABLE")
    print("=" * 92)
    print(f"  {'driver':<16} {'corr w/ intraday':>17} {'early':>9} {'late':>9}  agree")
    for d in drivers:
        c_all = df[d].corr(df["intraday"])
        c_e, c_l = early[d].corr(early["intraday"]), late[d].corr(late["intraday"])
        agree = "YES" if (c_e > 0) == (c_l > 0) else "no"
        print(f"  {d:<16} {c_all:>17.3f} {c_e:>9.3f} {c_l:>9.3f}  {agree}")

    print("\n  Also: does a BIG overnight gap predict the intraday session that follows")
    print("  (gap-and-go vs gap-fill)? Decided at the open, so tradeable.")
    df["gap_bucket"] = pd.qcut(df["overnight"], 5, labels=["big down", "down", "flat", "up", "big up"])
    print(f"\n  {'gap quintile':<14} {'n':>5} {'mean intraday bps':>18} {'t':>7} | {'early':>8} {'late':>8}  agree")
    for name, g in df.groupby("gap_bucket", observed=True):
        e, l = g[g["date"] < SPLIT], g[g["date"] >= SPLIT]
        if len(e) < 10 or len(l) < 10:
            continue
        agree = "YES" if (e["intraday"].mean() > 0) == (l["intraday"].mean() > 0) else "no"
        print(f"  {str(name):<14} {len(g):>5} {g['intraday'].mean()*10000:>18.1f} {tstat(g['intraday']):>7.2f} | "
              f"{e['intraday'].mean()*10000:>8.1f} {l['intraday'].mean()*10000:>8.1f}  {agree}")

    # ---- Q3: deciding at the close whether to hold overnight ----------------
    print("\n" + "=" * 92)
    print("Q3. Can anything known at TODAY'S CLOSE predict TOMORROW'S overnight gap?  TRADEABLE")
    print("=" * 92)
    print("  This is the highest-value question: the overnight window is where all the")
    print("  return lives, and you choose at the close whether to carry through it.\n")

    df["next_overnight"] = df["overnight"].shift(-1)
    # Everything below is strictly known at today's close.
    df["today_intraday"] = df["intraday"]
    df["today_gap"] = df["overnight"]
    df["us_same_day"] = df.get("SP500_FUT_chg", pd.Series(np.nan, index=df.index))

    conditions = {
        "today intraday UP": df["today_intraday"] > 0,
        "today intraday DOWN": df["today_intraday"] <= 0,
        "today gapped UP": df["today_gap"] > 0,
        "today gapped DOWN": df["today_gap"] <= 0,
    }
    valid = df.dropna(subset=["next_overnight"])
    base = valid["next_overnight"].mean()
    print(f"  Baseline next-overnight return: {base*10000:.1f} bps\n")
    print(f"  {'condition at close':<22} {'n':>5} {'next-ON bps':>12} {'vs base':>9} {'t':>7} | "
          f"{'early':>8} {'late':>8}  agree")
    for label, mask in conditions.items():
        sub = valid[mask.reindex(valid.index, fill_value=False)]
        if len(sub) < 30:
            continue
        e, l = sub[sub["date"] < SPLIT], sub[sub["date"] >= SPLIT]
        eb = valid[valid["date"] < SPLIT]["next_overnight"].mean()
        lb = valid[valid["date"] >= SPLIT]["next_overnight"].mean()
        e_rel = (e["next_overnight"].mean() - eb) * 10000 if len(e) >= 15 else float("nan")
        l_rel = (l["next_overnight"].mean() - lb) * 10000 if len(l) >= 15 else float("nan")
        agree = "YES" if (not np.isnan(e_rel) and not np.isnan(l_rel) and (e_rel > 0) == (l_rel > 0)) else "no"
        print(f"  {label:<22} {len(sub):>5} {sub['next_overnight'].mean()*10000:>12.1f} "
              f"{(sub['next_overnight'].mean()-base)*10000:>9.1f} {tstat(sub['next_overnight']):>7.2f} | "
              f"{e_rel:>8.1f} {l_rel:>8.1f}  {agree}")

    print("\n" + "=" * 92)
    print("'agree' is the whole test. Q1 is context only -- explaining a gap you cannot")
    print("trade earns nothing. Q2 and Q3 are the ones that could change what we do.")
    print("=" * 92)


if __name__ == "__main__":
    main()
