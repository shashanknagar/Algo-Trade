"""Tests calendar / expiry structure in the Indian indices.

WHY: weekly and monthly option expiry is a genuine structural feature of these
markets -- large open interest expires on a known schedule, and dealer hedging around
those strikes is a real mechanical force (unlike max-pain folklore). We have only
proxied it crudely so far with days_left_in_month.

DELIBERATELY DATA-DRIVEN ON THE WEEKDAY: NSE has changed weekly expiry days more than
once in recent years (and differently across indices), so rather than hardcode a day
and risk asserting something false, every weekday is measured and the data is allowed
to say which matters.

Tested:
  1. Day-of-week returns
  2. Last Thursday of the month (the long-standing monthly expiry) vs other days
  3. Week-of-month position
  4. Whether VOLATILITY (not just return) is elevated near expiry -- more relevant to
     us, since volatility is the thing this project has shown is predictable

Split early/late throughout; sign agreement across halves is the filter.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from algotrade import db, ml_dataset
from algotrade.config import load_postgres_config

SPLIT = date(2024, 1, 1)
WEEKDAYS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std() == 0:
        return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("date").reset_index(drop=True)
    d["ret"] = d["close"].pct_change()
    d["range_pct"] = (d["high"] - d["low"]) / d["close"]      # intraday range = realized vol proxy
    d["dow"] = d["date"].apply(lambda x: x.weekday())
    d["ym"] = d["date"].apply(lambda x: (x.year, x.month))

    # Last Thursday of each calendar month -- the long-standing monthly expiry.
    thursdays = d[d["dow"] == 3]
    last_thu = thursdays.groupby("ym")["date"].max()
    d["is_last_thu"] = d["date"].isin(set(last_thu.values))

    # Position within the month, in trading days
    d["dom_rank"] = d.groupby("ym").cumcount()
    d["week_of_month"] = (d["dom_rank"] // 5).clip(upper=4)
    return d.dropna(subset=["ret"])


def compare(d: pd.DataFrame, group_col: str, value_col: str, label: str,
            fmt_name=lambda x: str(x)) -> None:
    e_all, l_all = d[d["date"] < SPLIT], d[d["date"] >= SPLIT]
    e_base, l_base = e_all[value_col].mean(), l_all[value_col].mean()
    base = d[value_col].mean()
    print(f"\n  {label}  (baseline {base*10000:.1f} bps)")
    print(f"    {'group':<14} {'n':>5} {'mean bps':>10} {'vs base':>9} {'t':>7} | "
          f"{'early':>8} {'late':>8}  agree")
    for name, g in d.groupby(group_col, observed=True):
        e, l = g[g["date"] < SPLIT], g[g["date"] >= SPLIT]
        if len(e) < 10 or len(l) < 10:
            continue
        e_rel = (e[value_col].mean() - e_base) * 10000
        l_rel = (l[value_col].mean() - l_base) * 10000
        agree = "YES" if (e_rel > 0) == (l_rel > 0) else "no"
        print(f"    {fmt_name(name):<14} {len(g):>5} {g[value_col].mean()*10000:>10.1f} "
              f"{(g[value_col].mean()-base)*10000:>9.1f} {tstat(g[value_col]):>7.2f} | "
              f"{e_rel:>8.1f} {l_rel:>8.1f}  {agree}")


def main() -> None:
    pd.set_option("display.width", 200)
    config = load_postgres_config()
    with db.get_connection(config) as conn:
        frames = {inst: prepare(ml_dataset._load_full_ohlc(conn, inst))
                  for inst in ("NIFTY", "BANKNIFTY")}

    for inst, d in frames.items():
        print("\n" + "=" * 88)
        print(f"{inst}  ({len(d)} days)")
        print("=" * 88)

        compare(d, "dow", "ret", "1. RETURN by day of week", lambda x: WEEKDAYS.get(x, str(x)))
        compare(d, "is_last_thu", "ret", "2. RETURN: last Thursday (monthly expiry) vs rest",
                lambda x: "last Thu" if x else "other days")
        compare(d, "week_of_month", "ret", "3. RETURN by week of month",
                lambda x: f"week {int(x)+1}")
        compare(d, "dow", "range_pct", "4. VOLATILITY (daily range) by day of week",
                lambda x: WEEKDAYS.get(x, str(x)))
        compare(d, "is_last_thu", "range_pct", "5. VOLATILITY: last Thursday vs rest",
                lambda x: "last Thu" if x else "other days")

    print("\n" + "=" * 88)
    print("'agree' is the test. Note especially section 4/5: volatility is the thing")
    print("this project has shown IS predictable, so an expiry effect on volatility is")
    print("worth more to us than one on returns.")
    print("=" * 88)


if __name__ == "__main__":
    main()
