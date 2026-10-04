"""Tests the market-structure and rejection-resolution rules, same harness as
test_price_action.py: equities for power, indices for relevance, sign-aware,
date-clustered, early/late, three horizons.

Reuses the cached equity OHLCV frame from the price-action run when available, so
only the new structures need computing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from algotrade import db, ml_dataset, market_structure as ms
from algotrade.config import load_postgres_config
from test_price_action import HORIZONS, clustered, load_equities, SPLIT

BASE = ["open", "high", "low", "close", "volume", "date"]


def with_structure(df: pd.DataFrame) -> pd.DataFrame:
    d = ms.build_structure(df.sort_values("date").reset_index(drop=True))
    for h in HORIZONS:
        d[f"fwd{h}"] = d["close"].shift(-h) / d["close"] - 1
    return d


def report(all_: pd.DataFrame, title: str) -> None:
    for h in HORIZONS:
        col = f"fwd{h}"
        print("=" * 122)
        print(f"{title}  --  forward {h}-day, sign-aware edge vs same-period mean (bps)")
        print("=" * 122)
        rows = []
        for sig in ms.LONG_SIGNALS + ms.SHORT_SIGNALS:
            sign = 1 if sig in ms.LONG_SIGNALS else -1
            sub = all_[all_[sig].fillna(False) & all_[col].notna()]
            r = clustered(sub, all_[all_[col].notna()], col, sign)
            t, e = (r["t"] or 0), (r["edge_bps"] or 0)
            verdict = ("PASS" if r["agree"] == "YES" and abs(t) > 2 and e > 0
                       else "WRONG WAY" if r["agree"] == "YES" and abs(t) > 2 and e < 0 else "")
            rows.append({"signal": sig, "side": "L" if sign == 1 else "S", **r, "verdict": verdict})
        print(pd.DataFrame(rows).to_string(index=False))
        print()


def main() -> None:
    pd.set_option("display.width", 240)
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    with db.get_connection(load_postgres_config()) as conn:
        idx = {i: with_structure(ml_dataset._load_full_ohlc(conn, i)) for i in ("NIFTY", "BANKNIFTY")}
        if cache and cache.exists():
            eq = pd.read_pickle(cache)[["sym"] + BASE]
            print(f"Equities from cache: {len(eq):,} rows")
        else:
            print("Loading equities...")
            eq = load_equities(conn)
    print("Building structures per symbol...")
    eq = eq.groupby("sym", group_keys=False).apply(with_structure)
    print(f"  {len(eq):,} stock-days\n")

    print("Trend-state mix (equities):", eq["trend"].value_counts(normalize=True).round(3).to_dict())
    print("Occurrences (equities):")
    for k in ms.LONG_SIGNALS + ms.SHORT_SIGNALS:
        print(f"  {k:<32} {int(eq[k].sum()):>8,}")
    print()

    report(eq, "EQUITY UNIVERSE (the powered test)")

    med = eq.groupby("sym").apply(lambda g: (g["close"] * g["volume"]).median())
    eq["liq"] = eq["sym"].map(pd.qcut(med, 3, labels=["low", "mid", "high"]))
    print("=" * 122)
    print("HIGH-LIQUIDITY TERCILE ONLY -- forward 10-day (where stock futures exist)")
    print("=" * 122)
    hi = eq[(eq["liq"] == "high") & eq["fwd10"].notna()]
    for sig in ms.LONG_SIGNALS + ms.SHORT_SIGNALS:
        sign = 1 if sig in ms.LONG_SIGNALS else -1
        r = clustered(hi[hi[sig].fillna(False)], hi, "fwd10", sign)
        print(f"  {sig:<32} {'L' if sign == 1 else 'S'}  n={r['n_obs']:>6,}  edge={r['edge_bps']:>7}  t={r['t']:>6}  "
              f"early={r['early']:>7} late={r['late']:>7}  {r['agree']}")
    print()

    both = pd.concat([d.assign(sym=i) for i, d in idx.items()], ignore_index=True)
    report(both, "NIFTY + BANKNIFTY (what you trade; low power)")


if __name__ == "__main__":
    main()
