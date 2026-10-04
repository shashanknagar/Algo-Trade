"""Component-level autopsy: WHICH factor is dragging the hit rate down?

You asked to review the algorithms and understand why the success rate is low. The
aggregate number (29.7% against a 40% baseline) says the ensemble fails but not who
is responsible, so this takes the machine apart.

For every sampled date it records each factor's raw score alongside the eventual
triple-barrier outcome, then asks two questions the aggregate cannot answer:

  1. LONG vs SHORT. You are right that the system must trade both ways, and it does
     (roughly 2:1 bullish). But if one side is carrying all the losses, that is a
     different problem from "the whole model is blind", and it is invisible in a
     pooled hit rate.

  2. DOES EACH FACTOR PREDICT ANYTHING ON ITS OWN? Each factor is scored -1..+1 and
     the composite is a weighted vote. If a factor's high-score bucket does not
     outperform its low-score bucket, that factor is contributing noise -- and
     because the weights are fixed rather than fitted, noise gets an equal say with
     signal. This measures each one separately against forward return, which is a
     cleaner test than hit rate because it does not depend on barrier geometry.

A factor is only credible if its buckets are MONOTONIC and the sign holds in both
sample halves. That filter has been the reliable discriminator throughout this
project, and it is applied here too.
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

from algotrade import backtest_signal, db, signal_factors, signal_generator, trade_selection
from algotrade.config import load_postgres_config

SEEDS = [1, 2, 3, 7, 99]
N_PER_SEED = 250
HORIZON = 20
SPLIT = date(2024, 1, 1)
BREAKEVEN = 1 / (1 + signal_generator.RISK_REWARD_RATIO) * 100


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * 100, (c + h) * 100


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std() / math.sqrt(len(x)))) if len(x) > 2 and x.std() else float("nan")


def collect(conn) -> pd.DataFrame:
    rows = []
    for inst in ("NIFTY", "BANKNIFTY"):
        span = backtest_signal._get_valid_date_range(conn, inst, "positional")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT time::date FROM ohlc_bars WHERE trading_symbol = %s "
                "AND interval = '1day' AND time::date BETWEEN %s AND %s ORDER BY 1",
                (inst, span[0], span[1]))
            all_dates = [r[0] for r in cur.fetchall()]
        index_of = {d: i for i, d in enumerate(all_dates)}

        picked: set = set()
        for seed in SEEDS:
            picked |= set(random.Random(seed).sample(all_dates, min(N_PER_SEED, len(all_dates))))

        for k, d in enumerate(sorted(picked), 1):
            if k % 200 == 0:
                print(f"  {inst} {k}/{len(picked)}", flush=True)
            try:
                sig = signal_generator.generate_signal(conn, inst, d, "positional")
            except Exception:
                continue
            scores = {f["name"]: f["score"] for f in sig.factor_breakdown if f.get("available")}
            future = backtest_signal._load_ohlc_after(conn, inst, d, HORIZON)
            if future.empty or sig.entry_price is None:
                # Still record factor scores for neutral days -- forward return is
                # defined even when the algorithm declines to trade, and excluding
                # them would only measure factors on days the ensemble already liked.
                if future.empty:
                    continue

            entry = float(future.iloc[0]["open"]) if sig.entry_price is None else sig.entry_price
            fwd20 = float(future.iloc[-1]["close"]) / entry - 1

            res = backtest_signal.walk_forward_outcome(conn, sig, HORIZON) if sig.direction != "neutral" else None
            rows.append({
                "instrument": inst, "date": d, "direction": sig.direction,
                "confidence": sig.confidence_score,
                "trend": scores.get("trend"),
                "momentum_breakout": scores.get("momentum_breakout"),
                "oi_positioning": scores.get("oi_positioning"),
                "fwd20": fwd20,
                "outcome": res.outcome if res else "no_trade",
                "r": res.r_multiple if res else None,
                "entry_index": index_of[d],
                "resolve_index": index_of[d] + ((res.days_to_resolve or 0) if res else 0),
            })
    return pd.DataFrame(rows)


def hit_line(sub: pd.DataFrame, label: str) -> None:
    t = sub[sub["outcome"].isin(["target_hit", "stop_hit", "timed_out"])]
    if len(t) < 10:
        print(f"  {label:<28} n={len(t):>4}  too few")
        return
    k = int((t["outcome"] == "target_hit").sum())
    lo, hi = wilson(k, len(t))
    print(f"  {label:<28} n={len(t):>4}  hit={k/len(t)*100:>5.1f}%  CI [{lo:>5.1f},{hi:>5.1f}]  "
          f"E[R]={t['r'].mean():+.3f}  vs {BREAKEVEN:.1f}% baseline")


def main() -> None:
    pd.set_option("display.width", 200)
    with db.get_connection(load_postgres_config()) as conn:
        df = collect(conn)
    print(f"\n{len(df)} signal-days collected\n")

    print("=" * 96)
    print("1. LONG vs SHORT -- is one side carrying the losses?")
    print("=" * 96)
    live = df[df["outcome"] != "no_trade"]
    nonov = trade_selection.select_non_overlapping(live)
    hit_line(nonov, "ALL trades")
    for d in ("bullish", "bearish"):
        hit_line(nonov[nonov["direction"] == d], f"  {d} (long)" if d == "bullish" else f"  {d} (short)")
    print("\n  Also the raw direction mix:")
    print("  " + ", ".join(f"{k}={v}" for k, v in df["direction"].value_counts().items()))

    print("\n" + "=" * 96)
    print("2. DOES EACH FACTOR PREDICT ON ITS OWN? (forward 20-day return by factor score)")
    print("=" * 96)
    print("  A working factor is MONOTONIC low->high and holds its sign in both halves.")
    print(f"  {'factor':<20} {'bucket':<10} {'n':>5} {'fwd20 bps':>11} {'t':>7} | "
          f"{'early':>9} {'late':>9} {'agree':>6}")
    for fac in ("trend", "momentum_breakout", "oi_positioning"):
        sub = df[df[fac].notna()].copy()
        if len(sub) < 60:
            print(f"  {fac:<20} insufficient data ({len(sub)})")
            continue
        # Breakout is 0 on most days; bucket by sign instead of quantiles there.
        if sub[fac].abs().le(1e-9).mean() > 0.4:
            sub["bucket"] = np.select(
                [sub[fac] > 0, sub[fac] < 0], ["up-break", "down-break"], default="no-break")
            order = ["down-break", "no-break", "up-break"]
        else:
            sub["bucket"] = pd.qcut(sub[fac], 3, labels=["low", "mid", "high"], duplicates="drop")
            order = ["low", "mid", "high"]
        for b in order:
            g = sub[sub["bucket"] == b]
            if len(g) < 20:
                continue
            e, l = g[g["date"] < SPLIT]["fwd20"], g[g["date"] >= SPLIT]["fwd20"]
            agree = "YES" if len(e) > 10 and len(l) > 10 and (e.mean() > 0) == (l.mean() > 0) else "no"
            print(f"  {fac:<20} {b:<10} {len(g):>5} {g['fwd20'].mean()*10000:>11.1f} "
                  f"{tstat(g['fwd20']):>7.2f} | {e.mean()*10000:>9.1f} {l.mean()*10000:>9.1f} {agree:>6}")
        print()

    print("=" * 96)
    print("3. DOES THE COMPOSITE RANK BETTER THAN ITS PARTS?")
    print("=" * 96)
    print("  If the ensemble were adding value, high-confidence trades would beat")
    print("  low-confidence ones. Terciles of the confidence score:")
    if len(nonov) >= 45:
        n2 = nonov.copy()
        n2["band"] = pd.qcut(n2["confidence"], 3, labels=["low", "mid", "high"], duplicates="drop")
        for b, g in n2.groupby("band", observed=True):
            hit_line(g, f"  confidence {b}")


if __name__ == "__main__":
    main()
