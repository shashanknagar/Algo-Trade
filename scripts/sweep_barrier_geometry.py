"""Barrier-geometry sweep: can small stops + big targets make the signal profitable?

THE REQUEST: accept a low hit rate, but size the barriers so that the few winners pay
for all the losers -- profits measured in index POINTS, not percent.

THE TRAP THIS MUST EXPOSE RATHER THAN HIDE. On a random walk the hit rate adjusts to
exactly cancel any barrier geometry: P(target first) = 1/(1+RR), so expectancy is
zero at every RR. Widening the target to 5x lowers the breakeven to 16.7% -- but it
lowers the hit rate too, because the target is five times farther away. Wide targets
pay ONLY if the market actually trends in the direction of the signal for long
enough to reach them. That is an empirical question about this signal on this
market, and the answer is what this sweep measures.

THREE DISCIPLINES, because a sweep over ~50 configurations is a machine for finding
lucky ones:
  1. TRAIN/TEST. The best geometry is chosen on pre-2024 data and its performance is
     then reported on 2024+ data it never saw. The test number is the only one that
     counts.
  2. CONTROL ARM. At every geometry the algorithm's direction is compared with
     always-long. If always-long earns the same points, the geometry is harvesting
     market drift, not signal -- and you do not need an algorithm for that.
  3. RESOLVABILITY. Small stops bring back the daily-bar ambiguity problem. Any
     geometry whose outcomes are mostly decided by the tie-break rule is reported
     as unmeasurable, not as a result.

Signals are generated once per date and reused across every geometry, so the sweep
compares barriers, not samples.
"""
from __future__ import annotations

import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from algotrade import db, ml_dataset, signal_generator, trade_selection
from algotrade.config import load_postgres_config

SPLIT = date(2024, 1, 1)
SEEDS = [1, 2, 3, 7, 99]
N_PER_SEED = 200
STOP_MULTS = [0.5, 1.0, 1.5, 2.0]            # x ATR(14)
RR_LIST = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
HORIZONS = [20, 40]


def atr14(d: pd.DataFrame) -> pd.Series:
    c, h, l = d["close"], d["high"], d["low"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def resolve(bars: pd.DataFrame, i: int, side: int, entry: float, stop_pts: float,
            target_pts: float, horizon: int) -> tuple[str, float, bool, int]:
    """Returns (outcome, pnl_points, decided_by_tiebreak, bars_used). Stop wins ties."""
    stop = entry - side * stop_pts
    target = entry + side * target_pts
    fwd = bars.iloc[i + 1:i + 1 + horizon]
    for k, (_, b) in enumerate(fwd.iterrows(), 1):
        hi, lo = float(b["high"]), float(b["low"])
        hit_stop = lo <= stop if side == 1 else hi >= stop
        hit_tgt = hi >= target if side == 1 else lo <= target
        if hit_stop:
            return "stop", -stop_pts, bool(hit_tgt), k
        if hit_tgt:
            return "target", target_pts, False, k
    if fwd.empty:
        return "none", 0.0, False, 0
    last = float(fwd.iloc[-1]["close"])
    return "timeout", side * (last - entry), False, len(fwd)


def collect_signals(conn) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    bars_by_inst, rows = {}, []
    for inst in ("NIFTY", "BANKNIFTY"):
        bars = ml_dataset._load_full_ohlc(conn, inst).sort_values("date").reset_index(drop=True)
        bars["atr"] = atr14(bars)
        bars_by_inst[inst] = bars
        idx = {d: i for i, d in enumerate(bars["date"])}
        eligible = list(bars["date"][60:-max(HORIZONS) - 1])
        picked: set = set()
        for s in SEEDS:
            picked |= set(random.Random(s).sample(eligible, min(N_PER_SEED, len(eligible))))
        for k, d in enumerate(sorted(picked), 1):
            if k % 200 == 0:
                print(f"  {inst} signals {k}/{len(picked)}", flush=True)
            try:
                sig = signal_generator.generate_signal(conn, inst, d, "positional")
            except Exception:
                continue
            if sig.direction == "neutral":
                continue
            i = idx[d]
            rows.append({"instrument": inst, "date": d, "i": i,
                         "side": 1 if sig.direction == "bullish" else -1,
                         "entry": float(bars.at[i, "close"]), "atr": float(bars.at[i, "atr"])})
    return pd.DataFrame(rows), bars_by_inst


def evaluate(sig: pd.DataFrame, bars_by_inst: dict, stop_mult: float, rr: float,
             horizon: int, direction_rule: str) -> pd.DataFrame:
    out = []
    for r in sig.itertuples(index=False):
        side = r.side if direction_rule == "algorithm" else 1
        stop_pts = stop_mult * r.atr
        outcome, pnl, tie, used = resolve(bars_by_inst[r.instrument], r.i, side, r.entry,
                                          stop_pts, rr * stop_pts, horizon)
        if outcome == "none":
            continue
        out.append({"instrument": r.instrument, "date": r.date, "outcome": outcome,
                    "pnl_pts": pnl, "r": pnl / stop_pts, "tie": tie,
                    "entry_index": r.i, "resolve_index": r.i + used})
    df = pd.DataFrame(out)
    return trade_selection.select_non_overlapping(df) if not df.empty else df


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n": 0}
    return {
        "n": len(t),
        "hit": round((t["outcome"] == "target").mean() * 100, 1),
        "tie%": round(t["tie"].mean() * 100, 1),
        "E[R]": round(t["r"].mean(), 3),
        "pts/trade": round(t["pnl_pts"].mean(), 1),
        "total_pts": round(t["pnl_pts"].sum(), 0),
    }


def main() -> None:
    pd.set_option("display.width", 240)
    with db.get_connection(load_postgres_config()) as conn:
        sig, bars = collect_signals(conn)
    print(f"\n{len(sig)} directional signals ({(sig['side'] == 1).sum()} long, "
          f"{(sig['side'] == -1).sum()} short)\n")

    results = []
    for horizon in HORIZONS:
        for sm in STOP_MULTS:
            for rr in RR_LIST:
                for rule in ("algorithm", "always_long"):
                    t = evaluate(sig, bars, sm, rr, horizon, rule)
                    tr, te = t[t["date"] < SPLIT], t[t["date"] >= SPLIT]
                    results.append({
                        "horizon": horizon, "stop_x_atr": sm, "RR": rr, "rule": rule,
                        "breakeven%": round(100 / (1 + rr), 1),
                        **{f"train_{k}": v for k, v in stats(tr).items()},
                        **{f"test_{k}": v for k, v in stats(te).items()},
                    })
    res = pd.DataFrame(results)

    # ---------- 1. The full grid for the algorithm, in points ------------------
    algo = res[res["rule"] == "algorithm"]
    print("=" * 130)
    print("1. ALGORITHM -- every geometry.  Watch the hit rate FALL as RR rises; the")
    print("   breakeven falls too, and the question is which falls faster.")
    print("=" * 130)
    cols = ["horizon", "stop_x_atr", "RR", "breakeven%", "train_n", "train_hit", "train_E[R]",
            "train_pts/trade", "test_n", "test_hit", "test_tie%", "test_E[R]", "test_pts/trade", "test_total_pts"]
    print(algo[cols].to_string(index=False))

    # ---------- 2. Select on TRAIN, report on TEST ------------------------------
    print("\n" + "=" * 130)
    print("2. HONEST SELECTION: best geometry by TRAIN points-per-trade, then its TEST result")
    print("=" * 130)
    measurable = algo[algo["test_tie%"] < 20]      # exclude geometries daily bars cannot score
    best = measurable.sort_values("train_pts/trade", ascending=False).head(5)
    print(best[cols].to_string(index=False))
    print("\n   (geometries with >20% tie-decided trades are excluded as unmeasurable on daily bars)")

    # ---------- 3. Control: does always-long earn the same? ----------------------
    print("\n" + "=" * 130)
    print("3. CONTROL: at the top-5 geometries, algorithm vs ALWAYS-LONG on TEST")
    print("=" * 130)
    print(f"  {'geometry':<24} {'algo pts/trade':>15} {'algo total':>11} | "
          f"{'long pts/trade':>15} {'long total':>11} | {'algo adds':>10}")
    for _, b in best.iterrows():
        ctrl = res[(res["rule"] == "always_long") & (res["horizon"] == b["horizon"])
                   & (res["stop_x_atr"] == b["stop_x_atr"]) & (res["RR"] == b["RR"])].iloc[0]
        geo = f"h{int(b['horizon'])} stop{b['stop_x_atr']}xATR RR{b['RR']}"
        adds = b["test_pts/trade"] - ctrl["test_pts/trade"]
        print(f"  {geo:<24} {b['test_pts/trade']:>15.1f} {b['test_total_pts']:>11.0f} | "
              f"{ctrl['test_pts/trade']:>15.1f} {ctrl['test_total_pts']:>11.0f} | {adds:>+10.1f}")

    print("\n  'algo adds' is the only column that measures the signal. Positive points from")
    print("  a long-biased book in a rising market are the market's, not the model's.")


if __name__ == "__main__":
    main()
