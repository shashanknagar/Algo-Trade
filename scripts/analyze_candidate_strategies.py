"""Tests five researched strategies against this project's own data, rather than
adopting them on reputation.

CANDIDATES AND WHY EACH WAS SHORTLISTED:

  A. OVERNIGHT vs INTRADAY DECOMPOSITION (Cooper/Cliff/Gulen 2008, plus 2025
     replications). Finding: index risk premium is earned almost entirely OVERNIGHT
     while intraday returns behave like a random walk. If that holds for NIFTY it
     would explain a great deal -- our close-to-close model mixes a positive overnight
     component with a near-random intraday one.

  B. TIME-SERIES MOMENTUM, 12-MONTH (Moskowitz/Ooi/Pedersen). The best-evidenced
     momentum formulation, and NOT what we implemented: our trend factor is a
     20/50-day SMA crossover, a much shorter and noisier proxy.

  C. VOLATILITY TARGETING / SCALING. Sizing inversely to recent volatility reliably
     improves Sharpe in the literature. Note this improves RISK-ADJUSTED return
     without requiring any better prediction -- which matters here, because our
     prediction is the part that does not work.

  D. TURN-OF-MONTH EFFECT. A long-documented calendar anomaly, trivially testable,
     and if real it is nearly free to exploit.

  E. VRP ENTRY TIMING. Refines the ONE strategy that validated here rather than
     adding another directional signal. Does conditioning entry on VIX percentile or
     on the observable premium improve the variance-risk-premium harvest?

Everything is split early/late. In this project that replication test has been the
sole reliable filter: every finding that failed it later evaporated.
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from algotrade import db, ml_dataset, vrp_strategy
from algotrade.config import load_postgres_config

SPLIT = date(2024, 1, 1)
ANNUALIZER = math.sqrt(252)


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std() == 0:
        return float("nan")
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2 or x.std() == 0:
        return float("nan")
    return float(x.mean() / x.std() * ANNUALIZER)


def halves(d: pd.DataFrame):
    return d[d["date"] < SPLIT], d[d["date"] >= SPLIT]


def candidate_a_overnight(data: pd.DataFrame) -> None:
    print("\n" + "=" * 84)
    print("A. OVERNIGHT vs INTRADAY -- is the premium earned while the market is shut?")
    print("=" * 84)
    for inst, d in data.groupby("instrument"):
        d = d.copy()
        d["overnight"] = d["open"] / d["close"].shift(1) - 1
        d["intraday"] = d["close"] / d["open"] - 1
        d["c2c"] = d["close"] / d["close"].shift(1) - 1
        d = d.dropna(subset=["overnight", "intraday"])
        e, l = halves(d)
        print(f"\n  {inst}  (n={len(d)})")
        print(f"    {'component':<12} {'ann.return':>11} {'t-stat':>8} {'Sharpe':>8} | "
              f"{'early ann':>10} {'late ann':>10}  agree")
        for name in ("overnight", "intraday", "c2c"):
            ann = d[name].mean() * 252 * 100
            ea, la = e[name].mean() * 252 * 100, l[name].mean() * 252 * 100
            agree = "YES" if (ea > 0) == (la > 0) else "no"
            print(f"    {name:<12} {ann:>10.1f}% {tstat(d[name]):>8.2f} {sharpe(d[name]):>8.2f} | "
                  f"{ea:>9.1f}% {la:>9.1f}%  {agree}")
    print("\n  If overnight is strongly positive and intraday is flat/negative in BOTH")
    print("  halves, then WHEN you hold matters more than WHICH WAY you predict.")


def candidate_b_tsmom(data: pd.DataFrame) -> None:
    print("\n" + "=" * 84)
    print("B. TIME-SERIES MOMENTUM (12-month) vs our 20/50 SMA crossover")
    print("=" * 84)
    print(f"  {'instrument':<11} {'rule':<22} {'ann.return':>11} {'Sharpe':>8} | "
          f"{'early Sh':>9} {'late Sh':>8}  agree")
    for inst, d in data.groupby("instrument"):
        d = d.copy()
        d["ret"] = d["close"].pct_change()
        # 12-month TSMOM: long if trailing 252-day return positive, else short.
        d["tsmom_sig"] = np.sign(d["close"] / d["close"].shift(252) - 1)
        # Our existing rule, for a like-for-like comparison.
        d["sma_sig"] = np.sign(d["close"].rolling(20).mean() - d["close"].rolling(50).mean())
        d = d.dropna(subset=["tsmom_sig", "sma_sig", "ret"])
        for label, col in (("TSMOM 12m", "tsmom_sig"), ("SMA 20/50 (ours)", "sma_sig"),
                           ("buy & hold", None)):
            strat = d["ret"] if col is None else d[col].shift(1) * d["ret"]
            e, l = halves(d.assign(s=strat))
            agree = "YES" if (sharpe(e["s"]) > 0) == (sharpe(l["s"]) > 0) else "no"
            print(f"  {inst:<11} {label:<22} {strat.mean()*252*100:>10.1f}% {sharpe(strat):>8.2f} | "
                  f"{sharpe(e['s']):>9.2f} {sharpe(l['s']):>8.2f}  {agree}")


def candidate_c_voltarget(data: pd.DataFrame) -> None:
    print("\n" + "=" * 84)
    print("C. VOLATILITY TARGETING -- same signal, size scaled inversely to volatility")
    print("=" * 84)
    print("  Improves risk-adjusted return WITHOUT needing a better prediction.")
    print(f"\n  {'instrument':<11} {'variant':<26} {'Sharpe':>8} | {'early':>8} {'late':>8}  agree")
    for inst, d in data.groupby("instrument"):
        d = d.copy()
        d["ret"] = d["close"].pct_change()
        d["sig"] = np.sign(d["close"].rolling(20).mean() - d["close"].rolling(50).mean()).shift(1)
        d["rvol"] = d["ret"].rolling(20).std()
        target_vol = d["rvol"].median()
        d["scale"] = (target_vol / d["rvol"]).clip(0, 3)   # cap leverage at 3x
        d = d.dropna(subset=["sig", "ret", "scale"])
        variants = {
            "unscaled (ours)": d["sig"] * d["ret"],
            "vol-targeted": d["sig"] * d["ret"] * d["scale"],
            "long-only vol-targeted": d["ret"] * d["scale"],
        }
        for label, strat in variants.items():
            e, l = halves(d.assign(s=strat))
            agree = "YES" if (sharpe(e["s"]) > 0) == (sharpe(l["s"]) > 0) else "no"
            print(f"  {inst:<11} {label:<26} {sharpe(strat):>8.2f} | "
                  f"{sharpe(e['s']):>8.2f} {sharpe(l['s']):>8.2f}  {agree}")


def candidate_d_turn_of_month(data: pd.DataFrame) -> None:
    print("\n" + "=" * 84)
    print("D. TURN-OF-MONTH EFFECT (last 1 + first 3 trading days vs the rest)")
    print("=" * 84)
    print(f"  {'instrument':<11} {'window':<16} {'n':>5} {'mean bps':>9} {'t':>7} | "
          f"{'early':>8} {'late':>8}  agree")
    for inst, d in data.groupby("instrument"):
        d = d.copy().reset_index(drop=True)
        d["ret"] = d["close"].pct_change()
        d["ym"] = d["date"].apply(lambda x: (x.year, x.month))
        d["dom_rank"] = d.groupby("ym").cumcount()               # 0 = first day of month
        d["dom_from_end"] = d.groupby("ym")["dom_rank"].transform("max") - d["dom_rank"]
        d["tom"] = (d["dom_rank"] <= 2) | (d["dom_from_end"] == 0)
        d = d.dropna(subset=["ret"])
        for label, mask in (("turn-of-month", d["tom"]), ("rest of month", ~d["tom"])):
            sub = d[mask]
            e, l = halves(sub)
            agree = "YES" if (e["ret"].mean() > 0) == (l["ret"].mean() > 0) else "no"
            print(f"  {inst:<11} {label:<16} {len(sub):>5} {sub['ret'].mean()*10000:>9.1f} "
                  f"{tstat(sub['ret']):>7.2f} | {e['ret'].mean()*10000:>8.1f} "
                  f"{l['ret'].mean()*10000:>8.1f}  {agree}")


def candidate_e_vrp_timing(nifty: pd.DataFrame, vix: pd.DataFrame) -> None:
    print("\n" + "=" * 84)
    print("E. VRP ENTRY TIMING -- refining the one strategy that actually validated")
    print("=" * 84)
    df = vrp_strategy.build_vol_frame(nifty, vix)
    base = vrp_strategy.StrategyConfig(max_loss_vol_points=5.0, label="baseline")

    variants = {
        "baseline (always sell)": vrp_strategy.StrategyConfig(max_loss_vol_points=5.0),
        "only if IV > trailing RV": vrp_strategy.StrategyConfig(
            max_loss_vol_points=5.0, min_observable_premium=0.0),
        "only if premium > 2 pts": vrp_strategy.StrategyConfig(
            max_loss_vol_points=5.0, min_observable_premium=2.0),
        "only if premium > 4 pts": vrp_strategy.StrategyConfig(
            max_loss_vol_points=5.0, min_observable_premium=4.0),
        "prem > 2 + no VIX spike": vrp_strategy.StrategyConfig(
            max_loss_vol_points=5.0, min_observable_premium=2.0, max_vix_spike_5d=0.15),
    }
    print(f"  {'variant':<26} {'full n':>7} {'win':>6} {'mean':>7} {'Sharpe':>7} | "
          f"{'test n':>7} {'test win':>9} {'test Sh':>8}")
    test_df = df[df["date"] >= SPLIT]
    for label, cfg in variants.items():
        cfg.label = label
        full = vrp_strategy.summarize(vrp_strategy.simulate(df, cfg), cfg)
        test = vrp_strategy.summarize(vrp_strategy.simulate(test_df, cfg), cfg)
        if not full.get("n_trades"):
            continue
        print(f"  {label:<26} {full['n_trades']:>7} {full['win_rate']:>6.2f} "
              f"{full['mean_pnl']:>7.2f} {full['sharpe_approx']:>7.2f} | "
              f"{test.get('n_trades', 0):>7} {test.get('win_rate') or 0:>9.2f} "
              f"{test.get('sharpe_approx') or 0:>8.2f}")
    print("\n  Watch for filters that improve the full sample but shrink test n to")
    print("  single digits -- that is selection, not skill.")


def main() -> None:
    pd.set_option("display.width", 220)
    config = load_postgres_config()
    frames = []
    with db.get_connection(config) as conn:
        for inst in ("NIFTY", "BANKNIFTY"):
            d = ml_dataset._load_full_ohlc(conn, inst)
            d["instrument"] = inst
            frames.append(d)
        nifty = ml_dataset._load_full_ohlc(conn, "NIFTY")
        vix = ml_dataset._load_vix(conn)
    data = pd.concat(frames, ignore_index=True)
    print(f"Data: {len(data)} index-days, {data['date'].min()} to {data['date'].max()}")

    candidate_a_overnight(data)
    candidate_b_tsmom(data)
    candidate_c_voltarget(data)
    candidate_d_turn_of_month(data)
    candidate_e_vrp_timing(nifty, vix)

    print("\n" + "=" * 84)
    print("'agree' = kept its sign across both halves. That column is the whole test.")
    print("=" * 84)


if __name__ == "__main__":
    main()
