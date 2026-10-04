"""Does market breadth improve the VRP entry filter -- the one strategy that works?

THE HYPOTHESIS: vrp_strategy decides whether to sell volatility by comparing India
VIX to TRAILING realized vol. But what determines the P&L is FORWARD realized vol.
Breadth (fraction of stocks at highs minus at lows) correlated -0.184 with forward
vol *after* removing everything trailing vol explains -- so it carries information
the current filter is blind to. More stocks at lows => higher forward vol => selling
vol is more dangerous than the trailing-vol comparison suggests.

TWO WAYS TO USE IT, tested separately because they fail differently:
  A. FORECAST. Replace trailing RV with a fitted forecast of forward RV that includes
     breadth, and recompute the premium. Fitted on TRAIN only.
  B. GATE. A single rule -- skip entries when breadth is in its worst quintile. One
     parameter, no fitting, much harder to overfit.

POWER WARNING, stated up front because it governs how the output may be read: trades
are non-overlapping 21-day blocks, so the held-out period yields only ~30 of them.
Thirty trades cannot distinguish a Sharpe of 2.2 from 2.6. The FORECAST test (~1,600
daily observations) is the part of this with real statistical power; the strategy
comparison is a sanity check that the forecast change does not break anything, and
must not be read as proof that it helps.
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from algotrade import db, ml_dataset, vrp_strategy
from algotrade.config import load_postgres_config
from algotrade.vrp_strategy import StrategyConfig

CACHE = Path(sys.argv[1])
SPLIT = date(2024, 1, 1)
HORIZON = vrp_strategy.DEFAULT_HORIZON_DAYS


def build_breadth() -> pd.DataFrame:
    df = pd.read_pickle(CACHE)
    at_low = df["close"] <= 1.015 * df["lo20"]
    at_high = df["close"] >= 0.985 * df["hi20"]
    spike = df["vol_ratio"] > 1.3
    df = df.assign(quiet_low=at_low & ~spike, at_high=at_high, at_low=at_low)
    b = df.groupby("date").agg(
        quiet_low_pct=("quiet_low", "mean"),
        at_low_pct=("at_low", "mean"),
        at_high_pct=("at_high", "mean"),
        n=("close", "size"),
    ).reset_index()
    b = b[b["n"] >= 200].copy()
    b["net_hl"] = b["at_high_pct"] - b["at_low_pct"]
    # Percentile vs own trailing year: levels drift with the universe, ranks don't.
    # rolling() is strictly backward-looking, so this is knowable at entry.
    b["net_hl_rank"] = b["net_hl"].rolling(252, min_periods=120).rank(pct=True)
    b["quiet_low_rank"] = b["quiet_low_pct"].rolling(252, min_periods=120).rank(pct=True)
    return b[["date", "net_hl", "quiet_low_pct", "net_hl_rank", "quiet_low_rank"]]


def main() -> None:
    pd.set_option("display.width", 200)
    with db.get_connection(load_postgres_config()) as conn:
        nifty = ml_dataset._load_full_ohlc(conn, "NIFTY")
        vix = ml_dataset._load_vix(conn)

    df = vrp_strategy.build_vol_frame(nifty, vix, HORIZON)
    df = df.merge(build_breadth(), on="date", how="inner").sort_values("date").reset_index(drop=True)
    fit = df.dropna(subset=["forward_realized_vol", "trailing_realized_vol", "net_hl", "net_hl_rank"])
    train, test = fit[fit["date"] < SPLIT], fit[fit["date"] >= SPLIT]
    print(f"Aligned {len(fit)} days | train {len(train)} | test {len(test)} (held out)\n")

    # ---------- PART A: does breadth improve the forward-vol FORECAST? ----------
    print("=" * 96)
    print(f"PART A: forecasting the next {HORIZON} days of realized vol (the P&L driver)")
    print("=" * 96)
    specs = {
        "trailing RV only (current)": ["trailing_realized_vol"],
        "trailing RV + VIX": ["trailing_realized_vol", "implied_vol"],
        "trailing RV + breadth": ["trailing_realized_vol", "net_hl", "quiet_low_pct"],
        "trailing RV + VIX + breadth": ["trailing_realized_vol", "implied_vol", "net_hl", "quiet_low_pct"],
    }
    y_tr, y_te = train["forward_realized_vol"], test["forward_realized_vol"]
    models, base_rmse = {}, None
    print(f"  {'specification':<32} {'test RMSE':>10} {'vs base':>9} {'test corr':>10}")
    for label, feats in specs.items():
        m = LinearRegression().fit(train[feats], y_tr)
        pred = pd.Series(m.predict(test[feats]), index=test.index)
        rmse = float(np.sqrt(((pred - y_te) ** 2).mean()))
        base_rmse = rmse if base_rmse is None else base_rmse
        models[label] = (m, feats)
        print(f"  {label:<32} {rmse:>10.3f} {(base_rmse - rmse) / base_rmse * 100:>8.1f}% "
              f"{pred.corr(y_te):>10.3f}")

    m_bre, f_bre = models["trailing RV + breadth"]
    print(f"\n  breadth coefficients: " +
          ", ".join(f"{k}={v:+.2f}" for k, v in zip(f_bre, m_bre.coef_)))
    print("  A negative net_hl coefficient means deteriorating breadth predicts HIGHER")
    print("  forward vol -- the mechanism that motivated this test.")

    # ---------- PART B: does it improve the STRATEGY? ----------
    print("\n" + "=" * 96)
    print("PART B: strategy impact (UNDERPOWERED -- read the caveat below the table)")
    print("=" * 96)

    df_f = df.copy()
    for label in ("trailing RV + breadth", "trailing RV + VIX + breadth"):
        m, feats = models[label]
        ok = df_f[feats].notna().all(axis=1)
        col = "fc_breadth" if "VIX" not in label else "fc_breadth_vix"
        df_f.loc[ok, col] = m.predict(df_f.loc[ok, feats])

    variants: dict[str, pd.DataFrame] = {}
    variants["current: IV - trailing RV"] = df_f
    for col, name in (("fc_breadth", "IV - forecast(RV+breadth)"),
                      ("fc_breadth_vix", "IV - forecast(RV+VIX+breadth)")):
        v = df_f.copy()
        v["observable_premium"] = v["implied_vol"] - v[col]
        variants[name] = v

    # Gate variant: block entries outright when breadth is in its worst quintile.
    gated = df_f.copy()
    gated.loc[gated["net_hl_rank"] < 0.2, "observable_premium"] = -99.0  # forces a skip
    variants["current + breadth gate (worst 20%)"] = gated

    # DIAGNOSTIC, not a candidate strategy. The breadth-forecast variants underperformed,
    # and there are two very different explanations: breadth is useless here, or
    # substituting ANY fitted forecast for trailing RV breaks the filter. Running the
    # same substitution with a breadth-free forecast separates them. A good number here
    # is not a discovery -- it is the control arm.
    m_rv_vix = LinearRegression().fit(train[["trailing_realized_vol", "implied_vol"]],
                                      y_tr)
    ctrl = df_f.copy()
    ok = ctrl[["trailing_realized_vol", "implied_vol"]].notna().all(axis=1)
    ctrl.loc[ok, "fc"] = m_rv_vix.predict(ctrl.loc[ok, ["trailing_realized_vol", "implied_vol"]])
    ctrl["observable_premium"] = ctrl["implied_vol"] - ctrl["fc"]
    variants["[control] IV - forecast(RV+VIX)"] = ctrl

    print(f"  {'variant':<36} {'full n':>7} {'win':>6} {'mean':>7} {'Sh':>6} | "
          f"{'TEST n':>7} {'win':>6} {'mean':>7} {'Sh':>6} {'worst':>7}")
    for label, frame in variants.items():
        cfg = StrategyConfig(label=label)
        full = vrp_strategy.summarize(vrp_strategy.simulate(frame, cfg), cfg)
        te = frame[frame["date"] >= SPLIT]
        tst = vrp_strategy.summarize(vrp_strategy.simulate(te, cfg), cfg)
        if not full.get("n_trades"):
            print(f"  {label:<36} no trades")
            continue
        print(f"  {label:<36} {full['n_trades']:>7} {full['win_rate']:>6.2f} "
              f"{full['mean_pnl']:>7.2f} {full['sharpe_approx']:>6.2f} | "
              f"{tst.get('n_trades', 0):>7} {tst.get('win_rate') or 0:>6.2f} "
              f"{tst.get('mean_pnl') or 0:>7.2f} {tst.get('sharpe_approx') or 0:>6.2f} "
              f"{tst.get('worst') or 0:>7.2f}")

    print("\n  With ~30 held-out trades the standard error on Sharpe is roughly +/-0.4.")
    print("  Treat any difference smaller than that as noise, not improvement.")


if __name__ == "__main__":
    main()
