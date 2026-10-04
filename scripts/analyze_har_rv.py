"""Tests HAR-RV volatility forecasting as a replacement for the crude 'trailing
realized vol' currently used by the VRP strategy's entry filter.

WHY THIS MATTERS HERE: vrp_strategy.py decides whether to sell volatility by comparing
India VIX to TRAILING realized vol. But trailing RV is a mediocre forecast of the
volatility that will actually occur over the holding period -- and that future RV is
what determines whether the trade wins. A better forecast should mean a better filter.

HAR-RV (Corsi) regresses future realized volatility on realized volatility measured
over three horizons -- daily, weekly and monthly -- capturing the well-documented fact
that volatility has memory at several timescales simultaneously. The literature puts
it 35-40% ahead of GARCH(1,1) on forecast error.

TWO TESTS:
  1. Does HAR-RV forecast future 21-day realized vol better than the naive trailing
     value? (RMSE on held-out data)
  2. Does swapping it into the VRP entry filter actually improve the strategy? A
     better forecast is only worth having if it changes decisions for the better.

Fitted on TRAIN only, evaluated on TEST only.
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

SPLIT = date(2024, 1, 1)
HORIZON = vrp_strategy.DEFAULT_HORIZON_DAYS


def build(conn) -> pd.DataFrame:
    nifty = ml_dataset._load_full_ohlc(conn, "NIFTY")
    vix = ml_dataset._load_vix(conn)
    df = vrp_strategy.build_vol_frame(nifty, vix, HORIZON)

    r = df["log_ret"]
    ann = math.sqrt(252) * 100  # to vol points, matching VIX units
    df["rv_d"] = r.rolling(1).std().fillna(r.abs()) * ann      # daily component
    df["rv_w"] = r.rolling(5).std() * ann                       # weekly
    df["rv_m"] = r.rolling(22).std() * ann                      # monthly
    return df.dropna(subset=["rv_w", "rv_m", "forward_realized_vol", "trailing_realized_vol"])


def main() -> None:
    pd.set_option("display.width", 200)
    with db.get_connection(load_postgres_config()) as conn:
        df = build(conn)

    train = df[df["date"] < SPLIT]
    test = df[df["date"] >= SPLIT]
    print(f"Train {len(train)} days | Test {len(test)} days (held out)\n")

    # ---- Test 1: forecast accuracy -----------------------------------------
    feats = ["rv_d", "rv_w", "rv_m"]
    model = LinearRegression().fit(train[feats], train["forward_realized_vol"])
    coefs = dict(zip(feats, model.coef_.round(3)))
    print("=" * 78)
    print("TEST 1: forecasting the NEXT 21 days of realized volatility")
    print("=" * 78)
    print(f"  HAR-RV fitted coefficients: {coefs}  intercept={model.intercept_:.2f}")

    test = test.copy()
    test["har_forecast"] = model.predict(test[feats])
    naive = test["trailing_realized_vol"]
    actual = test["forward_realized_vol"]

    def rmse(pred):
        return float(np.sqrt(((pred - actual) ** 2).mean()))

    def mae(pred):
        return float((pred - actual).abs().mean())

    r_har, r_naive = rmse(test["har_forecast"]), rmse(naive)
    print(f"\n  {'forecaster':<28} {'RMSE':>8} {'MAE':>8} {'corr w/ actual':>16}")
    print(f"  {'naive: trailing 21d RV':<28} {r_naive:>8.2f} {mae(naive):>8.2f} "
          f"{naive.corr(actual):>16.3f}")
    print(f"  {'HAR-RV (daily+weekly+monthly)':<28} {r_har:>8.2f} {mae(test['har_forecast']):>8.2f} "
          f"{test['har_forecast'].corr(actual):>16.3f}")
    improvement = (r_naive - r_har) / r_naive * 100
    print(f"\n  RMSE improvement from HAR-RV: {improvement:+.1f}%")

    # ---- Test 2: does it improve the actual strategy? -----------------------
    print("\n" + "=" * 78)
    print("TEST 2: does swapping it into the VRP entry filter improve the strategy?")
    print("=" * 78)
    print("  A better forecast only matters if it changes decisions for the better.\n")

    # Rebuild the observable premium using the HAR forecast instead of trailing RV.
    df = df.copy()
    df["har_forecast"] = model.predict(df[feats])
    har_df = df.copy()
    har_df["observable_premium"] = har_df["implied_vol"] - har_df["har_forecast"]

    variants = {
        "current: IV - trailing RV": (df, 2.0),
        "HAR: IV - HAR forecast": (har_df, 2.0),
        "HAR, stricter (>3 pts)": (har_df, 3.0),
    }
    print(f"  {'variant':<28} {'full n':>7} {'win':>6} {'mean':>7} {'Sharpe':>7} | "
          f"{'test n':>7} {'test win':>9} {'test Sh':>8}")
    for label, (frame, min_prem) in variants.items():
        cfg = StrategyConfig(max_loss_vol_points=5.0, min_observable_premium=min_prem, label=label)
        full = vrp_strategy.summarize(vrp_strategy.simulate(frame, cfg), cfg)
        t = frame[frame["date"] >= SPLIT]
        tst = vrp_strategy.summarize(vrp_strategy.simulate(t, cfg), cfg)
        if not full.get("n_trades"):
            print(f"  {label:<28} no trades")
            continue
        print(f"  {label:<28} {full['n_trades']:>7} {full['win_rate']:>6.2f} "
              f"{full['mean_pnl']:>7.2f} {full['sharpe_approx']:>7.2f} | "
              f"{tst.get('n_trades', 0):>7} {tst.get('win_rate') or 0:>9.2f} "
              f"{tst.get('sharpe_approx') or 0:>8.2f}")

    print("\n  Forecast accuracy and strategy improvement are different things -- a")
    print("  sharper forecast that never changes which trades get taken is worth nothing.")


if __name__ == "__main__":
    main()
