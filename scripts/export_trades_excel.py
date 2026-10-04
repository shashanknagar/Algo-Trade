"""Exports the paper-trading ledger to Excel, with the success-rate summary.

Sheets:
  Trades        -- every recommendation and its outcome
  Summary       -- hit rate, expectancy and R-multiple stats, overall and per underlying
  DataCoverage  -- how much option-chain data has accumulated (the asset being built)

On hit rate: it is reported here because it was asked for, but read it alongside
expectancy. At the 1:1.5 risk:reward used by these trades, breakeven is 40% -- a 45%
hit rate is profitable and a 65% hit rate with a poor payoff ratio can still lose
money. Hit rate alone is the single most misleading number in trading.

Usage:
    python scripts/export_trades_excel.py [--out paper_trades.xlsx]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from algotrade import db
from algotrade.config import load_postgres_config

RISK_REWARD = 1.5
BREAKEVEN_HIT_RATE = 1 / (1 + RISK_REWARD)


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("ALL", trades)] + [(u, g) for u, g in trades.groupby("underlying")]
    for name, g in groups:
        resolved = g[g["status"].isin(["target_hit", "stop_hit"])]
        n_res = len(resolved)
        hits = int((resolved["status"] == "target_hit").sum()) if n_res else 0
        hit_rate = hits / n_res if n_res else None
        r = pd.to_numeric(g["r_multiple"], errors="coerce").dropna()
        rows.append({
            "scope": name,
            "total_recommendations": len(g),
            "open": int((g["status"] == "open").sum()),
            "resolved": n_res,
            "target_hit": hits,
            "stop_hit": n_res - hits,
            "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
            "breakeven_hit_rate": round(BREAKEVEN_HIT_RATE, 3),
            "profitable_vs_breakeven": (
                None if hit_rate is None else ("YES" if hit_rate > BREAKEVEN_HIT_RATE else "no")
            ),
            "expectancy_r": round(float(r.mean()), 3) if len(r) else None,
            "total_r": round(float(r.sum()), 2) if len(r) else None,
            "best_r": round(float(r.max()), 2) if len(r) else None,
            "worst_r": round(float(r.min()), 2) if len(r) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="paper_trades.xlsx")
    args = parser.parse_args()

    config = load_postgres_config()
    with db.get_connection(config) as conn:
        trades = pd.read_sql(
            "SELECT id, generated_at, underlying, recommendation, option_type, strike_price, "
            "expiry_date, direction, entry_price, underlying_spot, target_price, "
            "stop_loss_price, confidence_score, model_version, status, resolved_at, "
            "resolved_price, r_multiple FROM paper_trades ORDER BY generated_at DESC",
            conn,
        )
        coverage = pd.read_sql(
            "SELECT underlying, COUNT(*) AS snapshot_rows, "
            "COUNT(DISTINCT time) AS distinct_minutes, "
            "COUNT(DISTINCT strike_price) AS strikes_covered, "
            "MIN(time) AS first_captured, MAX(time) AS last_captured "
            "FROM option_chain_snapshots GROUP BY underlying",
            conn,
        )

    if trades.empty:
        print("No paper trades recorded yet -- run scripts/run_live_paper.py during "
              "market hours first.")
        summary = pd.DataFrame([{"scope": "ALL", "total_recommendations": 0}])
    else:
        summary = summarize(trades)

    out = Path(args.out)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        trades.to_excel(writer, sheet_name="Trades", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        coverage.to_excel(writer, sheet_name="DataCoverage", index=False)

    print(f"Wrote {out.resolve()}")
    print(f"  Trades sheet      : {len(trades)} recommendations")
    print(f"  DataCoverage sheet: {len(coverage)} underlyings with captured chain data")
    if not trades.empty:
        print("\nSummary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
