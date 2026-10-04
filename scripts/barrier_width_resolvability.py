"""How wide must barriers be before DAILY bars can actually decide the outcome?

THE PROBLEM THIS ANSWERS. At +50/-30 points the same signals on the same data score
a 25.2% hit rate under the pessimistic tie-break and 72.0% under the optimistic one.
Both are "correct" -- 47% of trades were settled on a bar where both barriers sat
inside the day's high-low range, and a daily bar records the high and the low but
never their ORDER. Which barrier was touched first is not in the data at all. The
tie-break rule invents it.

That 72% figure is exactly the hit rate this project has been chasing, and it is an
artifact. Reporting it would have been a fabrication dressed as a result.

So: generate each signal ONCE, then resolve it at a range of barrier widths under
both conventions. Where the two conventions converge, daily data is genuinely
answering the question. Where they diverge, it is not, and only intraday data can
settle it.

Widths keep the 5:3 reward:risk shape of your 50/30 request and scale it up, so the
breakeven stays 37.5% throughout and the hit rates remain directly comparable.
"""
from __future__ import annotations

import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from algotrade import backtest_signal, db, signal_generator, trade_selection, triple_barrier
from algotrade.config import load_postgres_config

SEED, N_DATES, HORIZON = 7, 400, 20
BREAKEVEN = 1 / (1 + 50 / 30) * 100
WIDTHS = [(50, 30), (100, 60), (150, 90), (200, 120), (300, 180), (500, 300), (750, 450)]


def resolve_at(direction, entry, up, down, future, tie_breaker):
    target = entry + up if direction == "bullish" else entry - up
    stop = entry - down if direction == "bullish" else entry + down
    out = triple_barrier.resolve(direction, entry, target, stop, future, up / down, tie_breaker)
    ambiguous = None
    if out.bars_to_resolve and out.status in ("target_hit", "stop_hit"):
        bar = future.iloc[out.bars_to_resolve - 1]
        hi, lo = float(bar["high"]), float(bar["low"])
        ambiguous = hi >= max(target, stop) and lo <= min(target, stop)
    return out, ambiguous


def main() -> None:
    pd.set_option("display.width", 200)
    rng = random.Random(SEED)
    rows = []

    with db.get_connection(load_postgres_config()) as conn:
        for inst in ("NIFTY", "BANKNIFTY"):
            span = backtest_signal._get_valid_date_range(conn, inst, "positional")
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT time::date FROM ohlc_bars WHERE trading_symbol = %s "
                    "AND interval = '1day' AND time::date BETWEEN %s AND %s ORDER BY 1",
                    (inst, span[0], span[1]))
                all_dates = [r[0] for r in cur.fetchall()]
            index_of = {d: i for i, d in enumerate(all_dates)}
            picks = sorted(rng.sample(all_dates, min(N_DATES, len(all_dates))))

            for k, d in enumerate(picks, 1):
                if k % 100 == 0:
                    print(f"  {inst} {k}/{len(picks)}", flush=True)
                try:
                    sig = signal_generator.generate_signal(conn, inst, d, "positional")
                except Exception:
                    continue
                if sig.direction == "neutral" or sig.entry_price is None:
                    continue
                future = backtest_signal._load_ohlc_after(conn, inst, d, HORIZON)
                if future.empty:
                    continue
                for up, down in WIDTHS:
                    for tb in ("stop", "target"):
                        out, amb = resolve_at(sig.direction, sig.entry_price, up, down, future, tb)
                        rows.append({
                            "instrument": inst, "date": d, "width": f"+{up}/-{down}",
                            "up": up, "tie": tb, "outcome": out.status,
                            "r": out.r_multiple, "ambiguous": amb,
                            "days": out.bars_to_resolve,
                            "entry_index": index_of[d],
                            "resolve_index": index_of[d] + (out.bars_to_resolve or 0),
                        })

    df = pd.DataFrame(rows)
    print(f"\n{len(df):,} resolutions across {len(WIDTHS)} widths x 2 conventions\n")

    print("=" * 100)
    print(f"CONVERGENCE OF THE TWO CONVENTIONS  (breakeven {BREAKEVEN:.1f}% for all widths)")
    print("=" * 100)
    print(f"  {'barrier':<12} {'% as of NIFTY':>14} {'tie-decided':>12} {'hit: STOP':>11} "
          f"{'hit: TARGET':>12} {'spread':>8} {'med days':>9}")

    for up, down in WIDTHS:
        w = f"+{up}/-{down}"
        cells = {}
        for tb in ("stop", "target"):
            sub = df[(df["width"] == w) & (df["tie"] == tb)]
            nonov = trade_selection.select_non_overlapping(
                sub[sub["outcome"] != "no_trade"], group_cols=("instrument",))
            traded = nonov[nonov["outcome"].isin(["target_hit", "stop_hit", "timed_out"])]
            cells[tb] = (traded["outcome"] == "target_hit").mean() * 100 if len(traded) else float("nan")
        sub = df[(df["width"] == w) & (df["tie"] == "stop")]
        amb = sub["ambiguous"].dropna().mean() * 100
        med = sub[sub["outcome"].isin(["target_hit", "stop_hit"])]["days"].median()
        nifty_pct = up / 20000 * 100      # rough: mid-sample NIFTY level
        spread = cells["target"] - cells["stop"]
        print(f"  {w:<12} {nifty_pct:>13.2f}% {amb:>11.1f}% {cells['stop']:>11.1f} "
              f"{cells['target']:>12.1f} {spread:>8.1f} {med:>9.0f}")

    print("\n  'spread' is how much of the hit rate is invented by the tie-break rule.")
    print("  A result is only real where that spread is small. Anything reported from a")
    print("  wide-spread row is a statement about the convention, not about the market.")
    print("\n  NOTE the direction of the trade-off: barriers wide enough for daily bars to")
    print("  resolve honestly are far wider than the 50/30 you asked for. Measuring 50/30")
    print("  properly needs intraday bars -- which is what the Pro 1-minute feed provides.")


if __name__ == "__main__":
    main()
