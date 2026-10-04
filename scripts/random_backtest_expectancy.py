"""Is the algorithm's positive expectancy an EDGE, or just long exposure to a rising market?

The random-sample run produced an apparent contradiction: a 34.1% hit rate (below the
40% random-walk breakeven) yet a POSITIVE +0.121R expectancy. Both are real. The
triple barrier has three exits, and timeouts resolve at whatever the market price is,
paying fractional R rather than a clean -1 or +1.5. So expectancy is not determined by
hit rate alone.

That makes expectancy the number that actually matters -- but it also opens an obvious
alternative explanation. The algorithm went long on 313 of 459 signals, and Indian
equities rose over the sample. Positive expectancy from a mostly-long book in a rising
market is BETA, not skill, and would be worth nothing.

The test: rerun the identical barriers, horizons and entry dates while replacing only
the algorithm's DIRECTION with dumb alternatives. If always-long scores the same, the
signal contributed nothing and the P&L was the market's, not the model's.
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

from algotrade import db, signal_generator, trade_selection, triple_barrier
from algotrade.config import load_postgres_config

SPLIT = date(2024, 1, 1)
N_DATES = 400
SEED = 7
HORIZON = 20


def load_bars(conn, instrument: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT time::date, open, high, low, close FROM ohlc_bars "
            "WHERE trading_symbol = %s AND interval = '1day' ORDER BY 1",
            (instrument,),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    return df


def resolve(bars: pd.DataFrame, i: int, direction: str, entry: float,
            target: float, stop: float) -> tuple[str, float, int]:
    """Walk forward from bar i+1. Stop wins ties -- the pessimistic convention."""
    fwd = bars.iloc[i + 1:i + 1 + HORIZON]
    for k, (_, bar) in enumerate(fwd.iterrows(), 1):
        if direction == "bullish":
            hit_stop, hit_tgt = bar["low"] <= stop, bar["high"] >= target
        else:
            hit_stop, hit_tgt = bar["high"] >= stop, bar["low"] <= target
        if hit_stop:
            return "stop_hit", -1.0, k
        if hit_tgt:
            return "target_hit", 1.5, k
    if fwd.empty:
        return "unresolved", 0.0, 0
    last = fwd.iloc[-1]["close"]
    risk = abs(entry - stop)
    r = ((last - entry) if direction == "bullish" else (entry - last)) / risk
    return "timed_out", float(r), len(fwd)


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / (x.std() / math.sqrt(len(x)))) if len(x) > 2 and x.std() else float("nan")


def main() -> None:
    pd.set_option("display.width", 200)
    rng = random.Random(SEED)
    rows = []

    with db.get_connection(load_postgres_config()) as conn:
        for inst in ("NIFTY", "BANKNIFTY"):
            bars = load_bars(conn, inst)
            idx = {d: i for i, d in enumerate(bars["date"])}
            eligible = [d for d in bars["date"][50:-HORIZON - 1]]
            picks = sorted(rng.sample(eligible, min(N_DATES, len(eligible))))
            for j, d in enumerate(picks, 1):
                if j % 100 == 0:
                    print(f"  {inst} {j}/{len(picks)}", flush=True)
                try:
                    sig = signal_generator.generate_signal(conn, inst, d, "positional")
                except Exception:
                    continue
                if sig.direction == "neutral" or sig.entry_price is None:
                    continue
                i = idx[d]
                entry, tgt, stp = sig.entry_price, sig.target_price, sig.stop_loss_price
                risk = abs(entry - stp)

                # Same entry, same risk, same horizon -- only the DIRECTION differs.
                for name, direction in (
                    ("algorithm", sig.direction),
                    ("always long", "bullish"),
                    ("always short", "bearish"),
                    ("coin flip", rng.choice(["bullish", "bearish"])),
                    ("inverted algo", "bearish" if sig.direction == "bullish" else "bullish"),
                ):
                    if direction == "bullish":
                        t2, s2 = entry + 1.5 * risk, entry - risk
                    else:
                        t2, s2 = entry - 1.5 * risk, entry + risk
                    out, r, days = resolve(bars, i, direction, entry, t2, s2)
                    if out == "unresolved":
                        continue
                    rows.append({
                        "variant": name, "instrument": inst, "date": d,
                        "outcome": out, "r": r, "entry_index": i,
                        "resolve_index": i + days,
                    })

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} resolved rows across {df['variant'].nunique()} variants\n")

    print("=" * 100)
    print("EXPECTANCY: algorithm vs direction-blind controls (identical entries/barriers)")
    print("=" * 100)
    print(f"  {'variant':<16} {'n':>5} {'hit%':>6} {'E[R]':>8} {'t':>7} | "
          f"{'early E[R]':>11} {'late E[R]':>10} {'agree':>6}")
    for name, g in df.groupby("variant"):
        nonov = trade_selection.select_non_overlapping(g)
        e = nonov[nonov["date"] < SPLIT]["r"]
        l = nonov[nonov["date"] >= SPLIT]["r"]
        agree = "YES" if len(e) > 5 and len(l) > 5 and (e.mean() > 0) == (l.mean() > 0) else "no"
        hit = (nonov["outcome"] == "target_hit").mean() * 100
        print(f"  {name:<16} {len(nonov):>5} {hit:>6.1f} {nonov['r'].mean():>8.3f} "
              f"{tstat(nonov['r']):>7.2f} | {e.mean():>11.3f} {l.mean():>10.3f} {agree:>6}")

    print("\n  If 'always long' matches or beats 'algorithm', the model added nothing and")
    print("  the positive expectancy was market drift. If 'inverted algo' also scores")
    print("  positive, the direction call is carrying no information at all.")


if __name__ == "__main__":
    main()
