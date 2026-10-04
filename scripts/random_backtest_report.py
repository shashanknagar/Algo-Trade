"""Random-sample evaluation of the CURRENT rule-based algorithm on stored data.

You asked to test the algo on random historical points. Five samples is an anecdote,
so this draws a large random sample and reports it against the only benchmark that
means anything here.

THE BENCHMARK MOVES WITH THE BARRIERS. For a driftless random walk the chance of
hitting target before stop is 1/(1+RR) where RR is the reward:risk ratio -- 40% at
the default 1:1.5, but 37.5% at a 50/30-point setting. Beating that number is the
bar, and quoting a hit rate without it is meaningless.

TWO NUMBERS ARE REPORTED AND THEY DIFFER FOR A REASON:
  * RAW: every sampled date. Inflated confidence -- 20-day horizons sampled from
    random dates overlap heavily, so nearby signals share almost all their forward
    window and are nowhere near independent observations.
  * NON-OVERLAPPING: one position at a time per instrument, which is what could
    actually have been traded. This is the honest n, and it is much smaller.

Wilson intervals are used rather than normal-approximation ones because they stay
correct at the small n this exercise inevitably produces.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from algotrade import backtest_signal, db, signal_generator, trade_selection, triple_barrier
from algotrade.config import load_postgres_config

SPLIT = date(2024, 1, 1)
BREAKEVEN = 1 / (1 + 1.5)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half) * 100, (centre + half) * 100


def invert(signal, barriers=None):
    """Flip the direction and rebuild the barriers around the same entry.

    Target and stop are direction-dependent, so flipping the label alone would leave
    a bearish call with a target ABOVE entry. Risk distance is preserved so the
    inverted trade carries identical geometry -- only the side changes. Neutral stays
    neutral: inversion must not manufacture trades the algorithm never took, or the
    two runs would not be comparable.
    """
    if signal.direction == "neutral" or signal.entry_price is None:
        return signal
    entry = signal.entry_price
    flipped = "bearish" if signal.direction == "bullish" else "bullish"
    signal.direction = flipped
    if barriers is not None:
        up, down = barriers.target_points, barriers.stop_points
    else:
        down = abs(entry - signal.stop_loss_price)
        up = down * signal_generator.RISK_REWARD_RATIO
    if flipped == "bullish":
        signal.target_price, signal.stop_loss_price = entry + up, entry - down
    else:
        signal.target_price, signal.stop_loss_price = entry - up, entry + down
    return signal


def ambiguous_fraction(sig, future) -> float | None:
    """Fraction of forward bars where BOTH barriers sit inside the day's range.

    THIS IS THE KEY DIAGNOSTIC FOR TIGHT BARRIERS. A daily bar records the high and
    the low but never their order, so when both barriers fall inside one bar we cannot
    know which was touched first -- the stop-wins-ties convention decides it by fiat.
    With ATR-scaled barriers that is a rare edge case. With barriers narrower than a
    typical daily range it becomes the common case, and the measured hit rate stops
    describing the signal and starts describing the tie-break rule.
    """
    if future.empty or sig.target_price is None:
        return None
    lo = future["low"].astype(float)
    hi = future["high"].astype(float)
    upper, lower = max(sig.target_price, sig.stop_loss_price), min(sig.target_price, sig.stop_loss_price)
    both = (hi >= upper) & (lo <= lower)
    return float(both.mean())


def backtest_one(conn, instrument, as_of_date, signal_type, inverted: bool, barriers=None,
                 tie_breaker: str = 'stop'):
    sig = signal_generator.generate_signal(conn, instrument, as_of_date, signal_type,
                                           barriers=barriers)
    if inverted:
        sig = invert(sig, barriers)
    horizon = signal_generator.HORIZON_TRADING_DAYS[signal_type]
    if sig.direction == "neutral" or sig.entry_price is None:
        return backtest_signal.BacktestResult(sig, "no_trade", None, None, None, None), None

    future = backtest_signal._load_ohlc_after(conn, instrument, as_of_date, horizon)
    rr = barriers.risk_reward if barriers else signal_generator.RISK_REWARD_RATIO
    outcome = triple_barrier.resolve(sig.direction, sig.entry_price, sig.target_price,
                                     sig.stop_loss_price, future, rr, tie_breaker)
    res = backtest_signal.BacktestResult(sig, outcome.status, outcome.resolved_date,
                                         outcome.resolved_price, outcome.r_multiple,
                                         outcome.bars_to_resolve)
    # Ambiguity only matters on the bar that actually decided the trade.
    decided_ambiguous = None
    if outcome.bars_to_resolve and outcome.status in ('target_hit', 'stop_hit'):
        bar = future.iloc[outcome.bars_to_resolve - 1]
        upper = max(sig.target_price, sig.stop_loss_price)
        lower = min(sig.target_price, sig.stop_loss_price)
        decided_ambiguous = float(bar['high']) >= upper and float(bar['low']) <= lower
    return res, decided_ambiguous


def run(conn, instrument: str, n: int, seed: int, signal_type: str,
        inverted: bool = False, barriers=None, tie_breaker: str = 'stop') -> pd.DataFrame:
    rng = random.Random(seed)
    span = backtest_signal._get_valid_date_range(conn, instrument, signal_type)
    if span is None:
        return pd.DataFrame()
    start, end = span

    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT time::date FROM ohlc_bars WHERE trading_symbol = %s "
            "AND interval = '1day' AND time::date BETWEEN %s AND %s ORDER BY 1",
            (instrument, start, end),
        )
        all_dates = [r[0] for r in cur.fetchall()]
    if not all_dates:
        return pd.DataFrame()

    index_of = {d: i for i, d in enumerate(all_dates)}
    picks = rng.sample(all_dates, min(n, len(all_dates)))

    rows = []
    for i, d in enumerate(sorted(picks), 1):
        if i % 100 == 0:
            print(f"    {instrument}: {i}/{len(picks)}", flush=True)
        try:
            res, ambig = backtest_one(conn, instrument, d, signal_type, inverted,
                                      barriers, tie_breaker)
        except Exception as exc:                      # a bad date must not kill the run
            print(f"    skipped {d}: {exc}")
            continue
        rows.append({
            "instrument": instrument,
            "date": d,
            "direction": res.signal.direction,
            "confidence": res.signal.confidence_score,
            "entry": res.signal.entry_price,
            "outcome": res.outcome,
            "r_multiple": res.r_multiple,
            "days": res.days_to_resolve,
            "ambiguous_frac": ambig,
            "entry_index": index_of[d],
            # A no-trade frees the book immediately; a live trade blocks it until resolved.
            "resolve_index": index_of[d] + (res.days_to_resolve or 0),
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> None:
    traded = df[df["outcome"].isin(["target_hit", "stop_hit", "timed_out"])]
    if traded.empty:
        print(f"  {label}: no resolved trades")
        return
    hits = int((traded["outcome"] == "target_hit").sum())
    n = len(traded)
    lo, hi = wilson(hits, n)
    rate = hits / n * 100
    exp = traded["r_multiple"].mean()
    edge = rate - BREAKEVEN * 100
    verdict = ("beats" if lo > BREAKEVEN * 100
               else ("below" if hi < BREAKEVEN * 100 else "INDISTINGUISHABLE from"))
    print(f"  {label:<26} n={n:>4}  hit={rate:>5.1f}%  95% CI [{lo:>5.1f},{hi:>5.1f}]  "
          f"edge={edge:+5.1f}pt  E[R]={exp:+.3f}   -> {verdict} baseline")


def main() -> None:
    global BREAKEVEN
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=400, help="random dates per instrument")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--type", dest="signal_type", default="positional")
    ap.add_argument("--invert", action="store_true",
                    help="flip every bullish call to bearish and vice versa")
    ap.add_argument("--target-points", type=float, default=None,
                    help="fixed target distance in index points (e.g. 50)")
    ap.add_argument("--stop-points", type=float, default=None,
                    help="fixed stop distance in index points (e.g. 30)")
    ap.add_argument("--tie-breaker", choices=["stop", "target"], default="stop",
                    help="who wins when both barriers fall inside one bar")
    args = ap.parse_args()

    pd.set_option("display.width", 200)
    barriers = None
    if (args.target_points is None) != (args.stop_points is None):
        ap.error("--target-points and --stop-points must be given together")
    if args.target_points is not None:
        barriers = signal_generator.FixedBarriers(args.target_points, args.stop_points)
        BREAKEVEN = 1 / (1 + barriers.risk_reward)

    with db.get_connection(load_postgres_config()) as conn:
        frames = [run(conn, inst, args.n, args.seed, args.signal_type, args.invert,
                      barriers, args.tie_breaker)
                  for inst in ("NIFTY", "BANKNIFTY")]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    mode = "INVERTED (bullish<->bearish)" if args.invert else "as-built"
    if barriers:
        mode += (f" | FIXED +{barriers.target_points:g}/-{barriers.stop_points:g} pts "
                 f"(R:R {barriers.risk_reward:.2f}, breakeven {BREAKEVEN*100:.1f}%)")
    mode += f" | ties -> {args.tie_breaker}"
    print(f"\nSampled {len(df)} random dates ({args.signal_type}, seed {args.seed})")
    print(f"Mode: {mode}\n")

    print("=" * 104)
    print("OUTCOME MIX")
    print("=" * 104)
    print(df["outcome"].value_counts().to_string())
    print("\ndirection: " + ", ".join(f"{k}={v}" for k, v in df["direction"].value_counts().items()))
    print(f"the algorithm declines to trade on {(df['outcome'] == 'no_trade').mean()*100:.0f}% of random days")

    resolved = df[df["outcome"].isin(["target_hit", "stop_hit"])]
    if not resolved.empty:
        print(f"median days to resolve: {resolved['days'].median():.0f}")

    if barriers is not None and not df.empty:
        # What the fixed distance actually means as a percentage varies hugely; show it.
        pct = (barriers.target_points / df["entry"].dropna() * 100)
        print(f"\n{barriers.target_points:g} points as % of index: "
              f"min {pct.min():.2f}%, median {pct.median():.2f}%, max {pct.max():.2f}% "
              f"-- a fixed point target is NOT a fixed difficulty")

    decided = df[df["ambiguous_frac"].notna()]
    if not decided.empty:
        amb = decided["ambiguous_frac"].mean() * 100
        print(f"\nTIE-DECIDED TRADES: {amb:.1f}% of resolved trades were settled on a bar where")
        print("BOTH barriers sat inside the day's range. A daily bar records the high and the")
        print("low but not their ORDER, so those outcomes were ASSIGNED by the tie-break rule,")
        print("not measured. Past ~30% the headline hit rate describes the convention as much")
        print("as the signal -- rerun with --tie-breaker target to see the other bound.")

    print("\n" + "=" * 104)
    print(f"HIT RATE vs THE {BREAKEVEN*100:.1f}% RANDOM-WALK BASELINE")
    print("=" * 104)
    report(df, "RAW (overlapping)")

    live = df[df["outcome"] != "no_trade"].copy()
    nonov = trade_selection.select_non_overlapping(live)
    report(nonov, "NON-OVERLAPPING (honest)")

    print("\n  Split by period -- an edge present in only one half is regime, not skill:")
    for name, sub in (("  early (<2024)", nonov[nonov["date"] < SPLIT]),
                      ("  late  (>=2024)", nonov[nonov["date"] >= SPLIT])):
        report(sub, name)

    print("\n  By instrument:")
    for inst, sub in nonov.groupby("instrument"):
        report(sub, f"  {inst}")

    if len(nonov) >= 30:
        print("\n  By confidence tercile (does the score rank anything?):")
        nonov = nonov.copy()
        nonov["band"] = pd.qcut(nonov["confidence"], 3, labels=["low", "mid", "high"],
                                duplicates="drop")
        for band, sub in nonov.groupby("band", observed=True):
            report(sub, f"  confidence {band}")

    print("\n" + "=" * 104)
    print("Read the NON-OVERLAPPING line, not the raw one. If its CI spans the baseline,")
    print("the sample cannot tell an edge from luck -- this project's binding constraint")
    print("throughout, and no amount of resampling fixes it.")
    print("=" * 104)


if __name__ == "__main__":
    main()
