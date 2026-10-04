"""Tests every price-action structure on the indices AND on the equity universe.

WHAT IS BEING ASKED OF EACH SIGNAL. A long signal should be followed by returns above
the same-period market average; a short signal by returns below it. So each row is
scored SIGN-AWARE: 'edge' is positive when the signal was right about direction,
whatever that direction was. A short pattern that precedes a rally shows up as a
negative edge, exactly as it should.

TWO DATA SETS, FOR TWO DIFFERENT REASONS.
  * EQUITIES (1,594 liquid names, ~2.2m stock-days) give the statistical POWER to
    say whether a pattern carries information at all. t-stats are clustered by date
    -- stocks move together, so pooled counts overstate independence by 5-10x.
  * INDICES are what you actually trade. Occurrence counts are tiny there, so the
    index results are read as 'consistent with the equity verdict or not', never as
    standalone proof.

The filter is unchanged from every other test in this project: monotone sign, both
halves agree, |t| > 2 on the honest n. Nothing softer has held up.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from algotrade import db, ml_dataset, price_action as pa
from algotrade.config import load_postgres_config
from algotrade.market_calendar import IST

SPLIT = date(2024, 1, 1)
HORIZONS = (5, 10, 20)
MIN_DAYS, MIN_TURNOVER = 500, 5_000_000


def load_equities(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute("SELECT trading_symbol, ohlc_bars.time, open, high, low, close, volume "
                    "FROM ohlc_bars WHERE source='nse_bhavcopy' AND close>0 ORDER BY 1, 2")
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["sym", "time", "open", "high", "low", "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["date"] = df["time"].apply(lambda t: t.astimezone(IST).date())
    df = df.drop(columns="time").dropna()
    st = df.assign(to=df["close"] * df["volume"]).groupby("sym").agg(n=("close", "size"), to=("to", "median"))
    keep = st[(st["n"] >= MIN_DAYS) & (st["to"] >= MIN_TURNOVER)].index
    return df[df["sym"].isin(keep)].reset_index(drop=True)


def with_signals_and_forward(df: pd.DataFrame) -> pd.DataFrame:
    d = pa.build_signals(df.sort_values("date").reset_index(drop=True))
    for h in HORIZONS:
        d[f"fwd{h}"] = d["close"].shift(-h) / d["close"] - 1
    return d


def clustered(sub: pd.DataFrame, all_: pd.DataFrame, h: str, sign: int) -> dict:
    """Sign-aware excess return vs the same-day cross-sectional (or same-period) mean,
    t-stat computed across DATES, not across stock-days."""
    base = all_.groupby("date")[h].mean()
    daily = sub.groupby("date")[h].mean()
    exc = (daily - base.reindex(daily.index)).dropna() * sign
    if len(exc) < 15:
        return {"n_obs": len(sub), "n_days": len(exc), "edge_bps": np.nan, "t": np.nan,
                "early": np.nan, "late": np.nan, "agree": "-"}
    e, l = exc[exc.index < SPLIT], exc[exc.index >= SPLIT]
    t = exc.mean() / (exc.std() / np.sqrt(len(exc))) if exc.std() else np.nan
    agree = "YES" if len(e) >= 10 and len(l) >= 10 and (e.mean() > 0) == (l.mean() > 0) else "no"
    return {"n_obs": len(sub), "n_days": len(exc), "edge_bps": round(exc.mean() * 1e4, 1),
            "t": round(t, 2), "early": round(e.mean() * 1e4, 1) if len(e) else np.nan,
            "late": round(l.mean() * 1e4, 1) if len(l) else np.nan, "agree": agree}


def report(all_: pd.DataFrame, title: str) -> None:
    for h in HORIZONS:
        col = f"fwd{h}"
        print("=" * 120)
        print(f"{title}  --  forward {h}-day, sign-aware edge vs same-period mean (bps)")
        print("=" * 120)
        rows = []
        for sig in pa.LONG_SIGNALS + pa.SHORT_SIGNALS:
            sign = 1 if sig in pa.LONG_SIGNALS else -1
            sub = all_[all_[sig].fillna(False) & all_[col].notna()]
            r = clustered(sub, all_[all_[col].notna()], col, sign)
            verdict = ("PASS" if r["agree"] == "YES" and abs(r["t"] or 0) > 2 and (r["edge_bps"] or 0) > 0
                       else "WRONG WAY" if (r["edge_bps"] or 0) < 0 and abs(r["t"] or 0) > 2 and r["agree"] == "YES"
                       else "")
            rows.append({"signal": sig, "side": "L" if sign == 1 else "S", **r, "verdict": verdict})
        print(pd.DataFrame(rows).to_string(index=False))
        print()


def main() -> None:
    pd.set_option("display.width", 240)
    with db.get_connection(load_postgres_config()) as conn:
        idx = {i: with_signals_and_forward(ml_dataset._load_full_ohlc(conn, i)) for i in ("NIFTY", "BANKNIFTY")}
        print("Loading equities...")
        eq = load_equities(conn)
    print(f"  {len(eq):,} rows, {eq['sym'].nunique():,} symbols. Building structures per symbol...")
    eq = eq.groupby("sym", group_keys=False).apply(with_signals_and_forward)
    print(f"  done: {len(eq):,} stock-days\n")

    counts = {s: int(eq[s].sum()) for s in pa.LONG_SIGNALS + pa.SHORT_SIGNALS}
    print("Occurrences in the equity universe:")
    for k, v in counts.items():
        print(f"  {k:<28} {v:>8,}")
    print()

    report(eq, "EQUITY UNIVERSE (the powered test)")

    # Tradeability: an edge confined to illiquid names is a spread illusion. Stock
    # futures exist only for the most liquid ~180 names, so the top tercile is the
    # only place a SHORT signal could actually be executed beyond intraday.
    med = eq.groupby("sym").apply(lambda g: (g["close"] * g["volume"]).median())
    eq["liq"] = eq["sym"].map(pd.qcut(med, 3, labels=["low", "mid", "high"]))
    print("=" * 120)
    print("LIQUIDITY TERCILES -- short-side signals, forward 10-day (bps, t)")
    print("=" * 120)
    print(f"  {'signal':<26} | " + " | ".join(f"{x:>20}" for x in ("low liq", "mid liq", "high liq")))
    for sig in pa.SHORT_SIGNALS:
        cells = []
        for liq in ("low", "mid", "high"):
            u = eq[(eq["liq"] == liq) & eq["fwd10"].notna()]
            r = clustered(u[u[sig].fillna(False)], u, "fwd10", -1)
            cells.append(f"{r['edge_bps']:>10} (t{r['t']:>6})" if not np.isnan(r["t"]) else f"{'too few':>20}")
        print(f"  {sig:<26} | " + " | ".join(cells))
    print()
    both = pd.concat([d.assign(sym=i) for i, d in idx.items()], ignore_index=True)
    report(both, "NIFTY + BANKNIFTY (what you trade; low power)")

    print("=" * 120)
    print("PASS = right direction, replicates across halves, |t|>2 on date-clustered n.")
    print("WRONG WAY = the same bar, but the market reliably went the OTHER direction.")
    print("Blank = noise at this sample size.")
    print("=" * 120)


if __name__ == "__main__":
    main()
