"""Learns an empirical hit-rate table keyed by market state, and uses it to gate
future trades -- the "learn from historical predictions with similar setups" loop.

The central safeguard here is SHRINKAGE. A raw per-state hit rate is wildly
unreliable when that state has only been seen a handful of times: 2 wins out of 3
occurrences is not a 67% edge, it's noise. So every state's rate is shrunk toward the
overall base rate using additive (empirical-Bayes / Laplace-style) smoothing:

    shrunk = (hits + alpha * base_rate) / (n + alpha)

With alpha=20, a state seen 3 times barely moves off the base rate, while one seen
300 times keeps almost all of its own signal. This is what stops the state table from
becoming an elaborate way to memorize noise -- the exact failure mode that made
gradient boosting's round-2 result unreproducible.

Discipline, same as ml_model.py: the table is built from the TRAIN period only and
applied unchanged to the held-out TEST period.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_ALPHA = 20.0


def build_state_table(train_df: pd.DataFrame, alpha: float = DEFAULT_ALPHA) -> tuple[pd.DataFrame, float]:
    """Returns (state_table, base_rate). state_table columns: state, n, hits,
    raw_hit_rate, shrunk_hit_rate, avg_r."""
    base_rate = float(train_df["target_hit"].mean())

    grouped = train_df.groupby("state").agg(
        n=("target_hit", "size"),
        hits=("target_hit", "sum"),
        avg_r=("r_multiple", "mean"),
    ).reset_index()

    grouped["raw_hit_rate"] = grouped["hits"] / grouped["n"]
    grouped["shrunk_hit_rate"] = (grouped["hits"] + alpha * base_rate) / (grouped["n"] + alpha)
    grouped = grouped.sort_values("shrunk_hit_rate", ascending=False).reset_index(drop=True)
    return grouped, base_rate


def apply_state_table(test_df: pd.DataFrame, state_table: pd.DataFrame, base_rate: float) -> pd.DataFrame:
    """Attaches each test row's learned (train-period) shrunk hit rate. States never
    seen during training fall back to the base rate -- not to an optimistic guess."""
    lookup = dict(zip(state_table["state"], state_table["shrunk_hit_rate"]))
    counts = dict(zip(state_table["state"], state_table["n"]))
    out = test_df.copy()
    out["state_expected_hit_rate"] = out["state"].map(lookup).fillna(base_rate)
    out["state_train_n"] = out["state"].map(counts).fillna(0).astype(int)
    out["state_seen_in_train"] = out["state"].isin(lookup)
    return out


def evaluate_state_gating(
    scored_test: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55),
    min_train_n: int = 0,
) -> pd.DataFrame:
    """Take a trade only when its state's learned hit rate clears the threshold (and,
    optionally, only when that state had at least min_train_n training examples)."""
    rows = []
    for t in thresholds:
        subset = scored_test[
            (scored_test["state_expected_hit_rate"] >= t)
            & (scored_test["state_train_n"] >= min_train_n)
        ]
        n = len(subset)
        if n == 0:
            rows.append({"threshold": t, "n_trades": 0, "hit_rate": None, "expectancy_r": None})
            continue
        rows.append(
            {
                "threshold": t,
                "n_trades": n,
                "hit_rate": round(float(subset["target_hit"].mean()), 3),
                "expectancy_r": round(float(subset["r_multiple"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


def persist_state_table(conn, state_table: pd.DataFrame, dimension_set: str, base_rate: float, alpha: float) -> int:
    """Stores the learned table so it's queryable and so live signal generation can
    read the same rates the backtest validated -- the feedback loop's memory."""
    import psycopg2.extras

    rows = [
        {
            "state": r.state,
            "dimension_set": dimension_set,
            "n": int(r.n),
            "hits": int(r.hits),
            "raw_hit_rate": float(r.raw_hit_rate),
            "shrunk_hit_rate": float(r.shrunk_hit_rate),
            "avg_r": float(r.avg_r),
            "base_rate": base_rate,
            "alpha": alpha,
        }
        for r in state_table.itertuples(index=False)
    ]
    sql = """
        INSERT INTO state_hit_rates
            (state, dimension_set, n, hits, raw_hit_rate, shrunk_hit_rate, avg_r, base_rate, alpha)
        VALUES
            (%(state)s, %(dimension_set)s, %(n)s, %(hits)s, %(raw_hit_rate)s,
             %(shrunk_hit_rate)s, %(avg_r)s, %(base_rate)s, %(alpha)s)
        ON CONFLICT (state, dimension_set) DO UPDATE SET
            n = EXCLUDED.n, hits = EXCLUDED.hits, raw_hit_rate = EXCLUDED.raw_hit_rate,
            shrunk_hit_rate = EXCLUDED.shrunk_hit_rate, avg_r = EXCLUDED.avg_r,
            base_rate = EXCLUDED.base_rate, alpha = EXCLUDED.alpha, updated_at = now()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows)
    conn.commit()
    return len(rows)
