"""Postgres connection + idempotent upsert helpers.

Every write here is an upsert keyed on the row's natural identity (instrument + time),
so a re-run after a crash or restart fills gaps rather than duplicating or dropping
rows -- this is the durability requirement from the plan: captured market data is
irreplaceable once Groww's 3-month intraday window rolls past it.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.extras

from .config import Config


@contextmanager
def get_connection(config: Config) -> Iterator["psycopg2.extensions.connection"]:
    conn = psycopg2.connect(config.postgres_dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def upsert_ohlc_bars(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO ohlc_bars
            (time, trading_symbol, exchange, segment, interval, open, high, low, close,
             volume, oi, source)
        VALUES
            (%(time)s, %(trading_symbol)s, %(exchange)s, %(segment)s, %(interval)s,
             %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(oi)s, %(source)s)
        ON CONFLICT (trading_symbol, segment, interval, time) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
            close = EXCLUDED.close, volume = EXCLUDED.volume, oi = EXCLUDED.oi,
            source = EXCLUDED.source, inserted_at = now()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows)
    return len(rows)


def upsert_ticks(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO ticks
            (time, trading_symbol, exchange, segment, ltp, bid, ask, bid_qty, ask_qty,
             volume, oi)
        VALUES
            (%(time)s, %(trading_symbol)s, %(exchange)s, %(segment)s, %(ltp)s, %(bid)s,
             %(ask)s, %(bid_qty)s, %(ask_qty)s, %(volume)s, %(oi)s)
        ON CONFLICT (trading_symbol, segment, time) DO UPDATE SET
            ltp = EXCLUDED.ltp, bid = EXCLUDED.bid, ask = EXCLUDED.ask,
            bid_qty = EXCLUDED.bid_qty, ask_qty = EXCLUDED.ask_qty,
            volume = EXCLUDED.volume, oi = EXCLUDED.oi, inserted_at = now()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows)
    return len(rows)


def start_capture_run(conn, job_name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO capture_runs (job_name, started_at, status) "
            "VALUES (%s, now(), 'running') RETURNING id",
            (job_name,),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def finish_capture_run(
    conn,
    run_id: int,
    status: str,
    rows_written: int,
    gaps_detected: int = 0,
    error_message: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE capture_runs
            SET ended_at = now(), status = %s, rows_written = %s,
                gaps_detected = %s, error_message = %s
            WHERE id = %s
            """,
            (status, rows_written, gaps_detected, error_message, run_id),
        )
    conn.commit()
