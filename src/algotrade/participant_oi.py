"""Free daily Participant-wise Open Interest report from NSE -- the FII/DII/Retail
positioning breakdown used by signal_factors.oi_positioning_factor.

Endpoint confirmed working directly (found by testing, not documented):
    https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{DDMMYYYY}.csv

One CSV per trading day. Rows are aggregated by Client Type (Client/Retail, DII, FII,
Pro, plus a TOTAL row we discard) across ALL index derivatives combined -- this is
market-wide, not separable per NIFTY-vs-BANKNIFTY. Confirmed available back to at
least 2023-01-03; 404s on weekends/holidays (treated as "no data", not an error).
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import psycopg2.extras
import requests

from . import db
from .config import load_postgres_config
from .logging_utils import get_logger

ARCHIVE_URL_TEMPLATE = "https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{date}.csv"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/csv,*/*",
}
CLIENT_TYPES = ("Client", "DII", "FII", "Pro")

_session = requests.Session()
_session.headers.update(_HEADERS)


def fetch_day(day: date) -> list[dict] | None:
    """Downloads and parses one day's participant-OI report. Returns None (not an
    error) if the file doesn't exist for this date (weekend/holiday/before archive
    start)."""
    url = ARCHIVE_URL_TEMPLATE.format(date=day.strftime("%d%m%Y"))
    response = _session.get(url, timeout=20)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    rows: list[dict] = []
    lines = response.text.splitlines()
    for line in lines[2:]:  # skip the two header lines
        fields = [f.strip() for f in line.split(",")]
        if not fields or fields[0] not in CLIENT_TYPES:
            continue
        try:
            rows.append(
                {
                    "date": day,
                    "client_type": fields[0],
                    "option_index_call_long": int(fields[5]),
                    "option_index_put_long": int(fields[6]),
                    "option_index_call_short": int(fields[7]),
                    "option_index_put_short": int(fields[8]),
                }
            )
        except (ValueError, IndexError):
            continue
    return rows


def upsert_day(conn, rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO participant_oi_daily
            (date, client_type, option_index_call_long, option_index_put_long,
             option_index_call_short, option_index_put_short)
        VALUES
            (%(date)s, %(client_type)s, %(option_index_call_long)s, %(option_index_put_long)s,
             %(option_index_call_short)s, %(option_index_put_short)s)
        ON CONFLICT (date, client_type) DO UPDATE SET
            option_index_call_long = EXCLUDED.option_index_call_long,
            option_index_put_long = EXCLUDED.option_index_put_long,
            option_index_call_short = EXCLUDED.option_index_call_short,
            option_index_put_short = EXCLUDED.option_index_put_short,
            inserted_at = now()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows)
    return len(rows)


def _last_backfilled_date(conn) -> date | None:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM participant_oi_daily")
        (result,) = cur.fetchone()
    return result


def backfill(conn, start: date, end: date, polite_delay_seconds: float = 0.3, logger=None) -> int:
    """Resumable: skips dates already in the DB by starting from the last backfilled
    date + 1 rather than re-fetching everything on re-run."""
    last = _last_backfilled_date(conn)
    if last and last >= start:
        start = last + timedelta(days=1)

    total = 0
    current = start
    while current <= end:
        if current.weekday() < 5:
            rows = fetch_day(current)
            if rows:
                written = upsert_day(conn, rows)
                conn.commit()
                total += written
                if logger:
                    logger.info("participant_oi %s: %d rows", current, written)
            time.sleep(polite_delay_seconds)
        current += timedelta(days=1)
    return total


def run() -> None:
    logger = get_logger("participant_oi_backfill")
    config = load_postgres_config()
    start_date = date(2023, 1, 1)
    end_date = datetime.now().date()
    with db.get_connection(config) as conn:
        written = backfill(conn, start_date, end_date, logger=logger)
    logger.info("participant_oi backfill finished: %d rows written", written)


if __name__ == "__main__":
    run()
