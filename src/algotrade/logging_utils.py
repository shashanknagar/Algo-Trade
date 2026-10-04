"""Structured run logging: stdout + a rotating log file + a capture_runs row per run,
so a missed or failed capture day is visible rather than silent.
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import db

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"


def get_logger(job_name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(job_name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        file_handler = logging.FileHandler(LOG_DIR / f"{job_name}.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
    return logger


@contextmanager
def capture_run(conn, job_name: str, logger: logging.Logger) -> Iterator[dict]:
    """Wraps a short-lived job (EOD/backfill) end-to-end: opens a capture_runs row,
    yields a stats dict the caller fills in (rows_written, gaps_detected), and closes
    the row as success or failed depending on whether the block raised.

    Not used by the intraday job, which runs for hours across many short-lived DB
    connections -- see capture_intraday.py for that job's own run tracking.
    """
    run_id = db.start_capture_run(conn, job_name)
    stats = {"rows_written": 0, "gaps_detected": 0}
    logger.info("Run %s started (id=%s)", job_name, run_id)
    try:
        yield stats
    except Exception as exc:
        logger.exception("Run %s failed", job_name)
        db.finish_capture_run(
            conn, run_id, "failed", stats["rows_written"], stats["gaps_detected"], str(exc)
        )
        raise
    else:
        db.finish_capture_run(
            conn, run_id, "success", stats["rows_written"], stats["gaps_detected"]
        )
        logger.info(
            "Run %s finished: rows_written=%s gaps_detected=%s",
            job_name, stats["rows_written"], stats["gaps_detected"],
        )
