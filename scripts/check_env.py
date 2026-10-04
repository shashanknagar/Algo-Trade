"""Sanity check: are the required environment variables set, can we reach Postgres, and
does Groww authentication actually work? Run this after setting env vars and starting
docker compose, before registering the Task Scheduler jobs.

Usage (after `pip install -e .` from the project root):
    python scripts/check_env.py
"""
from __future__ import annotations

import sys

from algotrade.config import ConfigError, load_config


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"FAIL: {exc}")
        return 1
    print("Environment variables OK.")

    import psycopg2

    try:
        conn = psycopg2.connect(config.postgres_dsn)
        conn.close()
    except Exception as exc:
        print(f"FAIL: could not connect to Postgres: {exc}")
        return 1
    print("Postgres connection OK.")

    try:
        from algotrade.groww_client import get_access_token

        token = get_access_token(config)
        print(f"Groww auth OK (access token acquired, length={len(token)}).")
    except Exception as exc:
        print(f"FAIL: Groww authentication failed: {exc}")
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
