"""Applies SQL files under db/migrations, in filename order. Safe to re-run -- every
migration is written with IF NOT EXISTS / ON CONFLICT so re-applying is a no-op.

Usage:
    python scripts/apply_migrations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psycopg2

from algotrade.config import load_postgres_config

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "db" / "migrations"


def main() -> None:
    config = load_postgres_config()
    conn = psycopg2.connect(config.postgres_dsn)
    try:
        with conn.cursor() as cur:
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                print(f"Applying {path.name} ...")
                cur.execute(path.read_text(encoding="utf-8"))
        conn.commit()
        print("All migrations applied.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
