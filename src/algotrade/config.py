"""Environment-driven configuration.

Credentials (Groww API key/secret, DB password) come from OS environment variables
only -- never hardcoded, never read from a .env file or any file in this repo. Set
them as Windows system or user environment variables (see README.md); this module
just reads whatever the process's environment already has and fails fast if a
required value is missing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing."""


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"Required environment variable '{name}' is not set. Set it as a Windows "
            "system/user environment variable (System Properties, or `setx NAME value`) "
            "before running any capture job -- see README.md."
        )
    return value


def _build_postgres_dsn(host: str, port: int, db: str, user: str, password: str) -> str:
    return f"host={host} port={port} dbname={db} user={user} password={password}"


@dataclass(frozen=True)
class PostgresConfig:
    """DB-only config -- for code that never touches Groww (signal_generator.py,
    backtest_signal.py, participant_oi.py's read paths) and shouldn't need to fail on
    a missing GROWW_API_KEY it doesn't use."""

    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str

    @property
    def postgres_dsn(self) -> str:
        return _build_postgres_dsn(
            self.postgres_host, self.postgres_port, self.postgres_db,
            self.postgres_user, self.postgres_password,
        )


@dataclass(frozen=True)
class Config(PostgresConfig):
    groww_api_key: str = ""
    groww_api_secret: str = ""


def load_postgres_config() -> PostgresConfig:
    return PostgresConfig(
        postgres_host=os.environ.get("POSTGRES_HOST", "localhost"),
        postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
        postgres_db=os.environ.get("POSTGRES_DB", "algotrade"),
        postgres_user=os.environ.get("POSTGRES_USER", "algotrade"),
        postgres_password=_require("POSTGRES_PASSWORD"),
    )


def load_config() -> Config:
    pg = load_postgres_config()
    return Config(
        postgres_host=pg.postgres_host,
        postgres_port=pg.postgres_port,
        postgres_db=pg.postgres_db,
        postgres_user=pg.postgres_user,
        postgres_password=pg.postgres_password,
        groww_api_key=_require("GROWW_API_KEY"),
        groww_api_secret=_require("GROWW_API_SECRET"),
    )
