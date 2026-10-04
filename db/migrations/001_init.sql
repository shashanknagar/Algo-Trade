-- Phase 0: capture-and-persist schema.
-- TimescaleDB hypertables for time-series data + pgvector extension staged for later phases
-- (feature/vector store) -- one consolidated Postgres instance per the plan, not four services.
-- Safe to re-run: every statement is idempotent (IF NOT EXISTS / ON CONFLICT).

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS instruments (
    trading_symbol   TEXT NOT NULL,
    exchange         TEXT NOT NULL,
    segment          TEXT NOT NULL,
    exchange_token   TEXT,
    groww_symbol     TEXT,
    name             TEXT,
    instrument_type  TEXT,
    lot_size         INTEGER,
    expiry_date      DATE,
    strike_price     NUMERIC,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (trading_symbol, exchange, segment)
);

-- Daily/EOD/backfilled bars. interval distinguishes granularity ('1day' for Phase 0;
-- finer intervals land here too if a paid intraday vendor is added in Phase 2+).
CREATE TABLE IF NOT EXISTS ohlc_bars (
    time             TIMESTAMPTZ NOT NULL,
    trading_symbol   TEXT NOT NULL,
    exchange         TEXT NOT NULL,
    segment          TEXT NOT NULL,
    interval         TEXT NOT NULL,
    open             NUMERIC,
    high             NUMERIC,
    low              NUMERIC,
    close            NUMERIC,
    volume           BIGINT,
    oi               BIGINT,
    source           TEXT NOT NULL,       -- 'groww_eod' | 'groww_backfill'
    inserted_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (trading_symbol, segment, interval, time)
);
SELECT create_hypertable('ohlc_bars', 'time', if_not_exists => TRUE);

-- Raw intraday ticks from the live WebSocket feed.
CREATE TABLE IF NOT EXISTS ticks (
    time             TIMESTAMPTZ NOT NULL,
    trading_symbol   TEXT NOT NULL,
    exchange         TEXT NOT NULL,
    segment          TEXT NOT NULL,
    ltp              NUMERIC,
    bid              NUMERIC,
    ask              NUMERIC,
    bid_qty          BIGINT,
    ask_qty          BIGINT,
    volume           BIGINT,
    oi               BIGINT,
    inserted_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (trading_symbol, segment, time)
);
SELECT create_hypertable('ticks', 'time', if_not_exists => TRUE);

-- Option-chain snapshots (populated once option-chain capture is added; table created
-- now so the schema doesn't need a breaking migration later).
CREATE TABLE IF NOT EXISTS option_chain_snapshots (
    time              TIMESTAMPTZ NOT NULL,
    underlying        TEXT NOT NULL,
    expiry_date       DATE NOT NULL,
    strike_price      NUMERIC NOT NULL,
    option_type       TEXT NOT NULL,      -- 'CE' | 'PE'
    ltp               NUMERIC,
    oi                BIGINT,
    volume            BIGINT,
    iv                NUMERIC,
    delta             NUMERIC,
    gamma             NUMERIC,
    theta             NUMERIC,
    vega              NUMERIC,
    inserted_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (underlying, expiry_date, strike_price, option_type, time)
);
SELECT create_hypertable('option_chain_snapshots', 'time', if_not_exists => TRUE);

-- Every capture run (intraday / EOD / backfill) logs here so a missed or failed day is
-- visible instead of silent -- this is the durability/observability requirement from the plan.
CREATE TABLE IF NOT EXISTS capture_runs (
    id               BIGSERIAL PRIMARY KEY,
    job_name         TEXT NOT NULL,       -- 'intraday_capture' | 'eod_capture' | 'daily_backfill'
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at         TIMESTAMPTZ,
    status           TEXT NOT NULL DEFAULT 'running',  -- running | success | failed
    rows_written     BIGINT NOT NULL DEFAULT 0,
    gaps_detected    INTEGER NOT NULL DEFAULT 0,
    error_message    TEXT,
    notes            TEXT
);
CREATE INDEX IF NOT EXISTS idx_capture_runs_job_started ON capture_runs (job_name, started_at DESC);

-- Retention policy defaults decided in the plan: raw ticks kept 6 rolling months,
-- option-chain snapshots kept 3 months. Daily bars (ohlc_bars) are kept indefinitely --
-- no retention policy on that table.
SELECT add_retention_policy('ticks', INTERVAL '6 months', if_not_exists => TRUE);
SELECT add_retention_policy('option_chain_snapshots', INTERVAL '3 months', if_not_exists => TRUE);
