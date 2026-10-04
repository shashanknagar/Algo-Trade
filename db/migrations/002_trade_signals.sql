-- Phase 1a: rule-based signal generation + backtesting.
-- Safe to re-run: every statement is idempotent (IF NOT EXISTS / ON CONFLICT).

-- One row per (date, client_type) from NSE's free daily Participant-wise Open
-- Interest report. Market-wide across all index derivatives combined -- NOT
-- separable per NIFTY-vs-BANKNIFTY, a real limitation noted in signal_factors.py.
CREATE TABLE IF NOT EXISTS participant_oi_daily (
    date                      DATE NOT NULL,
    client_type               TEXT NOT NULL,   -- 'Client' | 'DII' | 'FII' | 'Pro'
    option_index_call_long    BIGINT,
    option_index_put_long     BIGINT,
    option_index_call_short   BIGINT,
    option_index_put_short    BIGINT,
    inserted_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (date, client_type)
);

-- Generated signals (rule-based v1 -- see signal_generator.py) and their eventual
-- resolution via triple-barrier tracking. as_of_date is what the signal was
-- generated against (strictly no lookahead past this date); generated_at is when the
-- row was written, which differ for backtested signals run long after the fact.
CREATE TABLE IF NOT EXISTS trade_signals (
    id                BIGSERIAL PRIMARY KEY,
    generated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    as_of_date        DATE NOT NULL,
    instrument        TEXT NOT NULL,
    signal_type       TEXT NOT NULL,        -- 'positional' | 'intraday'
    direction         TEXT NOT NULL,        -- 'bullish' | 'bearish' | 'neutral'
    entry_price       NUMERIC,
    target_price      NUMERIC,
    stop_loss_price   NUMERIC,
    confidence_score  NUMERIC,              -- 0-100
    factor_breakdown  JSONB,                -- per-factor score + explanation, for explainability
    model_version     TEXT NOT NULL DEFAULT 'rule_v1',
    status            TEXT NOT NULL DEFAULT 'open',  -- open | target_hit | stop_hit | timed_out | no_trade
    resolved_at       TIMESTAMPTZ,
    resolved_date     DATE,
    resolved_price    NUMERIC,
    r_multiple        NUMERIC,
    notes             TEXT
);
CREATE INDEX IF NOT EXISTS idx_trade_signals_instrument_date
    ON trade_signals (instrument, as_of_date DESC);
