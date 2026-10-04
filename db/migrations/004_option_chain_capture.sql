-- Pro-plan option chain capture: extend the Phase 0 snapshot table with the fields
-- a paid subscription actually provides. Safe to re-run.

ALTER TABLE option_chain_snapshots ADD COLUMN IF NOT EXISTS bid NUMERIC;
ALTER TABLE option_chain_snapshots ADD COLUMN IF NOT EXISTS ask NUMERIC;
ALTER TABLE option_chain_snapshots ADD COLUMN IF NOT EXISTS oi_change BIGINT;
ALTER TABLE option_chain_snapshots ADD COLUMN IF NOT EXISTS underlying_spot NUMERIC;
ALTER TABLE option_chain_snapshots ADD COLUMN IF NOT EXISTS rho NUMERIC;
-- Raw payload of the first snapshot per run, so the response shape can be verified
-- against the normalizer rather than assumed (the SDK's docs have been wrong before).
ALTER TABLE option_chain_snapshots ADD COLUMN IF NOT EXISTS raw JSONB;

CREATE INDEX IF NOT EXISTS idx_option_chain_underlying_time
    ON option_chain_snapshots (underlying, time DESC);

-- Paper-trading ledger: every recommendation this system produces, and what actually
-- happened to it. Deliberately separate from trade_signals (which holds the
-- historical/backtest signals) so live paper recommendations are never confused with
-- backtested ones. NOTHING here is ever sent to a broker -- see no_trade_guard.py.
CREATE TABLE IF NOT EXISTS paper_trades (
    id                BIGSERIAL PRIMARY KEY,
    generated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    underlying        TEXT NOT NULL,
    recommendation    TEXT NOT NULL,       -- human-readable, e.g. 'Buy 24500 CE'
    option_type       TEXT,                -- 'CE' | 'PE' | NULL for non-option signals
    strike_price      NUMERIC,
    expiry_date       DATE,
    direction         TEXT,                -- 'bullish' | 'bearish' | 'neutral'
    entry_price       NUMERIC,             -- option premium at signal time
    underlying_spot   NUMERIC,
    target_price      NUMERIC,
    stop_loss_price   NUMERIC,
    confidence_score  NUMERIC,
    rationale         JSONB,               -- factor breakdown, for after-the-fact review
    model_version     TEXT NOT NULL DEFAULT 'paper_v1',
    status            TEXT NOT NULL DEFAULT 'open',   -- open|target_hit|stop_hit|timed_out|cancelled
    resolved_at       TIMESTAMPTZ,
    resolved_price    NUMERIC,
    r_multiple        NUMERIC,
    notes             TEXT
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades (status, generated_at DESC);
