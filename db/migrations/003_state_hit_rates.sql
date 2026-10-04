-- Phase 1b: empirical per-market-state hit-rate learning (state_learning.py).
-- Safe to re-run.

-- The learned "what happened last time the market looked like this" table.
-- One row per (market state signature, dimension set). Rebuilt from resolved
-- outcomes; shrunk_hit_rate is the base-rate-shrunk estimate actually used for
-- gating (see state_learning.py for why the raw rate alone is not trustworthy).
CREATE TABLE IF NOT EXISTS state_hit_rates (
    state             TEXT NOT NULL,
    dimension_set     TEXT NOT NULL,      -- 'coarse' | 'fine' (which dims form the key)
    n                 INTEGER NOT NULL,   -- historical occurrences of this state
    hits              INTEGER NOT NULL,   -- of which, target was hit
    raw_hit_rate      NUMERIC,            -- hits / n -- unreliable when n is small
    shrunk_hit_rate   NUMERIC,            -- base-rate-shrunk estimate -- use this one
    avg_r             NUMERIC,            -- average R-multiple in this state
    base_rate         NUMERIC,            -- overall training-period hit rate
    alpha             NUMERIC,            -- shrinkage strength used
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (state, dimension_set)
);

-- Every historical signal gets its state signature recorded alongside its outcome,
-- so the feedback loop can rebuild the table from trade_signals directly.
ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS market_state TEXT;
ALTER TABLE trade_signals ADD COLUMN IF NOT EXISTS state_dimension_set TEXT;
CREATE INDEX IF NOT EXISTS idx_trade_signals_state ON trade_signals (market_state);

-- One signal per (date, instrument, type, model version) -- lets bulk backfills of
-- historical predictions upsert instead of duplicating on every re-run.
CREATE UNIQUE INDEX IF NOT EXISTS uq_trade_signals_natural_key
    ON trade_signals (as_of_date, instrument, signal_type, model_version);
