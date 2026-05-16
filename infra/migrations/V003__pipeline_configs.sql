-- Pipeline configurations: every readings row is FK'd to the exact config that produced it,
-- so experiments (varying top-N peak count, decoder thresholds, etc.) can be sliced after the fact.

CREATE TABLE IF NOT EXISTS cosmos.pipeline_configs (
  id SERIAL PRIMARY KEY,
  fingerprint TEXT UNIQUE NOT NULL,
  params JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
