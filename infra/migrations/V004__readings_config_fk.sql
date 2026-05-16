-- Tag every reading with the pipeline config that produced it, and the raw hex it was decoded from.
-- raw_hex enables replaying decoding over history if decoder behaviour changes.

ALTER TABLE cosmos.readings
  ADD COLUMN IF NOT EXISTS pipeline_config_id INTEGER REFERENCES cosmos.pipeline_configs(id),
  ADD COLUMN IF NOT EXISTS raw_hex TEXT;

CREATE INDEX IF NOT EXISTS idx_readings_pipeline_config
  ON cosmos.readings (pipeline_config_id);
