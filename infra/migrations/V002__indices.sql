-- Create indexes to optimize common queries on the readings table

-- Indexes: timestamp index (for "last X minutes" searches), and per-aircraft index
CREATE INDEX IF NOT EXISTS idx_readings_ts ON cosmos.readings (ts DESC);
CREATE INDEX IF NOT EXISTS idx_readings_icao_ts ON cosmos.readings (icao24, ts DESC);

-- Index for looking up all readings for a specific aircraft
CREATE INDEX IF NOT EXISTS idx_readings_icao ON cosmos.readings (icao24);
