-- Create base schema and metadata table
CREATE SCHEMA IF NOT EXISTS cosmos;

-- Planes table: each row is a unique aircraft (ICAO24)
CREATE TABLE IF NOT EXISTS cosmos.planes (
  icao24 TEXT PRIMARY KEY,
  callsign TEXT,
  registration TEXT,
  type TEXT,
  operator TEXT,
  first_seen TIMESTAMP WITH TIME ZONE,
  last_seen TIMESTAMP WITH TIME ZONE,
  last_lat DOUBLE PRECISION,
  last_lon DOUBLE PRECISION,
  last_altitude INTEGER,
  last_velocity DOUBLE PRECISION,
  last_message JSONB DEFAULT '{}'  -- optional store of last raw message
);

-- Readings: partitioned table (top-level)
CREATE TABLE IF NOT EXISTS cosmos.readings (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  icao24 TEXT NOT NULL,
  callsign TEXT,
  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,
  altitude INTEGER,
  velocity DOUBLE PRECISION,
  heading DOUBLE PRECISION,
  squawk TEXT,
  raw JSONB,
  CONSTRAINT fk_plane_icao FOREIGN KEY (icao24) REFERENCES cosmos.planes (icao24) ON DELETE SET NULL
);
