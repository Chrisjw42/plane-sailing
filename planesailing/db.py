"""Postgres write path for the ADSB pipeline.

DBWriter holds a long-lived connection, ensures the pipeline_config row exists
on first contact, and writes one (icao24, cycle) snapshot per call to
write_cycle(). Failures are logged and swallowed so the acquisition loop keeps
streaming — the connection is dropped on error and re-established next cycle.
"""

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
import structlog

from planesailing.config import PipelineConfig

logger = structlog.get_logger()


@dataclass
class PlaneSnapshot:
    """One aircraft's merged state for a single decode cycle.

    Note that when we decode ADSB, we do so by looking at a recent history of messages for a given icao24.

    This is the result of that merge, which is what we write to the DB. So each snapshot is one row in the readings
    table, and also represents the latest state of the plane for upserting into the planes table."""

    icao24: str
    ts: datetime
    raw_hex: str
    raw: dict[str, Any]
    callsign: str | None = None
    altitude: int | None = None
    lat: float | None = None
    lon: float | None = None
    velocity: float | None = None
    heading: float | None = None
    squawk: str | None = None


def _connect() -> psycopg.Connection:
    return psycopg.connect(
        host=os.environ.get("PLANE_SAILING_DB_HOST", "localhost"),
        port=int(os.environ.get("PLANE_SAILING_DB_PORT", "5432")),
        dbname=os.environ["PLANE_SAILING_DB_NAME"],
        user=os.environ["PLANE_SAILING_DB_USER"],
        password=os.environ["PLANE_SAILING_DB_PASSWORD"],
    )


def _ensure_pipeline_config(conn: psycopg.Connection, config: PipelineConfig) -> int:
    """Upsert by fingerprint, return id. DO UPDATE no-op is how we RETURN on conflict."""
    fingerprint = config.fingerprint()
    params = Jsonb(config.as_dict())
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO cosmos.pipeline_configs (fingerprint, params)
            VALUES (%s, %s)
            ON CONFLICT (fingerprint) DO UPDATE SET fingerprint = EXCLUDED.fingerprint
            RETURNING id
            """,
            (fingerprint, params),
        )
        row = cur.fetchone()
        conn.commit()
        return row[0]


_UPSERT_PLANE_SQL = """
INSERT INTO cosmos.planes (
    icao24, callsign, first_seen, last_seen,
    last_lat, last_lon, last_altitude, last_velocity, last_message
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (icao24) DO UPDATE SET
    callsign      = COALESCE(EXCLUDED.callsign,      cosmos.planes.callsign),
    last_seen     = EXCLUDED.last_seen,
    last_lat      = COALESCE(EXCLUDED.last_lat,      cosmos.planes.last_lat),
    last_lon      = COALESCE(EXCLUDED.last_lon,      cosmos.planes.last_lon),
    last_altitude = COALESCE(EXCLUDED.last_altitude, cosmos.planes.last_altitude),
    last_velocity = COALESCE(EXCLUDED.last_velocity, cosmos.planes.last_velocity),
    last_message  = EXCLUDED.last_message
"""

_INSERT_READING_SQL = """
INSERT INTO cosmos.readings (
    ts, icao24, callsign, lat, lon, altitude, velocity, heading, squawk, raw,
    pipeline_config_id, raw_hex
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _upsert_plane(cur: psycopg.Cursor, s: PlaneSnapshot) -> None:
    cur.execute(
        _UPSERT_PLANE_SQL,
        (
            s.icao24,
            s.callsign,
            s.ts,
            s.ts,
            s.lat,
            s.lon,
            s.altitude,
            s.velocity,
            Jsonb(s.raw),
        ),
    )


def _insert_reading(
    cur: psycopg.Cursor, s: PlaneSnapshot, pipeline_config_id: int
) -> None:
    cur.execute(
        _INSERT_READING_SQL,
        (
            s.ts,
            s.icao24,
            s.callsign,
            s.lat,
            s.lon,
            s.altitude,
            s.velocity,
            s.heading,
            s.squawk,
            Jsonb(s.raw),
            pipeline_config_id,
            s.raw_hex,
        ),
    )


class DBWriter:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.conn: psycopg.Connection | None = None
        self.pipeline_config_id: int | None = None

    def _ensure_conn(self) -> bool:
        if self.conn is not None and not self.conn.closed:
            return True
        try:
            self.conn = _connect()
            self.pipeline_config_id = _ensure_pipeline_config(self.conn, self.config)
            logger.info(
                "db connected",
                pipeline_config_id=self.pipeline_config_id,
                fingerprint=self.config.fingerprint(),
            )
            return True
        except (psycopg.Error, KeyError) as e:
            logger.warning("db connect failed", error=str(e))
            self.conn = None
            return False

    def write_cycle(self, snapshots: list[PlaneSnapshot]) -> None:
        if not snapshots:
            return
        if not self._ensure_conn():
            return
        try:
            with self.conn.transaction():
                with self.conn.cursor() as cur:
                    for s in snapshots:
                        _upsert_plane(cur, s)
                        _insert_reading(cur, s, self.pipeline_config_id)
        except psycopg.Error as e:
            logger.warning(
                "db write failed, dropping batch", error=str(e), n=len(snapshots)
            )
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    def close(self) -> None:
        if self.conn is not None and not self.conn.closed:
            self.conn.close()
        self.conn = None
