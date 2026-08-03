"""Read recent rows from cosmos.readings.

Uses the same PLANE_SAILING_DB_* env vars as planesailing/db.py.
"""

from datetime import timedelta
import os

import pandas as pd
import psycopg


def _connect() -> psycopg.Connection:
    return psycopg.connect(
        host=os.environ.get("PLANE_SAILING_DB_HOST", "localhost"),
        port=int(os.environ.get("PLANE_SAILING_DB_PORT", "5432")),
        dbname=os.environ["PLANE_SAILING_DB_NAME"],
        user=os.environ["PLANE_SAILING_DB_USER"],
        password=os.environ["PLANE_SAILING_DB_PASSWORD"],
    )


def fetch_readings(days: int = 60, icao24: str | None = None) -> pd.DataFrame:
    """Return readings from the last `days` days, newest first.

    Pass icao24 to filter to a single aircraft.
    """
    sql = """
        SELECT id, ts, icao24, callsign, lat, lon, altitude, velocity,
               heading, squawk, raw_hex, pipeline_config_id
        FROM cosmos.readings
        WHERE ts >= now() - %(window)s
        {icao_filter}
        ORDER BY ts DESC
    """
    params: dict = {"window": timedelta(days=days)}
    icao_filter = ""
    if icao24 is not None:
        icao_filter = "AND icao24 = %(icao24)s"
        params["icao24"] = icao24

    with _connect() as conn:
        return pd.read_sql(sql.format(icao_filter=icao_filter), conn, params=params)


if __name__ == "__main__":
    df = fetch_readings(days=60)
    print(f"{len(df)} readings in the last 60 days")
    print(df.head())
