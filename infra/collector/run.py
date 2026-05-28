"""24/7 collector entrypoint for the ADSB pipeline.

This is *orchestration*, not application logic. It lives under infra/, consumes
``planesailing`` as a library, and is responsible only for:
  - turning ``COSMOS_*`` env vars into a :class:`PipelineConfig`,
  - starting the long-running :class:`ADSBStreamer`,
  - watching that rows keep landing in ``cosmos.readings``, and
  - exiting non-zero (so Docker's ``restart: unless-stopped`` policy brings us
    back) if collection stalls or the acquisition thread dies.

A "stall" is: no new row in ``cosmos.readings`` for ``COSMOS_STALENESS_SECONDS``.
This catches the silent-failure mode where the process is alive but the SDR has
stopped delivering usable data. Health transitions are appended to a text file
(``COSMOS_HEALTH_LOG_PATH``, bind-mounted to the host) so overnight stalls and
restarts are visible the next morning, as well as going to stdout for
``docker logs``.

A transient DB error is deliberately *not* treated as a stall: we log it and
keep running, so a brief Postgres blip doesn't trigger a restart loop.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

from planesailing.config import PipelineConfig
from planesailing.db import _connect
from planesailing.main import ADSBStreamer

logger = structlog.get_logger()


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def build_config() -> PipelineConfig:
    """Build a PipelineConfig from COSMOS_* env vars, defaulting to the dataclass defaults.

    Any unset var falls back to the PipelineConfig field default, so you only
    need to set the one or two params you're actually experimenting with. The
    resulting config's fingerprint is what links the rows this run produces back
    to their exact configuration in cosmos.pipeline_configs.
    """
    d = PipelineConfig()
    return PipelineConfig(
        sample_rate=_env_float("COSMOS_SAMPLE_RATE", d.sample_rate),
        target_sr=_env_float("COSMOS_TARGET_SR", d.target_sr),
        n_reads_per_acquisition=_env_int(
            "COSMOS_N_READS_PER_ACQUISITION", d.n_reads_per_acquisition
        ),
        n_reads_to_try_to_parse=_env_int(
            "COSMOS_N_READS_TO_TRY_TO_PARSE", d.n_reads_to_try_to_parse
        ),
        n_samples_per_read=_env_int("COSMOS_N_SAMPLES_PER_READ", d.n_samples_per_read),
        sleep_length_s=_env_float("COSMOS_SLEEP_LENGTH_S", d.sleep_length_s),
        sdr_gain=_env_str("COSMOS_SDR_GAIN", d.sdr_gain),
        n=_env_int("COSMOS_N", d.n),
    )


HEALTH_LOG_PATH = _env_str("COSMOS_HEALTH_LOG_PATH", "/var/log/cosmos/collector.log")


def _health_write(status: str, **fields) -> None:
    """Append one timestamped, greppable line to the health log file (and stdout).

    Format: ``<iso8601-utc> <STATUS> key=value key=value`` — designed so an
    overnight `grep STALE collector.log` immediately tells you when and why the
    collector restarted itself.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"{now} {status} {parts}".rstrip()
    try:
        path = Path(HEALTH_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(line + "\n")
    except OSError as e:
        logger.warning("could not write health log", error=str(e), path=HEALTH_LOG_PATH)
    print(line, flush=True)


def _last_row_age_seconds() -> tuple[float | None, datetime | None]:
    """Return (age_seconds, last_ts) for the most recent cosmos.readings row.

    Returns (None, None) when the table is empty. Raises on DB error so the
    caller can distinguish 'DB unreachable' (transient, keep running) from
    'confirmed stale' (exit).
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT max(ts) FROM cosmos.readings")
            row = cur.fetchone()
    last_ts = row[0] if row else None
    if last_ts is None:
        return None, None
    age = (datetime.now(timezone.utc) - last_ts).total_seconds()
    return age, last_ts


def main() -> None:
    staleness_s = _env_float("COSMOS_STALENESS_SECONDS", 900.0)  # 15 minutes
    interval_s = _env_float("COSMOS_HEALTH_INTERVAL_SECONDS", 60.0)

    cfg = build_config()
    _health_write(
        "STARTUP",
        fingerprint=cfg.fingerprint(),
        staleness_window_s=int(staleness_s),
        check_interval_s=int(interval_s),
    )
    logger.info(
        "collector starting", config=cfg.as_dict(), fingerprint=cfg.fingerprint()
    )

    ADSBStreamer.start_adsb_stream(config=cfg, persist=True, verbose=False)

    process_start = time.monotonic()

    try:
        while True:
            time.sleep(interval_s)

            # The acquisition runs in a daemon thread; if it died, the process is
            # alive but doing nothing — restart so we recover.
            thread = ADSBStreamer.thread
            if thread is not None and not thread.is_alive():
                _health_write("THREAD_DEAD", action="exit")
                sys.exit(1)

            try:
                age, last_ts = _last_row_age_seconds()
            except Exception as e:  # transient DB issue: log, keep running
                _health_write("DB_ERROR", error=type(e).__name__)
                continue

            # Grace period after (re)start: give the pipeline time to produce its
            # first row before we'd ever consider exiting. Equal to the staleness
            # window so a restart after a long outage isn't instantly re-killed.
            in_grace = (time.monotonic() - process_start) < staleness_s

            if age is None:
                if in_grace:
                    _health_write("STARTING", last_row="none")
                else:
                    _health_write("STALE", last_row="none", action="exit")
                    sys.exit(1)
                continue

            last_ts_str = last_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            if age > staleness_s and not in_grace:
                _health_write(
                    "STALE",
                    last_row_age_s=int(age),
                    last_row_ts=last_ts_str,
                    action="exit",
                )
                sys.exit(1)
            else:
                _health_write("OK", last_row_age_s=int(age), last_row_ts=last_ts_str)
    except KeyboardInterrupt:
        _health_write("SHUTDOWN", reason="interrupt")
        ADSBStreamer.stop_stream()


if __name__ == "__main__":
    main()
