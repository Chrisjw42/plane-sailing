"""The collector's watchdog (infra/collector/run.py): restart a stuck collector, not a quiet sky.

Since the decoder writes only aircraft a CRC-checked frame vouched for, rows stop being a heartbeat: on a quiet night
the collector can go 15-40 minutes without one while it is working (2026-11-01 02:53 and 03:29 UTC it was restarted
for exactly that). These tests drive run.main() with a fake clock, a fake streamer and a fake database:

  * a quiet sky (loop turning, no row for an hour) is NOT a stall;
  * a stuck loop (no completed cycle for COSMOS_STALENESS_SECONDS) IS, even while old rows look fresh enough;
  * a deaf receiver (loop turning, no row for COSMOS_ROW_STALENESS_SECONDS) still restarts it;
  * the health log's OK lines carry last_cycle_age_s next to last_row_age_s.
Needs plane-sailing's environment (run.py imports planesailing.main). PS_ROOT selects the tree.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("PS_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT))


@pytest.fixture
def run(monkeypatch, tmp_path):
    monkeypatch.setenv("COSMOS_HEALTH_LOG_PATH", str(tmp_path / "collector.log"))
    for k in ("COSMOS_STALENESS_SECONDS", "COSMOS_ROW_STALENESS_SECONDS", "COSMOS_HEALTH_INTERVAL_SECONDS"):
        monkeypatch.delenv(k, raising=False)
    spec = importlib.util.spec_from_file_location("collector_run", ROOT / "infra" / "collector" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Rig:
    """Fake clock + streamer + database. Each 60 s sleep is one watchdog check; the acquisition 'thread' completes
    a decode cycle during it if `turning`, and a row lands if `rows`."""

    def __init__(self, run, monkeypatch, minutes, turning, rows):
        self.now = 1_000_000.0
        self.minutes, self.turning, self.rows = minutes, turning, rows
        self.last_row = dt.datetime(2026, 11, 1, 1, 0, tzinfo=dt.timezone.utc)
        self.lines: list[str] = []
        self.checks = 0
        rig = self

        class Thread:
            def is_alive(self):
                return True

        class Streamer:
            thread = Thread()

            def start_adsb_stream(self, **kw):
                pass

            def stop_stream(self):
                pass

            def _batch_generate_hexes(self, **kw):
                return None

        self.streamer = Streamer()
        monkeypatch.setattr(run, "ADSBStreamer", self.streamer)
        monkeypatch.setattr(run.time, "monotonic", lambda: rig.now)
        monkeypatch.setattr(run.time, "sleep", self.sleep)
        monkeypatch.setattr(run, "_last_row_age_seconds", self.row_age)
        monkeypatch.setattr(run, "_health_write", lambda status, **f: rig.lines.append(
            " ".join([status] + [f"{k}={v}" for k, v in f.items()])))

    def wall(self):
        return dt.datetime(2026, 11, 1, 1, 0, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=self.now - 1_000_000.0)

    def sleep(self, s):
        if self.checks >= self.minutes:
            raise KeyboardInterrupt  # end of the scenario: run.main logs SHUTDOWN and returns
        self.checks += 1
        self.now += s
        if self.turning(self.checks):
            self.streamer._batch_generate_hexes(n=3)  # through whatever run.py installed on the instance
        if self.rows(self.checks):
            self.last_row = self.wall()

    def row_age(self):
        return (self.wall() - self.last_row).total_seconds(), self.last_row

    def exit_reason(self, run):
        try:
            run.main()
        except SystemExit:
            return self.lines[-1]
        return None


def test_quiet_sky_is_not_a_stall(run, monkeypatch):
    # 3 h: a row every few minutes for the first hour, then none for 70 minutes, then some again. Loop always turns.
    rig = Rig(run, monkeypatch, minutes=180, turning=lambda m: True,
              rows=lambda m: m < 60 or m > 130)
    assert rig.exit_reason(run) is None, rig.lines[-3:]


def test_stuck_loop_is_a_stall(run, monkeypatch):
    # rows fresh until minute 30 (the old row-age watchdog is happy until 45); the loop stops at minute 20
    rig = Rig(run, monkeypatch, minutes=120, turning=lambda m: m < 20, rows=lambda m: m < 30)
    last = rig.exit_reason(run)
    assert last is not None and last.startswith("STALE") and "no_cycle" in last
    assert rig.checks <= 20 + 16  # caught within the 15-minute window (+1 check)


def test_deaf_receiver_still_restarts(run, monkeypatch):
    rig = Rig(run, monkeypatch, minutes=6 * 60, turning=lambda m: True, rows=lambda m: m < 10)
    last = rig.exit_reason(run)
    assert last is not None and "no_rows" in last
    assert 4 * 60 <= rig.checks <= 4 * 60 + 12


def test_ok_lines_carry_the_cycle_age(run, monkeypatch):
    rig = Rig(run, monkeypatch, minutes=20, turning=lambda m: True, rows=lambda m: True)
    rig.exit_reason(run)
    oks = [ln for ln in rig.lines if ln.startswith("OK")]
    assert oks and all("last_cycle_age_s=0" in ln and "last_row_age_s=" in ln for ln in oks)
