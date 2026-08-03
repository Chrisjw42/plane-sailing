# Current state — 2026-08-03

Working notes from the 3 Aug session: the COSMOS_N experiment loop, the
ghost-row investigation, and the pipeline fixes that came out of it.

## COSMOS_N sweep experiment (armed, starts midnight tonight)

Goal: one full day (exactly 24h) of collection per value of `COSMOS_N`, through
the sequence `1,2,3,4,6,8,10,...,30` (17 values).

- `infra/experiment/cosmos_n_cycle.sh` is fired by cron daily at **00:00
  local** (`crontab -l`). Each firing advances the sequence, writes `COSMOS_N`
  into `infra/.env`, and rebuilds+force-recreates **only** the collector
  container. Postgres stays up throughout.
- State: `infra/experiment/state` — currently `start`, so tonight's midnight
  firing applies **n=1 with the fixed code** (the `--build` flag picks up the
  code changes below). n=1 runs Aug 4, n=2 Aug 5, …, n=30 runs Aug 20; the
  firing at 00:00 Aug 21 stops the collector and writes `done`.
- Flip history: `infra/experiment/cycle.log`. Stray cron/docker output:
  `infra/experiment/cron.log`. Collector health: `infra/collector/logs/collector.log`.
- Recovery: `cosmos_n_cycle.sh --apply-current` re-applies the state value
  without advancing; `--dry-run` previews. State is only advanced after the new
  container is verified running with the intended `COSMOS_N`, so a failed flip
  self-retries the next night.
- Analysis note: rows link to their config via `pipeline_config_id` /
  fingerprint. **Only use data with `ts >= 2026-08-04`** for the experiment —
  earlier data was collected with the pre-fix pipeline (no ghost filtering,
  broken raw_hex attribution), so row counts are not comparable across that
  boundary. Config fingerprints do NOT change with code changes, so the date
  filter is the only guard.

## Ghost-row findings (investigation, 3 Aug)

Of 7.1M readings, 63.9% had no lat/lon. Breakdown:

- ~3.1M rows belonged to **1.96M distinct icao24s that never produced a
  position** — overwhelmingly DF20/21/16 Mode S interrogation replies. These
  frames never contain position, and their icao24 is recovered by XOR-ing the
  address/parity field, so a single bit error silently invents a ghost
  aircraft (1.68M "aircraft" seen exactly once).
- ~1.4M no-position rows came from the 5,542 real (position-producing)
  aircraft: altitude/squawk/velocity between position fixes. Not map-usable but
  real data.
- Replay test (full day 2026-06-15, all stored `raw_hex` re-decoded through
  rs1090 with whole-day CPR pairing context): **zero** new aircraft, **zero**
  extra positions vs what the DB already held (3,000 replayed vs 43,920
  stored). The DB is write-only for the pipeline (only `SELECT max(ts)` for
  health), so old rows are never inputs to decoding. The "keep them to
  reconstruct positions later" hypothesis is dead.

Cleanup queries (written, **not yet run** — pick a cutoff before 2026-08-04 and
`VACUUM FULL cosmos.readings` afterwards to reclaim disk):

```sql
-- Option A: all position-less rows older than cutoff
DELETE FROM cosmos.readings
WHERE lat IS NULL AND lon IS NULL AND ts < '2026-07-01'::timestamptz;

-- Option B (sharper): only rows from never-positioned (ghost) aircraft
DELETE FROM cosmos.readings r
WHERE r.ts < '2026-07-01'::timestamptz AND r.lat IS NULL
  AND NOT EXISTS (SELECT 1 FROM cosmos.readings p
                  WHERE p.icao24 = r.icao24 AND p.lat IS NOT NULL);
```

`cosmos.planes` is still polluted with ~1.96M ghost aircraft rows; deleting
readings does not clean it (separate follow-up).

## Pipeline changes (3 Aug, live from tonight's midnight rebuild)

1. **raw_hex attribution bug fixed** (`planesailing/main.py`, `_write_cycle`):
   `rs1090.decode` silently *omits* undecodable messages from its output
   (measured: 700 returned for 1,000 in), so zipping the batch result against
   the input list misattributed `raw_hex`/timestamps to the wrong aircraft
   whenever any hex in a cycle failed to decode. Now decodes one message at a
   time so pairing is exact. (`decode_current_cached_messages` was audited and
   is unaffected — it never zips against inputs.)
2. **Ghost suppression** (`_write_cycle`): snapshots with neither a position
   nor an altitude are no longer written (drops squawk-only/empty DF21-type
   rows and stops their bogus icao24s polluting `cosmos.planes`). Rows with
   altitude are kept per decision on 3 Aug, even when position-less.

Verified: module lints/parses, docker image builds, and a stubbed-writer test
confirmed (a) an altitude-only DF20 row is kept with correctly-attributed
raw_hex despite an undecodable hex earlier in the batch, and (b) a squawk-only
DF21 row is dropped.
