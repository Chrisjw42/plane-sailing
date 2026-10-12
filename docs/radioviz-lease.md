# radioviz SDR leases (note for plane-sailing)

radioviz (`../radioviz`) borrows the RTL-SDR by **stopping the collector** for the
length of a "lease", then brings it back with `docker compose up -d collector`.
Every lease is logged in `radioviz.sdr_leases`, so readings gaps it causes are
explainable and excludable.

## The cycle script

On this branch (`radioviz-lease-aware`), `infra/experiment/cosmos_n_cycle.sh`
checks `~/.local/state/radioviz/sdr.lease` before touching the collector. If a
live radioviz process holds the lease it writes `sdr.release-request`, waits up
to 120 s (`RADIOVIZ_WAIT_S`) for the lease to clear, then proceeds either way.
Stale lease files (dead pid) are ignored. Nothing else changes.

Until this branch is merged, radioviz defends itself instead: it releases any
lease shortly before midnight and refuses new ones until the flip is done.

## Excluding lease windows from ADSB analysis

```sql
-- Readings that were NOT taken during (or within 2 min after) a radioviz lease.
SELECT r.*
FROM cosmos.readings r
WHERE NOT EXISTS (
    SELECT 1 FROM radioviz.sdr_leases l
    WHERE r.ts >= l.started_at
      AND r.ts <  COALESCE(l.ended_at, now()) + interval '2 minutes'
);

-- Collection lost to leases, per day.
SELECT date_trunc('day', started_at) AS day,
       sum(COALESCE(ended_at, now()) - started_at) AS sdr_lent,
       count(*) AS leases
FROM radioviz.sdr_leases
GROUP BY 1 ORDER BY 1;
```

The 2-minute tail covers collector start-up after a lease ends.
