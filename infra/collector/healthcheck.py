"""Docker HEALTHCHECK probe: exit 0 if a reading landed within the staleness window.

This is independent of run.py's own self-exit logic — its job is to surface
health in ``docker ps`` / ``docker inspect`` so you can see at a glance whether
the collector is producing data. It does NOT trigger restarts on its own; that
is handled by run.py exiting non-zero combined with the compose restart policy.

Exit 0 = a row landed within COSMOS_ROW_STALENESS_SECONDS (default 4 h); exit 1 =
stale, empty, or DB unreachable. Not COSMOS_STALENESS_SECONDS: that is run.py's
decode-loop heartbeat, which this separate process can't see, and a quiet night
goes 15-40 minutes without a row while the collector is healthy.
"""

import os
import sys
from datetime import datetime, timezone

from planesailing.db import _connect


def main() -> None:
    staleness_s = float(os.environ.get("COSMOS_ROW_STALENESS_SECONDS") or 4 * 3600.0)

    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT max(ts) FROM cosmos.readings")
                row = cur.fetchone()
    except Exception as e:
        print(f"healthcheck: db error: {e}", file=sys.stderr)
        sys.exit(1)

    last_ts = row[0] if row else None
    if last_ts is None:
        print("healthcheck: no rows", file=sys.stderr)
        sys.exit(1)

    age = (datetime.now(timezone.utc) - last_ts).total_seconds()
    if age > staleness_s:
        print(f"healthcheck: stale, last row {int(age)}s ago", file=sys.stderr)
        sys.exit(1)

    print(f"healthcheck: ok, last row {int(age)}s ago")
    sys.exit(0)


if __name__ == "__main__":
    main()
