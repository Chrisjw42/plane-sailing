#!/usr/bin/env bash
# Daily COSMOS_N cycling for the collector experiment.
#
# Each invocation advances COSMOS_N through SEQ, updates infra/.env, and
# recreates only the collector container (Postgres stays up). Cron fires this
# once a day, so each value collects for exactly 24 hours. After the final
# value has had its day, the collector is stopped and the state file is set to
# "done", making later firings no-ops.
#
# Usage:
#   cosmos_n_cycle.sh                  advance to the next value and apply it
#   cosmos_n_cycle.sh --apply-current  re-apply the state file's value (recovery
#                                      / end-to-end test; does not advance)
#   cosmos_n_cycle.sh --dry-run        log what would happen, change nothing
#
# State file holds the currently-running n, "start" (armed, first firing will
# apply SEQ[0]), or "done". It is only written after the new container is
# verified running with the right env, so a failed flip is retried from the
# same position at the next firing.

set -euo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin

INFRA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_FILE="${COSMOS_CYCLE_STATE_FILE:-$INFRA/experiment/state}"
LOG_FILE="${COSMOS_CYCLE_LOG_FILE:-$INFRA/experiment/cycle.log}"
ENV_FILE="$INFRA/.env"
LOCK_FILE="$INFRA/experiment/.lock"
COMPOSE=(docker compose -f "$INFRA/docker-compose.yaml" --project-directory "$INFRA")

SEQ=(1 2 3 4 6 8 10 12 14 16 18 20 22 24 26 28 30)

DRY_RUN=0
MODE=advance
case "${1:-}" in
  "") ;;
  --apply-current) MODE=apply-current ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

log() {
  local line
  line="$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
  echo "$line"
  [[ "$DRY_RUN" == 1 ]] || echo "$line" >> "$LOG_FILE"
}

# radioviz can borrow the SDR by stopping the collector (a "lease"). If we
# recreated the collector mid-lease it would fail to open the device and
# crash-loop, so ask radioviz to give the SDR back first and wait for it.
# A lease file whose pid is dead is stale: radioviz's reaper cleans those up,
# and we carry on. After the timeout we proceed regardless.
RADIOVIZ_STATE_DIR="${RADIOVIZ_STATE_DIR:-$HOME/.local/state/radioviz}"
RADIOVIZ_LEASE_FILE="$RADIOVIZ_STATE_DIR/sdr.lease"
RADIOVIZ_RELEASE_REQUEST="$RADIOVIZ_STATE_DIR/sdr.release-request"
RADIOVIZ_WAIT_S="${RADIOVIZ_WAIT_S:-120}"

radioviz_lease_live() {
  [[ -f "$RADIOVIZ_LEASE_FILE" ]] || return 1
  local pid
  pid="$(grep -o '"pid": *[0-9]*' "$RADIOVIZ_LEASE_FILE" | grep -o '[0-9]*$' || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

wait_for_radioviz_release() {
  radioviz_lease_live || return 0
  log "RADIOVIZ_LEASE active; requesting release (waiting up to ${RADIOVIZ_WAIT_S}s)"
  [[ "$DRY_RUN" == 1 ]] && return 0
  echo "{\"requested_by\": \"cosmos_n_cycle\", \"at\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > "$RADIOVIZ_RELEASE_REQUEST"
  local waited=0
  while radioviz_lease_live && (( waited < RADIOVIZ_WAIT_S )); do
    sleep 2
    waited=$((waited + 2))
  done
  rm -f "$RADIOVIZ_RELEASE_REQUEST"
  if radioviz_lease_live; then
    log "RADIOVIZ_LEASE still held after ${RADIOVIZ_WAIT_S}s; proceeding anyway"
  else
    log "RADIOVIZ_LEASE released after ${waited}s"
  fi
}

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "another cycle run holds the lock; exiting" >&2; exit 1; }

current="$(cat "$STATE_FILE")"

if [[ "$current" == "done" ]]; then
  log "COMPLETE experiment finished; nothing to do"
  exit 0
fi

if [[ "$current" == "start" ]]; then
  # Experiment armed but not yet begun: first firing applies the first value.
  next="${SEQ[0]}"
  current="(start)"
else

# Locate current in SEQ; a state value outside the sequence means something
# tampered with the file — fail loudly rather than guess.
idx=-1
for i in "${!SEQ[@]}"; do
  [[ "${SEQ[$i]}" == "$current" ]] && { idx=$i; break; }
done
if (( idx < 0 )); then
  log "ERROR state file value '$current' is not in the sequence"
  exit 1
fi

if [[ "$MODE" == "apply-current" ]]; then
  next="$current"
elif (( idx + 1 >= ${#SEQ[@]} )); then
  # The final value has now had its 24 hours: stop collecting.
  log "SEQUENCE_END n=$current has completed; stopping collector"
  if [[ "$DRY_RUN" == 1 ]]; then
    log "DRY_RUN would stop collector and mark done"
    exit 0
  fi
  wait_for_radioviz_release
  "${COMPOSE[@]}" stop collector
  echo "done" > "$STATE_FILE"
  log "COMPLETE collector stopped; all $((${#SEQ[@]})) values collected"
  exit 0
else
  next="${SEQ[$((idx + 1))]}"
fi

fi  # end of non-"start" handling

log "FLIP from n=$current to n=$next (mode=$MODE)"
if [[ "$DRY_RUN" == 1 ]]; then
  log "DRY_RUN no changes made"
  exit 0
fi

wait_for_radioviz_release

# Persist the new value where compose reads it, then recreate the collector.
# --build picks up any code changes since the last flip (cached no-op when the
# code is unchanged); --force-recreate guarantees a fresh container even when
# re-applying the same value; dependencies (db) are started if down but never
# recreated.
if grep -q '^COSMOS_N=' "$ENV_FILE"; then
  sed -i "s/^COSMOS_N=.*/COSMOS_N=$next/" "$ENV_FILE"
else
  echo "COSMOS_N=$next" >> "$ENV_FILE"
fi
"${COMPOSE[@]}" up -d --build --force-recreate collector

# Verify before committing state: container running, with the env we intended.
sleep 5
status="$(docker inspect -f '{{.State.Status}}' plane-sailing-collector)"
applied="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' plane-sailing-collector | grep '^COSMOS_N=' || true)"
if [[ "$status" != "running" || "$applied" != "COSMOS_N=$next" ]]; then
  log "ERROR post-flip check failed: status=$status applied=${applied:-none}; state not advanced"
  exit 1
fi

echo "$next" > "$STATE_FILE"
log "APPLIED n=$next collector=running"
