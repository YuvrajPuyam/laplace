#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Laplace live decision-twin — clean teardown. Reverses scripts/live/up.sh using
# the recorded state file (PIDs + Gilbreth JOBID). Idempotent and SCOPED: it only
# touches the job/processes this session started — never a broad scancel.
#
#   scripts/live/down.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
STATE_FILE="${LAPLACE_LIVE_STATE:-.laplace_live.state}"

if [ ! -f "$STATE_FILE" ]; then
  echo "[down] no $STATE_FILE — nothing to tear down."
  exit 0
fi

# shellcheck disable=SC1090
. "$STATE_FILE"

_kill_pid() {  # _kill_pid <pid> <label>
  local pid="$1" label="$2"
  [ -n "${pid:-}" ] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    echo "[down] stopped $label (pid $pid)"
  fi
}

_kill_pid "${ENGINE_PID:-}" "engine"
_kill_pid "${STREAM_PID:-}" "mock stream"
_kill_pid "${TUNNEL_PID:-}" "ssh tunnel"

if [ -n "${JOBID:-}" ]; then
  G_HOST="${LAPLACE_GILBRETH_HOST:-gilbreth.rcac.purdue.edu}"
  echo "[down] cancelling Gilbreth job $JOBID ..."
  ssh -o BatchMode=yes -o ConnectTimeout=20 "$G_HOST" "scancel $JOBID" 2>/dev/null || true
fi

rm -f "$STATE_FILE"
echo "[down] clean."
