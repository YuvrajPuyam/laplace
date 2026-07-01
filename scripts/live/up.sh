#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Laplace live decision-twin — ONE-COMMAND bring-up (Git Bash / Linux).
#
# Brings the whole chain up with a health gate at each hop and records every PID
# in a state file so scripts/live/down.sh can tear it down cleanly.
#
#   scripts/live/up.sh                 # LOCAL toy: mock PhysX stream + engine + viewer (no GPU, no SSH)
#   scripts/live/up.sh --gilbreth      # REAL: Isaac/PhysX on Gilbreth over an SSH tunnel
#   scripts/live/up.sh --scenario braess_dev
#
# The chain:  Three.js viewer  <->  engine (uvicorn)  <->  PhysX stream
#   --mock:      stream runs locally (renderer.physx_stream --mock)
#   --gilbreth:  stream runs in the Isaac container on a Gilbreth GPU node,
#                reached over  ssh -N -L <local>:<node>:<remote>.
#
# Then open the printed URL and click "Live".  Tear down with scripts/live/down.sh.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ---- config (override via env) ---------------------------------------------
SCENARIO="braess_dev"
MODE="mock"
ENGINE_PORT="${LAPLACE_ENGINE_PORT:-8013}"        # the live engine (viewer is served here)
STREAM_PORT="${LAPLACE_STREAM_PORT:-8765}"        # PhysX stream TCP port (local or remote)
STATE_FILE="${LAPLACE_LIVE_STATE:-.laplace_live.state}"

# Gilbreth (only used with --gilbreth; all overridable)
G_HOST="${LAPLACE_GILBRETH_HOST:-gilbreth.rcac.purdue.edu}"
G_HOI="${LAPLACE_HOI:-/scratch/gilbreth/gupta596/MotionGen/HOI/laplace}"
G_PART="${LAPLACE_SLURM_PARTITION:-a30}"
G_ACCT="${LAPLACE_SLURM_ACCOUNT:-csml}"
G_SIF="${LAPLACE_SIF:-$G_HOI/containers/isaac-sim-5.1.0.sif}"

while [ $# -gt 0 ]; do
  case "$1" in
    --mock) MODE="mock" ;;
    --gilbreth) MODE="gilbreth" ;;
    --scenario) SCENARIO="$2"; shift ;;
    --scenario=*) SCENARIO="${1#*=}" ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "[up] unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

if [ -f "$STATE_FILE" ]; then
  echo "[up] $STATE_FILE exists — a live session may already be running."
  echo "[up] run scripts/live/down.sh first (or delete the file if it is stale)." >&2
  exit 1
fi

# fresh state file; every started PID/JOBID is appended so down.sh can reverse it
: > "$STATE_FILE"
echo "MODE=$MODE"          >> "$STATE_FILE"
echo "SCENARIO=$SCENARIO"  >> "$STATE_FILE"
echo "ENGINE_PORT=$ENGINE_PORT" >> "$STATE_FILE"

# wait until an HTTP endpoint returns, up to N seconds; 0=fail
_wait_http() {  # _wait_http <url> <seconds> <grep-needle>
  local url="$1" secs="$2" needle="${3:-}" i=0
  while [ "$i" -lt "$secs" ]; do
    if body="$(curl -fsS "$url" 2>/dev/null)"; then
      if [ -z "$needle" ] || printf '%s' "$body" | grep -q "$needle"; then
        return 0
      fi
    fi
    sleep 1; i=$((i + 1))
  done
  return 1
}

# ---- 1. the PhysX stream (local mock, or remote Isaac via tunnel) -----------
PHYSX_ADDR="127.0.0.1:$STREAM_PORT"
if [ "$MODE" = "mock" ]; then
  echo "[up] starting LOCAL mock PhysX stream on :$STREAM_PORT (scenario=$SCENARIO) ..."
  $PYTHON -m renderer.physx_stream --mock --host 127.0.0.1 --port "$STREAM_PORT" \
          --scenario "$SCENARIO" --seconds 0 >runs/_live_stream.log 2>&1 &
  echo "STREAM_PID=$!" >> "$STATE_FILE"
else
  command -v ssh >/dev/null || { echo "[up] ssh not found" >&2; exit 1; }
  echo "[up] submitting the Isaac/PhysX stream job on Gilbreth ($G_PART/$G_ACCT) ..."
  JOBID="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$G_HOST" "
    cd '$G_HOI' &&
    sbatch --parsable --partition='$G_PART' --account='$G_ACCT' \
      --export=ALL,REPO='$G_HOI',SIF='$G_SIF',PHYSX_MODULE=renderer.physx_stream,PHYSX_ARGS='--host 0.0.0.0 --port $STREAM_PORT --scenario $SCENARIO --seconds 0' \
      scripts/gilbreth/physx.slurm" 2>/dev/null | tr -dc '0-9')"
  [ -n "$JOBID" ] || { echo "[up] sbatch failed (check partition/account/SIF)"; exit 1; }
  echo "JOBID=$JOBID" >> "$STATE_FILE"
  echo "[up] job $JOBID submitted; waiting for it to start + print READY (up to ~5 min) ..."
  NODE=""; i=0
  while [ "$i" -lt 100 ]; do
    line="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$G_HOST" \
      "grep -m1 'READY' '$G_HOI/physx_$JOBID.out' 2>/dev/null; squeue -j $JOBID -h -o %T 2>/dev/null" 2>/dev/null || true)"
    if printf '%s' "$line" | grep -q READY; then
      NODE="$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$G_HOST" "squeue -j $JOBID -h -o %N 2>/dev/null" 2>/dev/null | tr -d '[:space:]')"
      break
    fi
    printf '%s' "$line" | grep -qiE 'PENDING|RUNNING|CONFIG' || { sleep 3; i=$((i+1)); continue; }
    sleep 3; i=$((i + 1))
  done
  [ -n "$NODE" ] || { echo "[up] job never reported READY; see physx_$JOBID.out on Gilbreth"; exit 1; }
  echo "[up] stream READY on node $NODE; opening SSH tunnel localhost:$STREAM_PORT -> $NODE:$STREAM_PORT ..."
  ssh -N -o BatchMode=yes -o ConnectTimeout=20 -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      -L "$STREAM_PORT:$NODE:$STREAM_PORT" "$G_HOST" >runs/_live_tunnel.log 2>&1 &
  echo "TUNNEL_PID=$!" >> "$STATE_FILE"
  sleep 2
fi

# ---- 2. the engine (serves the viewer + relays the stream) ------------------
echo "[up] starting the engine on :$ENGINE_PORT ..."
LAPLACE_PHYSX_ADDR="$PHYSX_ADDR" \
  $PYTHON -m uvicorn engine.api:app --host 127.0.0.1 --port "$ENGINE_PORT" \
          --log-level warning >runs/_live_engine.log 2>&1 &
echo "ENGINE_PID=$!" >> "$STATE_FILE"

# ---- 3. health gate ---------------------------------------------------------
echo "[up] waiting for the engine to come up ..."
if ! _wait_http "http://127.0.0.1:$ENGINE_PORT/health" 30 '"ok"'; then
  echo "[up] engine did not become healthy — see runs/_live_engine.log" >&2
  exit 1
fi
echo "[up] waiting for the PhysX feed to be reachable ..."
if ! _wait_http "http://127.0.0.1:$ENGINE_PORT/health" 30 '"physx_reachable":true'; then
  echo "[up] WARNING: engine is up but the PhysX feed is not reachable yet."
  echo "[up]   mock: see runs/_live_stream.log   gilbreth: see runs/_live_tunnel.log"
fi

echo ""
echo "  ✓ live twin is up ($MODE)"
echo "  → open:  http://127.0.0.1:$ENGINE_PORT/twin?scenario=$SCENARIO   then click \"Live\""
echo "  → logs:  runs/_live_*.log     tear down:  scripts/live/down.sh"
echo ""
