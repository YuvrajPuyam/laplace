#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Laplace PhysX validation on Purdue Gilbreth (RCAC) — SLURM + Apptainer.
# Runs a renderer.physx_* module inside the Isaac Sim container on one A100.
#
#   PHYSX_MODULE  which module to run   (default renderer.physx_run)
#   PHYSX_ARGS    args for that module  (default = the drive step on braess_plan.json)
#
# Smoke first (boot + fit), then the real drive:
#   sbatch --export=ALL,PHYSX_MODULE=renderer.physx_probe,PHYSX_ARGS="--robots 12 --steps 600" scripts/gilbreth/physx.slurm
#   sbatch scripts/gilbreth/physx.slurm                          # the drive (defaults below)
#
# The A100 has no RT cores — that is FINE here: the agreement NUMBER is pure PhysX
# compute, no ray tracing. (Save an RT-core node for the separate photoreal render.)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=laplace-physx
# NOTE: partition + account are NOT hard-coded — pass them on the sbatch command line so
# this one script serves every cluster/allocation (scripts/live/up.sh and the rho* drive
# both do this):  sbatch --partition=a30 --account=csml ... scripts/gilbreth/physx.slurm
#SBATCH --gpus-per-node=1                # 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=physx_%j.out

set -euo pipefail
# apptainer is system-installed on Gilbreth (on PATH), not always an Lmod module — so a
# failed `module load` must NOT abort the job. Try the module, else fall back to PATH.
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
command -v apptainer >/dev/null || command -v singularity >/dev/null || { echo "[physx.slurm] no apptainer/singularity on PATH"; exit 1; }

# ── paths (override via env or edit) ────────────────────────────────────────
REPO="${REPO:-$HOME/laplace}"                                  # the repo checked out on Gilbreth
SIF="${SIF:-$HOME/containers/isaac-sim-5.1.0.sif}"             # built in the README (match D:\iv = 5.1.0)
CACHE="${CACHE:-${RCAC_SCRATCH:-$HOME}/isaac-cache}"           # Kit writes a LOT — keep it on scratch

PHYSX_MODULE="${PHYSX_MODULE:-renderer.physx_run}"
PHYSX_ARGS="${PHYSX_ARGS:-drive --plan runs/physx/braess_plan.json --max-accel 1.2 --max-sim-min 5 --out runs/physx/braess_phys.json}"

mkdir -p "$CACHE"/{kit,ov,cache,logs,data,documents,pip}

echo "[physx.slurm] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader || echo '?')"
echo "[physx.slurm] running: python.sh -m $PHYSX_MODULE $PHYSX_ARGS"

# ── run Isaac headless in the container ─────────────────────────────────────
# --nv exposes the GPU; binds redirect Kit's caches/home to writable scratch;
# EULA env must be set or first boot blocks on an interactive prompt.
apptainer exec --nv \
  --env ACCEPT_EULA=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PRIVACY_CONSENT=Y \
  --env LAPLACE_DRIVE_DBG="${LAPLACE_DRIVE_DBG:-}" \
  --bind "$REPO":/workspace/laplace \
  --bind "$CACHE/kit":/isaac-sim/kit/cache \
  --bind "$CACHE/ov":/root/.cache/ov \
  --bind "$CACHE/cache":/root/.cache \
  --bind "$CACHE/logs":/root/.nvidia-omniverse/logs \
  --bind "$CACHE/data":/root/.local/share/ov/data \
  --bind "$CACHE/documents":/root/Documents \
  "$SIF" bash -lc "cd /workspace/laplace && /isaac-sim/python.sh -m $PHYSX_MODULE $PHYSX_ARGS"

echo "[physx.slurm] done. retrieve runs/physx/*.json and run 'compare' on a CPU box."
