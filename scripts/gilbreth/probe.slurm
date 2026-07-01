#!/bin/bash
# Laplace asset probe on Gilbreth - list real Isaac warehouse props (packing tables,
# conveyors, pallets, forklifts, chargers) so render_in_env can wire stations to real assets.
#SBATCH --job-name=laplace-probe
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=probe_%j.out
set -euo pipefail
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

B=/scratch/gilbreth/gupta596/MotionGen/HOI/laplace
SIF="${SIF:-$B/containers/isaac-sim-5.1.0.sif}"
CACHE="${CACHE:-$B/isaac-cache}"
mkdir -p "$CACHE"/{kit,ov,cache,logs,data,documents}

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PRIVACY_CONSENT=Y \
  --bind "$B":/workspace/laplace \
  --bind "$CACHE/kit":/isaac-sim/kit/cache \
  --bind "$CACHE/ov":/root/.cache/ov \
  --bind "$CACHE/cache":/root/.cache \
  --bind "$CACHE/logs":/root/.nvidia-omniverse/logs \
  --bind "$CACHE/data":/root/.local/share/ov/data \
  --bind "$CACHE/documents":/root/Documents \
  "$SIF" bash -lc "cd /workspace/laplace && /isaac-sim/python.sh -m renderer.probe_assets" \
  > "$B/renderer/out/probe_assets.txt" 2>&1 || true

echo "[probe] done -> $B/renderer/out/probe_assets.txt"
