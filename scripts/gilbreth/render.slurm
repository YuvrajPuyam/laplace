#!/bin/bash
# Laplace photoreal render on Gilbreth - Isaac render_in_env inside the Apptainer container.
# Needs an RT-core GPU for good RTX (a10); a30/a100 also work (compute ray-tracing fallback).
# 24+ GB VRAM means none of the laptop's 8 GB OOM/crash issues.
#
#   # boot-test: one still (verifies the Simple_Warehouse asset fetches + RTX renders)
#   sbatch --partition=a10 --account=csml scripts/gilbreth/render.slurm
#   # full sequence: override RENDER_ARGS via --export (see render_seq usage)
#SBATCH --job-name=laplace-render
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=render_%j.out
set -euo pipefail
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
command -v apptainer >/dev/null || command -v singularity >/dev/null || { echo "[render] no apptainer/singularity"; exit 1; }

B=/scratch/gilbreth/gupta596/MotionGen/HOI/laplace
SIF="${SIF:-$B/containers/isaac-sim-5.1.0.sif}"
CACHE="${CACHE:-$B/isaac-cache}"
mkdir -p "$CACHE"/{kit,ov,cache,logs,data,documents} "$B/renderer/out"

# default = one full-res still (the boot-test). Override via --export=ALL,RENDER_ARGS="..."
RENDER_ARGS="${RENDER_ARGS:---env-usd /Isaac/Environments/Simple_Warehouse/full_warehouse.usd --tracks renderer/scenes/real_warehouse_demo_tracks.json --coord-map eval/dev_scenarios/real_full_warehouse.provenance.json --at-frame 120 --out renderer/out/gilbreth_still.png --width 1280 --height 720 --eye -13 1 4 --target -13 20 0.8}"

echo "[render] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "[render] args: $RENDER_ARGS"

apptainer exec --nv \
  --env ACCEPT_EULA=Y --env OMNI_KIT_ACCEPT_EULA=YES --env PRIVACY_CONSENT=Y \
  --bind "$B":/workspace/laplace \
  --bind "$CACHE/kit":/isaac-sim/kit/cache \
  --bind "$CACHE/ov":/root/.cache/ov \
  --bind "$CACHE/cache":/root/.cache \
  --bind "$CACHE/logs":/root/.nvidia-omniverse/logs \
  --bind "$CACHE/data":/root/.local/share/ov/data \
  --bind "$CACHE/documents":/root/Documents \
  "$SIF" bash -lc "cd /workspace/laplace && /isaac-sim/python.sh -m renderer.render_in_env $RENDER_ARGS"

echo "[render] done -> $B/renderer/out/"
