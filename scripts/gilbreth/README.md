# Running the PhysX agreement number on Purdue Gilbreth

Goal: land the headline **"fast sim agrees with full-physics PhysX to within X% at
contention scale"** (docs/PHYSX_VALIDATION.md). Gilbreth (RCAC) is **SLURM + Apptainer**,
and its A100s have **no RT cores** - which is exactly fine, because the number is pure
PhysX compute, not ray tracing.

The work splits across two machines:

| Step | Where | Needs GPU? |
|---|---|---|
| 1. `extract` - DES rollout -> leg plan | any CPU box (your laptop) | no |
| 2. `drive` - replay the plan under PhysX | **Gilbreth A100** (this folder) | yes |
| 3. `compare` - the agreement headline | any CPU box | no |

## One-time setup on Gilbreth

```bash
# a) get the repo on Gilbreth (only renderer/ + the plan are strictly needed for drive)
git clone <your-laplace-remote> $HOME/laplace      # or: rsync -a ./ gilbreth:~/laplace

# b) build the Isaac Sim container (match your local D:\iv version = 5.1.0).
#    isaac-sim on NGC may need `apptainer remote login` with an NGC API key first.
module load apptainer
mkdir -p $HOME/containers
apptainer pull $HOME/containers/isaac-sim-5.1.0.sif docker://nvcr.io/nvidia/isaac-sim:5.1.0

# c) edit scripts/gilbreth/physx.slurm: set --partition (run `slist`) and --account.
```

## The run

```bash
# 1. LOCAL (CPU) - generate the leg plan. Start with a SMALL window so the first
#    GPU run is cheap (3 min ~ a few hundred legs; 8 min ~ 670).
python -m renderer.physx_run extract --scenario braess_dev --seed 0 \
    --window-min 3 --out runs/physx/braess_plan.json
scp runs/physx/braess_plan.json gilbreth:~/laplace/runs/physx/

# 2a. GILBRETH (GPU) - SMOKE FIRST: does Isaac boot and do N robots fit? (~$0, minutes)
sbatch --export=ALL,PHYSX_MODULE=renderer.physx_probe,PHYSX_ARGS="--robots 12 --steps 600" \
    scripts/gilbreth/physx.slurm
#   check physx_<jobid>.out: it should print "[probe] RESULT {...\"fits\": true...}".
#   If the isaacsim.* import path errors, fix the NOTE-marked lines (older builds:
#   omni.isaac.core.*) before spending a drive job.

# 2b. GILBRETH (GPU) - the real drive run (defaults to braess_plan.json).
sbatch scripts/gilbreth/physx.slurm
#   check physx_<jobid>.out: "[drive] measured M/N legs ...". M close to N = good.

# 3. LOCAL (CPU) - pull the result and compute the headline.
scp gilbreth:~/laplace/runs/physx/braess_phys.json runs/physx/
python -m renderer.physx_run compare --plan runs/physx/braess_plan.json \
    --phys runs/physx/braess_phys.json --out runs/physx/braess_agreement.json
```

The last command prints, e.g.:
> *fast-sim and full-physics PhysX agree on per-leg travel time to within **X%** (mean abs
> error, CI90 ...%, n=... legs); systematic bias ...% (physics slower).*

## Before the number is publishable (the protocol that makes it mean something)

- **Calibrate + FREEZE `--max-accel`** on a *held-out* config first (PHYSX_VALIDATION.md B5).
  Otherwise "agreement" measures the tuning, not the abstraction. Pick the accel that makes
  PhysX match the DES on a non-Braess config, freeze it, *then* run Braess.
- **Report the direction honestly.** If physics *weakens* the Braess effect, that's a real
  result about the fast sim's abstraction, not a failure to bury.
- The `drive` half is GPU-unverified skeleton (like `physx_probe`); expect to fix a few
  Isaac-API lines on the first cluster run - that's why step 2a smokes it cheaply first.

## Troubleshooting Isaac-in-Apptainer

- **Boot hangs** -> EULA env not set. The script exports `ACCEPT_EULA / OMNI_KIT_ACCEPT_EULA /
  PRIVACY_CONSENT`; confirm they reached the container (`--env`).
- **Permission / read-only cache errors** -> a Kit cache path isn't bound to writable scratch.
  Add the offending path to the `--bind` list (the container's own README lists them; they
  drift between versions).
- **`CUDA`/driver mismatch** -> the container's CUDA must be <= the node driver; pick an Isaac
  tag compatible with Gilbreth's driver (check `nvidia-smi` in the job output header).
