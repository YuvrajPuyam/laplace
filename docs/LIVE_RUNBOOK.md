# Live decision-twin, runbook (Half A)

One command brings the whole live chain up; one command tears it down. Use the
**local mock** path to demo/develop without a GPU, and the **Gilbreth** path for
real Isaac/PhysX motion. The viewer, engine, and frame contract are identical in
both, only the source of motion changes.

```
Three.js viewer  <->  engine (uvicorn :8013)  <->  PhysX stream
                                                     mock:     local, kinematic (no GPU)
                                                     gilbreth: Isaac/PhysX on a GPU node,
                                                               reached over an SSH tunnel
```

## Quick start

```bash
make live            # local toy: mock stream + engine + viewer   (no GPU, no SSH)
# → open http://127.0.0.1:8013/twin?scenario=braess_dev  and click "Live"
make live-down       # stop everything
```

Real physics on the cluster:

```bash
make live-gilbreth   # submits the Isaac stream job, opens the SSH tunnel, starts the engine
make live-down       # scancels the job, closes the tunnel, stops the engine
```

Under the hood these call `scripts/live/up.sh [--gilbreth] [--scenario <id>]` and
`scripts/live/down.sh`.

## What "up" does, hop by hop (and how to tell which hop broke)

`up.sh` starts each hop, then **gates on a health check** before declaring success.
Every PID/JOBID it starts is written to `.laplace_live.state` (git-ignored) so
`down.sh` reverses exactly what was started, never a broad `scancel`.

1. **Stream**, `mock`: `renderer.physx_stream --mock` locally. `gilbreth`: `sbatch`
   the stream job, wait for it to print `READY`, then `ssh -N -L` a tunnel to its node.
   - broke? → `runs/_live_stream.log` (mock) or `runs/_live_tunnel.log` + `physx_<JOBID>.out` on Gilbreth.
2. **Engine**, `uvicorn engine.api:app` on `:8013` with `LAPLACE_PHYSX_ADDR` pointed at the stream.
   - broke? → `runs/_live_engine.log`.
3. **Health gate**, polls `GET /health`:
   - `"ok"` true → the engine is up;
   - `"physx_reachable"` true → the stream/tunnel is actually reachable (not just configured).
   If `physx_configured` is true but `physx_reachable` is false, the engine is fine and the
   **stream/tunnel** is the problem (hop 1).

The viewer's **Live** button enables only when `physx_reachable` is true, and the live
badge turns green only after the first real frame, so it never lies about the feed.

## Configuration (env overrides)

| var | default | meaning |
|-----|---------|---------|
| `LAPLACE_ENGINE_PORT` | `8013` | port the viewer/engine is served on |
| `LAPLACE_STREAM_PORT` | `8765` | PhysX stream TCP port (local and remote) |
| `LAPLACE_GILBRETH_HOST` | `gilbreth.rcac.purdue.edu` | SSH host (resolves user via `~/.ssh/config`) |
| `LAPLACE_HOI` | `/scratch/.../HOI/laplace` | the repo checkout on Gilbreth |
| `LAPLACE_SLURM_PARTITION` / `LAPLACE_SLURM_ACCOUNT` | `a30` / `csml` | GPU queue + allocation |
| `LAPLACE_SIF` | `$LAPLACE_HOI/containers/isaac-sim-5.1.0.sif` | Isaac Sim container |

## Notes
- The same `scripts/gilbreth/physx.slurm` dispatches both this live stream and the offline
  ρ\* drive job (via `PHYSX_MODULE`/`PHYSX_ARGS`); partition + account are passed on the
  `sbatch` command line, never hard-coded.
- If `up.sh` refuses to start because `.laplace_live.state` exists, a session is already
  running (or the file is stale), run `make live-down` first.

## Design note, the live feed is an intentional, infra-bounded choice

Streaming real Isaac/PhysX motion live from a SLURM cluster over an SSH tunnel is the
single biggest source of moving parts here (reconnect-with-backoff, per-hop health, the
one-command bring-up all exist to keep that link healthy). We kept it on purpose:

- **The live twin is the point.** Watching the LLM operate a real, physics-driven facility
  in real time is the experience this project is built to show, that "wow" is worth the
  operational surface.
- **It's the best achievable with the available hardware.** The dev box is below Isaac's
  minimums, so high-fidelity PhysX has to run on Gilbreth and reach the viewer over a tunnel.
  Given that constraint, the hardened reconnect + health + bring-up path is the clean answer,
  not over-engineering.
- **It degrades gracefully.** `make live` (the local mock loop) gives the full experience with
  zero cluster/SSH dependency, so a demo never hinges on the tunnel being up.
- **The fidelity-critical work does NOT depend on this link.** The DES-vs-PhysX agreement /
  ρ\* validation (Half B) runs PhysX offline as a batch job, benchmark numbers never ride
  the live stream (per CLAUDE.md: the renderer/live path is non-authoritative).

In short: live cluster-PhysX is a deliberate experience choice, isolated from the parts that
have to be exactly reproducible. If the infra bar drops (local Isaac-capable GPU), the same
frame contract lets the stream move on-box and most of this transport machinery retires.
