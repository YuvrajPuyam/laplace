# renderer/ — WS6: the Omniverse digital twin (domain-extensible)

The twin is built in three layers so it can be **retargeted to a new domain**
(hospital, factory, airport, traffic grid…) by writing one small adapter — the
Isaac rendering code never changes.

```
  domain config ──▶  domain pack  ──▶  TwinScene JSON  ──▶  Isaac builder  ──▶  RTX render
  (Contract A,       (config →          (portable,          (USD stage from      (.png / .usd /
   warehouse)         TwinScene +        self-describing,    primitives+catalog,   later .mp4)
                      asset catalog)     domain-agnostic)    domain-agnostic)
        main python env  │                    │                 Isaac venv (D:\iv)
```

## The three layers

1. **`twin_scene.py` — the contract.** `TwinScene` = nodes, lanes, props,
   agents, cameras, floor bounds, **plus an embedded asset catalog**. Plain
   dataclasses, JSON round-trip, zero third-party imports. This is the only
   thing both environments share. Because the catalog travels inside the
   scene, the renderer is fully domain-agnostic — hand it a hospital scene and
   it renders a hospital.

2. **`domains/<name>.py` — a domain pack.** `to_scene(config) -> TwinScene`
   plus a `CATALOG` (type → asset). Runs in the main env; the warehouse pack
   imports the sim's navgraph so geometry can never drift from the simulator.
   **Adding a domain = a new file here + a catalog. Nothing downstream
   changes.** `domains/REGISTRY` lists them by name.

3. **`build_stage.py` — the Isaac builder.** Runs in the Isaac venv (`D:\iv`).
   `TwinScene JSON → USD stage` (floor, lane markings, station props, agent
   prims, lights, camera) → headless RTX still. Imports **no** sim and **no**
   domain code — only `twin_scene` + `catalog`.

`catalog.py` is the fidelity/swap point: v1 uses clean primitives (boxes with
PBR colors — fast on an 8 GB laptop, no downloads). Point a type at a `usd`
reference instead of a `prim` to drop in real NVIDIA warehouse/robot assets;
nothing else changes.

## Run it

```powershell
# main env: scenario -> portable scene.json (optionally with a config patch)
python -m renderer.export_scene --scenario braess_dev --out renderer/scenes/braess_baseline.json
python -m renderer.export_scene --scenario braess_dev `
    --patch renderer/scenes/shortcut_patch.json --out renderer/scenes/braess_shortcut.json

# Isaac venv: scene.json -> USD stage + RTX still
$env:OMNI_KIT_ACCEPT_EULA = "YES"
D:\iv\Scripts\python.exe -m renderer.build_stage renderer/scenes/braess_shortcut.json `
    --out renderer/out/braess_shortcut.png --camera congestion_closeup `
    --usd-out renderer/out/braess_shortcut.usd
```

`--usd-out` writes the `.usd` stage so it can be opened directly in Omniverse
USD Composer / Isaac Sim GUI. `--eye X Y Z --target X Y Z` overrides the camera
for framing without re-exporting the scene.

## Animated replay (robots moving through the twin)

Three steps, same two-env split. The trajectories come from a real rollout
event log via the replay-sufficiency rule (`sim/replay.py`), so what you see
is exactly what the simulator computed.

```powershell
# 1. main env: rollout -> per-frame AMR (x,y)+heading over a sim-time window
python -m renderer.export_tracks --scenario braess_dev `
    --patch renderer/scenes/shortcut_patch.json --seed 0 `
    --t0 45 --window 2.5 --n-frames 220 --out renderer/scenes/braess_tracks.json

# 2. Isaac venv: animate the agent prims, capture a PNG sequence
$env:OMNI_KIT_ACCEPT_EULA = "YES"
D:\iv\Scripts\python.exe -m renderer.build_stage renderer/scenes/braess_shortcut.json `
    --animate renderer/scenes/braess_tracks.json --frames-dir renderer/out/frames `
    --width 960 --height 540 --eye 0 -14 13 --target 7.5 15 0.5

# 3. main env: encode the frames to MP4
python -m renderer.encode_mp4 --frames-dir renderer/out/frames `
    --out renderer/out/braess_motion.mp4 --fps 30
```

The side-by-side Braess hero (baseline flowing vs. shortcut clogging on the
same seed) = run steps 1-3 twice (with and without the shortcut patch) and
composite — pending.

## To add a new domain (the whole checklist)

1. Write `domains/<your_domain>.py` with `to_scene(config) -> TwinScene` and a
   `CATALOG`.
2. Register it in `domains/__init__.py` `REGISTRY`.
3. `python -m renderer.export_scene --domain <your_domain> --scenario … --out scene.json`
4. Render with the **unchanged** `build_stage.py`.

A `TwinScene` can also be hand-authored as JSON with no domain pack at all —
useful for one-off scenes or non-simulated twins. Unknown asset types fall
back to generic primitives, so a scene never has a silent gap.

## Roadmap (this workstream)

- ✅ scene contract, warehouse pack, Isaac stage builder, RTX still
- ✅ event-log replay driver (animate agent prims over time → MP4 clip)
- ⬜ side-by-side Braess replay (baseline vs shortcut, same seed)
- ⬜ real USD assets via catalog `usd` refs (cloud GPU for the money shots) — **B**
- ⬜ path-traced lighting + final hero renders (cloud GPU) — **C**
- ⬜ PhysX validation pass for the finalist config (Tier-1, cloud GPU)
