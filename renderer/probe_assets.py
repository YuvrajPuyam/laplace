"""probe_assets — ISAAC VENV: discover real warehouse/robot USD assets.

Boots Isaac headless just enough to reach the asset library, then lists the
.usd files under the robot and warehouse-prop directories so we can wire the
catalog to real assets (instead of guessing paths). Network is required the
first time (assets stream from NVIDIA content / local Nucleus cache).

  D:\\iv\\Scripts\\python.exe -m renderer.probe_assets
"""

from __future__ import annotations

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.client  # noqa: E402

from .build_stage import _assets_root  # reuse the resolver  # noqa: E402

root = _assets_root()
print(f"[probe] assets_root = {root}", flush=True)

# (dir, keyword filter | None). None = print all (capped).
DIRS = [
    ("/Isaac/Environments/Simple_Warehouse/Props", None),   # the warehouse's own SM_ props
    ("/Isaac/Props", None),                             # ALL prop subdirs (discover chargers etc.)
    ("/Isaac/Props/PackingTable", None),
    ("/Isaac/Props/Pallet", None),
    ("/Isaac/Props/Conveyors", None),
    ("/Isaac/Props/Forklift", None),
    ("/Isaac/Props/Sortbot_Housing", None),
    ("/Isaac/Props/Sektion_Cabinet", None),
]

if root:
    for d, kw in DIRS:
        url = root.rstrip("/") + d
        try:
            res, entries = omni.client.list(url)
            names = sorted(e.relative_path for e in entries)
            usds = [n for n in names if n.lower().endswith(".usd")]
            if kw:
                usds = [n for n in usds if any(k in n.lower() for k in kw)]
            subdirs = [n for n in names if "." not in n]
            print(f"\n[probe] {d}  ({res})", flush=True)
            if usds:
                for n in usds[:60]:
                    print("   usd: " + n, flush=True)
            if subdirs:
                print("   dirs: " + ", ".join(subdirs[:40]), flush=True)
            if not usds and not subdirs:
                print("   (empty or unreadable)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"\n[probe] {d}  ERROR {e}", flush=True)
else:
    print("[probe] no assets root — cannot list", flush=True)

print("\n[probe] done", flush=True)
app.close()
