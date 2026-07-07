"""FastAPI transport over the tool handlers (tools.md, episode lifecycle).

POST /episodes                       {scenario_id, question, budgets?, seed_base?}
POST /episodes/{eid}/tools/{tool}    params object -> result | error envelope
GET  /episodes/{eid}/trace           trace records (WS5 will add SSE)
GET  /health

Tool errors return HTTP 400 with the structured envelope — the agent reads
error.code, never the status line. Unknown episode/scenario is 404.

Run: uvicorn engine.api:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from .episode import Episode, RenderBackend
from .errors import ToolError
from .store import ScenarioStore
from .tools import dispatch

NOT_FOUND = ("unknown_scenario", "unknown_episode")


def _physx_reachable(addr: str | None, timeout: float = 1.0) -> bool:
    """Quick TCP probe of the PhysX stream address (host:port). Sync, so the sync
    /health endpoint (which FastAPI runs in a threadpool) never blocks the event
    loop. 'configured' (env set) and 'reachable' (this probe) are reported
    separately so the viewer can tell a missing config from a dropped tunnel."""
    if not addr or ":" not in addr:
        return False
    import socket as _socket
    host, _, port = addr.rpartition(":")
    try:
        with _socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def _live_terminal_event(result, error) -> dict:
    """Build the closing SSE event for a live 'ask' that produced NO report stage.
    A live episode must always end in something the user can read — never a silent
    halt. `result` is the runner's EpisodeResult (or None), `error` its error string."""
    # a genuine transport/agent error (not just "no report") -> show it as an error
    if error and "ended without an accepted report" not in error:
        return {"stage": "error", "kind": "failed",
                "title": "The agent hit an error", "detail": error}
    # otherwise it simply didn't reach a confident answer in budget — neutral, not a failure
    return {"stage": "report", "kind": "inconclusive",
            "title": "No confident recommendation",
            "detail": ("The agent finished without a confident recommendation — usually the "
                       "experiments it ran didn't show a clear enough difference, or it "
                       "reached its step budget first. Try rephrasing the question or "
                       "asking about a larger change.")}


def create_app(store: ScenarioStore | None = None,
               render_backend: RenderBackend | None = None,
               runs_dir: str = "runs") -> FastAPI:
    app = FastAPI(title="laplace-env", version="0.1.0")
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.store = store or ScenarioStore()
    app.state.render_backend = render_backend or RenderBackend()
    app.state.runs_dir = runs_dir
    # LRU-bounded so a long-lived engine can't leak memory one episode at a time;
    # evicted episodes' trace.jsonl / report.json stay on disk and the GET endpoints
    # fall back to reading them. EPISODE_CAP is generous — active episodes are touched
    # (move_to_end) on every access so only long-idle ones are ever dropped.
    app.state.episodes: "OrderedDict[str, Episode]" = OrderedDict()
    EPISODE_CAP = 256

    def error_response(e: ToolError) -> JSONResponse:
        return JSONResponse(status_code=404 if e.code in NOT_FOUND else 400,
                            content=e.envelope())

    @app.get("/health")
    def health():
        addr = os.environ.get("LAPLACE_PHYSX_ADDR")
        reachable = _physx_reachable(addr)
        return {"ok": True, "scenarios": app.state.store.ids(),
                "physx_configured": bool(addr),     # is a live PhysX address set?
                "physx_reachable": reachable,       # can we actually reach it right now?
                "physx": bool(addr) and reachable}  # deprecated alias: true only when truly usable

    @app.post("/episodes")
    def create_episode(body: dict):
        try:
            ep = Episode(
                scenario_id=body.get("scenario_id", ""),
                question=body.get("question", ""),
                store=app.state.store,
                budgets=body.get("budgets"),
                seed_base=int(body.get("seed_base", 0)),
                runs_dir=app.state.runs_dir,
                render_backend=app.state.render_backend,
                max_workers=int(body["max_workers"]) if body.get("max_workers") else None,
                config=body.get("config"),   # eval-harness inline config (held scenarios); None for public callers
            )
        except ToolError as e:
            return error_response(e)
        app.state.episodes[ep.episode_id] = ep
        while len(app.state.episodes) > EPISODE_CAP:     # bound memory; trace/report persist on disk
            app.state.episodes.popitem(last=False)
        ep.trace({"event": "episode_created", "scenario_id": ep.scenario_id,
                  "question": ep.question, "budgets": ep.budgets,
                  "seed_base": ep.seed_base})
        return {"episode_id": ep.episode_id, "baseline_hash": ep.baseline_hash,
                "budgets": ep.budgets}

    @app.post("/episodes/{eid}/tools/{tool}")
    def call_tool(eid: str, tool: str, params: dict):
        ep = app.state.episodes.get(eid)
        if ep is None:
            return error_response(ToolError("unknown_episode", f"no episode '{eid}'"))
        app.state.episodes.move_to_end(eid)              # mark active so the LRU never evicts it
        try:
            return dispatch(ep, tool, params)
        except ToolError as e:
            return error_response(e)

    @app.get("/episodes/{eid}/trace")
    def get_trace(eid: str):
        ep = app.state.episodes.get(eid)
        if ep is not None:
            app.state.episodes.move_to_end(eid)
            path = ep.dir / "trace.jsonl"
        else:                                            # evicted from memory -> read the persisted trace
            try:
                path = _episode_dir(eid) / "trace.jsonl"
            except ToolError as e:
                return error_response(e)
            if not path.exists():
                return error_response(ToolError("unknown_episode", f"no episode '{eid}'"))
        if not path.exists():
            return {"records": []}
        with open(path, encoding="utf-8") as f:
            return {"records": [json.loads(line) for line in f if line.strip()]}

    # --- live decision-twin (WS5): edit -> patch -> re-sim -> re-render -------
    def _twin_scene(body: dict) -> dict:
        from pathlib import Path as _P  # noqa: F401 (kept for symmetry)

        from renderer.export_tracks import sample_tracks
        from sim.config import apply_patch, fill_defaults, validate_config
        from ui.export_web_scene import build_scene

        sid = body.get("scenario_id", "")
        config = app.state.store.get(sid)
        if config is None:
            raise ToolError("unknown_scenario", f"no scenario '{sid}'")
        # station relocations: {station_id: node}
        for kind in ("pick", "pack", "charge", "dock"):
            for s in config["stations"][kind]:
                if s["id"] in (body.get("edits") or {}):
                    s["node"] = body["edits"][s["id"]]
        patch = dict(body.get("patch") or {})        # arbitrary dot-path patch (e.g. the agent's recommendation)
        if body.get("fleet") is not None:
            patch["fleet.amr_count"] = int(body["fleet"])
        if body.get("demand") is not None:
            patch["demand.arrival_rate_per_min"] = float(body["demand"])
        config = fill_defaults(apply_patch(config, patch) if patch else config)
        validate_config(config)                       # catch out-of-bounds edits
        warm = config["horizon"]["warmup_minutes"]
        tracks, hud = sample_tracks(
            config, int(body.get("seed", 0)), float(body.get("t0", warm + 30)),
            float(body.get("window", 6.0)), int(body.get("n_frames", 180)))
        scene = build_scene(config, tracks, hud)
        scene["metrics"] = {"throughput": hud["frames"][-1]["throughput"]
                            if hud["frames"] else None}
        return scene

    @app.post("/twin/simulate")
    def twin_simulate(body: dict):
        from sim.config import ConfigError
        try:
            return _twin_scene(body)
        except ToolError as e:
            return error_response(e)
        except ConfigError as e:        # a bad edit (off-grid node, out-of-range fleet/demand)
            return JSONResponse(status_code=400, content={"error": {
                "code": "validation_error",
                "message": "That change isn't valid for this facility.",
                "details": {"violations": e.violations}}})
        except Exception as e:  # noqa: BLE001 — reserve simulate_failed for genuine internal errors
            return JSONResponse(status_code=400, content={
                "error": {"code": "simulate_failed", "message": str(e)}})

    @app.post("/twin/command")
    def twin_command(body: dict):
        """Parse a typed operator COMMAND into an instant, validated edit.

        Returns {recognized: true, kind, summary, patch|edits} for a recognised canonical
        command (the viewer applies it to both layers immediately, then re-sims for the
        before/after). {recognized: false} means the grammar didn't match — the viewer routes
        it to the agent (a question, or phrasing the grammar doesn't cover: the hybrid path).
        A command that parses but yields an illegal config returns a structured validation_error."""
        from engine.commands import parse_command
        from sim.config import ConfigError, apply_patch, fill_defaults, validate_config

        sid = body.get("scenario_id", "")
        cfg = app.state.store.get(sid)
        if cfg is None:
            return error_response(ToolError("unknown_scenario", f"no scenario '{sid}'"))
        cmd = (body.get("command") or "").strip()
        if not cmd:
            return JSONResponse(status_code=400, content={"error": {
                "code": "missing_command", "message": "type a command to run"}})
        res = parse_command(cmd, cfg)
        if res is None:
            return {"recognized": False}            # let the agent handle it (question / complex phrasing)
        try:                                        # confirm the edit yields a legal config
            test = json.loads(json.dumps(cfg))      # deep copy
            for sid_node, node in (res.get("edits") or {}).items():
                for kind in ("pick", "pack", "charge", "dock"):
                    for s in test["stations"][kind]:
                        if s["id"] == sid_node:
                            s["node"] = node
            if res.get("patch"):
                test = apply_patch(test, res["patch"])
            validate_config(fill_defaults(test))
        except ConfigError as e:
            return JSONResponse(status_code=400, content={"error": {
                "code": "validation_error",
                "message": f"\"{res['summary']}\" isn't valid for this facility.",
                "details": {"violations": e.violations}}})
        return {"recognized": True, **res}

    @app.get("/twin/meta")
    def twin_meta(scenario: str):
        """Baseline params + layout summary for one scenario — NO simulation.

        Powers the viewer's pre-flight wizard so it can show real per-scenario
        defaults (fleet, demand, layout) before the heavy /twin/simulate call."""
        cfg = app.state.store.get(scenario)
        if cfg is None:
            return error_response(ToolError("unknown_scenario", f"no scenario '{scenario}'"))
        grid = cfg["layout"]["grid"]
        # station summary per type (id, node, slots) so the pre-flight can teach + show the
        # real facility before the run — the guided setup reads this, no simulation needed.
        stations = {kind: [{"id": s["id"], "node": s["node"], "slots": s.get("slots")}
                           for s in cfg["stations"].get(kind, []) or []]
                    for kind in ("pick", "pack", "charge", "dock")}
        return {"scenario": cfg["scenario_id"],
                "fleet": cfg["fleet"]["amr_count"],
                "demand": cfg["demand"]["arrival_rate_per_min"],
                "aisles": grid["aisles"],
                "cross_aisles": sorted(grid["cross_aisles"]),
                "aisle_length_m": grid["aisle_length_m"],
                "stations": stations}

    # --- staged-streaming decision twin (North Star S0-S3): ask -> stream ------
    def _episode_dir(eid: str) -> Path:
        if not eid or not all(c.isalnum() or c == "_" for c in eid):
            raise ToolError("unknown_episode", f"bad episode id '{eid}'")
        return Path(app.state.runs_dir) / eid

    @app.get("/twin/episodes")
    def twin_episodes():
        """Recorded decisions available to replay as a streamed report."""
        base = Path(app.state.runs_dir)
        out = []
        if base.exists():
            for d in sorted(base.iterdir()):
                tr = d / "trace.jsonl"
                if not (d.is_dir() and tr.exists()):
                    continue
                question = None
                try:
                    with open(tr, encoding="utf-8") as f:
                        question = json.loads(f.readline()).get("question")
                except (OSError, ValueError):
                    pass
                out.append({"episode_id": d.name, "question": question,
                            "has_report": (d / "report.json").exists()})
        return {"episodes": out}

    def _read_trace(tr: Path) -> list[dict]:
        """Robustly read a (possibly being-written) trace; skip partial last lines."""
        recs: list[dict] = []
        if not tr.exists():
            return recs
        try:
            with open(tr, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        recs.append(json.loads(line))
                    except ValueError:
                        pass                     # a line still being written
        except OSError:
            pass
        return recs

    @app.get("/twin/ask")
    async def twin_ask(replay: str = "", delay: float = 0.5, scenario: str = "",
                       q: str = ""):
        """Stream a decision as progressively-disclosed stages (SSE).

        REPLAY mode (?replay=<eid>) streams a recorded trace — no Max spend.
        LIVE mode (?scenario=<id>&q=<question>) runs the agent now and tails its
        trace into the SAME stage protocol — spends the Max window."""
        from engine.stages import trace_to_stages

        if replay:
            try:
                tr = _episode_dir(replay) / "trace.jsonl"
            except ToolError as e:
                return error_response(e)
            if not tr.exists():
                return error_response(
                    ToolError("unknown_episode", f"no trace for '{replay}'"))
            stages = trace_to_stages(_read_trace(tr))

            async def gen():
                for st in stages:
                    yield f"data: {json.dumps(st)}\n\n"
                    if delay:
                        await asyncio.sleep(delay)
                yield 'data: {"stage": "done", "t": -1}\n\n'

            return StreamingResponse(gen(), media_type="text/event-stream")

        # --- LIVE: run the agent and tail its trace --------------------------
        q = (q or "").strip()
        if not q:
            return JSONResponse(status_code=400, content={"error": {
                "code": "missing_question",
                "message": "live mode needs ?q=<question> (or ?replay=<episode_id>)"}})
        if len(q) > 2000:                           # reject, don't silently truncate (client caps too)
            return JSONResponse(status_code=400, content={"error": {
                "code": "question_too_long",
                "message": "Please shorten your question to 2000 characters or fewer."}})
        if app.state.store.get(scenario) is None:
            return error_response(ToolError("unknown_scenario", f"no scenario '{scenario}'"))

        runs = Path(app.state.runs_dir)
        box: dict = {}

        def _run_agent():
            try:
                from agent.runner import DEFAULT_MAX_TURNS, ClaudeAgentRunner
                runner = ClaudeAgentRunner(runs_dir=str(runs), max_turns=DEFAULT_MAX_TURNS)
                box["result"] = asyncio.run(runner.run(
                    question=q, scenario_id=scenario,
                    # attach to the EXACT episode this run creates (no glob race)
                    on_episode_start=lambda eid: box.__setitem__("episode_id", eid)))
            except Exception as e:  # noqa: BLE001 — surfaced as an SSE error stage
                box["error"] = f"{type(e).__name__}: {e}"

        threading.Thread(target=_run_agent, daemon=True).start()

        async def gen_live():
            yield ('data: ' + json.dumps({"stage": "plan", "kind": "starting",
                   "title": "Starting the agent", "detail": q, "t": 0}) + "\n\n")
            # attach to the exact episode the runner reports via on_episode_start
            eid, deadline = None, time.time() + 120
            while eid is None and time.time() < deadline and "error" not in box:
                eid = box.get("episode_id")
                if eid is None:
                    await asyncio.sleep(0.4)
            if eid is None:
                detail = box.get("error", "agent did not start in time")
                yield ('data: ' + json.dumps({"stage": "error", "kind": "failed",
                       "title": "Could not start", "detail": detail}) + "\n\n")
                return
            tr = runs / eid / "trace.jsonl"
            emitted, t0, report_seen = 0, time.time(), False
            while True:
                stages = trace_to_stages(_read_trace(tr))
                for st in stages[emitted:]:
                    yield f"data: {json.dumps(st)}\n\n"
                emitted = len(stages)
                if any(s["stage"] == "report" for s in stages):
                    report_seen = True                      # incl. a neutral 'inconclusive' report
                    break
                thread_done = "result" in box or "error" in box
                if thread_done and time.time() - t0 > 2:   # final read grace
                    break
                if time.time() - t0 > 1800:                 # hard cap
                    break
                await asyncio.sleep(0.6)
            if not report_seen:                             # never halt silently: emit a closing outcome
                term = _live_terminal_event(box.get("result"), box.get("error"))
                term["t"] = emitted
                yield "data: " + json.dumps(term) + "\n\n"
            yield 'data: {"stage": "done", "t": -1}\n\n'

        return StreamingResponse(gen_live(), media_type="text/event-stream")

    @app.get("/twin")
    def twin_page(scenario: str = "real_full_warehouse"):
        from pathlib import Path as _P

        from fastapi.responses import HTMLResponse
        tpl = _P("ui/viewer_template.html").read_text(encoding="utf-8")
        html = (tpl.replace("/*__SCENE__*/", "null")
                .replace("/*__BOOT__*/", json.dumps({"engine": "", "scenario": scenario})))
        # no-store: the template is re-read per request; a cached copy in the browser would
        # keep serving a STALE viewer after UI fixes (a plain F5 must always be current).
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    # --- live feed (Track L): stream real-time poses to the twin over a WebSocket ---
    @app.websocket("/twin/live")
    async def twin_live(ws: WebSocket):
        """Stream live robot poses to the viewer. Source today = local ORCA physics
        (real collision-avoidance motion, CPU); the Gilbreth PhysX stream swaps in behind
        the SAME frame contract. Poses go through the delayed-feed jitter buffer (Track L)
        so the playout is smooth and ready for a jittery cluster feed."""
        await ws.accept()
        scenario = ws.query_params.get("scenario", "braess_dev")
        try:
            n = int(ws.query_params.get("robots", "0")) or None
        except ValueError:
            n = None
        cfg = app.state.store.get(scenario)
        if cfg is None:
            await ws.send_json({"error": {"code": "unknown_scenario"}})
            await ws.close()
            return
        from sim.config import fill_defaults as _fd

        from engine.live_feed import SystemClock, physx_reader
        from engine.playout import Frame, JitterBuffer, PlayoutController, Relay
        from renderer.physx_stream import Fleet, step_mock

        relay = Relay(JitterBuffer(prune_margin=2.0),
                      PlayoutController(SystemClock(), delay=0.8))
        stop = asyncio.Event()
        link_state = {"s": "connected"}      # live-link health surfaced to the viewer (PhysX branch updates it)
        addr = os.environ.get("LAPLACE_PHYSX_ADDR")
        use_physx = ws.query_params.get("source") == "physx" and bool(addr)

        # control(msg): apply an operator change {fleet|demand|patch} to the live source so a
        # change made in the viewer takes effect on the SAME running twin (no rebuild/reboot).
        if use_physx:                               # real PhysX stream over TCP (bidirectional)
            host, port = addr.rsplit(":", 1)
            fq: queue.Queue = queue.Queue(maxsize=480)
            ctrl_q: queue.Queue = queue.Queue(maxsize=64)
            link_state["s"] = "connecting"
            threading.Thread(
                target=physx_reader,
                args=(host, int(port),
                      lambda fr: (not fq.full()) and fq.put_nowait(fr),
                      stop.is_set, ctrl_q),
                kwargs={"on_state": lambda s: link_state.__setitem__("s", s)},
                daemon=True).start()
            await ws.send_json({"hello": {"scenario": scenario, "source": "physx",
                                          "addr": addr}})

            def control(msg):                       # forward upstream to the Isaac stream
                try:
                    ctrl_q.put_nowait(msg)
                except queue.Full:
                    pass

            async def produce():
                while not stop.is_set():
                    for _ in range(16):
                        try:
                            fr = fq.get_nowait()
                        except queue.Empty:
                            break
                        relay.push(Frame(fr.get("t_sim", 0.0),
                                         {"agents": fr.get("agents", {}),
                                          "kpis": fr.get("kpis", {})}))
                    await asyncio.sleep(1 / 60)
        else:                                        # local stand-in: the SAME Fleet model, mock physics
            lf = Fleet(_fd(cfg), n or int(cfg["fleet"]["amr_count"]))
            await ws.send_json({"hello": {"scenario": scenario, "source": "orca_local",
                                          "robots": lf.ids[:lf.active]}})

            def control(msg):                       # apply in-process
                lf.apply_control(msg)

            async def produce():
                dt = 1 / 30
                while not stop.is_set():
                    step_mock(lf, dt)
                    fr = lf.frame()
                    relay.push(Frame(fr["t_sim"],
                                     {"agents": fr["agents"], "kpis": fr["kpis"]}))
                    await asyncio.sleep(dt)

        async def send():
            while not stop.is_set():
                s = relay.tick()
                if s.state is not None:
                    await ws.send_json({"t": s.playout_t, "status": s.status,
                                        "link": link_state["s"],
                                        "agents": s.state.get("agents", {}),
                                        "kpis": s.state.get("kpis", {})})
                await asyncio.sleep(1 / 15)

        async def _guard(coro_fn):
            """Run a producer/sender; if it dies, set stop so the peer task and the
            PhysX reader thread + TCP conn are released (no orphan thread/connection)."""
            try:
                await coro_fn()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — a broken feed must tear the whole WS down, not leak
                stop.set()

        prod = asyncio.create_task(_guard(produce))
        snd = asyncio.create_task(_guard(send))
        try:
            while not stop.is_set():
                txt = await ws.receive_text()       # operator control channel (fleet/demand/patch)
                try:
                    msg = json.loads(txt)
                except ValueError:
                    continue
                if isinstance(msg, dict):
                    control(msg)
        except WebSocketDisconnect:
            pass
        finally:
            stop.set()
            prod.cancel()
            snd.cancel()
            # await the cancelled tasks so their exceptions are retrieved (no "task
            # exception was never retrieved" warnings) and the reader thread is released
            await asyncio.gather(prod, snd, return_exceptions=True)
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 — already closed on the WebSocketDisconnect path
                pass

    return app


app = create_app(runs_dir=os.environ.get("LAPLACE_RUNS_DIR", "runs"))
