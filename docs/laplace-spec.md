# Laplace — an agent that answers operational questions by running experiments

**Name:** `laplace` (after Laplace's demon — the intellect that, knowing a system's state and dynamics, computes the future rather than guessing it). Repo tagline:
*"LLMs guess at dynamics. Laplace runs the experiment."*

> "An intellect which at a certain moment would know all forces that set nature in motion … for such an intellect nothing would be uncertain." — Laplace, 1814

**Author:** Yuvraj Puyam · **Target:** v1 public before SIGGRAPH 2026 (19 July) · **Budget:** ~6 weeks of evenings + Claude Code agents

---

## 1. Problem statement

> Operational decisions about physical facilities — fleet sizing, layout, staffing, buffers — are answerable by simulation, but simulation requires a scarce expert to design, run, and interpret the experiments. So the vast majority of these decisions are made by intuition instead, and intuition is systematically wrong about stochastic dynamics (queueing nonlinearity, congestion migration, Braess-type paradoxes). Meanwhile, people have begun asking LLMs these questions because LLMs are accessible — and LLMs answer fluently with fabricated numbers, because language models cannot execute dynamics.

**Hypothesis (the project exists to test this):** the expert-in-the-loop role decomposes into (a) *computing dynamics* — which the simulator already does — and (b) *reasoning about experiments*: translating questions into designs, allocating budget, judging significance, interpreting results. Part (b) is orchestration and judgment, the class of work LLM agents became competent at. Therefore an agent wrapped around a simulator can deliver expert-grade answers at intuition-grade cost.

**Falsifiable form:** on held-out scenarios with computable optima:
- **H1 (grounding gap):** agent+sim decision accuracy ≫ LLM-alone accuracy.
- **H2 (judgment gap):** agent+sim ≥ naive grid search at *equal rollout budget*.
- **H3 (trustworthiness):** the agent's stated confidence is calibrated (90% CIs cover truth ~90% of the time).

If H1 is small, LLMs didn't need the sim. If H2 is small, agency adds nothing over brute force. Either result is reported honestly.

### Positioning vs. prior art (cite these in the README)

| Work | What it does | What it lacks (our lane) |
|---|---|---|
| Simulation Agent framework (arXiv:2505.13761) | LLM front-end for exploring sim models/outputs | No held-out benchmark, no ablations, no calibration |
| KG+LLM DES bottleneck analysis (arXiv:2507.17273) | LLM reasoning over DES *output* knowledge graphs | Post-hoc analysis only; agent does not design/run experiments |
| LLM-driven DES model generation (ScienceDirect 2026; Reider & Lang) | LLM *generates* sim models from text | Model authoring, not decision-making; no decision benchmark |
| Simio + Claude 3.5 case study (in MDPI Algorithms 18(9):573 review) | Proof-of-concept LLM↔DES integration | Case study, not evaluated framework |

**Our differentiators (all four, by design):** (1) held-out known-optimum benchmark with three-way ablation; (2) calibration as a first-class metric; (3) full visual observability in a 3D twin (Omniverse); (4) the environment exposed as an MCP server so *any* agent can be benchmarked on it.

---

## 2. System overview

```
                    ┌──────────────────────────────────────────────┐
 user question ───▶ │  AGENT (Claude, tool-use loop)               │
 (natural language) │  plan → design experiments → run → analyze   │
                    │  → follow-up hypotheses → recommend          │
                    └───┬──────────────┬───────────────┬───────────┘
                        │ tools        │ tools         │ tools
                        ▼              ▼               ▼
                ┌────────────┐  ┌─────────────┐  ┌─────────────────┐
                │ FAST SIM   │  │ STATS TOOLS │  │ RENDERER        │
                │ (Tier-0)   │  │ compare,    │  │ Isaac Sim 5.x   │
                │ headless   │  │ power, CI   │  │ headless replay │
                │ Python DES │  └─────────────┘  │ → evidence MP4/ │
                │ ~1000× RT  │                   │   PNG           │
                └────────────┘                   └─────────────────┘
                        │                               ▲
                        └── event logs ─────────────────┘ (replay)

                ┌──────────────────────────────────────────────────┐
                │ OBSERVABILITY UI (web): agent trace · config     │
                │ diffs in 3D · live rollout board · side-by-side  │
                │ replays · final report                           │
                └──────────────────────────────────────────────────┘
```

**Architecture principle:** *thin agent, rich engine.* The agent holds no warehouse knowledge; all intelligence it can't fake lives in tools. Every agent action must be representable visually (constrains the tool schema — see §7).

**Fidelity tiers:**
- **Tier-0 (required, authoritative for v1):** custom headless discrete-event/kinematic hybrid sim. Pure Python/NumPy. Deterministic given seed. Target ≥500× realtime per rollout on one CPU core; rollouts parallelize across cores via multiprocessing.
- **Renderer (required):** Isaac Sim 5.x headless (`SimulationApp({"headless": True})`, container deployment supported) replaying Tier-0 event logs on the stock warehouse USD scene. Renders evidence clips/stills. *No physics authority in v1.*
- **Tier-1 (stretch):** Isaac physics validation of finalist configs; enables the "agent manages fidelity budget" claim. Do not start before Week 5 gate passes.

---

## 3. The environment: parameterized warehouse

One environment family, fully specified. Everything the agent can change is in the config; everything else is frozen.

### 3.1 World model (Tier-0 sim semantics)

- **Topology:** aisles + cross-aisles as a directed navgraph. Nodes = waypoints/stations; edges have length, max speed, and capacity (max simultaneous AMRs; exceeding capacity queues entrants — this is the congestion model).
- **Stations:** `pick` (N slots), `pack` (M slots), `charge`, `dock`. Each station: service time ~ Lognormal(μ, σ), finite queue.
- **AMRs:** speed, capacity 1 task, simple battery model (drains per meter; below threshold → must visit `charge`). Routing policy: configurable — `shortest_path` | `congestion_aware` (penalize occupied edges).
- **Demand:** orders arrive ~ Poisson(λ), each order = pick at random pick-station → deliver to pack-station chosen by policy (`round_robin` | `shortest_queue`).
- **Task allocation:** nearest-idle-AMR (frozen in v1).
- **Determinism:** all randomness from a single `numpy.random.Generator(seed)`. Identical (config, seed) ⇒ identical trajectory. **Common random numbers:** the same seed produces the same arrival stream across *different* configs (arrival RNG seeded independently of layout RNG) — this enables paired statistical comparisons.

### 3.2 Config schema (FROZEN WEEK 1 — Contract A)

```json
{
  "schema_version": "1.0",
  "scenario_id": "baseline_small",
  "layout": {
    "grid": {"aisles": 6, "aisle_length_m": 30, "cross_aisles": [0, 30]},
    "extra_edges": [{"from": "A3_15", "to": "A4_15", "bidirectional": true}],
    "edge_overrides": [{"edge": "A3_15->A4_15", "capacity": 1, "one_way": false}]
  },
  "stations": {
    "pick":  [{"id": "P1", "node": "A1_05", "slots": 1, "service_lognorm": [2.1, 0.4]}],
    "pack":  [{"id": "K1", "node": "A6_28", "slots": 2, "service_lognorm": [2.8, 0.6]}],
    "charge":[{"id": "C1", "node": "A1_00", "slots": 2}],
    "dock":  [{"id": "D1", "node": "A6_30"}]
  },
  "fleet": {"amr_count": 4, "speed_mps": 1.5, "battery_capacity_m": 4000,
            "routing": "shortest_path"},
  "demand": {"arrival_rate_per_min": 3.0, "pack_assignment": "round_robin"},
  "horizon": {"sim_minutes": 480, "warmup_minutes": 30}
}
```

Rules: configs are expressed as **diffs against a scenario baseline** when the agent proposes them (`{"base": "braess_med", "patch": {"fleet.amr_count": 5}}`). The engine validates patches against a JSON Schema and rejects out-of-bounds values with a machine-readable error (the agent must handle rejection).

### 3.3 Rollout results schema (FROZEN WEEK 1 — Contract B)

```json
{
  "config_hash": "ab12...", "seed": 17, "schema_version": "1.0",
  "metrics": {
    "throughput_orders_per_hr": 41.2,
    "p50_order_latency_min": 6.1, "p95_order_latency_min": 14.8,
    "amr_utilization_pct": 71.0,
    "station_wait_p95_min": {"P1": 2.2, "K1": 7.9},
    "edge_congestion_top5": [{"edge": "A3_15->A4_15", "occupancy_pct": 88}],
    "deadhead_pct": 23.5, "charge_downtime_pct": 4.1
  },
  "event_log_uri": "logs/ab12_s17.events.parquet"
}
```

**Event log** (parquet): `(t, entity_id, entity_type, event, node/edge, payload)` rows. This single format drives: Tier-0 metrics computation, the renderer replay, the live rollout board, and side-by-side comparisons. *One log format, four consumers — do not fork it.*

---

## 4. The agent

### 4.1 Tool API (the agent's entire world)

| Tool | Signature (abridged) | Notes |
|---|---|---|
| `get_scene_summary` | `(scenario_id) → text + config` | Layout, stations, fleet, demand; no metrics |
| `propose_config` | `(base, patch) → config_hash \| validation_error` | All edits are patches; errors are informative |
| `run_rollouts` | `(config_hashes[], n_seeds, horizon?) → results[]` | Paired seeds across configs (CRN). Debits budget |
| `compare_configs` | `(hash_a, hash_b, metric) → {diff_mean, ci95, p, effect_size, n}` | Paired analysis (CRN ⇒ paired t / Wilcoxon) |
| `power_check` | `(observed_effect, observed_var, target_power) → n_required` | "How many more seeds do I need?" |
| `render_evidence` | `(config_hash, seed, t_range, camera_preset) → mp4/png uri` | Renderer queue; async; debits render budget |
| `get_budget` | `() → {rollouts_left, renders_left, tokens_used}` | Budget ledger is engine-enforced, not honor-system |
| `submit_report` | `(report_json) → done` | Terminates the episode; schema in §6.4 |

**Budget enforcement:** each question comes with a rollout budget (default 200) and render budget (default 4). `run_rollouts` beyond budget fails. This makes H2 (vs. equal-budget grid search) honest and forces real allocation decisions.

### 4.2 Agent loop (single agent, v1)

System prompt contract: (1) restate the question as a decision space; (2) plan experiments before running any; (3) never report a number that did not come from `compare_configs`; (4) every claim in the final report must cite config hashes + seeds; (5) confidence must come from CIs, not vibes; (6) you may stop early if `power_check` says the budget cannot resolve the remaining uncertainty — say so.

Loop: `plan → propose/run → analyze → (follow-up hypotheses | refine) → report`. Cap: 25 tool calls or budget exhaustion. Trace every step (see §7).

**Explicitly not in v1:** multi-agent debate, planner/worker split, learned surrogates, memory across questions. (Roadmap §10.)

### 4.3 Question classes (v1: two)

- **Class A — What-if (capacity/layout/policy):** "Is a 5th AMR worth it?" "Should we open a cross-aisle between A3 and A4?" "One-way aisles: better or worse?" Decision space: small discrete set of candidate patches.
- **Class B — Diagnosis:** "Throughput dropped 18% vs. last quarter — why?" Agent receives degraded scenario + baseline event logs; must isolate cause among planted candidates (service-time variance ↑, an edge capacity ↓, λ ↑) by designing *discriminating* experiments.

Class A ships Week 3; Class B Week 4. No third class in v1.

---

## 5. Evaluation (the spine — build the harness BEFORE the agent)

### 5.1 Ground truth

Each benchmark scenario defines a **discrete decision space** of K candidate configs (K ≤ 8). GT = exhaustive pre-simulation: every candidate × 200 paired seeds; the optimum is the candidate with best mean primary metric, required to be significant at p<0.01 vs. runner-up (scenarios are tuned until this holds — a scenario whose truth is statistically ambiguous is a broken scenario). GT sweeps are cached and versioned; the suite is **held out** from agent prompt-engineering (dev on a separate dev-scenario set).

### 5.2 Scenario suite (15 scenarios, 3 tiers)

| Tier | Count | Design intent | Examples |
|---|---|---|---|
| **T1 — Clear** | 5 | Large effect; sanity tier | Add AMR when utilization is 95% (yes); add pick slot at the obvious bottleneck |
| **T2 — Subtle** | 5 | Effect ≈ noise; tests statistical discipline | 5th AMR when utilization is 78% (marginal); pack policy swap with small true gain |
| **T3 — Trap** | 5 | The intuitive answer is wrong | **Braess:** opening the attractive cross-aisle shortcut *reduces* throughput (capacity-1 edge concentrates flow). **Utilization cliff:** +10% demand looks linear, p95 latency explodes. **Variance trap:** faster-mean/higher-variance station underperforms slower/consistent one. **Local fix, global hurt:** adding slots at station X starves Y. **More-is-less:** 7th AMR increases congestion past saturation |
| Class B (diagnosis) | +5 | Cause isolation | One planted cause among 3 candidates each |

Trap design rule: each T3 scenario must come with a one-paragraph *mechanistic explanation* of why intuition fails — this doubles as demo narration and proves the trap is principled, not adversarial noise. **The Braess scenario is the demo centerpiece; design it first and tune it until the effect is large (≥10% throughput drop) and robust.**

### 5.3 Metrics

| Metric | Definition |
|---|---|
| **Decision accuracy** | % scenarios where recommended config = GT optimum |
| **Regret** | (GT-optimum metric − recommended-config metric) / GT-optimum, mean over suite |
| **Calibration** | Coverage of agent's stated 90% CIs vs. GT values; report coverage % and ECE-style gap |
| **Cost** | Rollouts used, tool calls, tokens, wallclock per question |
| **Diagnosis accuracy** (Class B) | % correct cause isolated |

### 5.4 Baselines (the three-way ablation)

1. **LLM-alone:** same model, same question, scene summary provided, *no sim tools.* Must still output a recommendation + confidence. (Expected: confident, frequently wrong on T2/T3 — quantifies H1.)
2. **Grid search @ equal budget:** uniformly allocate the rollout budget across the candidate set; pick best mean. No LLM. (Quantifies H2 — does the agent's *allocation and follow-up* beat uniform spend? Honest possibility: on small discrete spaces grid search is strong; the agent should win on budget-constrained T2 and on Class B, where there is no grid.)
3. **Agent+sim (ours).**
4. *(Optional ablation if time: agent without follow-up loop — one round of experiments only.)*

Run every arm with 3 question paraphrases × 3 agent seeds; report mean ± sd. Results table is the README centerpiece.

### 5.5 Statistical protocol (engine-enforced)

Common random numbers (paired seeds) for all config comparisons; paired t-test (fallback Wilcoxon) via `compare_configs`; warmup minutes excluded from metrics; CI reporting mandatory in the final report schema. The harness — not the agent — computes the eval metrics.

---

## 6. Observability & deliverable surfaces

### 6.1 Agent trace panel
Stream every loop step: plan, tool call + args (pretty-printed patch diffs), tool result summary, agent's interpretation. Persist as `trace.jsonl` per question; UI renders live via SSE/websocket.

### 6.2 Config diff in the twin
Any proposed config renders as a 3D scene mutation vs. baseline: added elements highlighted (new AMR, opened edge), removed ghosted, changed parameters badged. Implementation: config → scene-build is a pure function (Contract A makes this free); diffs computed structurally from the patch.

### 6.3 Live rollout board + replays
Grid of running/queued rollouts with live counters (throughput, queue depths) fed from event logs; congestion heatmap overlay (edge occupancy). **Side-by-side synchronized replay** of two configs on the same seed is the hero evidence artifact (Braess: shortcut clogging left, baseline flowing right).

### 6.4 Report schema (`submit_report`)

```json
{
  "question": "...", "recommendation": "Do not open the cross-aisle.",
  "primary_metric": {"name": "throughput_orders_per_hr",
    "baseline": {"mean": 41.2, "ci90": [40.1, 42.3]},
    "recommended": {"mean": 41.2, "ci90": [40.1, 42.3]},
    "rejected_alternative": {"config": "open_crossaisle", "mean": 36.7, "ci90": [35.5, 37.9]}},
  "mechanism": "Shortcut concentrates flow onto a capacity-1 edge; occupancy 88%, queueing dominates saved distance.",
  "confidence": 0.93,
  "evidence": [{"type": "side_by_side", "uri": "...", "configs": ["base", "crossaisle"], "seed": 17}],
  "experiments": [{"configs": ["..."], "seeds": "1-40 paired", "tool_calls": 14}],
  "caveats": ["Holds for arrival rates 2.5–3.5/min; not tested beyond."]
}
```

UI renders this as the final report page (plots from results, embedded evidence clips).

### 6.5 Renderer notes
Isaac Sim 5.x headless standalone (`SimulationApp({"headless": True})`); container image `nvcr.io/nvidia/isaac-sim` runs headless-only — fine for the render queue. Stock warehouse USD assets. Replay = scripted kinematic playback of event logs (set AMR prim transforms per timestep; no physics). Camera presets: overview, congestion-closeup, follow-AMR. Budget per question: ≤4 clips ≤20 s. **If Isaac fights the schedule (Week 5 gate), fallback renderer = the Three.js board with polished visuals; Isaac becomes post-v1.** The benchmark numbers never depend on the renderer.

### 6.6 MCP server
Expose the §4.1 tool API as an MCP server (`laplace-env`). One day of work, big positioning payoff: the benchmark becomes runnable by any MCP-capable agent, and the repo reads as *agent environment infrastructure*, not a one-off demo.

---

## 7. Workstreams (for Claude Code execution)

Freeze Contracts A (config schema) + B (results/event-log schema) + the §4.1 tool signatures on Day 1–2, by hand, before any agent writes code. All workstreams build against contracts + mock data; integration risk concentrates at the contracts, which are small.

| WS | Scope | Depends on | Agent-buildable? |
|---|---|---|---|
| **WS1 — Core sim** | Navgraph, entities, event engine, congestion, CRN seeding, metrics from event logs, multiprocessing runner | Contracts | Yes, with your review of dynamics correctness |
| **WS2 — Engine API + budget ledger** | FastAPI service wrapping WS1; tool endpoints; validation; MCP server | Contracts | Fully |
| **WS3 — Agent loop** | Claude tool-use loop, system prompt, trace logging | WS2 (or mocks) | Fully |
| **WS4 — Eval harness** | Scenario definitions, GT sweep runner + cache, baselines (LLM-alone, grid), metrics, results tables | WS1, WS2 | Fully; **you design the T3 traps** |
| **WS5 — UI** | Trace panel, diff view, rollout board, replays, report page (React+Three.js — reuse SITL patterns: SSE state machine, no-build frontend if desired) | Contracts (mock data) | Fully |
| **WS6 — Renderer** | Isaac headless replay service + render queue | Event-log format | Scripts yes; **Isaac wrangling is yours** |

**Your non-delegable list:** contract design, sim dynamics validation (does congestion behave like queueing theory says it should — validate WS1 against M/M/c closed forms on degenerate configs as a unit test), T3 trap design + tuning, Isaac setup, judging whether agent traces show real reasoning, final numbers sign-off.

Suggested `CLAUDE.md` preamble for the repo: *"This project is built from `docs/laplace-spec.md`. Contracts in `schemas/` are frozen; propose changes via PR description, never silently. Every sim behavior change requires the M/M/c validation tests to pass. The eval suite in `eval/scenarios/` is held out: never tune prompts against it; use `eval/dev_scenarios/`."*

---

## 8. Schedule & gates

| Week | Deliverable | Gate (go/no-go) |
|---|---|---|
| **1** | Contracts frozen; WS1 core sim running; M/M/c validation passing | Rollout ≥500× realtime; CRN pairing verified |
| **2** | WS2 engine API + budget; WS5 board on mock data; **Braess scenario built & tuned** | Braess effect ≥10% and significant at n=40 |
| **3** | WS3 agent answers a T1 question end-to-end; trace panel live | Agent beats LLM-alone on 3 dev scenarios |
| **4** | WS4 full harness; GT sweeps cached; Class B diagnosis working | Three-way ablation runs unattended overnight |
| **5** | WS6 evidence renders; side-by-side replay; report page | **Renderer gate:** Isaac clips by Friday or invoke Three.js fallback |
| **6** | Held-out run → results table; README; 90-s video; MCP polish; QR one-pager | Repo public; video posted |

Scope insurance (pre-authorized cuts, in order): Tier-1 physics validation → Class B diagnosis (ship Class A only) → Isaac renderer (Three.js fallback) → scenario count 15→10. **Never cut:** the three-way ablation, calibration metric, trace panel, Braess scenario.

---

## 9. Definition of done (v1)

1. `laplace ask "Should we open a cross-aisle between A3 and A4?" --scenario braess_med` returns a report (JSON + web page) citing experiments, CIs, and evidence.
2. README leads with the results table: decision accuracy / regret / calibration / cost for LLM-alone vs. grid vs. agent, on the held-out 15-scenario suite.
3. The 90-second video: question typed → agent trace → twin mutating → rollout board racing → Braess side-by-side → report. 
4. MCP server documented; a third party can point their own agent at the benchmark.
5. Everything reproducible: `make gt-sweeps && make eval` regenerates the table.

## 10. Roadmap (the slide, not the build)

Tier-1 physics validation in Isaac → agent-managed fidelity budgets → real-facility adapter (reconstruction → config; the scan-to-sim bridge) → continuous mode (telemetry-synced twin, standing questions) → multi-agent review. 

## 11. Risks

| Risk | Mitigation |
|---|---|
| Tier-0 sim dynamics unconvincing | M/M/c closed-form validation tests; congestion model reviewed Week 1 |
| Grid search ties the agent on Class A (H2 weak) | Budget-constrained T2 scenarios + Class B (no grid exists for diagnosis); report honestly either way |
| Isaac schedule sink | Renderer is non-authoritative; Week-5 gate + Three.js fallback |
| Agent prompt overfit to eval | Held-out suite; dev scenarios separate; paraphrase × seed protocol |
| LLM API cost during eval | Budget caps; cache GT sweeps; eval arms ≈ 15×3×3 episodes ≈ manageable |
