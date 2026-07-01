# Laplace tool API (FROZEN) — the agent's entire world

These eight tools are the only interface between the agent and reality. They
double as the MCP server (`laplace-env`) tool definitions, so signatures here
are written exactly as they will be exposed. All requests/responses are JSON.
Errors are structured, never prose-only: `{"error": {"code": "...", "message":
"...", "details": {...}}}` — the agent is expected to read and recover.

Conventions: `config_hash` always means the 12-hex canonical hash (Contract
B.1). Budgets are engine-enforced per question episode; exceeding them returns
`code: "budget_exhausted"`.

---

## 1. `get_scene_summary`
**Purpose:** orient the agent. No metrics — the agent must not get performance
numbers without spending rollouts.

```
params:  { "scenario_id": string }
returns: {
  "summary_text": string,        // ~200 words: topology, stations, fleet, demand
  "config": <Contract A object>, // the full baseline config
  "config_hash": string,
  "editable_bounds": object      // machine-readable: dot-path -> {min,max} | enum
}
errors:  unknown_scenario
```

## 2. `propose_config`
**Purpose:** register a config variant as a patch against a base. The ONLY way
configs enter the system.

```
params:  {
  "base": string,                 // scenario_id or a prior config_hash
  "patch": { <dot.path>: value }, // e.g. {"fleet.amr_count": 5,
                                  //       "layout.extra_edges": [...]}
  "label": string                 // human-readable, shown in UI ("5 AMRs")
}
returns: { "config_hash": string, "diff_summary": string }
errors:  validation_error (details = JSON-Schema violations, per-path),
         unknown_base, patch_path_not_editable
```
Patch semantics: dot-paths replace values wholesale (arrays replaced, not
merged). Defaults from Contract A are filled before hashing — semantically
identical configs always hash identically.

## 3. `run_rollouts`
**Purpose:** spend budget to observe reality.

```
params:  {
  "config_hashes": [string],     // 1..8 configs
  "n_seeds": int,                // 1..100 per config
  "horizon_minutes": int | null  // override; null = config default
}
returns: {
  "results": [<Contract B.1 object>],  // n_configs × n_seeds entries
  "seeds_used": [int],                 // PAIRED: same seed list for every config
  "budget": { "rollouts_spent": int, "rollouts_left": int }
}
errors:  budget_exhausted (partial results returned for completed work),
         unknown_config
```
Seeds are allocated by the engine from the episode's seed sequence — the agent
never picks seeds (prevents cherry-picking, guarantees CRN pairing).

## 4. `compare_configs`
**Purpose:** the ONLY legitimate source of comparative numbers. The agent's
system prompt forbids reporting any number that did not come from here.

```
params:  {
  "hash_a": string, "hash_b": string,
  "metric": string               // canonical name from Contract B.1
}
returns: {
  "n_pairs": int,
  "mean_a": number, "mean_b": number,
  "ci90_a": [number, number],    // ABSOLUTE 90% CI of config a's metric (full per-arm
  "ci90_b": [number, number],    // across-seed variance) — use for an absolute metric claim
  "diff_mean": number,           // b - a
  "ci95_diff": [number, number], // CIs of the paired DIFFERENCE (CRN-narrowed) — NOT
  "ci90_diff": [number, number], // valid as an absolute bound on a single config
  "p_value": number,             // paired t (Wilcoxon fallback when n<15 or
                                 // normality clearly violated; method reported)
  "method": "paired_t" | "wilcoxon",
  "effect_size_d": number,
  "warnings": [string]           // e.g. "orders_abandoned >5% in hash_b —
                                 // throughput comparison may be invalid"
}
errors:  insufficient_pairs (need >=5 common seeds), unknown_metric
```

## 5. `power_check`
**Purpose:** budget rationality. "Can my remaining budget resolve this?"

```
params:  {
  "observed_effect": number, "observed_sd_of_diff": number,
  "target_power": number         // default 0.8, alpha fixed 0.05
}
returns: { "n_pairs_required": int, "achievable_within_budget": bool }
```

## 6. `render_evidence`
**Purpose:** produce visual evidence for the report. Async; non-authoritative.

```
params:  {
  "kind": "clip" | "still" | "side_by_side",
  "config_hashes": [string],     // 1, or 2 for side_by_side (same seed enforced)
  "seed": int,                   // must be a seed with a retained event log
  "t_range_min": [number, number],  // <= 20 sim-minutes for clips
  "camera": "overview" | "congestion_closeup" | "follow_amr"
}
returns: { "job_id": string }    // poll via get_budget; uri appears in
                                 // render job listing when done
errors:  render_budget_exhausted, log_unavailable
```

## 7. `get_budget`
```
params:  {}
returns: {
  "rollouts_left": int, "renders_left": int,
  "tool_calls_used": int, "tool_calls_max": int,
  "render_jobs": [{ "job_id": string, "status": "queued|done|failed",
                    "uri": string | null }]
}
```

## 8. `submit_report`
**Purpose:** terminate the episode. Validated against the report schema
(spec §6.4); a rejected report is returned with violations and the agent gets
one repair attempt.

```
params:  <report object, spec §6.4>
returns: { "accepted": bool, "violations": [string] }
```
Hard validation rules: every numeric claim must reference a `compare_configs`
call id or a results entry; `confidence` must be consistent with reported CIs;
`evidence` uris must be completed render jobs; recommending a config never
rolled out is a rejection.

---

## Episode lifecycle (engine-side)

```
POST /episodes {scenario_id, question, budgets?} -> {episode_id}
  ... agent tool calls, all scoped to episode_id ...
submit_report -> episode closed; trace.jsonl + report.json persisted
```
Default budgets: 200 rollouts, 4 renders, 25 tool calls. The eval harness
creates episodes programmatically; the CLI (`laplace ask`) and the MCP server
are thin clients over the same endpoints.
