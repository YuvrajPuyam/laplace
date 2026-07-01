# Ablation, held suite (3 scenarios, budget 60, 12 GT seeds)

| arm | decision accuracy | mean regret | calibration (CI cov.) | cost (rollouts) |
|-----|------------------:|------------:|----------------------:|----------------:|
| LLM-alone (no sim) | 1.00 | 0.000 | 0.67 | 0 |
| Laplace agent | 1.00 | 0.000 | 1.00 | 84 |
| grid search | 1.00 | 0.000 |, | 84 |
| OCBA (strong OR) | 1.00 | 0.000 |, | 84 |

*Accuracy = fraction of decisions matching the GT optimum. Regret = normalized score gap (0 = optimum). Calibration = coverage of the arm's 90% CI (- = arm states no CI). Cost = sim rollouts used (LLM-alone uses none). All graded by the harness from GT sweeps, never from arm self-reports.*

## Per-trap detail

### `braess_holdout`, GT optimum: **mid_shortcut**
> Should we open a mid cross-aisle connecting all aisles at their midpoint, or leave the layout with cross-aisles only at the ends? Choose the option with higher throughput.

*The shortcut HELPS here (relieves the end-detour), defying the naive Braess fear that extra connectivity congests. The sim is deadlock-free, so added edges relieve load rather than create a worse equilibrium.*

| arm | picked | correct? | regret |
|-----|--------|:--------:|-------:|
| LLM-alone (no sim) | mid_shortcut | ✅ | 0.000 |
| Laplace agent | mid_shortcut | ✅ | 0.000 |
| grid search | mid_shortcut | ✅ | 0.000 |
| OCBA (strong OR) | mid_shortcut | ✅ | 0.000 |

### `pool_packzone`, GT optimum: **pooled_1x4**
> Pack capacity is four server-slots. Distribute them as four 1-slot pack stations (one per aisle), or pool them into a single 4-slot pack station? Choose the layout with the lower p95 order latency.

*Pack-side pooling effect: M/M/4 at the pack beats 4x M/M/1, even though distributing looks more local. Tuned to the subtle regime so distributed is stable-but-worse, not an obvious collapse.*

| arm | picked | correct? | regret |
|-----|--------|:--------:|-------:|
| LLM-alone (no sim) | pooled_1x4 | ✅ | 0.000 |
| Laplace agent | pooled_1x4 | ✅ | 0.000 |
| grid search | pooled_1x4 | ✅ | 0.000 |
| OCBA (strong OR) | pooled_1x4 | ✅ | 0.000 |

### `pool_pickzone`, GT optimum: **pooled_1x4**
> Pick capacity is four server-slots. Should they be DISTRIBUTED as four 1-slot pick stations (one per aisle, for short travel), or POOLED into a single 4-slot pick station? Choose the layout with the lower p95 order latency.

*Pooling beats distributing: M/M/4 has far lower queue wait than 4x M/M/1 at the same load, and in a compact zone the extra travel to one station is small. The locality intuition (spread for short trips) is the trap.*

| arm | picked | correct? | regret |
|-----|--------|:--------:|-------:|
| LLM-alone (no sim) | pooled_1x4 | ✅ | 0.000 |
| Laplace agent | pooled_1x4 | ✅ | 0.000 |
| grid search | pooled_1x4 | ✅ | 0.000 |
| OCBA (strong OR) | pooled_1x4 | ✅ | 0.000 |


## Calibration, corrected after the identity fix (read this)

The earlier held "agent CI coverage 0/3" was largely a **config-identity confound**, not
overconfidence: the agent's CI was being scored against the GT of the *canonical* candidate while
the agent had measured a *byte-different* config it built itself. After anchoring the agent to the
canonical patches + a config-hash identity detector (mismatched rows excluded from coverage), and
adding the Winkler **interval score** (width + miss penalty; lower=better; identity-independent):

| arm | accuracy | CI coverage (like-for-like) | interval score | 
|-----|---------:|----------------------------:|---------------:|
| Laplace agent | 1.00 | 1.0 (scored 1/3; 2 identity-mismatch) | **4.5606** |
| LLM-alone | 1.00 | 0.67 (scored 3/3) | 20.644 |

**Honest read:** on the identity-independent interval score the agent (4.5606) is
far better-calibrated than the bare LLM (20.644), sharper intervals, closer to
truth. Coverage is only scored where the agent provably measured the canonical config; it complied
on 1/3 (the LLM has no tools, so it estimates the canonical option directly
and is scored on all 3). The clean completion (expert rec) is to grade the agent's CI against the
GT recomputed for the agent's OWN config on its own seeds, removing the confound for all 3 rows.
