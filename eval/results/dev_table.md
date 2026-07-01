# Ablation, dev suite (4 scenarios, budget 60, 24 GT seeds)

| arm | decision accuracy | mean regret | calibration (CI cov.) | cost (rollouts) |
|-----|------------------:|------------:|----------------------:|----------------:|
| LLM-alone (no sim) | 0.75 | 0.250 | 0.75 | 0 |
| Laplace agent | 1.00 | 0.000 | 1.00 | 228 |
| grid search | 1.00 | 0.000 |, | 228 |
| OCBA (strong OR) | 1.00 | 0.000 |, | 228 |

*Accuracy = fraction of decisions matching the GT optimum. Regret = normalized score gap (0 = optimum). Calibration = coverage of the arm's 90% CI (- = arm states no CI). Cost = sim rollouts used (LLM-alone uses none). All graded by the harness from GT sweeps, never from arm self-reports.*

## Per-trap detail

### `braess_dev`, GT optimum: **mid_cross_aisle**
> Should we open a mid cross-aisle (a shortcut connecting all aisles at the midpoint) to improve throughput?

*Whether the extra connectivity raises throughput or (Braess) congests the network is decided by the sim, not intuition.*

| arm | picked | correct? | regret |
|-----|--------|:--------:|-------:|
| LLM-alone (no sim) | no_shortcut | ❌ | 1.000 |
| Laplace agent | mid_cross_aisle | ✅ | 0.000 |
| grid search | mid_cross_aisle | ✅ | 0.000 |
| OCBA (strong OR) | mid_cross_aisle | ✅ | 0.000 |

### `dc_pickzone_med`, GT optimum: **amr_6**
> How many AMRs does this pick zone need to keep p95 order latency under 10 minutes at peak demand?

*The optimum is the knee: the smallest fleet that still meets the p95 SLA; more AMRs are wasted capital, fewer breach it.*

| arm | picked | correct? | regret |
|-----|--------|:--------:|-------:|
| LLM-alone (no sim) | amr_6 | ✅ | 0.000 |
| Laplace agent | amr_6 | ✅ | 0.000 |
| grid search | amr_6 | ✅ | 0.000 |
| OCBA (strong OR) | amr_6 | ✅ | 0.000 |

### `mfc_compact`, GT optimum: **amr_4**
> How many AMRs does this micro-fulfilment zone need to keep p95 order latency under 12 minutes?

*Smallest fleet meeting the p95 SLA in a single-block zone.*

| arm | picked | correct? | regret |
|-----|--------|:--------:|-------:|
| LLM-alone (no sim) | amr_4 | ✅ | 0.000 |
| Laplace agent | amr_4 | ✅ | 0.000 |
| grid search | amr_4 | ✅ | 0.000 |
| OCBA (strong OR) | amr_4 | ✅ | 0.000 |

### `real_full_warehouse`, GT optimum: **amr_6**
> How many AMRs does the scanned warehouse pick zone need to keep p95 order latency under 10 minutes?

*Fleet knee on the real extracted footprint. NOTE: demand is an uncalibrated placeholder, treat results as RELATIVE (A-vs-B), not absolute, until demand is calibrated.*

| arm | picked | correct? | regret |
|-----|--------|:--------:|-------:|
| LLM-alone (no sim) | amr_6 | ✅ | 0.000 |
| Laplace agent | amr_6 | ✅ | 0.000 |
| grid search | amr_6 | ✅ | 0.000 |
| OCBA (strong OR) | amr_6 | ✅ | 0.000 |
