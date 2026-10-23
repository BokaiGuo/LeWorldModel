# PointMaze V19: fixed visual-motion routing for prospective repairability

## Question

V17 and V18 established a C1 result in one controlled high-motion,
velocity-alias mechanism: a causal visual-history model can recover fixed-bank
action preferences that action-only effects cannot. That alone does not answer
the C2 question: can a controller decide, *before terminal outcomes are
observed*, when to invoke this repair and when to retain a simpler fallback?

V19 tests one deliberately narrow mechanism-routed policy. It is not a claim
that arbitrary learned routers solve repairability.

## Frozen mechanism and policy

The fresh pool contains 832 roots split into two mechanisms:

| mechanism | roots | source condition | declared policy |
| --- | ---: | --- | --- |
| high-motion alias | 448 | position-matched, opposed incoming velocity | recurrent visual-history repair |
| direct control | 384 | position-matched, zero source velocity | action-only fallback |

The routing statistic is computed only from the causal, renderer-derived
`source_history_visual_state` XY trajectory:

\[
g(h_t)=\mathbb{1}\{\lVert x_t-x_{t-35}\rVert_2\geq0.02\}.
\]

When `g=1`, V19 uses the recurrent arm; otherwise it uses action-only. The
threshold, direction, visual source, model width (`32`), contrast weight
(`1.0`), temperature (`1.0`), candidate family, and root split were fixed
before V19 calibration/outcome labels. Calibration may only authorize or
reject this complete policy.

The mechanism labels are construction/audit metadata. They are not model
features, routing inputs, candidate-dependent switches, calibration targets,
or terminal-outcome labels.

## Isolation and admission

The static ledger passed the following conditions before an encoder/model fit:

- 256 train, 256 calibration, and 256 outcome roots; each split has 128
  high-motion and 128 direct-control roots;
- 64 disjoint high-motion design roots choose the shared 16-action bank;
- all causal history markers are renderer-visible;
- every held action is source-supported and fresh against prior contexts;
- high-motion pairs retain velocity winner reversal and reject the
  position/goal/global-effect proxy;
- zero-motion controls retain matched source position and negligible source
  velocity;
- root, design, and renderer-chart identities are disjoint from historical
  contexts as recorded by hash.

The shared bank is selected on high-motion design roots only. Therefore V19
does **not** preclaim that the direct-control stratum has independent
candidate-bank headroom; its outcome report must expose this stratum rather
than pool it into a positive aggregate.

## Decision rule and success boundary

The primary comparator is action-only on the same root and same candidate
bank. A positive V19 result requires all of the following after the fixed
policy is written:

1. calibration's all-root paired bootstrap lower bound for routed minus
   action-only regret is above zero as defined in the frozen receipt;
2. the outcome result is evaluated without changing threshold, route, model,
   actions, or split;
3. the outcome audit reports the aggregate and both mechanism strata, rather
   than presenting only the high-motion subset.

V19 remains a C2 mechanism experiment even if successful. It does not prove
universal repairability, calibrated online deployment, contact transfer,
closed-loop MPC, or an HTSC guarantee. A future C3 test must independently
calibrate the frozen post-repair model on a new candidate/root distribution.

## Sealed outcome and boundary update

The host-CUDA run completed after the route was frozen and calibration had
authorized it.  Its all-root, root-clustered result on 256 untouched outcome
roots was:

| comparison | regret improvement | 95% paired bootstrap CI |
| --- | ---: | ---: |
| fixed route vs. action-only | 0.09589 | [0.07647, 0.11494] |
| recurrent vs. action-only, high-motion roots | 0.19179 | [0.16106, 0.22217] |
| recurrent vs. action-only, direct-control roots | 0.00563 | [0.00350, 0.00801] |
| high-motion minus direct-control repair gain | 0.18615 | [0.15510, 0.21677] |

This is positive evidence for the narrow mechanism claim: the predeclared
visual-motion statistic predicts where the recurrent repair has substantially
more value than action-only prediction.  It does **not** establish that the
fallback is better than always running recurrent repair.  In fact, the route
was worse than always-recurrent by 0.00282 regret, with CI
[-0.00404, -0.00172], because recurrent also produced a small benefit on the
direct-control stratum.  Thus V19 is evidence of repair-gain heterogeneity,
not a completed selective-controller result.  Any deployment claim needs a
predeclared compute/risk budget or a new mechanism where the fallback has an
independent advantage.

The original runtime receipt inherited the generic C1 boundary string from
its reusable V11 driver.  The driver now binds a V19-specific C2 boundary for
future runs; the immutable V19 result itself is interpreted through the
read-only C2 outcome audits below, rather than by rewriting that receipt.

## Artifacts

- `data/point_maze_v63_c2_mixed_velocity_repairability_20261023/mixed_velocity_repairability_manifest.json`
- `results/pointmaze_mixed_velocity_repairability_v19_static_preflight_20261023.json`
- `experiments/decision_benchmark/run_pointmaze_mixed_velocity_repairability_v19_cuda.py`
- `experiments/gdd_wm/preflight_pointmaze_mixed_velocity_repairability_v19.py`
- `results/pointmaze_mixed_velocity_repairability_v19_20261023_cuda/v19_outcome_root_cluster_audit.json`
- `results/pointmaze_mixed_velocity_repairability_v19_20261023_cuda/v19_outcome_root_cluster_audit_route_comparison.json`
