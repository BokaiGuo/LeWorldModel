# PointMaze planning-shaped repairability: V5 mechanism audit and V7 protocol

## Status

V5 is a completed, execution-grounded visual-history/action control.  Its
root-level router was **not authorized** on independent calibration roots, so
the only authorized outcome policy is cost-only.  The findings below are a
post-execution mechanism decomposition, not a new deployment claim.

The V5 audit artifact is
[`planning_shaped_repairability_mechanism_audit.json`](../le-wm-main/results/pointmaze_planwam_certified_repairability_fresh_v5_p2_20261021_cuda/planning_shaped_repairability_mechanism_audit.json).

V7 executed on a fresh CUDA panel after its ledger was refreshed from 34 to 38
prior contexts. Its runner artifact is
[`report.json`](../le-wm-main/results/pointmaze_planning_shaped_advantage_fresh_v7_router_20261022_cuda/report.json)
and its completed post-execution decomposition is
[`planning_shaped_repairability_mechanism_audit.json`](../le-wm-main/results/pointmaze_planning_shaped_advantage_fresh_v7_router_20261022_cuda/planning_shaped_repairability_mechanism_audit.json).
The independent certification split did **not** authorize routing (one-sided
upper delta `+0.00582` against the then-selected arm). The raw, frozen router
would have selected planning on 8/64 outcome roots and obtained mean regret
`0.01407` versus cost `0.01814`, but this is descriptive only: it cannot be
promoted to a deployment result after failed certification.

V7 also exposed a protocol implementation defect: its documented fail-closed
fallback was fixed cost, but the runner selected planning as a calibration
incumbent when routing failed certification. Thus its authorized held-out arm
was planning (`0.02090` regret), not the declared cost fallback. The immutable
V7 artifact is retained without modification; future panels now always fall
back to cost, and V9 additionally gates training/calibration/release on actual
candidate disagreement.

## S2--S5 candidate-bank feasibility boundary

The opportunity-router sequence isolates a separate failure that occurs before
any learned router is fitted.  S2's expanded tail-robust bank passed the frozen
headroom gate (q10 best/second gap `0.001260`, sensitive-root fraction
`0.7031`) and its raw routed outcome had a favorable direction, although its
independent certification correctly rejected deployment.  Three independent
follow-ups then rejected at static preflight without materializing a model
panel:

| Panel | Candidate construction | q10 top-2 gap | Sensitive-root fraction | Result |
| --- | --- | ---: | ---: | --- |
| S3 | Expanded tail-robust | 0.000270 | 0.4844 | rejected |
| S4 | Ultra tail-robust, 1536 proposals | 0.000749 | 0.4844 | rejected |
| S5 | Direction-stratified tail-robust, 384 proposals | 0.000327 | 0.4844 | rejected |

All three retained the fixed q10 gate of `0.001`, exact shared in-support
actions, and eight legacy-summary collision pairs.  S5 also retained large
q10 action-cost spread (`0.17781`), so simply increasing global action spread
or proposal coverage does not repair the lower-tail near ties.  The immutable
S5 ledger is
[`pointmaze_opportunity_stability_s5_static_preflight_20261022.json`](../le-wm-main/results/pointmaze_opportunity_stability_s5_static_preflight_20261022.json).

This rules out treating further candidate-bank searches as a learned-router
repair.  The next valid mechanism must change available information (for
example, a verified probe/re-observation) or test counterfactual geometry
without requiring the same legacy-summary collision family to manufacture
large tail oracle margins.

## What the V5 negative result actually contains

| Quantity | Calibration | Outcome |
| --- | ---: | ---: |
| Roots | 48 | 32 |
| Cost/planning candidate disagreement | 14 (29.2%) | 7 (21.9%) |
| Planning-shaped better, conditional on disagreement | 8/14 (57.1%) | 5/7 (71.4%) |
| Frozen router effective recall | 6/8 (75.0%) | 2/5 (40.0%) |
| Frozen router effective precision | 6/9 (66.7%) | 2/3 (66.7%) |
| Unreleased, genuinely improvable roots | 2 | 3 |

The central structural fact is that 25/32 outcome roots chose the same
candidate under both scorers.  Those roots are not a valid test of a
planning-shaped repair: changing only the router cannot change their action.
Within the seven roots where a decision can change, candidate-bank headroom is
substantial (mean cost regret `0.0929`), and planning-shaped scoring lowered
regret on five.  The bottleneck is therefore the prospective router and its
calibration-to-outcome transport, not evidence that future-effect supervision
has no decision signal at all.

This does **not** authorize the raw V5 routed outcome (`-0.00341` mean routed
minus cost regret): the independent calibration certificate rejected routing,
and outcome labels cannot be reused to change that decision.

## V7: magnitude-aware, fresh-root test

The V5 logistic router learns only whether planning shaped is strictly better.
It gives a tiny improvement and a large improvement equal target weight.  V7
pre-registers a different routing target on cross-fitted train roots:

\[
 A(x) = r_{\mathrm{cost}}(x) - r_{\mathrm{planning}}(x).
\]

The new `advantage_ridge` router predicts `A(x)` from the same lawful
decision-time inputs as V5: frozen visual history, declared goal, candidate
action, and frozen scorer/effect disagreement geometry.  It never reads raw
source state or calibration/outcome terminal labels as input.  Calibration
still selects and independently certifies the release threshold; failure
falls back to the fixed cost arm.

The CUDA-only launcher is
[`run_pointmaze_planning_shaped_advantage_v7_cuda.sh`](../le-wm-main/experiments/gdd_wm/run_pointmaze_planning_shaped_advantage_v7_cuda.sh).
It first creates a new independent state/action extension with roots
`10296..10807`, then uses disjoint `128/192/64` train/calibration/outcome roots
and 64 separate candidate-design roots.  Existing V5/V6 roots and outcomes
are not reused.  MuJoCo extension generation is deterministic data creation;
the renderer, scorer arms, and advantage-router training are CUDA-only and the
launcher refuses a CPU fallback.

V7 uses the full ordered 20-step action sequence for both scorer arms and the
advantage router. This is required because its frozen candidate bank
deliberately contains eight pairs that collide under the legacy
mean/std/first/last action summary. The historical V5 summary interface
remains unchanged; V7 records its ordered action input contract explicitly so
an action-aliasing null cannot be reported as a failure of planning-shaped
repairability.

Before the GPU build it writes a fail-closed ledger that binds the exact prior
context bytes, the new state pool, root and action-bank seeds, declared root
range, candidate support, collision stress, and calibration partition.  A
stale ledger cannot be silently reused with a different root split or action
bank.

V7 passes only if the calibration certificate authorizes the frozen routed
policy **and** its authorized held-out outcome regret is lower than fixed
cost-only.  If it fails, the result distinguishes lack of repair signal from a
router that remains non-transporting; it must not be converted into a new
post-outcome threshold.

## Execution hardening before the first V7 CUDA panel

Two implementation conditions were audited before V7 was permitted to run.

1. V7 now uses `full_ordered_20x2` not merely as a recorded contract but at
   every scorer call: scorer fitting, cross-fitted router-label generation,
   calibration, and outcome evaluation. Earlier code used ordered sequences
   for fitting but accidentally supplied the legacy summary feature at
   cross-fit/evaluation. That would either cause a feature-dimension failure
   or invalidate the intended action-aliasing stress test. A regression test
   now rejects this mismatch.
2. The V5/V7 launchers no longer treat `nvidia-smi`/NVML telemetry as a GPU
   execution condition. They still fail closed on the stronger project
   preflight, which requires visible NVIDIA device nodes, CUDA initialization,
   a tensor allocation, and (for V7) the exact cuDNN GRU kernel. Thus a panel
   cannot silently run on CPU, while an optional NVML failure cannot create a
   false experimental null.
3. The V7 runner independently revalidates that train, calibration, and
   outcome root identities are pairwise disjoint at its execution boundary.
   It materializes outcome terminals only after cross-fit scorer training,
   router fitting, calibration threshold selection, and the release
   certificate are frozen. This prevents an upstream split-assembly mistake
   from turning an observed initial condition or outcome label into apparent
   prospective routing evidence.

Neither change is an efficacy result. Both prevent an infrastructure or
action-representation artifact from being misreported as a negative result of
prospective repairability.
