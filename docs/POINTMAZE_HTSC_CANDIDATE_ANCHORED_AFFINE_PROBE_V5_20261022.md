# PointMaze HTSC candidate-anchored affine local repair V5

## Mechanism targeted

V4 replaces an underdetermined residual basis with an observed affine action
chart.  Its task-safe probes are nonetheless continuous, goal-relative pulses,
whereas the planner ranks a frozen discrete action bank.  A full-rank chart
therefore need not support any action that the planner can actually select.

V5 changes only the information-action geometry.  The active arm is
`effect_adapter_rgb_online_affine_candidate_anchored_verified_local`.

1. Construct the immutable 32-action root planning bank before probing.
2. Retain only bank actions with positive goal projection, positive
   calibration-dynamics progress, and calibration-frozen maximum magnitude.
3. Greedily choose up to the calibration-frozen probe cap with a
   ridge-stabilized affine D-optimality criterion.  A full-rank three-action
   subset is admissible when the magnitude cap leaves only three safe forward
   bank entries; V5 never adds a lateral or backward action merely to fill a
   quota.
4. Record each probe's exact `probe_candidate_index`; the strict raw-trace
   validator rejects any action that is not bitwise-close to that bank entry.
5. Retain V4's finite calibration-selected leave-one-out gate, convex-hull
   support requirement, successor verification, quarantine, and physical
   block budget.

This makes local identification decision-support aligned, but it does not
establish that a local chart will improve task ranking or closed-loop reward.

## Frozen CUDA run contract

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
bash experiments/run_pointmaze_htsc_candidate_anchored_affine_probe_v5_cuda.sh
```

The launcher requires CUDA preflight through `run_with_torch_cudnn.sh`, writes
only to `results/pointmaze_htsc_candidate_anchored_affine_probe_v5_20261022_cuda/`,
and refuses to reuse that path.  It holds the V4 calibration roots, effect
policy, candidate-bank cardinality, pulse cap, and HTSC release rule fixed.
It excludes all roots declared by the V6 source manifest and additionally
excludes V4 held-out roots if a V4 provenance artifact exists; that conditional
V4 exclusion is also materialized in V5's root-provenance artifact.

GPU eligibility is determined by the project's real-allocation preflight
(visible device nodes, CUDA initialization, and tensor allocation), not by an
NVML-only `nvidia-smi` query. This preserves fail-closed GPU-only evidence
while avoiding a false launcher rejection on runtimes where PyTorch CUDA is
available but optional NVML telemetry is not.

Because the active runner uses the recurrent visual stack, the launcher also
executes a cuDNN GRU smoke test *before* creating its immutable output
directory.  This catches a host library shadowing PyTorch's bundled cuDNN
before a failed startup can consume the V5 artifact path and make a lawful
retry appear to reuse a partial panel.

Before any held-out root is opened, V5 also requires calibration-only evidence
that at least 75% of gate roots can execute exact candidate-anchored probes,
and at least one such root has a rank/LOO/hull repairable affine chart. The
coverage threshold is an eligibility gate, not a performance selector. If it
fails, the runner stops with held-out roots sealed rather than producing an
uninterpretable no-probe null result.

## CPU geometry preflight (not a controller result)

`experiments.preflight_pointmaze_candidate_anchored_probe` was added to
separate the existence of a safe information action from a later failure of
the learned repair/release path. It derives the position, velocity, target,
and candidate bank from calibration RGB and a dataset-matched renderer; the
environment state is used only for reset and causal no-op warmup. It neither
loads the learned effect policy nor evaluates a controller.

Two immutable, root-disjoint preflight artifacts were produced on 2026-10-22:

| Roots | cap | exact-bank, safe rank-3 chart availability |
| --- | ---: | ---: |
| calibration-tail held-out roots 1816--1831 | 0.25 | 4 / 16 (25.0%) |
| calibration-tail held-out roots 1816--1831 | 0.50 | 15 / 16 (93.75%) |
| V22 initially proposed CUDA roots | 0.25 | 10 / 64 (15.625%) |
| V22 initially proposed CUDA roots | 0.50 | 55 / 64 (85.9375%) |

The V22 roots were observed for geometry only and are now explicitly excluded
from the eventual V5 CUDA efficacy panel. The preflight establishes neither
local repair, action ranking, nor downstream benefit. It does show that a
future null at the 0.50 cap cannot be attributed solely to lack of a safe,
candidate-supported probe action.

## Evidence and acceptance rule

The trace audit must pass before reading any outcome.  The identifiability
audit reports affine rank, candidate-hull coverage, exact candidate-bank probe
membership, `repairable_before_planning`, and actual override use.  These are
mechanism measurements, not performance claims.

V5 is a positive repair only if fresh held-out roots show all of the following:

- a passing raw-trace contract;
- candidate-anchored probes and an actually exercised local override;
- no held-out regression against HTSC fallback under the same physical budget;
- a paired downstream benefit beyond probe execution or chart identifiability.

Otherwise the result identifies a new boundary: even decision-support-aligned
local excitation is insufficient to repair the remaining planner error.
