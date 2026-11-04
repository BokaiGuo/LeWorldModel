# Exact unicycle mechanism pilot

The fixed contract is in `PC_EXACT_UNICYCLE_DEV_PROTOCOL_20261102.md`.
This is a synthetic, exact-dynamics development panel, not frozen-LeWM,
robotics, independent-root, or gate evidence. The structured planner uses a
hand-specified five-angle bridge family and an analytical successor velocity
response. It is not yet the visual-model D2P compiler.

Across all 64 predetermined angular goals, 48 met the specified witness:
the first action had no immediate position improvement, increased the
goal-directed absolute velocity response compared with no turn, and improved
the best second-step terminal error over the zero-turn reference.

| Method | Mean final error | Structured wins / ties / losses |
|---|---:|---:|
| One-step greedy | 0.63677 | 48 / 16 / 0 |
| Random, 32 full sequences | 0.25959 | 34 / 0 / 30 |
| CEM, 4 x 8 samples | 0.32007 | 39 / 0 / 25 |
| Structured five-prefix program | 0.19389 | — |

Structured planning used 10 dynamics transitions, five terminal objective
evaluations and five analytic response evaluations per task. Random and CEM
each used 64 transitions and 32 objectives; greedy used two transitions and
two response evaluations. The separate diagnostic replay is not included in
those planner counts. This is a common transition upper bound, not equal
wall-clock, and response/Jacobian operations have not been normalized against
transition costs. The particular CEM has only two elites and four iterations;
this is not an exhaustive tuning or a strong-baseline comparison. Finite
differences confirm the analytical velocity response in the tests.

The result constructs a mechanism under known dynamics and exposes a
plausible test path; it cannot establish that the learned visual model encodes
the same effect or that D2P outperforms modern planners under equal total
resources. The 64 angle points are a deterministic grid, not 64 independent
real tasks. No statistical claim is made.

Artifacts:

- `le-wm-main/results/pc_exact_unicycle_development_20261102_v1/report.json`
- `le-wm-main/experiments/spectral_reachability/pc_exact_unicycle_development.py`
- `le-wm-main/tests/test_pc_exact_unicycle_development.py` — 3 tests passed.
