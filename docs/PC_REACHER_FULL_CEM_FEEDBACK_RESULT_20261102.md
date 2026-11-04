# Full-horizon CEM with real RGB feedback: development result

The fixed protocol is in `PC_REACHER_FULL_CEM_FEEDBACK_PROTOCOL_20261102.md`.
Both pre-existing development roots ran on host CUDA with the same frozen
checkpoint, root and goal RGB identity checks, five-step first packet and
actual-successor RGB/action-history replanning as the guarded spectral run.
The first CEM planned three packets; its full plan is saved but only its first
packet was executed before replanning. The pre/post model hashes agree.

| Root | Method | Initial qpos distance | After first packet | Final |
| --- | --- | ---: | ---: | ---: |
| 20263160 | full CEM + feedback | 1.003500 | 1.290389 | 0.943407 |
| 20263160 | guarded spectral selected | 1.003500 | 1.037149 | 0.558267 |
| 20263160 | zero prefix + feedback | 1.003500 | 1.010031 | 0.546408 |
| 20263161 | full CEM + feedback | 1.182600 | 0.692145 | 0.721176 |
| 20263161 | guarded spectral selected | 1.182600 | 1.267141 | 0.619442 |
| 20263161 | zero prefix + feedback | 1.182600 | 1.245444 | 0.598310 |

Full CEM has higher final qpos distance than selected by about 0.385140
and 0.101734 respectively; selected also loses to zero on both roots.
The second CEM prefix makes immediate physical progress, so this is not a
reconfiguration-mechanism comparison. The roots were used in development and
this baseline was designed after other outcomes were visible. Two paired roots
do not support statistical or generalization claims. A single CEM seed and
configuration also do not establish the best possible CEM result.

The initial planner reports 192 rollout candidates / 576 horizon-weighted
prediction samples per root, and the successor planner 32 / 64. This is not a
certified equal-compute comparison: the guarded-spectral approach additionally
uses 192 VJP rows and its compilation and encoder costs are not fully metered.
Neither cost type is converted to the other. A physical zero counterfactual
also runs in the shared executor for diagnostic purposes.

Artifacts:

- `le-wm-main/results/reacher_pc_full_cem_feedback_development_20261102_v1/report.json`
- `le-wm-main/experiments/spectral_reachability/run_reacher_pc_full_cem_feedback_development.py`
- `le-wm-main/tests/test_reacher_pc_full_cem_feedback_development.py`

The test suite checked that only the first packet is returned to the executor
while the entire plan and nominal call count are retained. It does not
independently certify the physical pipeline or complete planning ledger.
