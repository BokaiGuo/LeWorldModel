# CEM incumbent-retention diagnostic and physical replay

The old CEM implementation selects the best candidate in its final iteration,
not the best candidate across all iterations. With identical seeded candidate
generation, both archived final plans replayed exactly on host CUDA.

| Root | Best iteration (zero based) | Best predicted cost | Final-iteration cost | Original physical final qpos | Incumbent physical final qpos |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20263160 | 10 | 0.779444158 | 0.779566765 | 0.943406756 | 0.946565743 |
| 20263161 | 11 | 1.906306505 | 1.906306505 | 0.721175726 | 0.721175726 |

For root 20263160 the old selector did discard a lower model-cost sample; the
substitute's final physical qpos distance was worse by 0.003158987 under
actual-RGB suffix replanning. This is not evidence that incumbent retention
generally harms control: the model-cost difference is tiny, the resulting
prefix and observed successor differ, and only one known root changed.
For root 20263161, the entire raw action witness, observed successor context,
and reported physical metrics exactly match the archived run.

The diagnostic's best-ever rule uses the identical RNG stream, samples,
distribution updates and 192 initial candidate evaluations as the original
12-iteration CEM. Its physical runner executes the sealed first packet and
uses the original same-budget successor CEM. It charges the original planning
calls but does not include their wall time in its physical runner receipt.
This is a development replay, not a fair-budget certification, independent
test, or Planning Compiler gate result. Original outputs remain unchanged.

Four tests covering incumbent retention, invalid numerical costs, complete
plan preservation and horizon validation passed. They do not certify all
runtime cost accounting or establish a general CEM implementation defect.

Artifacts:

- `le-wm-main/results/reacher_pc_cem_incumbent_audit_20261102_v1/report.json`
- `le-wm-main/results/reacher_pc_cem_incumbent_physical_20261102_v1/report.json`
- `le-wm-main/experiments/spectral_reachability/reacher_pc_cem_incumbent_audit.py`
- `le-wm-main/experiments/spectral_reachability/run_reacher_pc_cem_incumbent_physical.py`
