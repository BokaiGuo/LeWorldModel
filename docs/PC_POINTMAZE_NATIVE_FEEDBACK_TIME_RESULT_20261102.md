# Native RGB feedback time-index development diagnostic

The native legacy PointMaze bridge was coupled to the frozen checkpoint on
host CUDA under `PC_POINTMAZE_NATIVE_FEEDBACK_TIME_PROTOCOL_20261102.md`.
Both branches choose the same first packet, execute it in separate simulator
instances, and use the actual raw post-step RGB at step 5 for replanning.
All four paired step-5 RGB hashes agree. Each method executes ten archive
actions. Simulator state was used for reset and separate offline physical
evaluation, not provided as a planner input.

| Exposed root | Remaining-horizon final position distance | Extended-horizon final position distance | Tail relation |
| --- | ---: | ---: | --- |
| 1880 | 0.266881 | 0.411040 | Different |
| 1881 | 0.209211 | 0.071525 | Different |
| 1882 | 0.842028 | 0.842028 | Same |
| 1883 | 0.028564 | 0.028564 | Same |

Distances are offline position-only replay labels from the separately audited
MuJoCo-2.2 backend, measured against the stored frame-10 position. The live
RGB bridge uses `mujoco-py` 2.1.2; same-action final-state equivalence between
those two backends was not freshly certified for these eight selected
trajectories. No native success criterion was measured. Thus these numbers
must not be represented as a native RGB success rate.

Remaining horizon wins one exposed root, extended wins one, and the other
two select identical tails. This small, shared-trajectory development panel
does not support a general winner or P1. It does show the complete
`frozen model → executable first packet → native post-step RGB → re-encode →
new model-selected tail → execution` engineering chain operating for this
specific experiment.

The methods have intentionally unequal budgets: 72 versus 128 prediction
samples and 136 versus 256 predicted block transitions per branch, plus two
branch-specific model encodes and one shared goal encode per root. Reported
elapsed time includes planning and live execution rather than a separately
certified complete cost ledger. The checkpoint receives a fixed raw zero
proprio input, not actual state; the normalizer was computed using the full
archive and its training-only provenance is unknown. Therefore this is a
missing-modality development ablation, not a fully certified RGB-only
benchmark, an equal-resource strong-baseline comparison, or independent
confirmation. Model parameter hashes were unchanged.

The bridge itself was checked on four fixed archive trajectories: initial,
step-5 and step-10 RGB images were byte-identical to the archive. Five
bridge tests plus one saved-decision/accounting test passed. These tests do
not establish unseen-action camera equivalence or planning efficacy.

- Runner: `le-wm-main/experiments/a0_pointmaze_native_feedback_time_20261102.py`
- Native bridge: `le-wm-main/experiments/pointmaze_native_rgb_bridge_20261102.py`
- Result: `le-wm-main/results/a0_pointmaze_native_feedback_time_20261102_DEV01.json`
- Result SHA-256:
  `0a94cb02813e916cdb5190e9a578fc0314a7e3e7a808479d5a4f0b92bc96a2d8`

Next certify same-action live endpoint equivalence, actual compute timing,
and the checkpoint's missing-proprio contract; then test genuine
dynamics-generated spectral programs against strong planners under fixed
resources on properly sourced independent roots.
