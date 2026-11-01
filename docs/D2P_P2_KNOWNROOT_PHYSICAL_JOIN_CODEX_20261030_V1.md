# P2 known-root physical source join

CPU-only saved-data join completed for the four known-root, two-arm
bank-bound CUDA plans. The output is
`le-wm-main/results/codex_p2_knownroot_physical_join_20261030_v1_CPU01.json`
(SHA-256 `68fadebe7cce9fca1b3af33dec1813de11534c447b24d6f35044e07746e34a1e`).
It is retrospective engineering material, not preregistration, physical
execution, equal-compute evidence, or scientific admission.

The join independently consumes each root report and matches the planner's
complete latent context, goal vector, metric projection, root RGB hash, goal
RGB hash, and raw action domain to the historical physical source.
The historical recipe fixes seed 202753010, oversample factor 4, and 15 public
warmup steps per branch. Eight selected raw10 plans are exported in
counterbalanced order; neither selection nor ordering uses physical outcomes.

The usable historical factory and executor live in
`p2_composition_predict_frontier_physical_20261027_v1.py`: they construct
`swm/PushT-v1` with `render_mode='rgb_array'`, validate the environment
evaluator and RoyalBlue guard, collect and fingerprint four roots, and
reproduce per-branch warmup and RGB context before dispatch. However that
entry's prepare/validate code only accepts a different, fixed planner report.
The present join is **not** connected to that physical entry, and does not
authenticate the actual runtime factory/source closure or provide the complete
physical protocol it expects. Root collection, warmup, action dispatch,
termination, RGB bytes, and physical call accounting have not been executed or
verified for this join. No physical command is ready to run.

The earlier generic bridge prototype has been explicitly blocked at its public
physical entry. Its private fixture seam and generic saved consumer must not
be treated as a native execution or strict physical evidence. An integration
must instead construct the old physical entry's expected plan/protocol from the
new strict consumer, preserve exact raw action witnesses, seal the actual
physical code closure, enforce its budget and guards, and verify saved RGB and
partial failure handling. Public step caps in the historical contract are 120
warmup and 80 selected steps; internal simulator work and historical
model/loading costs are not established by this join.

Tests: `cd le-wm-main && .venv/bin/pytest -q -p no:cacheprovider
tests/test_codex_p2_knownroot_physical_join_20261030_v1.py
tests/test_codex_p2_knownroot_physical_bridge_20261030_v2.py` -> 8 passed.
No GPU/model inference or physics was run; P1/P2 admission remains false.
