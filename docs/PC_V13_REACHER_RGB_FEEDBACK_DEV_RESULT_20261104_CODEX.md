# v13 Reacher RGB feedback: infrastructure-stopped development run

The new-root two-arm protocol was launched once. The program sealed its
manifest, then stopped at GPU preflight before loading the model, creating
the Reacher environment, collecting observations, or executing a planning
decision. The error reported that `/dev/nvidia*` devices were unavailable
inside the current sandbox. No physical control outcome or method comparison
was obtained; this is not a scientific negative result.

A separate read-only GPU allocation preflight succeeded outside the
restricted sandbox, using the declared NVIDIA device. It did not run or
retry the stopped experiment. The predeclared protocol prohibits retries
or replacing the root; the failure and partial manifest remain on disk.

The new Reacher adapter and shared-bank CEM interface passed three
parameterized synthetic-predictor tests for 15/10/5 horizons, including
carry and pair wiring. Those tests do not establish real Reacher feedback,
goal feasibility, source independence, or any PC gate.

Protocol: `PC_V13_REACHER_RGB_FEEDBACK_DEV_PROTOCOL_20261104_CODEX.md`.
Failure record and manifest:
`le-wm-main/results/pc_v13_reacher_rgb_feedback_dev_20261104_codex_failure.json`
and
`le-wm-main/results/pc_v13_reacher_rgb_feedback_dev_20261104_codex_manifest.json`.
Future new-protocol GPU work must check device and rendering permissions
*before* creating the run manifest; do not use this record as evidence
that either arm controlled the environment.
