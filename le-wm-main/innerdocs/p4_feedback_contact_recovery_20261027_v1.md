# Six saved feedback-suffix branches: contact geometry recovery arithmetic

New read-only consumer, tests and this innerdoc only; old geometry/compiler/
physical modules, reports, source seals, tests and central progress are immutable.
Uses experiment-code's saved-real-data audit, provenance and test/report pattern;
not a new neural experiment and no figures claiming efficacy. No applicable
AGENTS.md was found in the workspace ancestor/source/test/doc paths in this line.

## Frozen definitions before reading outcomes

Call the sealed `p4_saved_contact_geometry_20261027_v1.contract` BEFORE reading
new outcome-bearing reports. Exact reviewed official env SHA, two real T polygons,
COG-aware Pymunk static Body.local_to_world, circle radius15 and documented
collision_slop0.1 are unchanged. The old geometry report/source seal is verified,
and its contract must equal the newly recovered contract. Source/extension hashes
are retained. No environment/Space, reset, step, renderer, encoding, predictor,
gradient, training, download or new action selection. Pymunk is used only for
unattached shape transforms. Thresholds never use task success or observed pose.

All per-step label fields come from the same old geometry.label. Positive surface
clearance is actual polygon/circle separation in native 512-window units, NOT
point-to-center distance or 224 RGB pixels. Near-contact uses <=0.1 merely as a
geometric proxy, never real collision/force/dynamic-contact ground truth.

The summary reports exact step5 geometry, each step6..10 geometry, and every
suffix snapshot transition from previous not-near to near. A prefix already near
and remaining near is NOT labelled reentry. If it separates and returns, the later
entry is retained. Lack of saved near-contact does not exclude contacts between
snapshots. force_contact/causal_push remain null regardless of block motion.
Original pose is norm(block_xy-goal_xy)+20*abs(wrapped angle difference), with
unchanged evaluator source/function hash bound. Per-step pose, exact10 minus
prefix5 pose, and exact10 minus original role's zero-suffix endpoint pose are
descriptive arithmetic, not fitted labels. No early endpoint substitutes for10.

## Required binding chain and all slots

Default input is the completed six-branch report at
`results/pusht_feedback_suffix_physical_20261027_v1_prepare_v1/report.json`.
Read physical/compiler/feedback protocol+prepare receipts, verify source/archive/
args seals, their protocol SHA and every declared file binding. Reuse only the
existing pure `physical.plans_from_saved`/compiler.verify_plan arithmetic: verify
all saved spectral candidates/costs/actions/argmins and six contexts against
original feedback; no reimplementation of gradient generation and no runtime
prepare/run import calls. The resulting six action plans must exactly equal the
physical protocol, including selection hashes, root, role, arm, Q, latent target,
original prefix, predictor context and expected prefix image.

Physical root goal, original initial state, history/goal RGB identities and first
five executed state/action receipts bind to original capability role_zero branch.
All three roles x imagined/actual are retained even for aliases/negative results.
Read each durable branch, physical_report and final report; require exact equality.
Check every journal SHA, attempts/returns/observations order, raw command, initial
snapshot, PNG SHA and exact step5 pixels/pixel-hash gate before suffix dispatch.
Partial/blocked rows stay partial; no fabricated step10 or restoration attempt.

Optional twelve saved step5/10 encoding slots bind to files, exact image receipts,
root and observed step. Independently recompute original Q costs, latent MSE and
projected MSE from saved vectors; do NOT run a new encode. Predicted and encoded
costs are reported separately from geometry and physical pose. Better prediction
fit/model score does not imply recovered physical contact, and recovered geometry
or object motion does not imply task benefit. Actual-minus-imagined pose is paired
by role only at full endpoints. No hypothesis test, independence or efficacy claim.

All files read are SHA-tracked and rehashed at exit; changed historical inputs fail
closed. Source seals certify bytes, not proof of runtime or a mechanism. Output
P1/P4 admission always false. Saved official success is not used to tune thresholds
or select branches. Existing goals are used only in the original arithmetic.

## Commands and lifecycle

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_contact_recovery_20261027_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_contact_recovery_20261027_v1 --report results/pusht_feedback_suffix_physical_20261027_v1_prepare_v1/report.json --output results/p4_feedback_contact_recovery_20261027_v1_main_v1
```

If the physical report is absent, return awaiting_physical_report/exit3 and create
no output. Never trigger upstream execution. Otherwise exclusively create a tagged
new directory, seal this module/tests/innerdoc and explicit arithmetic/source
dependencies before reading outcomes, then persist report.json with exclusive
fsync and source verification. Existing directories cannot be reused. Success
exit0 means saved audit completed, not scientific admission; integrity_fail exit2
retains the new failure artifact. Fixture tests are explicitly nonphysical and
nonneural, supplemented by actual saved-report regression once available.
