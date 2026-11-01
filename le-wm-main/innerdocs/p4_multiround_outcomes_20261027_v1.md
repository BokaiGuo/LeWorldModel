# Saved multiround outcomes and history audit

This new bounded arithmetic consumer reads the completed or failed
`p4_multiround_feedback_runtime_20261027_v1` report/journal without editing it.
No neural calls, environment constructors, physics steps, new roots, downloads,
action selection, threshold fitting, outcome gating or scientific admission.
The experiment-code skill informs the contract/tests/fail-closed logging; its
generic requests for new experiments/figures do not apply to this arithmetic task.

## Inputs and integrity

The official installed PushT evaluator file must match reviewed SHA
`c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`.
The original `_pose_label` function must match source SHA
`916d0979838e502720940aca24f8dc717ce256f0db2e3ffdeccb457f3912e1de`.
These contracts are checked before opening outcomes. There is no new metric.
Official labels reuse the existing pure `audit_pusht_sealed_task_contract.labels`;
the installed evaluator is not instantiated/called. Its peculiar angular
arithmetic is preserved, not silently replaced by a different wrapped metric.
Original pose remains object XY norm plus 20 times wrapped absolute angle.
Official position error includes agent and object; official state distance
includes all seven coordinates, including pusher velocity. Thus official
state-distance improvement need not imply object pose improvement.

Verify runtime protocol, prepare/launch receipts, all source seals, original
data/witness/weight hashes, initial two-arm/six-slot census, report/journal SHA,
event indices/time ordering and final census. RGB PNG and raw pixel hashes,
actual guard pixel counts, action authorization and attempts/returns/observations
are checked. Read bound: at most 32 MiB and 2000 journal events; source/data
reads use the existing bounded Inputs implementation and rehash at exit.

At each reached boundary5/10/15, reconstruct the three RGB references and two
executed raw action packets from that arm's saved observations. Recompute the
original float32 time-major action normalization, with no extra /100. Check
the actual arm chooses its own observed context, the imagined arm chooses its
own previous predicted successor, older observed latents agree with their own
prior RGB-boundary history at the original 2e-5 numerical tolerance, and the
original selected prefix matches pixels at5. The consumer does not reencode
RGB or prove saved latents through an independent neural replay.

Check selection envelope SHA, live context SHA, original Q/goal SHA, packet
history, full saved bank/candidates and stable nonlinear argmin using existing
`verify_plan` algebra only. Independent-replay receipts and state/action shifts
are checked as saved witnesses, not rerun. Every physical command must match
the bound prefix or chosen packet, and durable selection must precede dispatch.
Setup and selected physics accounting remain separate; historical predictor,
derivative and encoder attempts are reconciled with durable events. No claim
of equal total compute or independent reconstruction of internal kernel work.

## Complete versus partial endpoints

Each arm retains initial and all observed primitive-state original labels.
`exact20` is populated only for a completed four-packet,20-step arm without
guard abstention or execution failure. `last_observed` is separately retained
for guard-aborted/environment-terminated/other failed trajectories. A missing
RGB after a returned or attempted command is explicitly endpoint-unknown.
Even a guard failure on the step20 observation does not become a successful
complete-horizon receipt. Blocked/initialization-failed slots have no endpoint.
Only two certified complete20 endpoints enter actual-minus-imagined differences;
partial trajectories are never substituted, padded or pooled into that contrast.

`guard_aborted` is not `execution_or_integrity_failure`, neither is a theory
negative or P4-admission decision. A corrupted binding fails the entire audit
closed. A valid saved partial run can pass arithmetic integrity while retaining
its partial status. Official success/object-only success, XY/angle/pose and
official seven-dimensional distance are reported separately without cherry-picking.
No force/causal/contact inference, significance or independent generalization.

## Commands

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_multiround_outcomes_20261027_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p4_multiround_outcomes_20261027_v1 --report results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1/report.json --output results/p4_multiround_outcomes_20261027_v1_main_v1
```

Output directory is new/exclusive. Source/test/this innerdoc and reviewed metric
sources are archived before arithmetic analysis. Success produces report.json;
failure produces failure.json with no replacement of old reports. Synthetic
fixtures only exercise audit logic and partial classification, not efficacy.
