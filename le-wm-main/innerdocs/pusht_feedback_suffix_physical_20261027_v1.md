# Six sealed feedback-suffix winners: physical consumer

This module consumes only the completed `pusht_feedback_suffix_compile_20261027_v1` report. It does not generate plans, call model.predict, train, reselect roots/actions, or modify any sealed source. All three original roles and both context arms are retained even when candidate IDs alias. Original qualified root index 1 is fixed; the other original roots are reconstructed solely to verify cohort identity/order, not scored for admission.

## Prepare contract

The actual compiler schema has no `gates` dictionary. Instead the consumer requires plans_sealed, failure=null, model_unchanged=true, partial_internal_work_unknown=false, 78 predictor calls/samples in 18 ordered context3/horizon1 scopes (2 compile +10 nonlinear candidate +1 independent replay per context), six completed Jacobians/1152 scalar VJP directions, 12 SVDs, zero new image encodes/physics steps. It calls the compiler's pure saved-plan verifier to reproduce the original spectral bank and requires independent winner replay. All 60 candidate commands, costs, stable argmins, original six contexts/prefixes and Q/goal are verified; only six winners are consumed, never reselected by outcomes.

Prepare verifies compiler/feedback source seals, protocols and input hashes, the original initial context/root RGB, feedback context files, and original root metadata via capability.prepare -> physical-v2.prepare. It exclusively creates a new output with protocol.json and prepare_receipt.json plus consumer/test/document/source archives. No simulator or neural work is executed by prepare; artifact/source verification is not efficacy evidence. Tests inject algebra fixtures and fake envs explicitly, never claim real physics or neural validation. Figures are intentionally omitted from this minimal consumer delivery.

## Physical execution contract

After main-thread review, run exclusively creates launch_receipt.json, reconstructs the original four roots using the original seed and matches their ordered fingerprints. For each of six predetermined winners, reset/warmup verifies original whole-stack RGB and goal RGB plus exact incoming raw packet history. Execute the exact saved 10x2 commands as ten separate guarded env.step calls: five original prefix commands, five winner suffix commands, no optimization or padding.

Every attempt is journaled before dispatch; returned steps are charged before rendering, so failures retain honest attempt/return counts. All step RGB PNGs and journal observations are persisted and fsynced. Crucially, after saving step5 observation, compare uint8 RGB pixels exactly to the original role's step5 PNG. A mismatch produces prefix_mismatch and stops BEFORE any step6 attempt. Neither state, phi nor outcome cost enters this gate; matching old RGB is a replay-integrity gate, not action or root selection. Failure/termination/visibility aborts retain partial trajectories and every planned branch slot. Maximum selected-path count is 60 returned steps; separate root reconstruction/reset/warmup is explicitly outside that count.

Official and pose metrics use unchanged `run_pusht_sampling_comparison_20261027_v2.official_checked` and `base.physical._pose_label` with original root goal. No new metric definition. Evaluator labels cannot affect action dispatch or branch order. Mismatch/partial endpoints must not be treated as full ten-step efficacy comparisons. P1_admission=false, efficacy_verified=false, significance_claim=false even on complete execution.

Optional `--encode-terminals` is frozen at prepare, not toggled mid-run: at most 12 CUDA image samples (step5/step10 per branch) using the existing exact-endpoint encoder after all six durable physical receipts exist. Partial paths never substitute early images for exact step10. Frozen model.eval/no-grad, TF32 disabled, before/after tensor-state integrity, model.predict explicitly forbidden; no CPU fallback. Persist per-slot failures rather than a fake success path.

## Commands

From le-wm-main, prepare only:

```bash
.venv/bin/python -m pytest -q tests/test_pusht_feedback_suffix_physical_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.pusht_feedback_suffix_physical_20261027_v1 prepare --planning-report results/pusht_feedback_suffix_compile_20261027_v1_prepare_v1/report.json --output results/pusht_feedback_suffix_physical_20261027_v1_prepare_v1 --encode-terminals
```

Only the main thread should authorize and execute the actual physical/CUDA stage:

```bash
.venv/bin/python -m experiments.spectral_reachability.pusht_feedback_suffix_physical_20261027_v1 run --output results/pusht_feedback_suffix_physical_20261027_v1_prepare_v1
```

Never reuse output directories or rerun after a partial launch. Inspect all six retained rows and input/source checks before interpreting the known-root calibration. This cannot establish independent P1 efficacy or a theoretical negative result.
