# Same-action CUDA diagnostic: saved audit supplement

This supplements `codex_p1_null_same_action_error_handoff_20261030_v1.md`.
That earlier handoff's statement that there is no separate consumer is now
superseded by this new read-only saved consistency auditor.

After separately reviewed host-CUDA execution, record the runner's reported
output SHA in `REVIEWED_REPORT_SHA` and retain the independently reviewed
`REVIEWED_PREPARE_SHA`. From the project directory:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 .venv/bin/python -B -s -m experiments.spectral_reachability.codex_p1_null_same_action_saved_audit_20261030_v1 --report /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p1_null_same_action_error_cuda_20261030_v1_run_MAIN01.json --report-sha "$REVIEWED_REPORT_SHA" --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/codex_p1_null_same_action_error_cuda_20261030_v1_prepare_MAIN01.json --prepare-sha "$REVIEWED_PREPARE_SHA"
```

The auditor performs no neural or physical execution. It rebuilds the current
prepared sources, checks externally supplied SHA values, all 16 row identities,
action normalization and history ordering, steps 5/10/15, saved latent
dimensions and MSE/L2 arithmetic, model hash fields, scientific flags and
recorded call counts. Partial, inconsistent or malformed files fail nonzero.
The audit is a consistency check of self-reported values, **not** independent
proof that CUDA executed, that claimed model calls occurred, or that latent
values correspond to model execution. Independent provenance and execution
attestation remain required if those claims matter.

The underlying diagnostic still lacks a cross-entrypoint global lock, an
independent host/GPU exclusivity attestation, a sealed full runtime
environment/transitive dependency census, and actual CUDA compatibility and
resource measurements. Different output names can permit duplicate runs.
Main-thread review, independent history/deduplication decision and GPU
coordination remain prerequisites. Do not remove or retry partial outputs
without an explicit disposition. Known roots and duplicate logical rows are
not independent confirmation, and none of these checks grants P1 admission.
