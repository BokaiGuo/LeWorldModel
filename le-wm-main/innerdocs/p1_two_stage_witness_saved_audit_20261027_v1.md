# Independent saved-array witness audit

No neural/physics execution, optimization, downloaded data, threshold selection
or changes to frozen source. The audit uses original prepared/source verification
only to bind inputs; its arithmetic checker never calls the producer's ridge,
bank generation, path runner, budget counter or neural backend.

## What is checked

- External terminal report SHA, original prepare SHA and source seals; exact
  ordered two original root/task identities and local report/journal SHA paths.
- All16 frozen prefix slots reconstructed from saved fullH2 root J/compile.
  All64 complete nonlinear paths use same original Q/goal, packet caps, float32
  tanh and normalized full-context chronology; actual Q MSE is recomputed.
- Each suffix192×10 J must bind its actual full predicted prefix context,
  paid zero-suffix nominal/AD forward, all192 saved reverse rows and own solve.
  Independent singular-system/normal-equation/KKT arithmetic checks the saved
  ridge result, scale choices, both radii and CPU bisection counters.
- Exact first-tie winner and six paid native packet/path receipts for fresh
  nominalH2/wholeH2/segmentedH2; all contexts/commands/cost agree with winner.
- Durable attempts/completions reconstructed independently and matched against
  both narrow/extended report ledgers, EACH stage and producer/witness budgets.
  Perroot120+6 predict/action_encoder,17J,3264 VJP,17SVD/solve. Two roots252
  new predict/action_encoder,34J,6528 VJP,34SVD/solve. Acquisition/direct costs
  remain separately inherited, not free or total-compute matched.

Numeric receipt tolerances are fixed float64 consistency tolerances, not tau,
plateau/escape thresholds or effect significance. Exact saved costs/commands/
context bindings remain exact. The audit reports unique prefix contexts and
suffix J arrays, pairwise Frobenius differences and differences vs initial H2
suffix J. It does NOT demand all16 J differ: repeated contexts/constant local
derivatives may legitimately coincide. Different J values alone do not prove
fresh graph execution or derivative correctness. Source-bound actual operation
receipts plus array consistency are not independent neural finite differences.

Full-H2 checks and partial failed roots are separate. Partial attempted counts
are retained as known work, never filled to252; unfinalized artifacts and unknown
work cannot become a full success. Missing terminal files are readiness gaps,
not negative scientific outcomes. The audit cannot replace rerunning actions
in physics, and contains no actual pose/official-task metric claims.

P1/model eligibility remain false. The reported continuing direct B1→B2
improvements are NOT direct stall. This candidate producer does not establish
finite-budget plateau, independent threshold authority or fresh confirmation;
causal repeated-frame padding is not observed-transition history.

## Execution after actual terminal, metadata/arithmetic only

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_p1_two_stage_witness_saved_audit_20261027_v1.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_two_stage_witness_saved_audit_20261027_v1 \
 --prepared results/p1_two_stage_witness_20261027_v1_prepared01 \
 --prepare-sha 1ade88fd784f2a1217ede182d8e46f99b3007b902406c97413f46ed70cc16106 \
 --actual results/p1_two_stage_witness_20261027_v1_cuda_main01 \
 --actual-sha EXTERNALLY_RETAINED_TERMINAL_REPORT_SHA \
 --output results/p1_two_stage_witness_saved_audit_20261027_v1_main01.json
```

No actual report hash is guessed before terminal. Output is exclusive NEW
small JSON (<1MiB), checks200MiB reserve and records three auditor source hashes.
Original report≤8MiB/journal≤16MiB are read only, never copied/rewritten.
Mock corruption tests validate the audit contract, not witness efficacy.
