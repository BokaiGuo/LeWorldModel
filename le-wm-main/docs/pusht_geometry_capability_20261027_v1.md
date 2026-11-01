# Frozen real prefix capability diagnostic (2026-10-27 v1)

Implementation-only handoff: no simulator or CUDA experiment has been run for
this diagnostic. This is a C2 mechanism-gap measurement on known roots, not a
deployed policy, P1 admission, significance test, or all-P completion claim.
Only the matching module, nonneural tests and this document are new files.
Main prefix generation and the separate all96 physical consumer are unchanged.

## Fixed experiment

At runtime, require the complete `plans_sealed` report of
`pusht_geometry_prefix_20261027_v3`, nonempty passing gates, original source
bindings and source seals. Run the original v3 `verify_plans(report, source)`;
the physical consumer's existing `prepare()` is imported only at runtime to
recover original root/RGB metadata. There is no failed-v2 fallback and no read
of the new all96 consumer's physical outcomes. If v3 is not sealed or that API
is unavailable, stop and await upstream completion; do not patch its source.

Qualify roots ONLY where the geometry arm's saved selected ID differs from its
saved greedy ID. Retain original root order, at most four roots, no physical
label filtering. Execute the geometry selected, greedy, and `prefix_0` roles
independently, even when greedy or selected aliases zero. Candidate IDs and
aliases remain in the sealed plans. Prefix commands are the exact saved first
five commands; verify bounded tanh-logit agreement without reconstructing them.

Each prefix gets a fixed bank of 41 suffixes: all-zero logits and both signs of
each of 10 independent full-packet coordinates at eps .05 and .1. Raw suffix
commands are `tanh(logits).reshape(5,2)`. Every branch plans exactly five prefix
plus five suffix primitive commands. The zero suffix is NOT the model's fitted
tail. No physical or neural optimization/reselection occurs.

One qualified root: 123 branches / 1230 planned primitive steps. Four: 492 /
4920. Reset/warmup and cohort reconstruction costs are separate from branch
step-attempt/return counters. Runtime uses `durable.physical_branch` unchanged:
original root fingerprints, source RGB context, renderer check, fresh-RGB guard
before every command, fsynced attempt/event/PNG/branch receipts. Early stop,
missing observation, failed step and failed setup remain explicit; no blind
continuation, root replacement, terminal padding or early-endpoint substitution.

Before any CUDA allocation, persist all branch slots, `physical_report.json`,
and `endpoint_receipts.json` containing every exact step5/step10 slot (including
missing slots). Exact images are obtained with full-packet `image_receipt`;
journal and PNG digests are verified. An observed step10 image remains distinct
from a branch's guard/full-horizon success flag. Official and pose evaluation
at each actual endpoint are reported alongside actual step count, never
relabeled as fixed-H outcomes when early. Per-probe actual endpoints include
role/candidate/alias bindings for selected-versus-greedy/zero comparison.

## Response and fits

Optional `--encode-terminals` uses full-packet `encode_observations`, on HOST
CUDA only, frozen/eval checkpoint and unchanged-weight checks. `model.predict`
is forbidden; no new prediction, training, readout, or CPU neural fallback.
Prefix and terminal encoding attempts/failures are retained independently.
Saved model nominal-zero-tail vectors are only encoder comparison references;
they are NOT used as physical FD observations. Missing exact terminal encodings
leave real-Q measurements blocked. Physical-phi measurements need no encoder.

For each role and each epsilon, use exact step10 vectors:
`J[:,i]=(z_plus_i-z_minus_i)/(2*eps)`. Preserve paired minus/plus methods,
vectors and denominator, raw J, projected K=QJ, singular values, and both raw
and projected residuals. Q is the original 192-dimensional projection and the
target is the original compilation `goal_vectors[0]`, never a learned readout.

The separate fixed evaluator feature is
`phi(s)=[s[2],s[3],20*cos(s[4]),20*sin(s[4])]/512`, with the same transform on
the original simulator goal. It is a surrogate mechanism diagnostic, explicitly
NOT official success. State labels do not enter dispatch or root selection.

For BOTH spaces and BOTH epsilons fit the SAME exact bounded ridge solver:
`min ||r-Ku||^2 + .002||u||^2`, `||u|| <= sqrt(5)*.75`.
Own residual is target minus that role's zero-suffix terminal; common residual
is target minus the zero-role zero-suffix terminal, held fixed across all
roles/epsilons. For Q space both are projected by Q. Report fit parameters,
remaining mean-square cost, ridge penalty, total fit cost, norm and multiplier.
Fitted controls are never executed. Compare selected-minus-greedy and
selected-minus-zero own/common fit costs at each epsilon, plus response and fit
epsilon sensitivity. There is no rank-based P1 pass, oracle admission, inferred
official efficacy, independent-root generalization, or significance claim.

## Disk and execution barrier

Before output creation or simulator creation, reject a planned estimate above
1 GiB OR a post-estimate free reserve below 512 MiB. No override or deletion.
Estimate is 64 MiB global overhead plus 3 MiB per branch: conservative allowance
for eleven 224x224 RGB PNGs, journals, physical/encoding JSON and analyses.
One root estimates 433 MiB; four estimate 1540 MiB and therefore STOP even with
ample free space. Zero qualified roots produce no physical or CUDA execution.
Free space and cumulative output size are rechecked before each branch; a
late disk stop materializes remaining missing slots as space permits. This is
a planning guard, not a filesystem quota against unrelated concurrent writers.
Do not delete assets to pass it. Use a NEW output directory, isolated from the
main all96 consumer. No automatic scheduling or shared GPU lock is introduced;
the main task owns scheduling after review and upstream sealing.

## Exact launch (main scheduler only, after review)

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```bash
.venv/bin/python -m experiments.spectral_reachability.pusht_geometry_capability_20261027_v1 --planning-report results/pusht_geometry_prefix_20261027_v3_cuda_v1/report.json --output results/pusht_geometry_capability_20261027_v1_cuda_v1 --encode-terminals
```

Read-only preparation/disk preflight (no environment/model allocation): append
`--preflight-only`. This still requires a fully sealed upstream v3 report.

Nonneural contracts only:

```bash
.venv/bin/python -m pytest -q tests/test_pusht_geometry_capability_20261027_v1.py
```

Sources, test, document, imported execution/guard helpers and upstream planning
dependencies are snapshotted in runtime seals before root reconstruction or
branch execution. Input/protocol/source hashes are checked before physics,
before encoding and at exit. Exclusive writes prohibit reusing an output.

## Review freeze

Nonneural validation on 2026-10-27: **28 passed in 8.87s**, including the
existing sealed v3 MODEL report/compilation regression. This test does not call
consumer `prepare`, construct an environment/model, or read physical outcome
reports. Other execution tests use only a small fake environment contract.
No real physics or CUDA launch was performed. The separate all96 task and
main prefix sources were not modified.

SHA256 at handoff (runtime seals independently recheck these files):

- Module: `67125e09b109b06f4680fff2ac05a5144907f3a7c3b5dd39c9f9b2c9731f963e`
- Tests: `3f89c7b8811a595f603f3e00e3ec6b6ab554f19362f039812bc11397e55da26d`

Encoding-disabled or incomplete endpoint measurements are explicitly reported
as `partial_diagnostic_recorded`, with completed/expected response group counts.
Integrity failures remain `integrity_fail`; neither status grants P1 admission.
