# Same-bank conditional composition versus spectral-only: development bridge

This document specifies a CPU-testable development bridge, not permission
to run CUDA, physical trials, or admit P1/P2. It was authored after B0 and
the prior repeated-control outcomes were known; the B0 asset is not a new or
independent root.

## Existing work and new mechanism

The earlier same-bank common optimizer is a repeatability control: both arms
start from the same saved candidate and call the same objective; labels alone
do not implement a composition ablation. The historical conditional-suffix
code does compile a suffix from its predicted prefix context, but uses a
different 40-candidate contract and no shared eight-update optimizer. This
bridge combines the relevant operations under a new development contract.

The input is the verified isolated B0 handoff's fixed 16-candidate
`joint_multistart` bank, in index order, with the same SHA-bound B0 context
and goal. The bank was generated using full-horizon local spectral directions;
the control is **fixed joint-spectral candidates plus common optimization**,
not the separate historical 40-candidate no-composition SRA algorithm.

For every candidate:

1. The control keeps its entire 10x2 float32 action as the initializer.
2. The composition arm takes precisely the same first 5x2 raw actions,
   predicts the complete intermediate latent *and action history*, and
   linearizes the terminal 192-vector with respect to the original 5x2 raw
   suffix at that intermediate context. A damped ridge solve, damping 0.002
   and L2 update radius 0.5, proposes a suffix update; the result is projected
   to the raw [-1,1] box. Prefix bytes remain unchanged.
3. Each initializer enters the existing same eight-step projected gradient
   optimizer (step 0.05, nine scored iterates, earliest tie) using the same
   frozen-transition terminal MSE objective and gradient callback. The
   optimizer may alter both packets after initialization; thus this measures
   a **conditional-composition initialization**, not a constraint that the
   final program preserve the initial prefix. Zero suffix changes are legal
   when the Jacobian or residual entails them.

The bridge uses raw suffix linearization and an update ball. It is not a
byte-for-byte reproduction of the historical compiler's zero-logit
linearization, atanh-space joint ball, or 40-candidate schedule. These choices
were made after B0 outcomes and require a separately frozen plan before any
future real execution.

## Accounting and failure boundary

For a complete schedule the common optimizer requests per arm 144 objective
scores, 128 gradients and 288 packet transitions. Composition initialization
adds 16 prefix transitions, 32 suffix forward transitions (nominal and
Jacobian forward), 16 192-output Jacobians and 16 bounded ridge/SVD solves;
the Jacobian API requests 192 output directions per candidate (3072 total).
These are dispatch/algorithm counts, not measured backward kernels, FLOPs,
energy, I/O or equal total compute. Shared bank generation is inherited;
its source cost is not re-executed or matched. The ledger records attempts,
returns and failures, including the full intermediate-context digest and
initial-action hashes. Its historical `composition_inherited` category name
is reused for **new** conditional compilation events in this bridge.

On a failed dispatch the exception propagates; no candidate is skipped or
retried and earlier ledger records remain. This is an in-process bridge:
the caller owns exclusive output creation, fsync ledger lifetime, partial
failure report, per-stage cost caps, locked model loading, source/runtime
sealing, model-entrypoint hooks and independent final result consumption.
The current native `CappedMeter` accepts only common-optimizer categories;
it cannot directly meter model hooks during the new compilation category.
A routed instrumented model meter and a full source-bound host runner,
including postchecks and a no-overwrite protocol, are still required.
Do not treat this interface as a CUDA-ready authorized command.

The fixed-bank loader checks the saved handoff and pair bytes and the
previous isolated-consumer proof. It is not by itself a complete live
authentication of every historical transitive dependency. In the current
worktree an additional old SRA-contract test fails closed because the
historically bound shared file `run_reacher_d2p_p0_goal_progress.py` has
changed. This bridge does not modify or restore it. Resolve that source
boundary through an independently authenticated read-only historical
environment before considering a real run.

## Runnable CPU contract

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```sh
.venv/bin/python -B -m pytest -q -s -p no:cacheprovider tests/test_p2_same_bank_conditional_suffix_ablation_20261102_exclusive.py tests/test_p2_same_bank_native_objective_20261102_exclusive.py
```

Actual CPU validation: **10 passed**. The fixture uses the real fixed
candidate bytes and context/goal with a synthetic differentiable transition;
16/16 compiled initial actions differed from their same-candidate controls.
The tests also check retained prefix bytes, binding rejection, prefix-context
sensitivity, common-gradient finite differences, event counts and
fail-closed transition failure. This is an implementation/fixture difference,
not evidence of changed native-model actions or physical outcomes. An
additional run including the old SRA contract tests produced 14 passed and
one historical-source-binding failure described above.

Source SHA-256:
`77d3423d4cb88fb05db546d2dc3c8a98adc219ba0aaecba9e26dce4edce75307`.
Test SHA-256:
`2cbc02af7ca0ac13e1ffb9fc665de9877e884634f21bfb90f7db319ed89f137e`.
These hashes document this CPU development snapshot, not a sealed future
native runtime. No GPU or real neural inference was executed in this work.
