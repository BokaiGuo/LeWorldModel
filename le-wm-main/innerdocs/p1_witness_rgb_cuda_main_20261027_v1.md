# Fixed all24 actual physical checkpoint native encoding

MAIN execution only, existing encoder+projector24 each, batch1 frozen CUDA.
No image dedup, predictor, AD, physics, training, Q fitting or action selection.
Fixed8 physical branches x initial/step5/step10 in original plan order. Prepare
checks the actual complete16reset/80step/88render physical report, source closure,
original full-H2 command/prediction bindings and each PNG's file/pixel SHA. It
uses goal/Q from the original source task, not future state labels. Actual state
diagnostics are evaluation-only and never passed to the model.

Predeclared analysis: native192 prediction error plus original Q predicted and
actual RGB distance to goal for each checkpoint; retain independent wrapped
angle/object/agent positional diagnostics. All raw latent vectors are fsynced in
the module-entry journal before scoring. This permits separating wrong prediction
from Q/physical-goal mismatch; neither pattern by itself proves causal root cause
or theory failure. Zero-arm predicted contexts are the paid original zero-H2
incumbent, not the assertion that zero command means zero model dynamics.

No independent cohort, direct plateau or equal-compute comparison. P1 staysfalse.
Source history inherits the original metadata; a separate actual-history audit
verified this batch's measured stationary history equals its original padding
bytes, not that all padded histories are valid. Output4MiB, reserve200MiB;
exclusive prepare/output and no automatic retry. Exceptions preserve existing
journal; an unwritable disk cannot guarantee final report persistence. No
watchdog/resource-matched timing claim. Source/test/doc hashes bound before run.

Use inner cwd module CLI: `prepare --output NEW_PREPARE`, then
`run --prepared NEW_PREPARE --prepare-sha PRINTED_SHA --output NEW_RUN`.
Tests only validate pure arithmetic and rejection paths; CUDA evidence requires
the actual main-owned receipt and an independent saved-journal recomputation.
