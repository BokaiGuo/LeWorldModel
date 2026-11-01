# Observed-prefix physical suffix finite differences

Implementation delivered; **NOT EXECUTED**. No CUDA allocation, new physical
inference, training, adapter, selector change, approval request or P1 admission
was performed by this task. Existing sealed sources remain unchanged.

## Frozen diagnostic

Read original sealed execution and compilation provenance, plus the completed
observed one-shot control report. For each of four original roots, choose only
the originally compiled `selected_candidate_id` and `prefix_0` (not observed
outcome winners). Require distinct selected/zero IDs; never silently deduplicate
or choose a replacement. This produces exactly eight prefix conditions.

Each condition has nine suffix probes: parameter zero, then ±each of two axes
at epsilon .01 and .05. Every probe independently resets/warms the same root,
replays the unchanged sealed prefix and checks its RGB SHA256 and exact state
against the original physical prefix receipt. It then executes exactly five
identical raw commands `tanh(float32(params))`, computed on HOST CUDA. The
final real RGB is CUDA-encoded once. Expected totals: 72 physical branches,
72 actual terminal encode calls, zero model predict calls. `model.predict` is
explicitly replaced with a counted failure guard during execution; saved J is
not recalculated or blessed by another model call.

Known Q is the saved projection operator, required to match the sealed
compilation exactly. At each epsilon the physical latent central difference
is `(z_plus-z_minus)/(2*float(float32(epsilon)))`, in PARAMETER coordinates.
Do not divide by live-command displacement: the saved observed model J
already differentiates through tanh, packet repetition and normalization.
Report physical J, raw/projected J-error MSE and column norms, plus actual-zero
versus saved model-zero nominal MSE. Epsilon-specific comparisons are diagnostic
values, not threshold tests. Failed/missing signed probes make that comparison
unavailable; they are not dropped, imputed, or used to select another prefix.

## Provenance and output

Before first CUDA allocation, validate upstream completed integrity reports,
current source seals, compilation/execution/observed hashes, checkpoint/config
bindings, individual observed stageplan hashes, root/plan IDs, frozen Q and
root-recreation bank/upstream digests. Create an exclusive new run directory,
freeze protocol and source seal, and verify that seal before allocating CUDA.
Checkpoint parameters, buffers, eval/frozen state, source and input hashes are
checked again after execution. No CPU neural fallback exists.

Each attempted probe has an individual JSON receipt, including errors, prefix
replay identities, actual parameters and all five raw commands. Successful
probes additionally save terminal RGB as PNG with pixel/file hashes, state,
latent vector, object pose labels and official/object-only labels. Installed
official evaluator results must match the independent label helper. Per-root
and final reports retain the fixed probe population and counters. Ordinary
per-probe exceptions are recorded and execution continues through the fixed
schedule; startup/root-set provenance failure aborts instead of substituting
data. A failed run leaves its exclusive output directory intact: never reuse
or overwrite it. Images are terminal-only (no videos), keeping output modest.

## Interpretation boundary

This tests local physical-vs-model action derivatives after an exactly replayed
observed prefix. It is distinct from array/solver/compile-protocol correctness.
Passing array algebra does NOT make the original J scientifically correct.
Finite differences at two scales in contact dynamics may disagree and cannot
establish differentiability, causation, global planning efficacy or P1 success.
All roots are known calibration roots. No gate, selector or efficacy threshold
is changed, and `P1_admission=false` is unconditional.

## Tests and parent launch command

Pure array/static tests use no neural inference or environment execution:

```sh
.venv/bin/python -m pytest -q tests/test_pusht_observed_suffix_physical_fd.py
```

Initial RED was missing-module collection failure. Final result: 10 passed.
Coverage: fixed 9/72 schedule, selection independent of outcomes, float32
parameter central-FD arithmetic, failed/missing/duplicate/nonfinite probes,
malformed saved J/Q, source-seal-before-CUDA ordering and no J recomputation.
Static checks are not a live runtime validation; parent must review and
coordinate/approve HOST CUDA execution separately.

From `/home/gbk/code/LeWorldModel/le-wm-main`, the exact proposed command is:

```sh
.venv/bin/python -m experiments.spectral_reachability.run_pusht_observed_suffix_physical_fd \
  --execution /home/gbk/code/LeWorldModel/le-wm-main/results/pusht_d2p_sealed_program_execution4_v1_20261026_cuda/report.json \
  --observed /home/gbk/code/LeWorldModel/le-wm-main/results/pusht_observed_oneshot_suffix_control4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/pusht_observed_suffix_physical_fd4_v1_20261026_cuda
```

The command has not been run or submitted for approval by this task.
