# Saved-suffix budget guard HOST CUDA integration probe

Prepared only; **not launched**. No new physical execution, observed-prefix
replay, training, adapter, P1/P3 admission, or compute-advantage claim.

Runner: `le-wm-main/experiments/spectral_reachability/run_pusht_d2p_budget_guard_cuda.py`.
Consumes only the sealed metric-program compilation report, all four original
root hashes and all sixteen saved receipts per root. Its input report SHA is
hardcoded to `d5f544689a9d2b5c6807b2454c1a98b9db6d88bb374badc457bdcc6defa87c8a`.
Also verifies upstream current-source seal and all input bindings, official
weights SHA, exact current config SHA and model/context/action config fields.
No candidate selection or altered compilation. Existing sealed modules untouched.

Before any CUDA allocation/model load: validate receipts, import dependency
definitions without constructing a model, seal upstream source set plus loaded
experiment code, configured model constructor/backbone source, new runner/test/doc;
write exclusive protocol and revalidate inputs/seals. Framework binaries are not
archived; torch version is recorded. The bounded source sealer fails closed if
file count/size limits are exceeded. Only HOST CUDA invocation is supported;
loader stages checkpoint bytes on CPU as required by the existing compatibility
loader but all neural prediction uses CUDA (no inference fallback).

Per root: construct frozen Binding from protocol digest, original root SHA,
official weight/config digests and fixed method; use PredictionBudget limit16.
Convert saved context_emb, context_actions, executed_suffix_params to CUDA
float32 and call existing p1._suffix_terminal, one batch/horizon-one per receipt.
Compare full 192-vector with saved predicted_terminal_vector using fixed
absolute tolerance 1e-5, relative tolerance zero. After all sixteen matches,
repeat first receipt: require PredictionBudgetExceeded before underlying
predict entry. An independent entry counter verifies exactly16 delegated calls;
budget receipt must show samples16, calls16, blocked1, failed0. Four roots imply
64 actual model.predict calls and four blocked attempts (action-encoder work
can occur before the blocked predict; this is NOT a total-compute cap).

Hash all model parameters and buffers before/after, including nonpersistent
buffers. Check eval/frozen state, input bindings and both source seals after
execution. Exclusive new output contains protocol/source seal, per-root
receipts and final report; failed integrity returns nonzero. Exceptions during
the prediction loop produce an integrity-fail report, never a pass; failure
before allocation/setup may leave only the prospective protocol/source seal.
No weight copies or rendered images are saved. Metadata-only tests do not
establish actual CUDA integration success; parent must approve and run.

## Verified preparation

New probe tests plus budget/instrumentation/resource-ledger regressions:
**147 passed**. Read-only CPU metadata precheck verified the actual four-root
report SHA, all input/checkpoint/config bindings and upstream current-source
seal. Dependency discovery found 62 source files, 1,296,982 bytes total,
maximum single file 250,825 bytes, within the source sealer limits. No CUDA
allocation, model construction, physical execution or integration run occurred.
No dependency on the separate composition/observed-prefix replay probe.

## Parent launch command (not executed here)

Working directory: `/home/gbk/code/LeWorldModel/le-wm-main`.

```bash
.venv/bin/python -m experiments.spectral_reachability.run_pusht_d2p_budget_guard_cuda \
  --compilation /home/gbk/code/LeWorldModel/le-wm-main/results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/pusht_d2p_budget_guard_probe4x16_v1_20261026_cuda
```

Output must not already exist. No overwrite/resume/delete mode. New host
approval is required; no background process has been created by this task.
