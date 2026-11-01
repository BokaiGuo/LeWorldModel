# P2 H15 CPU spectral backend with durable instrumentation

The existing `NativeTree` / `NativeSignal` model backend explicitly requires a
CUDA device; this task does not instantiate it or relax its checks. The new
independent backend instead executes the real NumPy H15 spectral-bank and
candidate-scoring computation on saved signal/terminal arrays. Its bounded
ridge and candidate construction follow the existing `bounded_ridge` and
`p2_h15_sra_prepare_20261029.spectral_bank` equations; an independent CPU test
compares its entire bank with the original implementation.

The backend invokes the durable driver for the outer program compile, nested
ridge solve, each of its two actual `numpy.linalg.svd` calls, and each of the 40
ordered candidate score calculations. The native ledger `metric_calls` axis
is exclusively reserved by this backend for candidate score evaluations and
its cap is tightened to the contract `selection_evaluations` cap at driver
construction. Mapped SVD/solve caps are also enforced before entry. Completion
requires the full 40-score sequence. Invalid inputs, source drift, a failed
kernel, an exceeded cap or incomplete sequence fail-stop. Each scope carries
the new adapter source hash. No global or sealed code is changed.

The saved integration diagnostic consumes pinned H15 host and child reports
and their hash-chained original journals. For each known root it rebuilds the
CPU bank from the saved Jacobian/nominal/goal and scores saved terminal vectors;
it compares the bank, all 40 scores and winner to the saved report. Its contract
and identity are **synthetic and retrospective**: this is not preregistration,
a new model run, a physical run or a budget-fair comparison. Each independent
new durable journal is immediately recounted. The integration verifier checks
saved report hashes, driver journal hashes, event scopes and source hashes; it
does not recompute the bank or independently recertify the full old source
closure.

The new ledger records per root 2 SVD calls, 1 ridge solve, 1 bank compile and
40 scores. It does not include the earlier model inference, saved-data loading,
source audit or any physical operation. Adapter-lifetime wall and process CPU
time are captured in the report; they exclude driver creation/finalization
and are not full planning time. Process CPU time can include other threads;
no per-operation CPU or FLOP estimate is substituted. FLOPs, GPU peak, neural
work, inherited work, full planning wall, and physical cost remain unknown;
zero hook counts in this *new CPU-only scope* do not describe the original run.
Source and event hashes do not prove coverage or causal execution and must be
independently retained to be useful as anchors.

The driver interface is usable by this CPU backend, not by the sealed
CUDA host runner with its exact ledger-type/cap assertions. A real neural
integration requires a separate new runner and verified backend wiring,
same-consumed model/task bindings, appropriate device authorization, and
instrumentation for still-uncovered axes. No equal-budget, P1, or P2 admission
is established.

The fixed diagnostic artifact is
`results/p2_h15_cpu_saved_integration_20261030_codex_CODEX01/report.json`,
SHA256 `95d0f3f74eec7852b0f021531976ad28a8201eeb9daa7bd2aea9a418c5a0ffbc`.
Verification:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
 .venv/bin/python -m experiments.spectral_reachability.p2_h15_cpu_saved_integration_20261030_codex \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_h15_cpu_saved_integration_20261030_codex_CODEX01 \
 --verify-sha256 95d0f3f74eec7852b0f021531976ad28a8201eeb9daa7bd2aea9a418c5a0ffbc
```
