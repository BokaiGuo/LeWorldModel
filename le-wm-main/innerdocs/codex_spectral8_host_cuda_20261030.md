# Known-root spectral8 host CUDA launcher

This launcher implements only the structured spectral8 H3 engineering arm.
The other contract arm entries are placeholders, not executed comparisons.
It does not authorize physical execution, training, P1 admission, or a fairness
claim. The existing four-arm ridge proxy entry is unchanged.

The CPU-only prepare is
`results/codex_spectral8_host_cuda_20261030_PREP01.json`, SHA-256
`d5f57f81e176d0eaa34574ab49b128c9b2637637e1c150a94fc3412bd6ea2958`.
It pins existing base prepare and inherited root slot 0, weights/config and
expected model state, 16 named source files, eight ordered candidate IDs,
two common suffix steps (rate 0.05), float32 stats, H3 and per-arm caps.
At run time it rebuilds the prepared inputs/sources before CUDA and after work;
model state is checked before and after. The named list and inherited bindings
are not a full Python, CUDA, NumPy, native library or model-class source closure.
Hardware identity is a `cuda:0` policy digest, not hardware attestation.

After human inspection and explicit authorization, run from
`/home/gbk/code/LeWorldModel/le-wm-main`:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_spectral8_host_cuda_20261030 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/codex_spectral8_host_cuda_20261030_PREP01.json \
  --sha256 d5f57f81e176d0eaa34574ab49b128c9b2637637e1c150a94fc3412bd6ea2958 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_spectral8_host_cuda_20261030_HOST01 \
  --authorization MAIN_AUTHORIZES_SPECTRAL8_SINGLE_ARM_CUDA_ONLY
```

The output path must not exist. A successful adapter writes a bounded, fsynced
hash-chained `candidates.jsonl` containing signal, all eight raw and optimized
candidates and final selection, plus a durable `ledger/events.jsonl` and
`report.json`. Partial failures exit nonzero and normally write a failure
report; filesystem/finalization failures can prevent the report. Ledger is
independently consumed in process; there is no independent semantic trace
consumer or external attestation. The trace's `complete_adapter` does not
alone prove later model/source postchecks passed; check report status.

Discrete caps: 75 action-encoder and predict calls/samples, 208 backward/VJP
rows, one Jacobian marker, one actual CPU float64 SVD, eight replay markers,
and eight selection evaluations. Neural inference/AD use the host CUDA model;
SVD is CPU NumPy. The wall cap is 600 seconds after CUDA preflight, peak
allocated/reserved caps are 4/6 GiB, and the trace limit is 16 MiB. These are
cooperative checks, not preemptive hard limits. Disk floor is 256 MiB; budget
does not cover all setup, tensor math, transfers, internal model operations,
time or energy. The full standalone scope label in the inherited ledger schema
does not make that missing work metered.

Review the fixed raw action contract and checkpoint normalizer provenance
before execution. An out-of-domain physical state remains possible even for
bounded commands; this launcher has no physical or RGB path. Leading spectral
degeneracy is rejected, and real checkpoint CUDA ABI/numerical correctness of
this new launcher has not been tested. No individual or joint result here
establishes original D2P equivalence or equal-budget performance.
