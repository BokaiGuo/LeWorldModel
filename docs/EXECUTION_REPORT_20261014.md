# LeWorldModel execution report — 2026-10-14

This report records the execution state after applying the review criteria. It
does not promote an experiment to scientific support merely because a command
started or produced partial artifacts.

## GPU preflight and failure boundary

- Host device: NVIDIA GeForce RTX 5060 Ti was previously visible to PyTorch
  (`torch 2.12.0+cu130`, CUDA tensor probe passed).
- Current host state: `nvidia-smi` fails with `Driver/library version mismatch`;
  kernel module is `595.84` while the visible user-space NVML library is
  `595.99`.
- Five consecutive fresh-process CUDA allocations subsequently failed with
  `RuntimeError: No CUDA GPUs are available`, despite the stale
  `torch.cuda.is_available()` result. The current learned GPU protocol is
  therefore **environment-blocked**, not CPU-completed.
- No learned GPU experiment was silently rerouted to CPU.

### Host-permission recovery on 2026-10-14

The restricted process continued to fail CUDA initialization, but a
host-permission probe confirmed that the device node and runtime are usable
from the host: a real tensor allocation succeeded on an NVIDIA GeForce RTX
5060 Ti, and the compatibility loader placed `pusht/lewm` on `cuda:0`. A fresh
rich export then completed 1,536 rows with the declared multi-horizon and
counterfactual fields.

The capped recovery protocol is running in a fresh directory:
`le-wm-main/results/full_spectral_transfer_cuda_recovery_20261014/`. It is not
yet scientific evidence until all stages and the final manifest complete. The
earlier blocked directories remain blocked artifacts.

## Code repair applied

`le-wm-main/experiments/io_utils.py` now loads named legacy LeWM checkpoints
through the compatibility key-remap path instead of first invoking the
incompatible upstream loader. CUDA initialization is kept GPU-first and raises
on a genuine device failure. The compatibility loader was observed placing the
model on `cuda:0`. The fail-closed probe is implemented in
`le-wm-main/experiments/gpu_preflight.py`; targeted tests and the
host-permission CUDA probe pass. Learned entry points now default to explicit
`cuda` and do not silently fall back to CPU.

The repair is a runtime robustness fix; it is not evidence that the learned
model has better dynamics or planning.

## Main protocol runs

| Run | Execution state | Evidence state |
| --- | --- | --- |
| `results/full_spectral_transfer_cuda_review_20261014` | stopped after 6,160 rich-export batches; resumable chunks retained | partial / interrupted |
| `results/full_spectral_transfer_cuda_review_capped200k_20261014` | theory stages completed; latent export blocked by CUDA initialization | blocked |
| `results/full_spectral_transfer_cuda_review_capped200k_loaderfix_20261014` | theory stages completed; latent export blocked after the same host failure | blocked |
| `results/gpu_batch_probe_256_rich_20261014` | 10-batch CUDA rich export completed | smoke/protocol validation only |
| `results/gpu_recovery_probe_20261014` | host-permission CUDA rich export completed, 1,536 rows / 1,000 saved | smoke/protocol validation only |
| `results/full_spectral_transfer_cuda_recovery_20261014` | capped GPU protocol running; chunks retained | running / not yet evidence |

The successful probe verified the intended rich schema, including
`dynamics_h3/h5`, predicted and residual horizons, physical state, goals,
counterfactual fields, episode identifiers, and timestep. It is not a full
dataset result.

## Existing evidence carried forward

The review matrix remains the scientific source of truth:

- matched transfer diagnostic: positive point estimate but preregistered
  sign-flip gate not passed;
- RGB successor prediction: improved prediction, no closed-loop planning proof;
- PushT contact/action intervention: within-mode action-effect identification
  remains the bottleneck;
- strict visual pose closure and universal certificate claims remain open or
  negative;
- GDD-WM remains proposed/not validated until teacher provenance,
  geometrizability, student-generated candidates, and independent planning
  consequences are executed.

## Newly isolated artifact failure in PushT repair/MPC

The earlier `metrics: null` files from the contact-predictor/MPC family are
not interpretable as model failures.  `predict_latent_sequence` was decorated
with `torch.enable_grad()`, so it overrode the caller's `torch.no_grad()` in
evaluation.  The MPC runner then attempted to convert a graph-attached tensor
to NumPy and exited after creating only its initial protocol artifact.  The
function now respects the caller's autograd context, and the regression test
checks both no-grad evaluation and differentiable training paths.  The MPC
runner also writes atomic progress checkpoints after predictor repair and each
policy, so a future interruption is represented as an incomplete run rather
than a null-metric result.  This repairs the experiment protocol; it does not
yet establish positive pose closure or online planning.

## CPU-safe validation completed during the GPU block

- The initial repository suite recorded during the GPU block passed: `629 passed, 3 skipped`;
  the post-governance regression is recorded below as `632 passed, 3 skipped`.
- The refreshed oral-level learned-panel aggregator completed with
  `panel_gate=blocked`, `strict_matched_status=gated_single_environment`, and
  only one identifiable matched stratum. It does not establish the required
  multi-architecture, multi-environment conditional effect.

## Resume condition

After the NVIDIA user/kernel libraries are made version-consistent (or the host
is rebooted into the matching driver), the restricted-shell CUDA path should be
rechecked. The host-permission recovery run may continue in the meantime. Do not reuse the blocked
run as positive evidence unless its input lock and all stages validate.

## Governance repair validation

After the review criteria were converted into runtime contracts, the following
checks passed:

- explicit claim-matrix evidence manifest, checksum rejection, unified manifest
  and runtime-state tests: `11 passed`;
- full repository regression suite: `632 passed, 3 skipped`;
- Bash syntax check and dry-run side-effect check: passed; dry-run did not create
  its requested output directory.

These are software/reproducibility results only. They do not promote the
blocked GPU run, the PushT pose-closure result, or any universal planning claim.
