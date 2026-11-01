# Four-arm known-root CUDA engineering entry

This is an exploratory engineering variant, not an independent confirmation,
not a physical experiment, and not evidence of fair total costs. Structured
is a local ridge proxy, not the D2P spectral candidate bank; native is a
zero-center CEM variant, not a validated native baseline. Root slot 0 is fixed
before execution from the existing prepared authority, not chosen by outcome.

From `/home/gbk/code/LeWorldModel/le-wm-main`, create a new preparation on CPU:

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m experiments.spectral_reachability.codex_four_arm_host_cuda_20261030 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_host_cuda_20261030_PREP01.json
```

The output prints the exact file SHA-256. After reviewing the generated
contract, sources, task, physical raw action bounds, checkpoint, and resource
limits, a human may run the following on the host, replacing the placeholder
with that SHA. Both paths must be new; never reuse a previous output.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_four_arm_host_cuda_20261030 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_host_cuda_20261030_PREP01.json --sha256 PREPARE_FILE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/codex_four_arm_host_cuda_20261030_HOST01 --authorization MAIN_AUTHORIZES_FOUR_ARM_ENGINEERING_CUDA_ONLY
```

Raw candidates have shape `[B,T,5,2]`, are clamped to the existing engineering
contract's raw `[-1,1]` box, then normalized by per-coordinate mean/std at the
model boundary. Confirm that this raw box is actually appropriate for the
bound task before execution; it is not inferred from normalized values.
Histories are already normalized. No physical action is executed.

Each arm has its own durable ledger and candidate trace. The trace saves all
generated candidates before scoring and the final scores/selection upon success.
An interrupted generation may lack intermediate replay scores. The report
retains statuses for all four arms; a partial run exits nonzero. Failure to
write or finalize files, including disk exhaustion, can prevent a report.
Trace hashes are recorded, but no independent trace consumer is provided.

The fixed per-arm discrete caps include at most 30 prediction and action-encode
calls/samples, 192 backward/VJP rows, and 12 selection and replay calls.
Wall cap is 600 seconds after CUDA preflight; peak allocated and reserved
memory caps are 4 and 6 GiB. Checks are cooperative and cannot preempt an
ongoing operation or guarantee a hard wall/memory limit. Disk floor is 256 MiB;
trace is limited to 16 MiB per arm. Initialization, model loading, serialization,
CPU work, and other overhead are not exhaustively metered. Actual hardware
identity, complete source/dependency closure, four-arm total-cost equality,
and real checkpoint CUDA correctness of this new entry remain unverified.
P1 admission is unchanged.
