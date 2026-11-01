# P5 RGB CUDA offline review gate

The review gate does not grant execution authority or import torch. It pins the
existing runner, consumer, bridge, checkpoint and fixed 164-branch input;
materializes and hashes only the two allowed history RGB frames and five raw
actions per branch; checks fixed root/condition ordering; and records off/on
denominators and basic action-component range diagnostics separately. Matching
off/on model-input hashes are only pixel identity diagnostics, not evidence
that the conditions or their training distributions are equivalent.

The earlier consumer checks chain, order, fixed input identity, finite outputs
and denominators, but does not reject extra event fields. The new strict review
checks canonical JSON and exact per-event and backend key sets before invoking
that consumer. A test constructs a self-consistently rehashed extra field that
the earlier consumer accepts and the new gate rejects. This protects against
accidental or undeclared fields, not adversarial semantic data hidden inside
strings or values; it does not prove a GPU operation occurred.

The strict journal review requires the SHA from a separately stopped and
sealed journal, retains incomplete slots, produces a new exclusive file and
never opens evaluation labels. Source pin changes fail closed and require
review and a new gate revision, not editing these pins after execution.
The input checker cannot prove actual tensors used in a GPU kernel; logs remain
self-reported. No simulated test result should be mistaken for a real run.

Open issues for an execution decision remain: a reviewer must choose device,
GPU-memory and wall-time supervision externally. The old runner's limit is
checked between calls only; CUDA initialization and hangs are not hard bounded.
The checkpoint's historic renderer implementation and training/panel visual
distribution equivalence remain unproven. off/on cannot be pooled as identical.
Successful preflight or journal review is not P5 scientific admission.

Offline command (no CUDA):

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p5_rgb_cuda_review_gate_20261030_codex_v1 preflight --output-name p5_rgb_cuda_review_gate_20261030_codex_v1_MAIN01.json
```

After a separately authorized run has stopped, use `review-journal`, specifying
its `--run-name`, independently calculated `--journal-sha256`, and an exclusive
`--output-name` with this gate's prefix and `.json` suffix.
