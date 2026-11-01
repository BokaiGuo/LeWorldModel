# P1 same-model cp5 H3 CEM candidate

Status: code/contract candidate on one already observed development root; no
official reproduction, independent confirmation, physical result, P1 passage,
or demonstrated fair complete-controller comparison.

## Existing baseline audit

`p1_constructive_matched_sampling_cuda_20261027_v1` uses same-model CEM 2×16
on two earlier blind tasks with H2 and 66 predict/action-encoder calls each,
including a selected replay. Its forward-count match excludes AD/J and inherited
encoding, so it is not equal total compute. `p1_constructive_mppi_cuda_20261027_v2`
uses the project MPPI kernel, two iterations ×16, H2 and 72 calls, dispatching
the final weighted nominal; it explicitly disclaims official MPPI and total
compute equivalence. Its physical and RGB adapters provide real selected-image
receipts on those earlier roots, not a matched current centered H3 cp5 baseline.
The first-feedback-carry study uses actual RGB at cp5/10/15 and charges extra
scoring, but its internal arms are not certified external strong baselines.
Thus none establishes a fair full-controller strong-baseline comparison for
this current root. Earlier known outcomes are not new confirmations.

## Frozen candidate

Use the existing `structured` actual cp5 from the five-prefix sealed physical
intervention, no choice based on physical outcome. The preflight rebuilds the
existing own-suffix preparation and verifies its physical/source closure. The
actual cp5 PNG bytes and decoded pixels are rechecked; the same checkpoint,
image transform, goal, centered quadratic Q, 3-frame context, actual warmup
and prefix action history, and original action mean/std are used. The cp5 image
is freshly encoded, not replaced by a stored latent. Physical state is not a
planner input. Existing context and goal/model preparation are inherited
upstream costs and must not be called free.

H3 open-loop15 CEM: NumPy PCG64 seed 20261101, four rounds of 16 independent
Gaussian 3×5×2 raw-action proposals, initially mean zero/std 0.5; clip to
[-1,1], replace slot 0 with the current mean. After each full round select four
lowest model-Q costs with index tie break, update elementwise population
mean/std, clip std to [0.05,1]. Choose lowest model Q across all 64 candidates,
stable index tie break, then replay that winner once; seal all banks, scores,
updates and selected sequence. No outcome-based suffix selection. This is a
local CEM implementation, not an official CEM/MPPI reproduction.

One durable per-node ledger caps and charges one RGB encoding, 65 complete
H3 evaluations, 195 predict calls/samples and action-encoder calls, 65
replay/selection events; no AD/J/SVD/physical steps. Attempts are charged
before dispatch and failure is retained. Model load, inherited target/goal
construction, earlier physical prefix and cp5 acquisition, file/source
validation and offline diagnostics are outside the node ledger; synchronized
elapsed time and peak allocated GPU memory are recorded, not FLOPs or energy.
65 calls to replay/selection are bookkeeping events, not extra neural calls.
This budget is not equal to the structured H3 node's heterogeneous costs;
no fairness or competitive-strength claim follows. The selected suffix is
model-only and requires a separate reviewed physical replay; no actual cp5/10/15
closed-loop CEM controller or full-prefix comparison is implemented here.

## CPU validation and main-thread commands

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
CUDA_VISIBLE_DEVICES='' .venv/bin/pytest -q tests/test_p1_cp5_cem_baseline_20261101_leibniz.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p1_cp5_cem_baseline_20261101_leibniz dry-prepare
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.spectral_reachability.p1_cp5_cem_baseline_20261101_leibniz prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cp5_cem_baseline_20261101_leibniz_PREP_NEW.json
```

After review, record the printed prepare SHA and use it in the explicitly
authorized host-CUDA run. The output directory must be new:

```bash
.venv/bin/python -m experiments.spectral_reachability.p1_cp5_cem_baseline_20261101_leibniz run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cp5_cem_baseline_20261101_leibniz_PREP_NEW.json --sha256 PREPARE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_cp5_cem_baseline_20261101_leibniz_RUN_NEW --authorization MAIN_AUTHORIZES_CP5_CEM_HOST_CUDA --authorize-host-cuda
```

This implementation turn did not run GPU or physics. CPU fixtures test the
optimizer contract, not real model quality or environment performance.
