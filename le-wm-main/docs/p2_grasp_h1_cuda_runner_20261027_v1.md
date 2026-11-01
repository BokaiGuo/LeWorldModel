# GRASP official-adapter subset: frozen LeWM H1 host CUDA contract

Only new files. No CUDA, neural forward, physics, checkpoint download or training
was executed to prepare this runner. `--prepare` uses standard-library reads and
hashes only. Main reviews and executes `--execute-cuda` explicitly. Four existing
roots are calibration; this is neither full official reproduction nor P0/P2
admission. It does not validate the old finite-response chi or replace H3 results.

## Algorithm and comparison boundary

Uses the sealed `p2_grasp_reference_core_20261027_v1.py`, whose actual prepare
file SHA is `608bffb9d83a8e033e5ec0cc2b30c82a8084d1bfe5f51a7ab11a3cf3a70be151`.
Its official source contract is documented in
`p2_grasp_official_contract_20261027_v1.md`. No sealed source is edited.
Official LeWM adapter parameters: 750 joint iterations, virtual-state LR .1,
action LR .005, virtual Gaussian noise .005, sync every 50 iterations, 10 sync
steps at LR .05. Rollout T=3 and local RNG seed 20261027 are explicit local
settings, not claims about published-paper evaluation. No tuning on outcomes.

The state is the **entire original 192-dimensional model output embedding**
(after its own encoder/projector), not Q or any spectral projection. GRASP has
two full 192-D intermediate virtual states and three action tokens. Each lifted
edge takes exactly one state. Sync and independent verification start from the
last real state only, then use generated context lengths 1,2,3. This **initial
H1 contract is different from real H3**. The baseline is explicitly
`action_adam_h1`: zero actions, 900 serial terminal-MSE Adam steps, LR .005,
same H1 reset, same independent goal0, same three-token horizon, no feedback,
no bounds, no multistart, no discarded ties. It is not renamed GRASP.

| Per root/arm | GRASP H1 | Action-Adam H1 |
|---|---:|---:|
| lifted transitions | 2250 | 0 |
| serial optimization transitions | 450 | 2700 |
| independent H1 verification transitions | 3 | 3 |
| backward API calls | 900 | 900 |
| processed context tokens incl. H1 verification | 3156 | 5406 |
| separate original-real-H3 diagnostic transitions | 3 | 3 |
| separate diagnostic context tokens | 9 | 9 |

All four roots/two arms: 21624 H1 transitions, 7200 backward calls, plus 24
real-H3 diagnostic transitions; expected predictor AND action-encoder samples
21648 each, processed context tokens 34320 each including diagnostic. No fresh
RGB/goal encoding. Inherited encoding costs are unknown. **This is not equal
wall time, FLOPs, Jacobian work, graph structure, or full-pipeline compute.**
No explicit Jacobian/VJP routine is invoked; backward API calls are not a
claim about equal internal derivative work. CUDA accounting comes from actual
predictor/action-encoder/visual-encoder hooks plus core backward receipts.

## Inputs, frames, normalization, model

`pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json` is bound by
SHA `d5f544689a9d2b5c6807b2454c1a98b9db6d88bb374badc457bdcc6defa87c8a`.
The packet includes all three latent frames and two between-frame action
packets, but optimization reads only the last state. Goal0 comes from variant
zero (offset [0,0]) of the independently encoded intervention goal, exact-value
joined by root SHA and checked against the compilation. No goal is synthesized
from the current state or optimization trajectory. Context RGB provenance is
the upstream **aggregate history RGB hash**, not invented individual-frame
hashes. Goal RGB hash is the independent goal0 hash, not necessarily the old
root-anchored report's original goal image. Both entire source reports and the
compilation's source seal are verified and frozen as inputs.

Mean/std are the exact float32 public constants from
`experiments/run_pusht_counterfactual.py`: [-.0087,.0068] / [.2019,.2002].
Prepare AST-checks that source without importing its physics dependencies.
These are supplied public statistics, **not certified checkpoint-training
statistics**. Each normalized 10-vector is reshaped time-major into five 2-D
commands and inverted as `raw = normalized*std + mean`. There is **no additional
division by 100**, clipping, tanh, or action substitution. The optimizer is
officially unconstrained; raw commands outside [-1,1] are preserved and marked
`physical_box_supported=false`. Such a plan cannot silently enter a bounded
physical consumer. Even in-box plans still require a new reviewed consumer.

Prepared model bindings refer to already present official weights SHA
`48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`
and config SHA
`2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`.
Prepare does not load/rehash the large checkpoint. Host execution rehashes both
before and after, requires CUDA frozen/eval float32 official LeWM, compares the
loaded state exactly with compatibility-remapped checkpoint tensors, and
hashes every parameter/buffer before/after (including nonpersistent buffers).
No learned CPU inference is allowed; CPU checkpoint comparison is not forward
inference. Already loaded main-thread model is supported through
`execute(prepared, expected_sha, output, model=model)`, with the same checks.

## Durable outputs and real replay exit

Exclusive output only. `started.json` always enumerates eight root-arm slots,
alternating order by root, without deduplication. `events.jsonl` fsyncs each
transition request **before** its call, including the actual normalized action,
then completion/failure. Actual module requests/completions and context lengths
are separately journaled. All evaluated actions are retained, not just winners.
This is not an optimizer-resume checkpoint; it does not claim to save every
unevaluated optimizer tensor. Final actions are journaled before diagnostics.
Each finished arm has its own JSON, final normalized [3,10] actions, exact raw
[15,2] `plan.commands`, command hash, H1 verification states, H1 terminal MSE,
core receipt, module work, failure fields and replay status.

The mandatory extra replay is a **model diagnostic**, starting with the
source-bound original real three-frame context and its two between-frame
actions, then applying the exact final normalized tokens for three predictions.
No new physical state or image is fabricated. It has separate states/MSE,
separate transition accounting, and `included_in_h1_comparison=false`.
The exported exact 15-command path is the real-environment replay exit for a
future new durableguard consumer, which must rebuild the same root and actual
frames, verify hashes/normalizer, execute each command with guards and retain
partials. The runner itself never calls physics; no old 10/15-step endpoint or
old physical outcome is substituted.

An arm failure retains its partial receipt and submitted actions and stops the
run; every remaining slot stays explicit `not_started`. A failed diagnostic
retains the valid H1 plan and verification but cannot become a successful row.
`report.json` records all eight slots, including preflight failure, and false
admissions. Its `paired_h1` rows retain all four root pairs and their statuses;
only fully completed same-root arms enter the descriptive H1 MSE difference
(GRASP minus action-Adam; negative is lower MSE). Real-H3 diagnostic scores are
never pooled into it. Hard kill/power loss can interrupt final report writing; the fsynced
started census and append journal remain recovery evidence, not a success.

## Review and launch

Run non-neural checks from repository root:

```bash
.venv/bin/python -m pytest -q tests/test_p2_grasp_h1_cuda_runner_20261027_v1.py tests/test_p2_grasp_reference_core_20261027_v1.py tests/test_p2_grasp_official_contract_20261027_v1.py
```

Offline prepare (already delivered in exclusive `results/..._prepared`):

```bash
.venv/bin/python -m experiments.spectral_reachability.p2_grasp_h1_cuda_runner_20261027_v1 --prepare --output results/p2_grasp_h1_cuda_runner_20261027_v1_prepared
```

After review, main supplies the **actual file SHA** from `sha256sum` (not an
in-memory return-object hash) and launches once into a NEW output:

```bash
.venv/bin/python -m experiments.spectral_reachability.p2_grasp_h1_cuda_runner_20261027_v1 --execute-cuda --prepared results/p2_grasp_h1_cuda_runner_20261027_v1_prepared --expected-prepare-sha256 <DELIVERED_ACTUAL_SHA> --output results/p2_grasp_h1_cuda_runner_20261027_v1_cuda_main_v1
```

Remaining gaps: host H1 differentiability/shape/runtime stability and actual
counts are unexecuted; unconstrained raw support is unknown; public-normalizer
certification and original-cache encoding costs remain unresolved; no physical
performance or held-out evidence, nativeJEPA execution, full DINO GRASP path,
official-repository numerical equivalence or published-table reproduction.
