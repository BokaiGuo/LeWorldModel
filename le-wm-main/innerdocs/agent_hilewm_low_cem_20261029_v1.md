# Hi-LeWM paid RGB/high-FIRST → installed low-CEM (normalized integration)

## Scope and frozen inputs

Exclusive source/tests/this document use `agent_hilewm_low_cem_20261029_v1`; no earlier files/results are edited. All four original slots remain root0/off, root0/on, root1/off, root1/on. Two known roots, no fresh roots; no outcome, reward, evaluator state or success is an input to search. This is **integrated normalized-low-CEM**, not official Hi-LeWM baseline performance, full policy, common-backbone equivalence, physical benefit or P1 admission.

Bound paid parent: `agent_hilewm_rgb_high_low_20261028_v1`, source SHA `ad7e19e8c1024c51227307b219b32bdc448e85ca27b6a3a4cc8e5708258bda2d` (tests/doc also pinned). Actual manifest SHA `a3255d38e9cf718f9560c1b9e381abe38750edd9e91d3f04e5ad332cf414d7e9`, report SHA `083d0d9aa190b561e6ccf865b09d7224b97d404b16ba38c792bd8af386f5622a`, receipt SHA `40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0`; exact paths are constants in source. Every build/verify reconstructs **entire** parent consume and compares its full returned dictionary with the persisted receipt, then rebuilds its manifest. It does not accept merely a status flag or selected receipt fields.

Each slot uses the parent's actual encoded `z_init` and actually selected FIRST high waypoint (not high terminal). Original NPY files, descriptors, raw float32 hashes, slot RGB provenance, export authority and complete source closure are pinned. Literal vectors in the new manifest are roundtripped float32 copies of these paid arrays, not new encodings. Parent resources remain a separate inherited field: 8 encoder/8 projector, 12 high-cost, 56 high-predictor, four selected high rollouts, and original four fixed low panels/eight low-predictor calls. They are not erased or charged as new work.

## Real runtime and original-equation subset

`seam.load_runtime()` imports installed CUDA CEM only after explicit MAIN flag and complete saved verify. No worker production prepare or neural execution was performed. CEM source is pinned `a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b`. `loader.construct_cuda()` builds the existing tensor-only owned graph; no pickle execution or foreign model class import. Existing 425 tensor values/export/IR and installed/source hashes are checked before and after.

Original source: sealed `results/p1_hilewm_text_contract_20261027_v1/texts/package__code__h_le_wm__models__jepa.py.source.txt`, `rollout_low` lines354–441, `get_cost_low` lines475–505. Source semantics are reproduced over owned nodes, **not** claimed as original foreign-class bytecode parity. The embedded owned root's existing high-only `get_cost` is not patched. New `OwnedLow` accepts expanded batch1/samples1..64/H1..16, width10, single initial latent/history; fixed production uses S8/H2. Literal parent model audit binds context3 and width10. Original `_plan_low` single `z_init`/zero grouped `a_hist` is preserved, not RGB history3 or an invented observed trajectory.

For each future group, append its outgoing width10 action **before** full action encoding. Explicit original kernel1 Conv1d → action embed MLP; take last `min(3,z_length,embedded_action_length)` context; low_predictor → flattened low_pred_proj → last predicted latent → append. Complete action-encoder input/output, predictor input/output, projection input/output and each cost's full rollout are persisted immediately. No analytic cost substitute: squared terminal minus original FIRST target summed over192 is the actual native cost. Saved consumer independently reconstructs all autoregressive deliveries and terminal from native outputs, and validates cost using a conservative gamma194 float32 arithmetic bound (including exact zero residual). This bound is **not** neural output parity tolerance.

`LowBatchBridge` is a new own-schema tensor-only installed-solver shim. It preserves original metadata `planner_level=low` and restores it after CEM adds sample expansion; never forges high-bridge schema. It checks exact fingerprints/expanded zero sample strides and same CUDA device, debits each attempt/candidate row before validation/dispatch, retains failed attempts with no refund. No global monkeypatch. Installed CEM receives only tensor info; no string metadata used to infer batch length.

## Fixed search and accounting

Each of four slots: samples8, iterations3, topk3, H2, primitive width2 × block5 = grouped width10, seed1729+slot index, var_scale1, full zero CUDA init `(1,2,10)`, no actor/warm start. Box[-1,1] provides synthetic normalized dimensional metadata; installed CEM **does not clamp** and these are not certified physical action limits. H2 is this preregistered integration budget, not the official planner's full budget.

Per-iteration full native costs/candidates and all nine CEM callback arrays are persisted. Consumer verifies three smallest costs, unique elite indices/gather, mean candidate at index0, mean/STD chains, lifecycle and returned action/mean/var/logged elite-average costs. `var` is actual **sample STD (ddof1)**, not variance. Only CEM NumPy reducer reproduction uses frozen seam 8eps atol/rtol; not used for neural output comparisons or native cost algebra.

New resources (not FLOPs):

| Axis | All4 |
|---|---:|
| low-cost API attempts/returns | 12 |
| candidate rows (B×S per cost) | 96 |
| action encoder / low predictor / low projection calls, each | 24 |
| predictor sample rows | 192 |
| predictor context tokens | 288 |
| full action-encoder context tokens | 480 |
| flattened low projection rows | 288 |
| encoder / pixel projector / high predictor / AD / physics | 0 |

Per module attempted/returned/input rows/context tokens, bridge attempts/candidate rows, inherited resources and total wall are separate. FLOPs remains unknown. No same-FLOPs advantage or physical-horizon advantage claimed. Final actions are the final CEM **mean**, not a previously evaluated winner candidate: no extra replay, `final_mean_evaluated=false`, logged costs are final elite average, never labelled final mean cost.

Exact full grouped `(1,2,10)` → primitive normalized `(1,10,2)` reshape is saved. Official receding1 group gives first packet `(1,5,2)`, also saved/independently checked. No inverse transform is performed.

## Raw inverse and current hard stop

Original action `StandardScaler` mean_/scale_, flags, dtype, XY/raw units and training-dataset/process provenance have **not** been authenticated for this checkpoint. No training statistics are fitted, no common-backbone constants borrowed, no missing fields filled. `inverse_transform(normalized, authority)` intentionally rejects every input in this version, including self-reported `certified:true` dictionaries. There is no registered complete certified authority to accept. A future independently bound authority consumer/version is required before raw inverse delivery can be implemented. This is an explicit unsupported physical-release branch, not a fabricated certificate and not a prerequisite blocker for normalized search. Every consume row has `raw_actions=null`; all reports/receipts keep physical_release/full_policy/scientific/P1 admission false. No run flag bypasses this.

## Persistence and failure handling

New output directory and receipt paths must be absolute direct children of `results`, with own prefix and exclusive creation. Limit32MiB including arrays/report, floor200MiB, wall600s, final JSON reserve4MiB. Original float32/int64 NPY only; no dtype coercion for native buffers. Before allocation, NPY headers are checked against actual data length and bound; full file SHA, array raw SHA, shape/dtype, finiteness and EOF are checked. Strict consumer checks complete directory census and complete array-reference census, unknown event kinds/order/parents, all4 denominators, reconstructed resources, frozen tensors/export and source closure. Guard: inference/no-grad/no autocast, deterministic CUBLAS `:4096:8`, both TF32 off, MATH SDP only (flash/efficient/cudnn SDP off), before/after equality. Cooperative wall/cap failures preserve partial rows/events/report when final reserve permits; no retry/refund or replacement slot. Strict complete consume refuses partials.

## MAIN commands (not executed by worker)

Run from `/home/gbk/code/LeWorldModel/le-wm-main`. Source/tests/doc must be frozen first. Existing directory/receipt names must not be reused. Obtain manifest/report SHA from the actual successful preceding command; never enter a future SHA.

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_low_cem_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_low_cem_20261029_v1_prepare_MAIN01
```

Set `TASK_LOW_MH` to the printed manifest SHA, then:

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_low_cem_20261029_v1 verify --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_low_cem_20261029_v1_prepare_MAIN01/manifest.json --manifest-sha256 "$TASK_LOW_MH"
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_low_cem_20261029_v1 run --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_low_cem_20261029_v1_prepare_MAIN01/manifest.json --manifest-sha256 "$TASK_LOW_MH" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_low_cem_20261029_v1_cuda_MAIN01 --main-authorized
```

Set `TASK_LOW_RH` to the actual run's printed report SHA, then:

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_low_cem_20261029_v1 consume --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_low_cem_20261029_v1_prepare_MAIN01/manifest.json --manifest-sha256 "$TASK_LOW_MH" --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_low_cem_20261029_v1_cuda_MAIN01/report.json --report-sha256 "$TASK_LOW_RH" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_low_cem_20261029_v1_consume_MAIN01.json
```

Worker tests: `env OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_low_cem_20261029_v1.py`. RAM literal arrays, symbolic shape execution (no model values), precise subprocess cold import prohibition and temporary saved fixtures. Includes coherent wrong-cost mutations, sample-STD vs variance, original action timing, variable S/H bounds, complete all4 and resource/census rejection, non-certified inverse refusal, NPY bomb-header rejection, and actual frozen source SHA reads. Fixtures do not constitute neural or scientific evidence. Production prepare/run/consume, numerical parity and physical controls remain unexecuted by worker.

Actual worker results: own suite78 passed in0.64s; joint suite with frozen RGB/high-low and real-high-CEM tests162 passed in3.65s; CLI `--help` exit0. No production model/physics/prepare execution.
