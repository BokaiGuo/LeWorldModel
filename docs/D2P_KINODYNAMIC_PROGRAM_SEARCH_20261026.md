# Model-native kinodynamic program beam search — 2026-10-26

## Implemented baseline

`le-wm-main/experiments/spectral_reachability/d2p_kinodynamic_program_search.py` implements actual bounded deterministic tree expansion through a supplied replay callback. It is not another metadata gate. `search_programs` accepts an explicit frozen primitive bank, full root model context, replay callback, saved-goal target-metric callback, maximum primitive depth, beam width and hard attempted-expansion cap. There are no physical outcomes, oracle states, learned adapters or latent-waypoint inputs.

Read `experiments.audit_pusht_reachability_horizon.select_horizon_oracle` as a structural reference only; neither that module nor its physical successor/cost functions is imported. The implementation uses `CudaReplay`'s `(context, raw_actions) -> successor_context` interface, the existing numeric context validator, and the existing `PredictionBudgetExceeded` exception. It does not import torch, launch CUDA, train, download, or modify any existing kernel, replay adapter, live RGB executor, or sealed source.

## Exact search semantics

- Every bank entry is finite raw `[T,2]` PushT action data bounded by `[-1,1]`, with `T >= 5` and `T % 5 == 0`. Multi-packet entries are permitted. Depth counts primitives, not packets or physical steps. No implicit padding, clipping, reparameterization, appended zero action or witness dtype conversion is performed. Input bank order is the frozen tie-break order.
- A node contains its complete generating action sequence, primitive-index sequence, full latent/action context, depth and finite target cost. Each expansion invokes replay on a private copy of the parent context and **only the next primitive**. It never replays the full prefix from an advanced context. The action witness concatenates the exact original primitive values; the CUDA adapter retains its own declared float32 conversion contract.
- Only model-context keys are accepted: layout, context length, latent/action dimensions and aligned latent/action histories. Root and successor layouts must match. No arbitrary waypoint is proposed or optimized. Replay and metric receive isolated copies, so callback argument mutation cannot alter sibling inputs, root inputs, bank contents or retained node contexts.
- At each depth all available frontier/bank edges are attempted unless a hard cap or external prediction-budget exception stops dispatch. Successful children are sorted by `(cost, depth, lexicographic primitive-index sequence)`; only `beam_width` nodes survive to the next layer. The returned best plan is the best finite **nonempty visited** plan across all depths, including nodes subsequently pruned. Thus a shorter plan may beat every depth-limit plan. That is explicit best-visited selection, not a fallback policy.
- The root is never a candidate and is not scored. If no finite nonempty plan exists, `best_plan=None`; the caller must not deploy a zero action or stored tail as an implicit substitute. Different program lengths require an externally frozen target-cost/horizon protocol; do not silently compare unequal task semantics.
- Attempt count increments before callback dispatch, including failures and blocked prediction-budget attempts. A replay/validation/metric exception is recorded with stage, type, message, depth and primitive indices. Ordinary errors allow remaining siblings but set terminal status `failed` unless subsequent exhaustion supersedes it. Nonfinite, bool, complex, string and nonscalar metric results fail the edge. Invalid search inputs raise before dispatch.

## Status and coverage

`completed` means the scheduled bounded beam procedure finished without failed edges or cap interruption, **not** exhaustive tree search or optimality. `budget_exhausted` means another required expansion was prevented by the expansion cap, or replay/metric raised `PredictionBudgetExceeded`. An exact expansion-cap boundary that finishes the final scheduled layer is `completed`, not falsely exhausted. `failed` means at least one ordinary callback/validation failure occurred without later exhaustion; a best successful plan can still be returned with that explicit failure status.

Receipt fields include attempted and failed counts, replay-vs-metric failures, complete per-attempt diagnostics, scheduled/attempted coverage per layer, finite children, retained/pruned counts, and `partial_coverage`. Beam pruning, any failure or interruption sets partial coverage. `optimality_claim`, `official_paper_faithfulness` and `p1_admitted` are always false. This is an explicitly defined local model-native beam-search baseline, not a faithful reproduction of any named kinodynamic paper. Finite graph tests do not establish physical efficacy.

## Parent integration (not run here)

Freeze the bank using only calibration/training or prospective procedural rules before opening outcomes. Bind its content, saved-goal metric, root, checkpoint, action normalization, replay source and search hyperparameters in the parent protocol. A callable's type cannot prove that it is outcome-independent or CUDA-backed; trusted parent code must enforce this. Do not pass evaluator state/reward/success or use a physics oracle through callbacks.

Use the existing `pusht_composition_cuda_replay_v2.CudaReplay(model, device)` with an explicit host CUDA device and its frozen-eval validation. Construct `target_metric(context)` from the saved goal in the same frozen coordinate/metric contract, not physical endpoint measurements. Wrap the same model instance with existing `PredictionBudget` and an active planning phase before calling `search_programs`; preserve the guard's complete receipt separately. All predict calls must traverse that instrumented model attribute. Do not reset it per edge/depth or convert expansion counts into predict samples. One primitive can contain multiple packets, and failed replay may already consume prediction work.

Prediction-budget exhaustion stops the search immediately even if an earlier finite best plan exists. Such a plan remains a diagnostic best-visited witness, not evidence that the whole planned search completed. The caller must check status/coverage and its preregistered handling rule before any physical execution. This module makes no deployment decision, GPU attestation, P1 admission or scientific-stage change. Parent owns the live RGB executor and later CUDA integration.

## Validation

Run from `le-wm-main`:

```sh
.venv/bin/python -m pytest -q tests/test_d2p_kinodynamic_program_search.py
```

Tests cover finite exhaustive graph traversal and exact replay witness, a non-greedy two-edge route, deterministic ties, beam pruning, strict depth/width/expansion limits, exact-cap completion versus interruption, replay and metric failures, external prediction-budget interruption, no fallback when all candidates fail, callback nonmutation, multi-packet precision, no clipping/padding, and outcome-field/layout rejection. These are explicitly requested non-neural fixtures, not neural CPU fallback or physical results. Actual CUDA integration is left to the parent; no GPU was launched by this task.

Final combined validation added `tests/test_pusht_composition_cuda_replay_v2.py` and `tests/test_d2p_baseline_protocol_v2.py`: **86 passed in 0.27s** (40 new search cases). The first fixture run exposed NumPy scalar values in toy successor contexts, which violate the existing portable JSON context validator; fixture successors now export Python floats, matching the real adapter's `.tolist()` boundary. No existing validator was weakened.
