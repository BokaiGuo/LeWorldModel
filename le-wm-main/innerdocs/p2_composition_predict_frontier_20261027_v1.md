# P2 predict-call frontier extension: frozen bridge v1

Main-directed frontier budgets; implementation-thread predeclared seeded tail,
pending main review under authorized implementation.
Three NEW exclusive files only; no old sealed/source/report/test/shared-doc edits,
no CUDA/physics/model loading/download/training. This is a candidate-generation,
source-contract and exact-call budget bridge, NOT a ready CUDA planner or P2 pass.

## Why a NEW runner contract is required

Real paired-v2 `run_arm` binds `(40,20)` logits, `(40,2,5,2)` raw commands,
`range(40)`, fixed winner replay paths40/41/42; prepare/accounting/exporter also
freeze40. Changing array length, reusing old selected_actions or resealing old
prepared would violate the source contract. Bridge hashes EXACT old paired/SRA
sources and AST-checks dispatch/terminal sites before showing budget results.
The old minimal suffix compiler is usable as a primitive, not an80-slot runner.

## Frozen nested candidates, no outcome-selected tail

Four original used root SHA identities fixed in source. Preserve original model,
goal0/Q, normalizer/raw[-1,1], H2 ten-command action domain and full predictor
context. Open-loop predicted-prefix recompile only; no observed-prefix feedback.
Do not introduce goal-patch Q or physics labels into this protocol.

Original40 are deterministic and have NO random seed: zero1+DLS3+top9 mode x
two original radii x both signs36. Validate exact logits/labels/float32-tanh
mapping and inherited canonical right-vector signs/basis, original goal/Q hashes.
Joint40 is exactly the first40 of Joint80. Composition keeps exactly original40
prefixes, preserving the original paired ablation. No root/outcome reads choose
candidate IDs, directions, signs, radii, seeds or budgets.

Tail40: PCG64(20261027+root_index), draw10 independent9D Gaussian vectors,
normalize each, form their linear combination of inherited top9 spectral right
vectors; each gets original radii sqrt(10)*.25 and sqrt(10)*.75 and signs−/+.
Keep all40 even if duplicated/degenerate; no redraw/rank tuning or pruning.
Zero norm strictly refuses. Inherited null modes follow original LAPACK basis.
This is a seeded MULTI-MODE spectral extension, not random-search renamed as SRA;
no fresh Jacobian/SVD or learned adapter. Whole20 logits retain joint L2 cap
sqrt(10)*.75, then float32 tanh. Not per-packet balls or equal search distributions.
Saved source bank digest, goal/Q digest, root and seed bind every generated bank.
No generic arbitrary-root/count API: frozen40/80 points only.

Pure `candidate_bank` is a mockable generation recipe and explicitly marks
source_verified=false. `bound_candidate_bank` (used by CLI) additionally requires
the exact predeclared canonical digest of each archived root bank, including its
goal/Q/nominal/Jacobian/basis. Cross-root bank substitution or changed Q refuses.
This verifies saved input arrays, NOT a fresh root compilation. Main's new runner
must separately pay, record and source-bind fresh root compile; cached old arrays
must never receive fresh4predict/Jacobian credit. No outcomes are read here.

## Exact calls checked against real legacy control flow

Legacy root compile calls its two-packet terminal once nominally and once as the
vectorized-Jacobian forward:4predict shared, not2. Each old run_arm calls replay
on prefix5 then suffix5 for all40:80 ordinary predicts. Composition suffix_cuda
adds nominal1 and Jacforward1 at EACH fixed prefix:80 extra predicts. Each winner
is segmented5+5 and whole10:4predict, despite THREE ledger.call sites.
The inherited _Ledger.call splits whole10 into TWO separate five-command replay
adapter invocations/events, so there are FOUR underlying replay dispatches.
Tests execute actual sealed `run_arm(fixture=True)` with counted NumPy callbacks,
including mock nominal/Jacforward inside suffix callback: measured84 joint and
164 composition exclusive, ledger84 ordinary both. No real AD/forward is run.
RootAD count is source-level terminal invocation proof, not measured GPU work.

| Per root | Joint40 | Joint80 | Composition40 |
| --- | ---: | ---: | ---: |
| shared root compile attribution | 4 | 4 | 4 |
| real candidate packet forwards | 80 | 160 | 80 |
| suffix nominal+ADforward | 0 | 0 | 80 |
| winner double replay | 4 | 4 | 4 |
| standalone predict / action encoder | 88 | 168 | 168 |
| standalone Jacobian | 1 | 1 | 41 |
| mathematical output VJP directions | 192 | 192 | 7872 |

New point unique pair332/root, NOT336 (shared4 charged once); four-root1328
predict +1328action_encoder,3984context tokens each. Each exclusive164 plus
shared4. Selection80+40=120/root,480/four roots. Unique Jac41/root164 total,
7872 mathematical VJP/root31488 total. SVD2 shared +positive-radius suffix count,
ridge1 shared +positive-radius suffix count; actual40 suffix solves must report
their radius-dependent counts (old actual was40 positive/root). Generation CPU
draws/normalizations/combinations/scalings explicitly additional, not free.
Unknown backward engine calls, FLOPs, synchronized walltime, upstream RGB/Q/cache
construction remain unknown. Predict matched168/168 is NOT total-compute matched.

Budget-point labels are fixed BEFORE new execution: historical88/168 paired-v2
and new168/168. Do not mix old/new roots or old physical endpoints as same run.
Nested80 can give its first40 best predicted cost as a diagnostic WITHOUT extra
replay. Any fresh deployable40 winner still needs its own4predict; it cannot be
added free to the168 contract. Model-minimum monotonicity80≤40 is deterministic
selection arithmetic, NOT proof of improved actual Q/pose or P2.

## Precise remaining interfaces (no authority gap for bridge)

1. NEW source-bound prepare retains source/root/context/model/Q/action bindings.
2. NEW main-reviewed CUDA runner consumes both banks, joint80 x2 forwards and
   conditional40 x4, each winner4, one shared root compile; no padding/refunds.
3. Dual entry predispatch guards: shared4, exclusive164 each, unique332/root,
   durable attempts/completion/errors, actual batch/context metadata; Jac/VJP/
   SVD/solve/selection and generation costs separate. Failure retains partials
   and blocks deployable winner, no retry/selection on incomplete candidates.
4. NEW all-slot writer/exporter:80 joint one-edge,40 composition two-edge, full
   contexts, local derivative/solve, stable nonlinear selector and exact whole/
   segmented winner. Old40 exporter prohibited for new bank. All source-bound.
5. NEW frontier report/auditor distinguishes stored historical point, nested40
   predicted diagnostic, fresh168 point and actual resource ledger.
6. After main CUDA review/execution, NEW independent physical consumer executes
   fresh selected controls; no reused terminal, no P1/P2 admission.

Bridge alone cannot produce a deployable selected_actions, completed CUDA
receipt or physical-ready flag. No scientific or neural prerequisites are
silently filled. Main can now implement/review these interfaces in NEW names.

## Commands, small output only

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p2_composition_predict_frontier_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p2_composition_predict_frontier_20261027_v1
```

CLI default prints source-bound budget/remaining-interface JSON only. Optional
`--bank-json EXACT_FROZEN_ROOT_BANK.json --root-sha256 ORIGINAL_ROOT_SHA --count 80`
prints generated bank to stdout; no files copied/prepared,
no report/outcome discovery, no weights loaded. Mock tests are not CUDA evidence.
