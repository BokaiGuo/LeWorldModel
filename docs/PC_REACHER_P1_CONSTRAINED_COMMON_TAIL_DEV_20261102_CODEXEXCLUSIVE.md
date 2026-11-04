# P1 fixed-budget common-tail constraint search

This is a new outcome-informed **model-only development hypothesis** over
the two already-exposed roots. It neither modifies the P1 gate nor uses any
physical outcome to select an action. No training or physical run is
authorized by this document.

## Motivation and fixed inputs

The prior common-tangent host-CUDA run
`reacher_pc_common_tangent_cuda_20261102_v1/report.json` has SHA-256
`d85a12443483e2268a0849ad72b6afccf3b68804a536df06bb0f780620f41589`.
Both roots completed and returned `null`: none of the six non-neutral
candidates satisfied the existing strict conjunction. Model tangent FD
relative errors were 0.000583 and 0.000482, respectively. Positive-tangent
proposals improved adjusted-tail model scores but violated absolute-tail
requirements. This motivates searching the joint constraint set; it does
not prove that any solution exists or that the physical mechanism is absent.

Use the same source/checkpoint/goal-Q/root identity contract and the same
five ordered two-token parameter-space suffixes as the prior run. Baseline
means the zero **parameter** prefix; do not call it zero physical action
without decoding the action chart. Use no candidate, physical score, or
outcome from prior physical experiments as an optimizer input. The prior
CUDA record explains the development hypothesis but is not a tuning or
data-dependent seed source.

## Fixed search

For the common-tail immediate score `I`, final scores `F[0..4]`, and
their neutral-prefix counterparts `I0,F0`, let `d=F-F0` and
`r=d[1:]-d[0]`. The unchanged strict selector requires
`I >= I0+1e-6`, `mean(d) <= -1e-6`, every `d <= 0`,
`mean(r) <= -1e-6`, and every `r <= 0`.

The search minimizes the sum of squares of positive violations of exactly
those 12 inequalities. Four deterministic starts are `+0.06 e0`, `-0.06 e0`,
`+0.06 e1`, `-0.06 e1`, in that order, in the existing 10D bounded-action
**parameter** chart. Each start receives exactly four normalized gradient
updates of length 0.05, with projection after each update into the L2 ball of
radius 0.20 around the neutral parameter. A zero gradient produces a zero
update; no early stop, adaptive line search, alternate start or retry is
permitted. Evaluate and retain the initial point and all four post-update
points for every start, plus the neutral reference: 21 ordered rows in all.
Duplicate points remain separate receipts.

Only the unchanged exact strict selector above may mark a row eligible.
Among eligible rows, select minimum sum of the five final deltas, tie by
proposal index; otherwise return `None` without a zero fallback. A loss
improvement or a constraint violation close to zero is not admission. The
constraint loss is unnormalized and may emphasize some constraints; local
gradient optimization offers no feasibility or global-optimality guarantee.

Nominal per-root search budget is **21** prefix evaluations, **210**
suffix-token predictor samples, and **16** scalar prefix VJPs. Inherited
goal-Q construction, context/goal encoding, device sync, backward internal
operations, memory, wall time, model loading and root reconstruction must
be separately charged. This is not equal to the old 9-prefix/90-token/2-VJP
compiler budget and must not be presented as equal-total-compute evidence.

## CPU regression

From `le-wm-main`:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_reacher_pc_p1_constrained_common_tail_20261102_codexexclusive.py tests/test_reacher_pc_p1_common_tail_tangent_20261102_codexexclusive.py tests/test_reacher_pc_p1_common_tail_response_census_20261102_codexexclusive.py
```

These tests use analytic functions and verify fixed counts, a feasible
example, an infeasible/no-fallback example, the old selection logic,
per-tail counterexamples and CPU-neural refusal. They do not show real model
or physical improvement.

## Required future CUDA integration

The CUDA-only adapter is
`reacher_pc_p1_constrained_common_tail_20261102_codexexclusive.compile_reacher`.
The existing `run_reacher_pc_common_tangent_cuda.py` is a useful reference,
but cannot be reused unchanged: it binds the prior proposal SHA and assumes
seven rows. A new exclusive host launcher and saved consumer must be written
and reviewed before any separately coordinated CUDA use. In particular:

1. Authenticate source preflight, checkpoint and action stats, model
   parameter hash, device, root/goal/history and the inherited goal-Q Gram and
   FD receipts. Bind the new search code, strict selector and this protocol;
   the old `_02` preflight binds the *old* proposal and is not sufficient
   authority for the new search.
2. Use fixed roots `[20263160, 20263161]`, fixed starts/budget and exactly one
   search per root; save all 21 actions, constraint vectors or reproducible
   inputs, costs, traces, duplicates and explicit per-root errors. Check
   model parameters before/after. Do not change constants after a no-result.
3. If applying the prior separate response census, replay all 21 candidates
   with the same five suffixes and record all predicted endpoint vectors,
   bounded normalized/raw actions and exact bytes. This diagnostic adds
   21 prefix and 210 suffix-token samples per root; it is non-gating and its
   response spectra do not certify physical controllability.
4. Freeze the model-only report and full resource ledger before any separate
   physical protocol. An eligible predicted point is not physical evidence:
   retain every failure and apply unchanged gates only to properly sourced,
   fixed-denominator independent confirmation and appropriate baselines.

No GPU run, real-model output, physical result, equal-budget claim or P1
admission is created by this protocol.
