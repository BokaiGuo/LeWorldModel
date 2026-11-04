# P1 common-tail response census and counterexample

Status: CPU analytic tests and diagnostic implementation only. No neural
model, GPU, physical rollout, training, gate or historic result was changed.
The existing candidate generator and its preflight receipt remain intact.

## Counterexample

Consider a differentiable two-coordinate prefix parameter `x` with immediate
cost `1+x[0]`. Let `g=1+x[1]` and `z=sqrt(2+0.1*x[1])`. The neutral-tail
endpoint is `(0,0,z)` and the four active-tail endpoints are `(±g,0,z)` and
`(0,±g,z)`. With identity Q and mean squared goal-zero error, the neutral-tail
cost is `(2+0.1*x[1])/3`, and each active-tail cost adds `g**2/3`. The fixed model-only selector
can choose a candidate with positive `x[0]` and negative `x[1]`: immediate
cost increases, every common-tail cost decreases, and each active-minus-neutral
tail cost decreases. Yet the finite action response vectors are `(±g,0,0)`
and `(0,±g,0)`, so the chosen candidate has strictly smaller response
singular values than the neutral prefix.

The example is synthetic; its scores and response vectors use the same Q and
the same predicted endpoints. It
shows that the selection rule does **not logically guarantee** improved
physical suffix ability, model controllability, or task success. Nor does
smaller response norm by itself prove reduced useful controllability.

## New non-gating diagnostic

The new census consumes **predicted** terminal coordinate matrices for the
same five fixed tails under the neutral and candidate prefixes. It applies
the existing goal-Q metric, forms the two centered +/- finite-action response
columns, and reports both singular spectra, Frobenius-norm ratio and
per-singular-value ratios. When a baseline magnitude is zero the
corresponding ratio is `null`, not infinity or an imputed success.

It does not select, reject, reorder or modify candidates. It cannot infer
physical outcomes, global/nonlinear controllability, or the eventual effect
of a separately optimized real-feedback suffix. Those require predeclared
physical common-tail interventions and equal-budget planning comparisons.

For a future model-only runner, record all five predicted endpoint vectors
before converting them to scalar Q costs; use exactly the same tail ordering
and action chart for each prefix. Call this diagnostic for all fixed
candidates, preserve source and compute receipts, then separately determine
whether any diagnostic hypothesis warrants a **new** predeclared protocol.
Do not retroactively change the existing selector, threshold, preflight or
outcome denominators.

CPU tests:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_reacher_pc_p1_common_tail_response_census_20261102_codexexclusive.py tests/test_reacher_pc_p1_common_tail_tangent_20261102_codexexclusive.py tests/test_reacher_pc_p1_common_tail_preflight_20261102_codexexclusive.py
```

The future host-CUDA model launcher described in the original development
protocol is still not implemented or authorized. This census does not close
that implementation gap or open P1.
