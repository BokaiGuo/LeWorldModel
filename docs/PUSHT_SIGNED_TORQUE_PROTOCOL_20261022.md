# PushT signed torque-response confirmation

The completed 2026-10-21 panel classified which of `+y` or `-y` produced the
larger **absolute** zero-drift-corrected orientation response. Its fresh held
test result was not confirmed (`0.6042` balanced accuracy; lower confidence
bound `0.4792`). More importantly, that label is a sensitivity target: it
cannot tell a controller which torque reduces a signed angle error.

The initial signed draft still used world-frame `+y/-y` pulses. This made its
action meaning depend on block orientation and the contacted T edge, so it
could again mix opposite torque semantics into a single label. It is retained
as an unrun draft, not used for confirmation.

The v2 independent protocol keeps the RGB/action-history observation contract,
but uses the actual contact geometry to create a matched pulse pair:

\[
a_{\pm}=\alpha\,(-\hat r) \pm \beta\,R_{90}(-\hat r),
\]

where \(r\) is the block-centred contact radius. The pair has equal norm, the
same radial-inward contact component, and opposite geometric moments. The
protocol renames the actions by moment sign before constructing the label:

\[
\operatorname{sign}\bigl(\Delta\theta_{\mathrm{positive\ moment}}-
\Delta\theta_{\mathrm{negative\ moment}}\bigr).
\]

It therefore tests whether a visual history can identify the **direction** of
the local torque response. The three new high seed bands, balanced cohorts,
fixed readouts, and confirmation thresholds are declared in
[`run_pusht_signed_temporal_torque_confirmation_cuda.sh`](../le-wm-main/experiments/run_pusht_signed_temporal_torque_confirmation_cuda.sh).

For every source root, the zero, positive-moment, and negative-moment arms
also share the exact same environment reset seed, state, and goal. Thus the
declared action is the sole counterfactual difference; a future stochastic
environment implementation cannot silently turn reset randomness into a
signed-torque label. If a high seed band cannot form the predeclared balanced
cohort, the runner writes `rejected_pre_outcome_balanced_cohort` with aggregate
counts and does not fit a decoder, collect later-split labels, or materialize a
test score.

This is still only a necessary offline action-effect test. A positive result
would not establish action ranking over a full bank, contact preservation,
angle closure, or closed-loop planning. A negative result would show that the
remaining failure is already present at signed local-response observability,
after controlling the contact-frame action basis but before multi-step
planning.

## Candidate-bank matched V3

The radial V2 pair is a clean physical mechanism probe, but it is not the same
action family as the nearest-polygon-boundary bank used by the mode-aware
closed-loop controller.  On the asymmetric T, block-centred radial and
boundary-normal/tangent coordinates can disagree.  V2 therefore cannot be
used to attribute a later candidate-bank ranking result.

`run_pusht_candidate_bank_signed_temporal_torque_confirmation.py` resolves
that contract mismatch without rewriting V2: it asks the deployed
`candidate_actions(..., boundary_action_frame=True)` builder for its actual
full 19-action bank (`tangent=[.06,.10,.16]`,
`normal=[.02,.04,.08]`), extracts the exact `.08/.16` mirror pair from that
bank, and names the pair by actual geometric moment sign. A missing mirror or
a changed bank is rejected before labels or decoder fitting. The new fresh
CUDA-only protocol is launched with:

```bash
bash experiments/run_pusht_candidate_bank_signed_temporal_torque_confirmation_cuda.sh
```

This V3 result will answer whether RGB history distinguishes signed response
for the **same action semantics the ranking controller must choose among**.
It still cannot itself establish a learned ranking or pose closure; those
remain downstream, contingent on the candidate-bank/horizon feasibility gate.

V3 additionally records a transparent two-member transduction audit. For each
held root, it evaluates both declared desired-turn signs and maps the frozen
probability that the positive-moment pulse has the more positive response into
a `plus` or `minus` choice. The report includes local pair-ranking accuracy and
the observed signed-response gap lost by a wrong choice. This only checks the
sign-to-action interface: it is not full-bank regret, goal-pose closure,
contact retention, or a closed-loop result.

## Candidate-bank/horizon gate before learned pose ranking

The earlier small-bank privileged audit found that a contact root can remain
strictly pose-unreachable even when a local torque contrast is measurable.
Consequently, a learned ranker's `0/N` closure result is not interpretable as
an action-identifiability failure unless the exact candidate bank and horizon
first pass a privileged feasibility ceiling.

`experiments/preflight_pusht_candidate_horizon.py` now freezes that condition:

1. It creates disjoint, unique, **nonterminal** calibration and outcome
   boundary-contact roots.
2. It applies the same ordered contact-frame bank, pulse length, macro-step
   budget and three declared oracle horizons to every root.
3. Calibration reachability alone authorizes or rejects a future learned
   panel. Outcome oracle reachability is diagnostic and cannot alter that
   authorization.
4. A learned panel must provide the identical protocol digest and ordered
   root/action-bank digests. A mismatch, terminal root, or failed calibration
   ceiling is fail-closed as `candidate_or_horizon_limited`.

The CUDA-only launcher is:

```bash
bash experiments/run_pusht_candidate_horizon_preflight_cuda.sh
```

It retries CUDA preflight a bounded number of times and exits without a CPU
fallback. Oracle success remains a privileged upper bound, never evidence of
an RGB ranker, frozen LeWM planner, pose closure, or causal dynamics.

The same contract is now consumed by the mode-aware learned closed-loop
runner, rather than being left as a separate diagnostic.  Once an authorized
preflight exists, the only compatible invocation must specify its exact
contact-frame bank and execution budget, for example:

```bash
.venv/bin/python -m experiments.run_pusht_mode_aware_shared_candidate \
  --output results/pusht_mode_aware_preflight_matched_cuda \
  --device cuda --root-protocol boundary_contact \
  --oracle-feasibility-preflight results/pusht_candidate_horizon_preflight_20261022_cuda/preflight.json \
  --held-out-roots 32 --pulse-steps 2 --macro-pulse-steps 2 \
  --closure-steps 18 --lookahead 2 --inward-magnitude 0.08 \
  --tangent-magnitudes 0.06 0.10 0.16 \
  --contact-normal-magnitudes 0.02 0.04 0.08
```

The runner recreates the predeclared outcome roots and hashes the initial
ordered candidate bank on every root before fitting or evaluation.  It does
not receive successor states or oracle-reachability labels at runtime.

For prospective execution, use the serial GPU-only wrapper instead of running
the two stages manually:

```bash
bash experiments/run_pusht_mode_aware_preflight_matched_cuda.sh
```

It runs the feasibility ceiling, reads only its calibration authorization, and
starts the learned loop only when that authorization is true.  If the ceiling
fails, no learned outcome artifact is created and the correct result is
`candidate_or_horizon_limited`, not a negative verdict on representation or
ranking.
