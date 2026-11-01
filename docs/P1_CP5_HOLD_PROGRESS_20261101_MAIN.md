# P1 cp5 hold comparison: development evidence

This note joins already saved runs; it does not introduce a new selection
rule or an independent confirmation. All three prefixes are from a known
development root. The current P1 admission remains closed.

The same optimizer was applied after executing each prefix, with that
branch's actual cp5 RGB and action history. The later hold experiment also
scored each branch's own old plan tail using the same model and allowed it
to compete with the optimized tail. Each arm recorded one cp5 encoding, 532
predictor calls, 532 action-encoder calls and 256 reverse requests; the
counts include hold scoring but not all prior prefix-generation costs.

| Prefix | Hold model MSE | Optimized model MSE | Chosen | Physical result |
| --- | ---: | ---: | --- | --- |
| selected | 0.602856 | 0.208488 | optimized | no success by step 15 |
| zero | 1.477664 | 0.191065 | optimized | no success by step 15 |
| same-bank greedy | 0.710846 | 0.081224 | optimized | native and wrapped success at step 14 |

The previous fixed-plan run reported native and wrapped success at step 15
for the selected prefix with its old tail. The hold run selected the new
tail, which failed on this development instance. This is a concrete
model-score versus physical-outcome discrepancy for that selected prefix,
not independent replication or a general measurement of model accuracy.
The greedy run stopped at native success; its unexecuted step 15 is not
imputed. Different historical prefix costs prevent a claim of equal
end-to-end budget.

Evidence:

- `le-wm-main/results/codex_p1_diag32_cp5_hold_20261101_exclusive_run_CODEXHOLD01/report.json`
- `le-wm-main/results/codex_p1_diag32_cp5_hold_verify_20261101_exclusive_CODEXHOLD01.json`
- `le-wm-main/results/codex_p1_diag32_cp5_saved_paired_audit_20261101_exclusive_CODEX01.json`
- `docs/P1_BOUNDED_JOINT_SEARCH_20261101_MAIN.md`

The saved-artifact verifier checks the recorded bindings; it is not an
independent rerun of model inference or physics. The prospective total-cost
guard is still fail-closed for host operations not covered by the guard.
Historical reset coverage and checkpoint training-data membership remain
unknown. Do not admit a new independent confirmation until the actual
resource path, frozen method and source contract are established.

The paired audit additionally checks cp5 state, body snapshot, RGB, goal,
prefix and tail identity against saved physical branches. It found no saved
physical execution of the greedy arm's own hold tail in its searched panel.
Repeated executions of the same selected hold action sequence are not
independent task instances.

## Guarded visual probe

The saved-input host CUDA probe completed for structured and CEM visual
handlers. Each recorded one encoder and one projector return and an unchanged
model hash. The saved root encoding matched exactly. Its report is
`le-wm-main/results/p1_visual_guard_host_20261101_leibniz_RUN_LEIBNIZ01/report.json`.
This is not execution of the complete planning host or a physics experiment;
the guard's overall coverage and fairness flags remain false. An initial
invocation using relative paths failed the prepare-path check before the
probe; an absolute-path invocation succeeded.

## Presealed cp10 feedback comparison

The fixed control and cp10 real-RGB rescoring arms were run on host CUDA
and in the physical environment for all three known-root prefixes.
Both arms scored the same two remaining action candidates; the control
kept the cp5 selection, while the feedback arm selected by the fresh cp10
model score. In all three cases the fresh score still preferred the
optimized tail, so the executed controls and physical results coincided.

| Prefix | cp10 hold MSE | cp10 optimized MSE | Outcome in each arm |
| --- | ---: | ---: | --- |
| selected | 0.5202 | 0.2191 | no success at step 15 |
| zero | 0.9797 | 0.1262 | no success at step 15 |
| same-bank greedy | 0.2355 | 0.0395 | native and wrapped success at step 14 |

The greedy step 15 is unobserved after native termination. Each complete
episode used two RGB encodings, six action-encoder calls and six predictor
calls, with no backward calls; cached upstream candidate construction is not
balanced. This is a single known root, not six independent tasks, and does
not establish a fair end-to-end comparison. A saved-artifact verifier
checks binding and dispatch but does not rerun model or physics.
The historical successful hold and the selected optimized tail reach
different cp10 states: the physical pair compares complete cp5 tails, not
both candidate suffixes from the same cp10 state. The cp10 result establishes
an unchanged decision, not whether the local physical ranking is correct.

- `le-wm-main/results/codex_p1_cp10_rgb_refresh_control_20261101_exclusive_PRESEAL_CODEX01.json`
- `le-wm-main/results/codex_p1_cp10_rgb_refresh_runner_20261101_exclusive_run_CODEX01/report.json`
- `le-wm-main/results/codex_p1_cp10_rgb_refresh_verify_20261101_exclusive_CODEX01.json`

## Same-cp10 physical counterfactual

The two last-packet candidates were then physically executed after the same
ten actions for each known-root prefix. Saved verification matched the cp10
state, body snapshot, RGB, goal and preceding actions within each pair.

| Prefix | Hold final joint XY | Optimized final joint XY | Outcome |
| --- | ---: | ---: | --- |
| selected | 45.421 | 35.118 | both fail at step 15 |
| zero | 120.244 | 33.563 | both fail at step 15 |
| same-bank greedy | not comparable | not comparable | both succeed, at steps 12 and 14 respectively |

For selected and zero, the optimized last packet also had the lower
physical joint error at the observed common endpoint. The earlier success
of the complete selected hold tail therefore cannot be attributed to
selecting the wrong final packet from the optimized trajectory's cp10 state.
The greedy stop times differ and must not be treated as simultaneous
measurements. This is one development root, not independent confirmation.
The verifier checks saved records and does not independently replay physics.

- `le-wm-main/results/codex_p1_cp10_fixed_counterfactual_20261101_exclusive_PREP_CODEX01.json`
- `le-wm-main/results/codex_p1_cp10_fixed_counterfactual_20261101_exclusive_run_CODEX01/report.json`
- `le-wm-main/results/codex_p1_cp10_fixed_counterfactual_verify_20261101_exclusive_CODEX01.json`

## Same-cp5 two-packet prediction diagnosis

Host CUDA inference compared both complete tails from the same actual cp5
context with the already recorded RGB trajectories. Goal MSE is lower
better; physical errors are in the environment's position units.

| Step | Tail | Open-loop predicted MSE | Real RGB latent MSE | Pusher / object error |
| --- | --- | ---: | ---: | --- |
| 10 | hold | 0.922 | 0.809 | 81.45 / 66.31 |
| 10 | optimized | 0.967 | 1.057 | 98.76 / 87.54 |
| 15 | hold | 0.603 | 0.048 | 10.45 / 12.97 |
| 15 | optimized | 0.208 | 0.084 | 34.53 / 6.40 |

At cp15 the open-loop model ordering reverses relative to the real RGB
latent and joint position error; the optimized tail has the smaller object
position error alone. The model predicts each tail more favorably after
refreshing with that tail's actual cp10 RGB (hold 0.137, optimized 0.219),
but these observations come from different cp10 states and cannot both be
obtained by a single online execution before choosing the cp5 tail.

The diagnostic encoded 21 real RGB frames and used six action-encoder and
six predictor calls on CUDA. Model predictions are at packet boundaries, not
invented per-frame predictions. It is a retrospective, single-root
diagnostic; the saved verifier did not rerun inference or physics.

- `le-wm-main/results/codex_p1_cp5_tail_rgb_diagnosis_20261101_exclusive_run_CODEX01/report.json`
- `le-wm-main/results/codex_p1_cp5_tail_rgb_diagnosis_verify_20261101_exclusive_CODEX01.json`

## Single-packet selection and matched cp10 reversal

A development comparison, designed after the above observations and sealed
before execution, used the same two candidates and actual RGB feedback for
each prefix. Under selected, cp5 single-packet scoring chose the hold first
packet, but fresh cp10 scoring chose the optimized last packet; the result
failed. The double-packet rule also failed. Zero failed under both rules;
greedy succeeded under both at step 14. Zero and greedy selected identical
actions across rules and do not provide independent repetitions. Each arm
used two RGB encodings, six action-encoder and six predictor calls; upstream
candidate costs were not equalized.

A subsequent saved-record audit matched the selected single-packet branch to
the historical complete hold trajectory at cp10, including all first ten
actions, state, body snapshot, RGB pixels and goal. From that same saved
cp10 state, the model ranked the optimized last packet ahead of the hold
packet (goal MSE 0.12705 versus 0.13734), yet the two actual executions
gave joint XY 35.306 and failure versus 16.657 and success, respectively.
Both returned step 15; hold terminated successfully at that step. This is
different from the optimized-cp10 comparison above: the two matched tests
start at different cp10 states. It is a retrospective single-root comparison,
not proof of generalization or P1 admission.

- `le-wm-main/results/codex_p1_cp5_q1_feedback_runner_20261101_exclusive_run_CODEX01/report.json`
- `le-wm-main/results/codex_p1_cp5_q1_feedback_verify_20261101_exclusive_CODEX01.json`
- `le-wm-main/results/codex_p1_selected_cp10_hold_audit_20261101_exclusive_CODEX01.json`

## Local-perturbation scoring development comparison

After observing the above reversal, a separate experiment sealed the
candidate-scoring rule before executing all three prefixes. It compared
nominal cp10 scoring with a fixed local-action-perturbation score; both arms
received the same calls and used the same available candidate tails. Actual
execution used the original selected candidate, not a perturbed action.

| Prefix | Nominal / robust cp10 choice | Nominal outcome | Robust outcome |
| --- | --- | --- | --- |
| selected | optimized / hold | failure at step 15 | native and wrapped success at step 15 |
| zero | optimized / optimized | failure at step 15 | failure at step 15 |
| same-bank greedy | optimized / optimized | success at step 14 | success at step 14 |

Each arm recorded two RGB encodings and 14 action-encoder and predictor calls.
Historical candidate-generation costs remain outside equalized accounting.
The same-prefix cp10 identity was checked, and greedy's step 15 was not
observed after termination. This reproduces previously observed physical
outcomes on the same development root; it is not independent evidence,
calibrated uncertainty, proof of a mechanism, or permission to open P1.

- `le-wm-main/results/codex_p1_cp10_robust_feedback_runner_20261101_exclusive_ACTIVATION_CODEX01.json`
- `le-wm-main/results/codex_p1_cp10_robust_feedback_runner_20261101_exclusive_run_CODEX01/report.json`
- `le-wm-main/results/codex_p1_cp10_robust_feedback_verify_20261101_exclusive_CODEX01.json`

## Root1 development follow-up

The next numbered known development root used reconstructed DLS H3
candidates, not root0's diag32 generation method. In a separately sealed
six-arm real-feedback run, nominal and robust scoring chose the same actions
for every prefix. Selected and greedy also had identical prefix and
candidate actions, so they are not independent examples.

| Prefix | Both scoring arms | Last observed step | Joint XY |
| --- | --- | ---: | ---: |
| selected | success | 14 | 19.113 |
| zero | failure | 15 | 29.109 |
| greedy | success | 14 | 19.113 |

Each arm used two RGB encodings, 14 action-encoder calls and 14 predictions;
complete upstream candidate-generation costs remain uncertified. The four
successful branches terminated at step 14, so step 15 was not observed.
The saved verifier checked same-prefix cp10 state, snapshot, RGB and goal
bindings; it did not repeat inference or physics. This follow-up does not
distinguish the scoring rules, prove independent generalization, establish
end-to-end budget fairness, or admit P1.

- `le-wm-main/results/codex_p1_root1_candidate_binding_20261101_exclusive_CODEX01.json`
- `le-wm-main/results/codex_p1_root1_robust_feedback_runner_20261101_exclusive_run_CODEX01/report.json`
- `le-wm-main/results/codex_p1_root1_robust_feedback_verify_20261101_exclusive_CODEX01.json`
