# Push-and-return task generation and Euclidean control

## Task preparation

Two predefined root/object-goal combinations, with pusher goal reset to initial
position. The renderer first tried native state7 reset (rejected because reset
advances physics) and full snapshot restoration (still three pixels different
on root0, rejected without relaxing tolerances). Literal original placement,
warmup and ten witness actions then reproduced original cp10 images exactly.
A NumPy-to-Pymunk tuple ABI error was corrected before completing generation.

Target receipt:
`le-wm-main/results/p1_return_target_cuda_20261101_main_MAIN06.json`,
SHA256 `5bd5f5bbe82fe085cf26ceef12dc8ff6e0bb2fbc784e241ae4f5f8e4f693ed8a`.
Both images were encoded on host CUDA with unchanged frozen weights.
Historical replay used only previously fixed commands; it is not a new planning
success. Counts/commands are retained in target generation metadata.

Root1 is already successful under the unchanged native predicate (initial joint
XY error14.07463). Its four slots are explicitly ineligible, not replaced and not
counted as control successes. Root0 is eligible (initial joint XY error101.67640).

## Actual control results

All four root0 arms completed20; success0/4. Same prefix-protected diverse carry
method as before; no outcome-based candidate modification.

| Arm | cp5 pusher error change | cp5 object position error change | cp5 joint XY change |
|---|---:|---:|---:|
| structured | +63.81355 | -37.47189 | -11.15354 |
| native | +42.22434 | -33.89369 | -21.81784 |
| random | +53.12809 | -27.09446 | -10.10643 |
| immediate_progress | +44.38319 | -16.93390 | -6.01470 |

Pusher regression alone is not no-immediate-task-progress: every arm improves
joint XY error. No non-greedy capability claim follows. Reachability of the new
target within20 is unproven, so zero successes is not a theoretical impossibility.

Joined analysis:
`le-wm-main/results/p1_return_goal_analysis_20261101_main_MAIN01.json`.
Saved node validation:
`le-wm-main/results/p1_return_goal_saved_20261101_main_MAIN01.json`,
16 actual nodes with action/source selection and ledger/hook checks.

## Goal-only metric candidate

The existing goal-encoder co-metric was measured on the eligible target:
`le-wm-main/results/p1_goal_cometric_cuda_20261101_main_MAIN01.json`,
SHA256 `3ce857733d7ac413aa322e96fa4955f6db94bea53b55add7a5b0527148a21925`.
The unperturbed embedding matched, the original finite-difference thresholds
passed and frozen weights stayed unchanged. Construction uses13 encoder forwards
and192 input VJP rows. This is not control validation or P1 admission.

Next prospective controller uses Q consistently in all six relevant stages,
with unchanged evaluator and separately reported metric-construction cost.
