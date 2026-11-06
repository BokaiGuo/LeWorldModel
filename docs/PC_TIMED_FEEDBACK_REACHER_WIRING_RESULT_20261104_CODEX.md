# Timed feedback entry point: Reacher RGB wiring result

The prespecified one-task diagnostic completed three real RGB feedback
decisions for each arm. Both arms started with identical saved images and
action history and used the common metered entry point at 15/10/5 remaining
steps. Decision, call ledger and trace were sealed before each dispatch;
old tails were re-scored in the next observed context. The saved-data audit
checked hashes, actions, packet counts, budgets, clock, carry and selected
chains. A second audit checked the initial RGB/history match, successor
image bindings and final paired choices. Same-backend physical replay
reproduced the saved image histories and physical distances.

| Arm | Initial distance | After 5 | After 10 | After 15 |
| --- | ---: | ---: | ---: | ---: |
| v17 | 1.086652 | 0.497179 | 1.050470 | 1.404377 |
| Multistart CEM | 1.086652 | 0.622860 | 0.898435 | 1.468866 |

Both arms finished farther from the physical target than at the initial
state in this task. This single exposed development task does not
establish task feasibility, an advantage, or a gate result. Planning and
evaluation shared a process; the checks are not an isolation or independent
replication certification.

The first incumbent audit used a strict `1e-7` selected-score tolerance
and failed on the CEM second decision. Inspection showed the *same exact
float32 selected action* scored `0.6650121212` in an earlier batch and
`0.6650122404` in the final pair, a difference of `1.1921e-7`. The audit
now reports selected-minus-best recorded score rather than silently
classifying a numerical tolerance as a lost chain; a separate check
confirms the saved selection is the actual minimum of the final pair.
The source of the cross-batch numerical difference is not established.

Raw artifacts and audits:
`le-wm-main/results/pc_timed_feedback_reacher_wiring_20261104_codex/`.
Protocol: `PC_TIMED_FEEDBACK_REACHER_WIRING_PROTOCOL_20261104_CODEX.md`.
The one-attempt stopping rule is reached; do not retry or replace the root.
