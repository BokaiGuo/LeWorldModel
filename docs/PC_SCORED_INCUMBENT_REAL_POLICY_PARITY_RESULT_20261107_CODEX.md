# Scored-incumbent output contract: real PushT policy regression

On the declared seed-85 four-task development panel, two separate GPU-backed
PushT evaluation processes used the historical scored-incumbent solver and
the action-only wrapper. Both completed seven planning decisions and 173
physical steps. The solver audit fields other than elapsed time, selected
actions and costs, sorted physical traces and evaluation metrics matched under
the declared comparison. Both runs reported one success in four tasks.

The audit stores selected actions, selection metadata, initial costs and final
branch costs, not all candidate actions or all intermediate candidate scores.
Therefore the real-policy comparison does not independently prove candidate
pool identity. The prior CPU/GPU synthetic tests did compare candidate batches.
The runner fitted the same scaler from the same dataset for both arms, but
did not record before/after checkpoint or normalizer hashes; model immutability
is not independently certified by this regression. The task rows were chosen
with `eval.num_eval=4`, not the earlier 50-task panel, and their independence
from training and earlier exposure is unverified.

Gym action-casting and Lance fork-safety warnings appeared in both runs. The
evaluator uses physical state for reset and measurement; this test does not
establish process isolation or certify pixel provenance. It demonstrates
compatibility with the installed policy and bounded outcome parity only: it
does not explain the three failures, establish a search improvement, or
change any PC0–PC5 decision.

The full traces, audits, metrics and comparison are in
`le-wm-main/results/pc_scored_incumbent_real_policy_parity_20261107_codex/`.
