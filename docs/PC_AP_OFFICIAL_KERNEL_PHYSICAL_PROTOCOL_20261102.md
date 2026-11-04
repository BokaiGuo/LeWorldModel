# Pinned AP CEM kernel: real-RGB development protocol

Before this run, fix both known development roots 20263160 and 20263161,
the synthetic Reacher goal, model, action normalization, and the same memory
episodes 0, 2, 3 and 25-step retrieval-control input as
`PC_AP_REACHER_REAL_RGB_DEVELOPMENT_PROTOCOL_20261102.md`. Each root has two
arms: observed-memory target and final goal. Both run two five-action
decisions, execute and reobserve actual RGB/history, and use the pinned AP
torch CEM update with 16 candidates, two iterations, four elites and the
same per-root/per-decision generator seed. Each arm must run regardless of
outcomes in another arm. No episode is selected by results.

The callback uses the same frozen host-CUDA predictor and a CUDA-to-NumPy
bridge as the recorded-RGB kernel smoke. Counts and elapsed time include
that bridge; memory preparation and full model/resource cost are reported
separately and are not claimed matched. Raw simulator state is used only
for goal construction and evaluation; the planner receives only RGB, goal,
executed action history, frozen model and allowed observation memory.

Preserve all step-level qpos-distance, reward, termination and truncation
records. Early stop or exception must remain an explicit non-complete arm,
not a filled endpoint or a successful full-horizon run. Record model/runner/
kernel hashes, memory RGB hashes, paired root/goal hashes, each planning
seed, actions, retrieval target identity, actual prediction counts and wall
time. No official memory, query set or original 300-by-30 budget is claimed.
These roots and method choices have been exposed during development; the
result cannot establish independent generalization, P1 or full AP fidelity.
