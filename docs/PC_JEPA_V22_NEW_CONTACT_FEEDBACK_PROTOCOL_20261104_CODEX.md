# v22 native JEPA contact feedback: fixed development protocol

Use the sole seed 2026110451 and the constructed, separately replayed
witness described in `PC_JEPA_V22_NEW_CONTACT_WITNESS_PROTOCOL_20261104_CODEX.md`.
The goal RGB is the witness terminal image. Witness actions and physical
state are evaluator-only and excluded from the isolated planner mounts.
If the witness is illegal, trivial, or fails exact replay, do not run
the planner. The seed, normalizer and checkpoint provenance are not
certified independent.

Attempt each arm once: v22 response and the shared-bank multistart CEM.
Both receive the same initial history, goal RGB, normalized proprioception
and previous action history, and the same 1,920 prediction-token budget
at each fixed-terminal horizon 15/10/5. Re-score the prior selected tail
under the new observation, seal each decision before executing the first
five raw actions, and save full prediction batches, actions, contexts,
costs and physical evaluator records. Use the common v22 timed planner.
Encoding and synchronized planning time are separate; tokens do not
represent complete compute cost.

Require the existing network-isolated worker sandbox and strict model
loading; do not fall back to a privileged planner. Stop after one attempt
per arm, including any infrastructure or execution failures; no retries,
seed replacement or horizon extension. Report object position, angle,
agent position and the established joint-position/angle success rule.
Audit input, trace, resource, execution bindings and same-backend replay.
This single constructed task is an integration diagnostic, not a
performance comparison, independent replication, or PC gate result.
