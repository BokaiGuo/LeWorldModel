# E1 prospective shared executable target: bounded seam

Existing E1 six-slot work used different arbitrary, FIRST and compiled targets.
Its completed tracking and physical results do not establish a same-target
intervention. The frozen controller core `optimize(task,root,arm,target,dispatch,sink)`
accepts a target separately; its score combines prefix target MSE and terminal
task-goal MSE. Its existing `arm` names identify proposal sources, not three
distinct control algorithms. Feeding one target to those names with identical
seed and inputs is not a meaningful three-method comparison.

Prospectively MAIN must specify, before outcomes, an actual intervention and
control, fixed roots and horizon, target generation/selection and resource caps.
One target per root is then delivered byte-identically to both controllers;
the common task, history, action mapping and goal are fixed. Witness actions
are excluded from controller input unless an explicitly different intervention
and information budget are preregistered. Seal proposal, target, witness and
all plans before evaluating native outcomes. Use same-time actual RGB, native
state and original goal metrics; keep missing/early-stop slots, no winner or
post-hoc threshold. Record generation, controller, replay, encoder and physics
costs separately; FLOPs unknown. Known roots are engineering calibration,
not independent confirmation. The scientific gate stays false.

An action witness plus model prediction is only model reachability, not native
executability. A claim of native attainability needs a previously authorized,
independently recorded same-root natural replay, exact action/time and RGB/state
provenance, and a preregistered arrival criterion. Existing receipts do not
automatically provide this new prospective criterion or authorize new replay.
Never select a target retrospectively from observed physical outcomes.

The pure `assemble(records, interventions)` seam checks fixed root order, exact
little-endian f32[192] bytes and SHA, finite target and explicit source hashes;
it copies one payload for every intervention and isolates witness references.
Its hashes are *untrusted declarations*, not verification of files or receipts.
It intentionally returns `production_authorized=false` and cannot release a
plan. It is not a production driver or authority-generating manifest. A real
adapter must authenticate task/history/goal, witness and complete receipt, bind
an approved prospective choice and independently enforce the information and
resource boundaries; no exact future authority is assumed here.

RAM tests only: `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e1_shared_executable_target_20261029_v1.py`. No NN, physics, training, source mutation or fullstrict.
