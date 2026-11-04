# Reacher nonlinear spectral prefix refinement: development protocol

The preceding response-screening experiment did not change the selected
actions. This pre-execution development protocol asks whether differentiable
nonlinear lookahead can generate new candidates from the same guarded
spectral directions on the two already outcome-exposed roots. It is not a
confirmatory experiment or a replacement for the existing P1 gate.

Keep the complete guarded bank, including neutral and unchecked DLS.
Independently refine up to the first four guarded spectral candidates in bank
order. For each, fix the same five one-packet suffix parameters (zero and
±0.3 repeated along each of the two actuator axes). Optimize the prefix
parameters for four Adam iterations at learning rate 0.08, projecting each
iterate into a radius-0.25 Euclidean ball around its spectral initialization.
The loss is a temperature-0.01 soft minimum of the five predicted terminal
goal-coordinate MSEs plus 10 times the squared hinge for immediate predicted
goal cost below the neutral prefix's immediate cost. The hinge is a proposal
heuristic, not proof of non-greediness or physical capability.

Append each refined prefix to the unmodified guarded bank. Apply the existing
predicted-successor CEM selector, seed and budget per candidate unchanged.
Record all initial/final prefix parameters, losses, finite-value checks,
candidate origins, selected index, forward sample counts, backward calls,
source hashes and model hashes. Charge refinement separately from guarded
SVD and suffix CEM. This is not equal-total-budget to the original method;
no state-based cost may be fed to the planner.

Only if a genuinely distinct action is selected should a separately sealed
development physical execution be considered; compare original and new
selected actions, zero and greedy, with the same observed feedback and
planner. Same-action common-suffix interventions and own-tail optimization
remain distinct. Do not consume independent confirmation roots before a full
method and resource contract are frozen.
