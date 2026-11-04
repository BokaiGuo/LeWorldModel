# Spectral-seeded feedback CEM: interface and development contract

Add an opt-in, training-free candidate source for the existing five-action
PointMaze shadow-RGB feedback planner. This document freezes the algorithmic
contract for a new development component, not a scientific confirmation or a
claim of matched total compute. Do not change the SHA-bound smoke executor or
its saved receipt.

The model-facing caller supplies only a differentiable two-component projected
task prediction as a function of the ten **normalized** action coordinates,
the frozen two-component action mean/std, and the existing CEM noise and
scoring callback. The callback is responsible for using RGB-derived context
and the frozen model; simulator state, endpoint labels and physical outcomes
must not be inputs. At raw zero action, differentiate the two predicted task
coordinates with respect to normalized actions, then apply the exact
normalization chain rule to obtain the 2×10 response to raw actions. Compute
an SVD, choose the top nonzero right singular direction, fix its sign using
its first largest-magnitude coordinate, and scale it to raw infinity norm one.
Create four **raw** five-action candidate programs with amplitudes
`+0.25,-0.25,+0.5,-0.5`, verify raw bounds `[-1,1]`, and convert to
normalized coordinates exactly once. If the response is nonfinite,
non-differentiable or has top singular value <= 1e-8, fail closed.

Reuse the historical CEM's 3×16 innovations, prior, normalized bounds,
first-candidate mean, four elites, mean/std updates and two-final-candidate
comparison. The spectral arm replaces slots 1–4 only in the first generation;
the random arm has no injection. Save candidate, score, elite and update traces,
all raw/normalized injected candidates, response Jacobian and singular values.
No environment execution is authorized before a caller separately binds
inputs, records resource use and validates the action-clock contract.

Count separately response forward calls, reverse-vector gradient calls, SVDs,
CEM model samples and batches, RGB encodes, and wall time. The spectral arm
has extra structure cost; matching CEM samples is not equal-total-compute
evidence. A later comparison must let controls use equivalent resources under
a fixed policy, plus same feedback and execution horizon. Tests cover
normalizer chain rule by finite differences, sign and bounds, fail-closed
behavior, trace reconstruction, and exact no-injection parity with the
existing CEM. Tests do not establish neural or physical performance.
