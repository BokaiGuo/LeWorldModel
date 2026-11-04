# Nonlinear spectral-prefix refinement: development result

The fixed protocol is `PC_REACHER_SPECTRAL_PREFIX_REFINEMENT_PROTOCOL_20261102.md`.
Both previously exposed roots completed on host CUDA; model parameter hashes,
source checkpoint and RGB root/goal bindings were checked. This is a
model-only development experiment, not a physical test or P1 admission.

| Root | Original candidates | Added refinements | Selected index | Same as previous selected |
|---|---:|---:|---:|---|
| 20263160 | 6 | 4 | 2 | yes |
| 20263161 | 8 | 4 | 5 | yes |

All eight refined candidates have lower immediate predicted goal cost than
their respective neutral controls. The unchanged selector requires immediate
cost at least 1e-6 *higher* than neutral, so none is eligible; the penalty in
the refinement loss did not enforce this constraint. This is a concrete
model-side candidate/selector mismatch, not evidence of a physical mechanism
failure. Some unrefined candidates are eligible and the original selections
remain unchanged. The guard's non-greedy requirement is a property of this
development selector, not a replacement for the physical P1 definition.

For each root, refinement used 17 prefix and 85 suffix forward samples plus
16 backward calls, separate from the existing guarded spectral compiler. The
augmented CEM selector used 660 and 792 predictor samples respectively,
versus 396 and 528 for the original selector. Encoder and total resources
are not certified. The increase is not an equal-budget comparison.

No new selected action exists to justify replaying prior physical trajectories
as fresh evidence. The next development change should enforce feasibility
explicitly during candidate generation or retain feasible iterates, with a
new protocol and separate cost accounting; it must not reinterpret the soft
penalty as a guarantee or alter the old results.

Artifact: `le-wm-main/results/reacher_pc_prefix_refinement_development_20261102_v1/report.json`.
The corresponding unit and adjacent regression suite passed 16 tests; these
tests do not measure efficacy.
