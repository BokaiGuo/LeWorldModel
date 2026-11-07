# RGB image boundary feedback parity: development result

The predeclared engineering regression replayed exposed TwoRoom roots 10–13
using the four existing strategies and the frozen LeWM model on GPU. It replaced
the old boundary with `RGBImageLeWM`, while retaining the existing planner,
seeds, evaluator resets, and feedback rules. The new output was saved before
the historical output was opened for comparison.

Across 16 arm trajectories, 70 decisions, 307 environment steps and 196 model
receipts, the comparison found zero differences under the protocol: exact
inputs, candidates, selections, actions, and termination; score tolerances
`atol=1e-4, rtol=1e-5`; physical-distance tolerance `atol=1e-6`. All 16
trajectories terminated successfully, matching the historical run. This result
does not compare timing or audit fingerprints, which intentionally differ.

The four boundary and track test modules passed: 47 tests. Gym emitted existing
reset observation-type and NumPy-casting warnings; this regression did not
establish their wider implications.

The runner explicitly uses the new boundary. Other entry points remain
unchanged; this is not a claim that the boundary is globally enforced. Shape
checks do not establish image provenance or prevent a privileged caller from
encoding state as pixels. Evaluator state is used only to reset the simulator,
not supplied to the planner in this experiment.

The panel was previously exposed and small. These results support bounded
implementation parity only, not superiority, independent confirmation, or
PC0–PC5 passage. The existing method still needs a demonstrated incremental
benefit over the strongest applicable baseline.
