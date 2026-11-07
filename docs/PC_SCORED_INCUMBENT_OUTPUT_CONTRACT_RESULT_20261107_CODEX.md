# Scored-incumbent output contract: result

The historical PushT scored-incumbent solver returns its selected executable
chain as both `actions` and `mean`, and returns the final best-mean branch's
standard deviation as `var`. When a sampled candidate wins, those fields need
not describe the selected candidate's proposal distribution. This is an output
contract defect; it does not by itself establish incorrect action selection or
explain the negative planning comparison.

Recounting the existing 78-decision audit: 18 selected a final mean and 60
selected a sample. In nine of the 60, the sample came from a different branch
than the final best-mean branch; 14 samples came from a round before the last.
These categories overlap, and the audit does not save the sampling standard
deviations needed to reconstruct the selected sample's distribution.

`ExplicitPrefixSelectedActionOnly` calls the unchanged historical solver and
returns only its selected actions, evaluated costs, and an explicit
`selected_action_distribution: unavailable`. It intentionally omits `mean`
and `var` so an incompatible consumer fails rather than receiving misleading
statistics. The installed `stable_worldmodel` policy currently consumes
`actions`; other consumers expecting `mean` or `var` require adaptation.

CPU and CUDA tests used the same deterministic synthetic scorer, two prefix
modes and three seeds: all five candidate batches, selected actions, selected
costs and audit records matched exactly between implementations. Together with
the existing scored-incumbent tests, **14 tests passed**. This is not a
real-checkpoint or full-environment rerun; the historical 35/50 result remains
unchanged. The new wrapper is opt-in, not globally installed, and no PC gate
is changed.
