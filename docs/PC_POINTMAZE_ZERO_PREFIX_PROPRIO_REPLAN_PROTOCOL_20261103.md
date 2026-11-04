# Zero-prefix RGB feedback replan, matched tail optimizer

On all exposed development roots 1888–1895, execute five raw zero actions
from the same reset as the saved proprio-spectral CEM experiment. Observe
real RGB at each step, compute causal cp4-to-cp5 RGB position/velocity using
the identical camera and static geometry, then encode the true cp5 RGB and
replan a five-step tail with the same frozen model, RGB-derived goal
proprio token with zero goal velocity, score, CEM optimizer and 3×16 tail
innovations as each saved random/spectral arm. Execute the chosen tail.

Bind all checkpoint, input and source hashes. Save full histories, CEM
traces, raw actions, counts, timings and terminal RGB. Compare terminal
RGB/positions with separate legacy-render and position-backend replays,
and check the zero-prefix cp5 state against the saved common-suffix panel.
Only after planning is complete, evaluate using physical goal position and
report strict offline terminal distance `<0.5`, including failures.

For each root compare the separately optimized zero-prefix result with the
already executed random/spectral prefixes and their separately optimized
tails. All three tail planners have the same samples and feedback
opportunity; prefix planning costs differ, and no full total-compute
equality or greedy/strong baseline is claimed. Roots are exposed and their
training provenance remains unknown. This is not independent P1 evidence,
nor does an optimized-tail difference alone prove non-greedy mechanism.
