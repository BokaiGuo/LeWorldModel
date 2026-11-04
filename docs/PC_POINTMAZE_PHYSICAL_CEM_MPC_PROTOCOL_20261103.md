# PointMaze privileged physical-CEM MPC: development protocol

Before this run, fix roots 1884–1887 and the sealed source
`a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json`, SHA-256
`ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade`,
including its source NPZ and initial/tail Gaussian innovations. On each
root use the same 3 iterations × 16 candidates, four elites, action bounds,
two five-action packets, then replan one five-action packet at cp5. Unlike
the visual planner, this diagnostic uses the privileged physical initial
state, physical transition simulator, cp5 state and goal position; candidate
cost is Euclidean distance to the fixed frame-10 goal position.

Use the audited position replay backend for all search and compare split
cp5/cp10 and continuous ten-action replay of each chosen chain; flag full
state differences, not merely position. Independently execute each chosen
chain in the native RGB bridge and save cp5/cp10 images and hashes; that
bridge does not expose simulator state, so physical cross-backend equivalence
for these actions remains unproven. Track every simulated candidate,
environment step, planner generation, elapsed time, and saved arrays.

The simulator is an approximate oracle with a position-only audited contract,
not a formally perfect dynamics certificate. It accesses privileged state and
task position, so it is not an RGB policy; equal candidate counts do not mean
equal computing cost. Previously exposed development roots are not independent
confirmation. Frame-10 position distance is not native success; do not choose
new roots or change parameters based on results.
