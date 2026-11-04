# Prospective development: spatial objective in real-feedback planning

Use archive roots 1884–1887, frames 0 and 10 as reset and RGB goal. These
roots are fixed by number before this run, not selected by outcomes. A
bounded search over PointMaze planning notes and local `a*pointmaze*` scripts
did not find them, but historical use across the entire project and model
training membership are not certified: label them development, not independent.

At each root compute the same unweighted 2D-to-pooled-visual Jacobian as the
prior A1 experiment (two JVPs and SVD), and generate five bounded constant
five-step packets along its top singular vector. Generate matching five-slot
seeded random-direction candidates and five axis candidates as in A1. For each
of the three banks, separately choose the minimum predicted pooled visual MSE
and minimum predicted full-spatial visual MSE at goal frame 10, giving six
arms. Ties go to the first index. This is a score comparison; neither arm is
the full task-weighted D2P method.

Each arm starts from the same native environment reset, executes its
selected five-step prefix, encodes the actual step-5 RGB, and independently
scores the same fixed eight-tail bank with its assigned metric. Execute the
chosen five-step tail and capture actual step-10 RGB. No simulator state or
physical labels are used by candidate selection. Once all arms are finished
for a root, evaluate their endpoints using the position-only offline physics
backend, with archive frame-10 position as the target. Record all scores,
actions, RGB hashes, physical endpoints, prediction and encoding counts,
JVP/SVD cost and elapsed time. Check same-prefix branches replay identical
RGB. Do not omit failures or tune choices based on outcome.

Each branch scores five prefixes and eight tails, with identical candidate
counts across metrics, but spectral candidates incur extra two JVPs and an
SVD; scoring two metrics from a shared prediction is not equal-total-budget
certification. The checkpoint uses fixed raw-zero proprio and a single-frame
context; normalizer statistics use the full archive. Camera calibration,
checkpoint training data, root development history and source provenance
are not independently certified. Position distance is not native success.
No claim of non-greedy mechanism, official baseline parity, independent
generalization or P1 is supported by this exploratory run.
