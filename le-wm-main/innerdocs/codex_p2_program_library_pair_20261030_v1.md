# P2 fixed program-library comparison: kernel, not yet a run

Known cp15 context and goal are shared. Compile one 192x20 H2 Jacobian and one
SVD at the common context. Fix four 5-step spectral primitives for each of two
positions before candidate execution. Enumerate their 4x4 ordered Cartesian
chains, so a library element is reused across several chains. Compare against
16 fixed full10 joint spectral proposals from the same right singular basis.
Stable first minimum selects a terminal objective. Both arms make 16 two-packet
candidate rollouts and one two-packet winner replay: 34 packet returns and 17
scores each, with shared compilation charged once.

The chain witness contains both literal float32 action blocks, identities and
full-context input/output hashes. This is fixed open-loop composition of
spectral action primitives, not proof of independent task-level skills,
arbitrary-depth composition, or feedback during the chain. The static library
is derived from the joint root Jacobian, not pretrained independently; report
this when interpreting the comparison. Same operation limits do not establish
equal GPU FLOPs, wall, energy or model-load cost. Degenerate SVD bases can vary.

The callback kernel and separately implemented saved-structure validator
are tested on synthetic callbacks. A CUDA-only H2 transition/Jacobian adapter
is implemented, but has not been tested against the real model. It requires
the caller to provide the meter. Neither module is a production driver.
Before execution it still needs: pinned model/task/context and CUDA source
prepare; actual model-entry and SVD accounting and synchronized timing;
lock held before initialization through context cleanup;
durable failed/no-retry journal; independent saved verifier binding the chain
to actual raw actions and model context; and a new physical adapter. The saved
validator checks candidate/context arithmetic, not actual CUDA source identity
or the future journal. No fixture
output may be treated as an experiment. The user authorized exploration with
P1 still closed; no P1 or P2 gate is changed.

The library explicitly keeps degenerate or identical primitives: zero
subvectors are not redrawn. A future report must disclose the number of unique
action sequences; 16 scheduled combinations need not mean 16 distinct actions.
