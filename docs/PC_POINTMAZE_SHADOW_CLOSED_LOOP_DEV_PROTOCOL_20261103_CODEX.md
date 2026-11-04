# Shadow bridge planner integration: frozen development smoke

This is an exposed-development integration test, not a new independent
evaluation or evidence of control superiority. Select exactly source indices
0 and 8, names `r8343_a0_p2` and `r8345_a0_p2`, from the pre-existing
`pc_causal_mixed_rejected_physical_20261103_DEV01` source and cp12 source
binding. Retain each fixed 12-action prefix. Initialize both the shadow and
legacy native RGB bridges from the same environment-setup state and apply
identical actions to both at every tick. The legacy bridge is a same-action
render-clock control only; its RGB, simulator state, and any physical
position do not enter the planner or influence action selection.

At cp12 run five actual feedback iterations on the shadow bridge. At each
iteration derive causal proprio only from the two most recent **shadow RGB**
frames and their fixed 0.05-second interval. Feed the current image and this
RGB-derived proprio to the existing frozen model, then use its existing
single-frame, one-packet (five-action) unroll and `cem` implementation. The
scoring projection, RGB goal encoding, action normalization and clipping
match the existing cp12 executor. Use zero raw prior and a fresh fixed CPU
noise seed `2026110400 + 10*source_index + iteration`; CEM has 3×16 candidates
plus 2 final comparisons, four model batches and 50 scored samples per
decision. Execute only the first action of the selected five-action plan,
then re-encode the new shadow observation before the next decision. There are
2×5=10 decisions, 500 scored candidates in 40 model batches; encode one goal
and 6 observations per root (including the final feedback), 13 encodes total.
No parity calls, JVPs, training or parameter updates are permitted.

Before each environment dispatch, record a durable per-decision snapshot
containing the RGB feedback pair, derived RGB state, seed/noise, all CEM
candidate/score/elite traces, final selection and action; journal attempt and
return separately. Record both rendered streams, full action history, per-step
score and model/physical timing, checkpoint/normalizers, GPU/GL identity,
parameter hashes before/after, model calls and commands. Finish with 2×2
bridge resets and 68 high-level commands (34 per bridge); each command is five
native integration ticks. Record raw-vs-shadow RGB comparison only as a
rendering observation, not as a physical-state error or outcome metric.
No true physical position, setup state or native raw frame is accepted by the
planner adapter. No physical-position replay or performance claim is made.

Require CUDA and the host NVIDIA renderer; verify locked source files and model
normalization. Stop on any malformed observation, clock/action mismatch,
nonfinite score, source/hash drift, parameter mutation, GPU error or partial
worker result; never auto-retry an uncertain command. Use exclusively new
outputs; preserve the journal and completed snapshots on any failure. The
adapter test uses isolated fake bridges and does not itself prove model or
physics behavior.
