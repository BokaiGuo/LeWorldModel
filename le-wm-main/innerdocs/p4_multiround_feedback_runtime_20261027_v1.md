# Fixed known-root multiround feedback runtime — prepare-only handoff

This new runtime is a single reused root1 calibration, not P1 or P4 admission.
No old source, report, seal, metric, outcome gate or central progress file is
modified. This delivery runs only non-neural fixtures and model-free prepare.
The main thread must review before invoking the CUDA/physics `run` entry point.

## Frozen experiment

Use original root SHA
`23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050`,
original selected prefix5, original latent goal and Q, original reset RGB and
between-boundary raw actions. Selected is the minimal fixed prefix role, not
an outcome-based new selection. Both arms independently reset to the same root.
At boundaries 5, 10 and 15 compile exactly ten full10 spectral candidates and
execute the stable nonlinear original-Q argmin's five primitive commands.
full10 means ten logits mapped to a 5x2 packet; it does not mean ten physical
steps. Each full arm has 20 steps, 40 selected-path steps total. There are no new
root draws for inference: the original deterministic four-root reconstruction
is checked in its entirety, only original root1 is used for these two arms.

Actual uses its own three observed RGB boundaries and two between-state packets.
Imagined rolls its own predicted context from the original sealed latent input,
without injecting live observations. Both encode the same number of observed
RGB frames at boundaries for explicit observation cost; the imagined selection
does not read these embeddings. Once actions diverge, neither arm borrows the
other arm's old latents, images, actions, selections or endpoints. Actual older
latents are reencoded from its own older RGB boundaries, not patched from the
imagined stream. Fixed original goal/Q is used at every boundary, never refit.

## Existing machinery and witnesses

The runtime calls `pusht_full_packet_compiler_20261027.make_transition` and
`spectral_bank` for normalized packet derivatives, bounded ridge and signed
spectral proposals. It calls the existing suffix compiler's `verify_plan` for
saved bank reproduction, tanh action mapping, nonlinear costs and stable argmin.
It does not implement a replacement gradient generator. Independent selected
replay uses the sealed `CudaReplay`. Encodes use existing `encode_cuda` and
the original image transform/public action normalization. No extra /100.

Each plan's `prefix` remains the original prefix for compatibility with the
upstream verification schema, NOT a claim that those five commands were just
executed at later boundaries. The selection envelope separately binds actual
incoming raw packets, boundary, full live context, original metric, all ten
candidates/Jacobian and selected successor. SHA and full envelope are fsynced
before physical dispatch. All six compile slots, including blocked slots, are
persisted before allocation. Step5 must exactly match the saved original
selected-prefix pixels before first suffix compilation or dispatch.

## Guard, accounting and partial durability

The unchanged RoyalBlue RGB guard authorizes each primitive command only from
fresh visible RGB. It cannot prevent the first out-of-view transition. Any
missing observation, environment termination, encoding/compilation/replay
failure, or integrity failure stops that arm; no retry, substitute shorter
endpoint, new action, new root, or outcome-based continuation. Other scheduled
arm slots remain present. Completed packets alone advance RGB/action history.

On a full run: 80 predict samples (2 fixed-prefix predictions + 78 compilation/
bank/replay), 6 reverse Jacobians with 1152 scalar VJP directions, 12 SVD calls,
6 encode calls/18 frames and 40 selected-path steps. Actual visual-encoder
invocations/frames and action-encoder invocations, returns and input context
tokens (240 on a full run) are hooked. Encode requests and
completion counts are distinct; failed internal encoder/backward/SVD work is
explicitly unknown rather than zero. Every setup env.step during original-root
collection and reset/warmup is also journaled and counted separately; this is
not an equal-total-compute claim. Setup steps do not enter selected-path counts.
No terminal20 encoding or neural CPU fallback is performed.

Append-only fsynced events preserve step attempts, returns before render,
state/RGB observations, raw history transitions, chosen versus observed versus
predicted contexts, derivative/candidate work, model-call scopes and failures.
RGB files and directory entries are fsynced before journal references. The
exclusive launch receipt prevents resuming/repeating partially dispatched arms.
A hard interruption may leave no final report, but durable events remain;
in-flight kernel/primitive internal work cannot be certified completed.

Prepare revalidates original capability/feedback input bindings and source
seals without creating an environment/model, seals source/test/this innerdoc,
upstream preprocessing/derivatives and installed environment Python sources,
and saves planned census and zero-execution receipt. Framework native binaries
are not archived as Python sources. A seal is byte integrity, not execution proof.

## Commands (repository root: le-wm-main)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_multiround_feedback_runtime_20261027_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p4_multiround_feedback_runtime_20261027_v1 prepare --output results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p4_multiround_feedback_runtime_20261027_v1 preflight --output results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1
```

Only main after review (NOT executed for this delivery):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p4_multiround_feedback_runtime_20261027_v1 run --output results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1
```

Non-neural fake backends/environments test orchestration only, not real model
behavior, contact force, dynamics or efficacy. Even a complete future runtime
is not independent evidence of long-horizon feedback advantage. Task-motion
alignment, mechanism and budget-complete independent comparisons remain open.
