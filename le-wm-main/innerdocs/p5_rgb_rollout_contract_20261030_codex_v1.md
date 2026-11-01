# P5 RGB fixed-panel rollout: preparation and proposed CUDA contract

This is an offline preparation, not an execution authorization or scientific admission.
The candidate is the local `VisualCausalWorldModel`, not JEPA or DINO-WM.
Its prior training used physical state labels and intervention outcomes; runtime
model observation here is only two RGB frames plus actions. No new training,
neural inference, CUDA, or physics is performed by this preparation.

## Action and renderer provenance

The training dataset generator in `experiments/run_causal_visual_pusht_world_model.py`
passes each raw float32 action directly to `env.step` and stores the same action;
the checkpoint's prior strict-load receipt records `action_repeat=1`. The physical
runner constructs `swm/PushT-v1` and the inherited mixed-mode runner sends a copy
of the raw action to `env.step`. The installed PushT implementation currently has
`relative=True`, `action_scale=100`: the action is a dimensionless relative
command, mapped to a target equal to current agent position plus 100 times the
command, in the environment's 512-wide canvas coordinates. This is not a direct
velocity or a JEPA-normalized action. Do not multiply by 100 or normalize before
the model. No clipping is introduced here. The exact old environment package
bytes are not established by the checkpoint, so current source is not proof of
historical implementation identity or matching training support.
The inspected installed `stable_worldmodel/envs/pusht/env.py` SHA-256 is
`c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2`;
the inherited `agent_mixedmode_physical_20261028_v1.py` SHA-256 is
`e4fac2479254db07c34de5d9f88fbe0d19606b8ce2b1f8d7d5e42cbad8a042ae`.

Training constructs the environment with default renderer arguments; current
installed defaults are resolution 224 and `with_target=True`. The saved physical
runner explicitly requests resolution 224 and `with_target=False`. Renderer goal
visibility also depends on `render_goal`; equality of the training and panel
image distributions is therefore not established. The panel's off/on conditions
must remain separate; equal physical panel does not guarantee equal appearance.
Current renderer first resizes its canvas to 224; training's `resize_image` uses
OpenCV INTER_AREA to 64, HWC to CHW uint8, then `tensor_images` casts float32
and divides by 255. This bridge checks saved RGB PNG bytes and decoded pixels
and repeats that second transformation. The actual OpenCV version is recorded
in the previous preparation, and its resulting tensors are hash-checked here.

## Proposed fixed execution ABI

For each indexed branch 0–163 in order, use only saved plan-step 4 and 5 RGB
(global steps 14 and 15) and the literal five raw float32 actions plan steps
6–10. The validated tensors are previous/current `[1,3,64,64]`, action
`[1,5,2]`, little-endian float32. No real future image, physical state,
position, velocity, goal image, or target enters the model. Encode once with
`encode_state(image, previous_image)`, then apply `predict_next` sequentially
for each raw action to the preceding normalized six-dimensional state.
The checkpoint's previously verified flags disable temporal dynamics and
previous-action dynamics. Apply its saved mean/std only to interpret predicted
state, never to transform actions. This is open-loop model-state extrapolation,
not RGB generation and not JEPA latent equality. Original model planning code
also recursively calls `predict_next`, but that does not establish accuracy.

A future CUDA runner requires separate approval and review. It must recheck
checkpoint, source, exact input hashes, pinned configuration, strict loading,
frozen/eval parameters, CUDA device and no CPU fallback; execute per branch
at most one encoder and five transition calls under inference/no-grad, with
no retry, training, optimizer, environment or adaptive branch selection. Record
attempt and return events with branch/step/input hashes, device, dtype, finite
six-vector outputs, checkpoint and software identity. Predetermine and document
resource/time caps before approval; none are authorized here. Preserve all 164
rows and five step slots even on failures, with explicit unattempted/failed
statuses; never silently shrink the denominator or substitute outputs.
Seal the full prediction artifact before separately opening/scoring future
labels. No runner or score is implemented or authorized by this contract.
Planned counts are 164 encoder and 820 transition calls; actual counts are zero.
This serial accounting does not assert equal cross-model compute or matched
observation privileges with a position/velocity-using JEPA checkpoint.

The input validator checks content, shape, dtype, finiteness and forbids
additional keys. The preparation binds the prior saved panel and complete
physical receipt by SHA, validates exact raw10 and frame identities, and checks
all 80 pair schedules, common pair history and single-coordinate changes.
It opens only the two history PNGs, not future PNGs. The known panel is not
an independent fresh-root replication. No P5 scientific claim follows.
