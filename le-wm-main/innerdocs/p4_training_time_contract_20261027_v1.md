# P4 training-time contract versus saved rolling context

Scope: independent source-derived temporal audit of one fixed calibration root and
both actual/imagined arms at decision boundaries 5/10/15. No neural forward,
model construction, simulator construction/steps, action reselection, new outcome
gating, or changes to prior sealed files. Normalization/camera/Q are not reaudited.

## Independent training derivation

The visible `config/train/lewm.yaml` specifies history=3 and num_preds=1;
`config/train/data/pusht.yaml` specifies frameskip=5 and num_steps=history+num_preds.
For episode offset E and arbitrary local start s, the installed base dataset
reads the half-open chunk [E+s,E+s+20), not a chunk rounded to a multiple of five.
HDF5 and Lance sample observations at E+s+[0,5,10,15], but retain every action
row before reshaping (4,5,2) into (4,10). The order is time-major:
`a0x,a0y,a1x,a1y,...,a4x,a4y`.

`train.lejepa_forward` supplies observation tokens 0:3 and action tokens 0:3 to
predict, and compares all three predicted tokens to observation tokens 1:4.
The fourth action token is not required for that context/target loss. It is not
the action incoming to the first RGB that must be prepended. The current wrapper
attaches incoming actions to returned states, but `World.collect` rotates this
column one row left before writing, yielding outgoing actions for interior rows.
Its current on_step callback records post-step states, not necessarily reset;
the rotated last action is padding/unusable as a genuine next-state transition.
This is a source contract, NOT proof of which collector produced the old cache.

Action Embedder transposes (B,T,10) to (B,10,T), applies Conv1d(kernel=1,stride=1),
and restores (B,T,D). It mixes ten channels within one packet, not the three
history times. Predictor keeps token positions, applies causal attention and
same-position conditioning; `LeWM.predict` projection flattens batch/time for
per-token projection and restores their order. Consequently selecting `[:, -1]`
uses the third training prediction label: one packet after the latest RGB.

The audit executes hash-allowlisted *extracted functions only* with NumPy/axis
metadata stand-ins. It never imports training or stable_worldmodel modules.
The training probe stops at prediction/target subtraction, before computing any
loss. Dataset probes use nonzero episode offset97 and local start2, and batched
overlap/order [0,1,0], so a zero-offset/time-channel mistake cannot trivially pass.
Lance numeric cached columns are exercised; image row stride is source-inspected,
not a new image decoding/camera experiment.

## Comparison with actual P4

The root-anchored bootstrap completes three five-command warmup packets and uses
their last three boundary frames. Relative to the resulting decision root, RGB
history is [-10,-5,0] with two outgoing between-boundary packets. The prefix
executes action indices [0,5). Journal primitive_step1 is raw action index0;
observation primitive_step5 follows raw actions0..4.

| Decision | RGB state steps | Two between intervals | New packet | Last predict target |
| --- | --- | --- | --- | --- |
| 5 | -5,0,5 | [-5,0),[0,5) | [5,10) | state10 |
| 10 | 0,5,10 | [0,5),[5,10) | [10,15) | state15 |
| 15 | 5,10,15 | [5,10),[10,15) | [15,20) | state20 |

Each row binds saved observed frame hashes to those boundaries, incoming raw
packets to dispatched commands, selected new packet to the following five
dispatches, boundary event order, latent history shift and imagined recurrence.
Actual conditions on saved observed latents; imagined conditions on its rolling
predicted latents at the same nominal times. No neural prediction is recomputed.
Prior validated outcomes seals/input hashes are checked, without rerunning the
outcomes arithmetic or its camera/normalization/Q checks. Receipt equality alone
is not the training justification: source-derived training token labels above
provide the independent meaning of these same row/packet indices.

## Evidence boundary and missing provenance

The cached official checkpoint has model architecture config (num_frames3,
action input_dim10), not the original full training-run YAML, source manifest,
frame-skip/num-pred/dataset identity. Width10 alone does not prove frameskip5.
The visible training YAML names a Lance cache; this host has the HDF5 expert cache
instead. The audit reads only two bounded 20-row numeric chunks and terminal
action NaN masks from that real cache; it saves their dtype/shape/bytes hashes,
schema and file stat, NOT a whole-file SHA or claim of historical identity.

Thus consistency of visible training interface with saved P4 can pass while the
checkpoint's original training contract remains unproven. No temporal bug is
invented from missing provenance; no original-checkpoint equivalence is inferred
from passing fixtures. Repeated warmup pixels alone do not timestamp frames;
the source packet loops and ordered dispatch events establish relative step
counts. These are outer env.step counts, not physics substeps or established
wall-clock dataset cadence. No scientific P1/P4 admission follows.

## Reproduce (nonneural only)

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_training_time_contract_20261027_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p4_training_time_contract_20261027_v1 --output results/p4_training_time_contract_20261027_v1_main_v1
```

Output is exclusive: report.json and source_seal with this module/test/doc plus
reviewed training/dataset/collector/model/consumer source snapshots. Never reuse
an existing output or edit any source after final seal. Report contains six
boundary slots and all probe/source evidence. Next executable gap is read-only
recovery of the checkpoint's original training config and dataset/source manifest,
not another neural or physics rerun.
