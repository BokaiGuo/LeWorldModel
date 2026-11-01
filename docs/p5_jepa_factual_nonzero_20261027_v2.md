# Fixed nonzero native factual pilot v2

New exclusive files; v1/runtime seals untouched. No outcomes read, physics,
neural work, action search or training by this agent. Main alone reviews/runs.
Reuses exact SHA-bound v1 journal/capture/profiling helpers, not its outcome
report or archived episode1. Seed2026102701 is shared with the zero pilot;
this is not a fresh unseen root/cohort or independent prospective efficacy test.

## Frozen commands and units

Exactly15 native env arguments u, three packets of five xy rows:

```text
packet0: [.12,.04], [.08,-.06], [-.05,.09], [.11,.03], [-.04,-.07]
packet1: [-.07,.10], [.03,.13], [.09,-.02], [-.11,.05], [.06,-.08]
packet2: [.05,-.12], [-.09,.07], [.13,.02], [.02,-.05], [-.06,.11]
```

These literal float64 values are fixed before nonzero execution, not selected
from zero/nonzero outcomes. No root/contact filtering, replacing or retrying.
Physics consumes u directly; internal pixel displacement=100*u and absolute
PD target=current agentxy+100*u per tick. No velocity conversion, tanh, clipping,
normalized-coordinate execution or extra u/100. All packets differ, are
asymmetric, and the final actual/mean/negative/reversed model packets are
pairwise different. This algebra is not action-fidelity evidence.

Two independently constructed official envs a/b use identical v1 config and
fixed seed BEFORE reset. Same32 native RGB frames/302Space.step(.01)/30control
calls;1.51s each including reset. Both complete trajectories retained before
exact16paired state/proprio/RGB/full-public-state/transition comparisons. No
threshold. v1 semantic shape sorting, full-declared-public-state/private-contact
cache boundary, durable pre-operation journal, failure/no-resume policy and
unmodified source profiling apply. No physical counterfactual branches.

## Stable output schema and downstream CUDA paths

Default directory `results/p5_jepa_factual_nonzero_20261027_v2/factual_main_v1/`.
`report.json` schema=`p5_jepa_factual_nonzero_20261027_v2`, success status
`fresh_native_nonzero_factual_recorded`, otherwise
`fresh_native_nonzero_factual_failed`. v1-compatible `rows[].exact` and
`native_call_receipts`, `journal_sha256`, `source_hashes_unchanged` are retained.
`consumer_inputs.a/b` bind each role's new consumer manifest path and SHA.

Raw paths unchanged: `a|b/frame_00..15/{rgb.npy,state.npy,proprio.npy,
full_public_state.json,transition.json}` and `a|b/command_01..15.json`.
Command01 advances frame00->01 with u[0]; packet0 advances00->05,
packet1 advances05->10, packet2 advances10->15. Every command includes raw u,
pixel delta, actual initial agentxy, absolute target, packet/action index,
pre-public-state SHA. Entire `schedule_env_u.npy` is durable BEFORE construction.

Each role has `consumer/inputs_manifest.json`. Paths in it are relative to
the **run root**, not consumer directory. `frames` maps indices0/5/10/15 to
raw RGB/state/proprio/full-public-state paths and raw file hashes. Added files:

| File under a/consumer (also b/consumer) | Shape/dtype and meaning |
| --- | --- |
| raw_env_commands.npy | 15x2 float64 exact executed native u |
| raw_pixel_displacements.npy | 15x2 float64,100*u |
| dataset_pixel_displacements.npy | 15x2 float32 projection of100*u |
| normalized_packets.npy | 3x10 float32, five xy steps flattened row-major |
| controls_packets.npy | 4x3x10 float32, arms actual/normalized_zero/negative_normalized/reverse_steps |
| states_frames.npy | 4x7 float32 native states at0/5/10/15 |
| raw_proprio_frames.npy | 4x4 float32 at0/5/10/15 |
| normalized_proprio_frames.npy | 4x4 float32, same frame order |
| context_proprio.npy | 3x4 float32 normalized proprio at0/5/10 ONLY |

Action normalization is explicitly the native dataset float32 arithmetic:
`raw_pixel32=(100*u64).astype(float32)`;
`z=(raw_pixel32/float32(100)-ACTION_MEAN32)/ACTION_STD32`.
This records float32 projection/inverse error; it does not reconstruct actual
commands from z or pretend the roundtrip is exact. Native source literal
stats are AST-read and checked, never importing the dataset/pickle loader.
Normalized controls preserve first two packets/proprio; mean is normalizedzero,
not rawzero. These files are model inputs only, never executed physically.

Consumer must verify successful report/source/manifest/journal/asset hashes
before use. RGB remains native224uint8; preprocessing/official encoder identity
and model/CUDA loading are the separate consumer's responsibility. Context
RGB/proprio uses0/5/10, predictor action histories are all three packets; frame15
is the already-collected successor label, NOT a planning input. No frame15
state/proprio/RGB used to select schedule or alter predictor history. Recorded
nonzero data/identity alone does not establish model conditional accuracy,
counterfactual ground truth, physical control benefit or P5 admission.

## Prepare/tests/main-only run

Original environment, project root `/home/gbk/code/LeWorldModel/le-wm-main`:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p5_jepa_factual_nonzero_20261027_v2.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_factual_nonzero_20261027_v2 prepare
```

After review, with delivered frozen manifest SHA:

```sh
PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_factual_nonzero_20261027_v2 run --confirm-physics --manifest-sha256 MANIFEST_SHA256
```

Existing output refuses overwrite. Tests use only property/constant-array
interface fixtures with no physics implementation. Prepare reads sealed
sources/manifests/stats only, never zero/nonzero outcomes. No physics/neural
imports on prepare/module import. This version remains immutable after freeze.
