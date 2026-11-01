# P5 prospective conditional action fidelity v5

## Frozen scope

Only val episodes **1–8**, in that order, context RGB/proprio at **0,5,10**, actual RGB
target **15**. Episode0 is already-opened calibration and excluded. No tuning, root
replacement, additional episodes, training, simulator control, or CPU neural fallback.
The experiment-code skill is used for reproducible protocol/receipts, not its training
or plotting steps. Existing v4 and external sources remain immutable.

This is a held-out-episode *local validation diagnostic*, not a new dataset or an assertion
that these installed validation episodes were never inspected by any historical study.
Preparation decodes RGB but never encodes targets or observes fidelity/ordering outcomes.

For each episode, retain true historical packets **0:5,5:10**, true normalized context
proprio (agent x/y + velocity), and actual decoded context frames. Four native predictor
calls differ only in the normalized last packet **10:15**, reshaped time-major **5×2**:

| Arm | New packet |
| --- | --- |
| actual | actual normalized recorded packet |
| normalized_zero | all normalized coordinates zero: native mean, NOT raw zero |
| negative_normalized | negate normalized coordinates, NOT raw actions |
| reverse_steps | reverse five temporal steps; preserve each step's x/y order |

Normalization remains `(raw/100−ACTION_MEAN)/ACTION_STD`, using literal constants
from SHA-bound native dataset source without importing/executing its unsafe class.
Normalized zero corresponds to approximately raw-array primitive action `[-0.87,+0.68]`.
Negation reflects around the native mean. No clipping or renormalization. Identical
controls (e.g. zero or time-constant actual packets) stay in the experiment and may tie.

## Primary recorded quantities

Each arm is compared to **actual frame15 patch features** from the same frozen HF-backed
encoder. The score is global 256×384 patch-feature MSE, float64 difference/square/mean
on CUDA. Report all four scores per row, each control-minus-actual difference, and
whether actual is strictly lower/equal/higher. Exact score ties retained without epsilon.
Summary contains eight-row lower/equal/higher counts and mean paired differences;
no significance, calibration-based thresholds, effect-size gate, or success claim.
No arm ordering is required for engineering completion: poor actual scores remain intact.
If any row is incomplete, no complete-eight scientific comparison summary is emitted.

These are comparisons against the observed successor under recorded actions, NOT physical
counterfactual targets for manipulated actions. They test whether conditioning on actual
actions better predicts that observed feature target; they do not prove control benefit,
causal dynamics correctness, generalization, or Full-D2P/P5 admission.

## Loading, checks, and execution accounting

Reuse v4's fixed SHA (checked before import), safe `weights_only=True` data arrays,
native modules, checkpoint SHA, strict actual state_dict key/shape/dtype checks,
DINOv2 cache SHA/header check, offline HF shim, deterministic native preprocessing,
paired native/source SAME-HF interface comparison, frozen/eval CUDA state receipts.
Do not open dataset pickle or execute its loader. Decord reads only four required frames.

Native predictor and proprio encoder are instantiated exactly as v4, strictly loaded;
native `DinoEncoder` uses the same forced local HF shim. Three model objects are used;
per-episode successful neural calls are **2 encoder computations** (native plus direct
same-HF paired interface), **1 proprio**, **4 predictor**. Hooks record actual successful
calls, including both HF-model calls. Complete eight totals must be native_encoder8,
HF_model_total_including_native16, proprio_encoder8, predictor32. No synthetic hidden states.

Before CUDA, verify all eight source/data byte seals and re-decode/recompute every selected
input against the prepared manifest. Missing/short video or missing array blocks CUDA;
preparation retains all eight episode IDs, never substitutes a row. Runtime failure retains
failed row and remaining not_run rows, aborting without retry or selection. All completed
rows/arms remain recorded; parameters/buffers and source/data seals are rechecked after run.
Returned status may be validation_failed even when a subset completed. Setup/loader failures
before the row loop fail by exception; they cannot be interpreted as completed evidence.

The source/native HF pairing is the SAME HF model, not independent official DINOv2
equivalence. Imported native/shim sources are covered by v4 source bindings. Dependency
implementations are not claimed byte-frozen as a complete third-party software closure.
`official_DINO_equivalence_proven=false`, `P5_admission=false`, `control_benefit_claim=false`.

## Prepare and main-only CUDA command

From `/home/gbk/code/LeWorldModel/le-wm-main`:

```bash
.venv/bin/python experiments/spectral_reachability/p5_jepa_action_fidelity_20261027_v5.py --prepare --output results/p5_jepa_action_fidelity_20261027_v5/manifest_frozen.json
.venv/bin/python experiments/spectral_reachability/p5_jepa_action_fidelity_20261027_v5.py --validate-cuda --manifest results/p5_jepa_action_fidelity_20261027_v5/manifest_frozen.json --output results/p5_jepa_action_fidelity_20261027_v5/pusht_action_fidelity_cuda_main_v1.json
```

No mode flag defaults to non-neural prepare. No command overwrites existing output.
Prepared manifest binds protocol, module/test/doc, v4/v3 sources and receipts, checkpoint,
encoder cache, native source tree, three arrays, all eight videos, decoded RGB and complete
raw/normalized histories/control packets BEFORE any CUDA call. It also hashes pre-existing
calibration receipts but does not re-encode or reselect calibration samples.

Worker executes only non-neural tests and preparation, never the second command.

```bash
.venv/bin/python -m pytest -q tests/test_p5_jepa_action_fidelity_20261027_v5.py
```

Tests cover frozen protocol, episode exclusion, history/unit/step-axis preservation,
native-mean zero/negative semantics, identical controls and ties, absent/nonfinite costs,
all-root summary refusal after a failure, short-video retention without replacement,
source/protocol drift, CUDA-required refusal, restricted loading policy, and actual fixed-eight
video/array reading without neural forwards. Tests are contract evidence, not fidelity results.
