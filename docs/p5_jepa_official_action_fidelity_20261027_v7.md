# Official-native encoder consumer: new v7 protocol

This is an explicitly new deployment-interface protocol, not a retroactive validation of v5. The eight validation episodes were already opened by v5/v6: they remain fixed for paired implementation continuity, NOT a new independent holdout. No P5 admission or physical control benefit is claimed, irrespective of ordering results.

## Read-only v6 review

The host report records paired_equivalence_failed in all eight episodes, with exact preprocessing agreement, strict loads, eight official/eight HF forwards, and unchanged finite CUDA/frozen/eval state. Native positional max absolute difference is 0.014400579035282135; patch RMSE ranges from 0.10976287113960083 to 0.12541542794258165. This is not roundoff-scale equivalence. Official source uses bicubic scale_factor=(16.1/37,16.1/37), installed HF uses output size=(16,16). This is a concrete upstream discrepancy corroborated by positional diagnostics, but existing measurements do not isolate attention backend differences or establish interpolation as the sole cause. v6 negative evidence is preserved and SHA-bound in this new manifest; no native offset, threshold, interpolation, or old report is changed.

## Fixed consumer

Use unmodified official commit 7764ea0f912e53c92e82eb78a2a1631e92725fc8 source and native dinov2_vits14(pretrained=False), with image configuration 518, patch 14, 37×37 positional grid, offset .1, antialias false, zero registers and unchunked blocks. The existing cached HF safetensors are strictly mapped using the sealed v6 exhaustive 223→175 mapping (Q/K/V concatenation in order). This authenticates the official implementation/source/license, not independent official checkpoint provenance. No HF model or old HF encoder shim is constructed or invoked; no torch.hub download path is used. Official forward(is_training=True) exposes the native feature dictionary without altering the model.

Fixed episodes 1–8, frames 0/5/10/15 from installed official extracted PushT val videos. Preparation uses the sealed safe decoder and SHA-restricted weights_only array loader, never the native dataset class or unsafe dataset pickle. Reproduce v5 full RGB histories, native state/proprio normalization, and raw actions 0:15 normalized as (raw/100 - native mean)/native std into three chronological 5×2 packets. Save/replay all decoded RGB SHA, actual normalized/raw array SHA and values, all four packet arms and frame lengths. Fail preparation if any fixed file/frame/history is missing; never substitute roots.

Native preprocessing: RGB float32 /255, full-frame 224 resize and ImageNet normalization through sealed native make_transforms; independent explicit bilinear/normalization check ≤1e-6. A single official encoder call per episode receives all four real frames, producing finite (4,256,384) patch tokens in original frame/spatial order. Context uses first three as (1,3,1,16,16,384); target is the fourth frame's actual official patch tokens (1,256,384). Target features are not hidden states invented for the test. Full feature and target SHA receipts are saved after forward (CPU copying for hashing is not CPU neural execution).

Native proprio embedding consumes normalized frame0/5/10 proprio (1,3,4), expanded to (1,3,256,16). Native released PushT predictor/proprio constructors are unchanged from v5 and checked against the config-derived v3 schema before any strict state load; checkpoint weights load SHA-bound with torch.load(weights_only=True,map_location='cuda'), no allowlisting/unsafe fallback. Prepare uses restricted meta/FakeTensor inspection only. All model construction/initialization, parameter storage and neural operations require CUDA; no CPU neural fallback, training, or model forward during prepare.

Four predictor arms preserve all context RGB features, proprio and the first two action packets. Only normalized last packet 10:15 changes: actual; normalized zero (native action mean, NOT raw zero); negative normalized packet (NOT negative raw actions); reverse five steps, preserving x/y order. Each predicts the last successor patch field. Target global patch MSE uses float64 CUDA difference/square/reduction against frame15 official features. Save all four MSEs, control-minus-actual differences and exact lower/equal/higher ordering. No success threshold for efficacy, no episode selection, no retry. Identical controls and ties remain present. These controls have no observed counterfactual physical targets, so this records observed-target conditional fidelity, not causal control outcomes.

Exactly 8 official encoder, 8 proprio and 32 predictor successful forwards are counted by hooks, with per-row ordered traces. There is no paired HF forward and no redundant official equivalence comparison. Native official SDPA math (XFORMERS_DISABLED=1), math SDPA context also applying to predictor, TF32 disabled; this numerical backend is explicit in the new protocol, not claimed to reproduce v5 arithmetic. Frozen/eval/finite/all-CUDA state receipts and complete parameter-byte hashes are compared before/after. Imported official and native JEPA module paths must reside in the correct repositories and their actual source files must be in pre-CUDA seals. Official source itself is never modified. Source/checkpoint/config/video/array/report hashes are rechecked after CUDA. A failed row remains failed and subsequent fixed roots remain not_run; no incomplete summary is accepted.

## Commands and freeze

From /home/gbk/code/LeWorldModel/le-wm-main, non-neural:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p5_jepa_official_action_fidelity_20261027_v7.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_official_action_fidelity_20261027_v7 --output results/p5_jepa_official_action_fidelity_20261027_v7/manifest_frozen.json
```

Main reviews the frozen manifest and alone executes CUDA:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_official_action_fidelity_20261027_v7 --validate-cuda --manifest results/p5_jepa_official_action_fidelity_20261027_v7/manifest_frozen.json --output results/p5_jepa_official_action_fidelity_20261027_v7/pusht_official_action_fidelity_cuda_main_v1.json
```

The frozen manifest binds this source/test/doc, old sealed dependencies, v6 source/provenance/negative report, all original v5 inputs/histories, configs and checkpoint metadata. CUDA replay must exactly reproduce preparation before model construction. Existing files are not overwritten. official_conditional_accuracy_recorded means all fixed rows were measured under the new official consumer and receipts/counts passed; it does not imply actual beats controls or meaningful prediction. All old source/docs/reports remain sealed.
