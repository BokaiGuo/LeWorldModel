# Independent official DINOv2 / HF pairing (v6)

This is an implementation-equivalence prerequisite, not P5 admission, a predictive accuracy experiment, or evidence of control benefit. Both implementations receive the same already cached HF weights; there is no independently authenticated official weight file. It cannot establish official checkpoint provenance.

Official source commit 7764ea0f912e53c92e82eb78a2a1631e92725fc8 comes from the exact codeload archive supplied by main. Preparation verifies its SHA256, bounds and safe members, compares every archive file against the extracted tree without extraction, rejects extra Python files, and saves the Apache LICENSE hash and all source hashes. Source remains unchanged. Fresh-process imports reject any preloaded dinov2 and verify imported module locations. Xformers is disabled through its native environment switch, not a source shim. Official native torch SDPA math and HF eager attention execute independently.

Episodes 1–8, frames 0/5/10/15, remain fixed, including all outcomes. v5's sealed real data decoder and native preprocessing are reused; no dataset class, unsafe pickle, training, synthetic hidden states, or CPU neural fallback. Preparation replays the v5 manifest and decodes real videos, reads safetensor headers only, and builds no model. Existing sealed v3/v4/v5 files are read-only.

Mapping is exhaustive: 223 HF tensors map to 175 official tensors. Query, key, value weights/biases concatenate in that order. Layer scales, norms, projection, MLP, class/mask/position embeddings map explicitly. Header coverage, source dtype/shapes, and actual native state_dict keys/shapes must match before strict loading. Models are constructed on CUDA, weights load directly from safetensors to CUDA, with no downloads. Frozen/eval/all-CUDA finite state receipts are compared before/after; all source/input hashes are rechecked.

Both positional grids stay 37×37 (constructor image size 518), while real input is 224. Official offset .1 and antialias false are retained. IMPORTANT: installed HF uses bicubic output size 16×16, whereas official uses scale factor 16.1/37. This is a concrete source-level mismatch, not yet a measured output mismatch. Neither implementation is patched to conceal it. Per-episode positional errors are diagnostic only, not a tuned gate.

Predeclared gates: native preprocessing vs explicit full-frame bilinear/ImageNet normalization max absolute error ≤1e-6; normalized class AND all 256 patch tokens max absolute error ≤1e-4 AND global RMSE ≤1e-5, all eight episodes required. Four-frame token order is preserved and per-frame maximum errors saved. Finite/shape failures block. Numeric mismatches retain all eight rows and yield paired_equivalence_failed; the command can finish successfully while the scientific gate fails. Do not equate exit0 with equivalence. No threshold overrides, alternate resolutions, retry/selection, HF interpolation replacement, or offset changes. Exactly eight official and eight HF encoder forwards; diagnostic positional methods are not extra encoder forwards. No predictor/encoder performance claim is inferred from load.

From /home/gbk/code/LeWorldModel/le-wm-main:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p5_jepa_official_equivalence_20261027_v6.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_official_equivalence_20261027_v6 --output results/p5_jepa_official_equivalence_20261027_v6/manifest_frozen.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_official_equivalence_20261027_v6 --validate-cuda --manifest results/p5_jepa_official_equivalence_20261027_v6/manifest_frozen.json --output results/p5_jepa_official_equivalence_20261027_v6/pusht_official_vs_hf_cuda_main_v1.json
```

Only main executes the final CUDA command after review. Source/test/doc become sealed by the prepare manifest; any edits require a new version. No CUDA result exists at preparation time.
