# Fresh-native factual CUDA action-fidelity consumer

Main implementation, new version, old sources/seals unchanged. A prerequisite is
the fixed-seed nonzero native collector v2, both independent repeats complete and
all16 paired frames exactly identical. Bind full source manifests, hash-chained
physical journal, all32 durable lossless asset receipts and every actual file.
Only role a feeds the neural computation; role b authenticates repeatability,
not an independent task sample. Both are retained in the input manifest.

Use native env arguments u in correct units, projecting to the source dataset's
float32 pixel representation: pixel32=(100*u64).astype(float32), followed by
z=(pixel32/100-source_mean32)/source_std32. This is not an extra division of u
by100; preserve this rounding convention and require exact agreement with the
collector's exported packets. Source-native proprio has x,y,vx,vy, normalized by unchanged
native constants. Frames0/5/10 form the context and frame15 is the factual target.
All15 commands form three time-major five-step packets. Hold two history packets
and proprio fixed; replace only last packet with normalized-zero, negative
normalized, or reversed-step controls. Require all controls differ from actual.

Reuse sealed v8 loader solely to construct/load the identical official DINO/native
dynamics/proprio modules on HOST CUDA. Do not use HF interpolation or physics
overlay for neural process. No CPU fallback, training, adapters or optimization.
MATH SDPA, TF32off, frozen eval; exact count1 encoder (4frames),1proprio,4predicts.
Save full four-frame native patch features and every arm successor as float32
arrays, so later arithmetic can independently reconstruct global float64 MSE.
Failures retain the report/call counts; failed counts/source/model checks cannot
be converted into a scientific negative.

This tests local factual action sensitivity, not counterfactual physical control
or prospective task benefits. One seed is calibration, not P5 admission. The
target is already observed, but never used to optimize/select any action.

From project root, after actual source-native nonzero run exists:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_jepa_fresh_fidelity_main_20261027_v11.py
mkdir -p results/p5_jepa_fresh_fidelity_main_20261027_v11
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_jepa_fresh_fidelity_main_20261027_v11 prepare --output results/p5_jepa_fresh_fidelity_main_20261027_v11/manifest_frozen.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_jepa_fresh_fidelity_main_20261027_v11 cuda --output results/p5_jepa_fresh_fidelity_main_20261027_v11/manifest_frozen.json
```

The last command must run explicitly on host CUDA after source/test/doc seal
review. Existing output is never overwritten; new run requires new output.

Failure-recording v11: retain every completed arm array/cost even if a later arm
fails; pre-forward hooks count attempts including failures, post-hooks count
completed forwards. Final hash/model checks are individually protected and
cannot replace the original failure. Immutable successful v10 is not edited.

