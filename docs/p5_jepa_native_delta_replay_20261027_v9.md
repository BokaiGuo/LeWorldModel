# P5 native fixed-delta nonlinear replay (v9)

This new version consumes the completed, sealed v8 episode1 response and unchanged
ridge delta. It does not modify v8, choose a direction, tune damping, train, execute
actuator commands, or admit P5. The real frame15 target was already observed: this
is a model-fitting and nonlinear-compilation diagnostic, not prospective control.

Before sealing, non-neural replay verifies all v8 gates, source/data seals, saved
numeric hashes/shapes, exact target-minus-nominal residual and the unchanged .01
ridge solution (rtol1e-10/atol1e-12). The new manifest seals this source, tests,
document, v8 manifest/report and every saved numeric artifact. Outputs are exclusive.

Main host CUDA only, same official native encoder/cached strict-mapped weights,
native frozen predictor and proprio encoder as v8. There is no HF encoder shim,
CPU neural fallback, action tanh/clipping or new normalization. Reencode four real
frames and three real proprio contexts; require exact v7 feature hashes and v8
nominal successor hash. Three predictor calls: nominal, fixed v8 delta cast once
to float32 normalized coordinates, and independent exact replay of that candidate.
No derivatives, line search, alternative candidate or outcome filter. Record actual
nonlinear target patch MSE, delta from nominal and difference from frozen linear
prediction, whether better or worse. Float64 cost reductions; report exact replay,
actual module counts and unchanged model/source/data receipts. Scope remains the
single already-used episode; cached weight provenance remains conditional.

From le-wm-main:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p5_jepa_native_delta_replay_20261027_v9.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_jepa_native_delta_replay_20261027_v9 --output results/p5_jepa_native_delta_replay_20261027_v9/manifest_frozen.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p5_jepa_native_delta_replay_20261027_v9 --validate-cuda --manifest results/p5_jepa_native_delta_replay_20261027_v9/manifest_frozen.json --output results/p5_jepa_native_delta_replay_20261027_v9/episode1_native_delta_cuda_main_v1.json
```

Only the last command runs models and requires approved host CUDA. Non-neural
fixtures/preparation alone establish no real derivative or control capability.
