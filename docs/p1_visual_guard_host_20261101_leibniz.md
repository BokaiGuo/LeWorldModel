# P1 visual Sink host wiring and saved-RGB CUDA probe

The structured call is `codex_four_arm_rgb_physical_20261030.episode` at
`moving.encode_boundary`; the CEM call is inside
`codex_p1_full_cem_host_20261101_exclusive.execute_one.encode_current`, where
`moving` is taken from `runner.rgb.moving`. The structured wrapper clones the
actual episode with a private moving proxy; the CEM runner proxy substitutes
only its private `rgb.moving`. The proxy invokes the exclusive Sink adapter,
leaving the sealed functions and actual moving module unchanged. A complete
controller/physical host is deliberately blocked; these are call-site
adapters, not a full experiment runner.

The fixed integration probe reloads the frozen model independently for both
arm labels and calls the real CUDA visual Handler and actual Sink on the two
sealed actual cp0 PNGs. Both saved images have the same root/goal/model
binding. The input is saved RGB, not a new live observation; no physics,
planner, candidate selection, or controller is executed. This checks module
pre-hook interaction with the real handler and its `Sink.begin/end`; it does
not check a live `encode_boundary` or entire host. Each arm has a precommitted
one-load, one-encode vector. All other dimensions are zero for this limited
probe only, not for a P1 experiment.

Source/prepared hashes, current model hash, image bytes, actual CUDA device,
the pre-existing saved root latent and post-run model hash are checked. The
cooperative GPU lock covers the probe. No training. Results remain development
only, `closed_unverified` and not a coverage/fairness certificate. Existing
guard readiness gate remains false: target provenance, physics, controller,
autograd and other host costs remain unknown/uncovered.

Tests are CPU-only fake callbacks with no neural inference:

```sh
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_visual_guard_host_20261101_leibniz.py tests/test_p1_visual_sink_guard_20261101_leibniz.py
```
