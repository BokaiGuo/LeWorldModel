# History, candidate provenance, and paired common suffixes

## History interface

The checked-in configuration associated with this local checkpoint specifies
three historical frames, `frameskip=5`, `action_skip=1`, normalized actions
and rollout window three. Its dataset slicer concatenates the five
chronologically ordered actions between sampled frames; training passes
observation and action sequences to the predictor with a one-frame loss
shift. The checkpoint weights and local configuration are hash-bound, but
the checkpoint's original training-run provenance is not independently
certified. This is evidence for testing a real-history window, not proof
that one-frame inference is an implementation bug or that three frames
are the only valid inference interface.

All 48 previously saved chains were rescored with exactly the same current
RGB, target and actions. The history arm adds RGB frames at five-action
intervals and the two corresponding observed five-zero-action packets.
The previous arm uses one current frame and window one. The single-frame
scores reproduced their sealed source within 6.38e-7. An independent native
physical replay verified all 15 warmup zero actions leave position and
velocity unchanged on each start, and all saved RGB frames were identical.
These checks permit pairing this stationary panel with prior outcomes.

| Start | Single selected index | Three-frame selected index | Single / history pairwise reversals | Mean absolute proxy / physical distance discrepancy, single / history |
| --- | ---: | ---: | ---: | ---: |
| (0.8, 0.8) | 4 | 4 | 6 / 7 | 0.1605 / 0.1282 |
| (0.8, 1.8) | 1 | 3 | 7 / 10 | 0.1282 / 0.1152 |
| (2.8, 0.8) | 10 | 10 | 9 / 7 | 0.1751 / 0.1322 |
| (2.8, 1.8) | 11 | 11 | 11 / 4 | 0.1714 / 0.1320 |

There are 66 pairs per root. The projected model score and physical distance
are not identical measurement spaces, so their absolute discrepancy is
diagnostic, not a calibrated model prediction-error estimate. Adding
history changes frames, action slots and attention window together; these
results do not isolate the effect of the action packet. The stationary zero
history is also not a test of moving histories. The historical direct
candidate is privileged, so the selected index comparison is not a
deployable method evaluation.

## Same suffix across prefixes

Each array below is absolute terminal distance for suffixes 0–5; suffix 0 is
ten zero actions. The second array is each terminal minus that same
prefix's suffix-0 terminal. Arrays in the same root column use identical
actions in identical order. `selected` refers to the *previous* single-frame
decision, and `direct` is a privileged diagnostic reference.

| Start | Prefix | Absolute terminal distances | Changes relative to own zero suffix |
| --- | --- | --- | --- |
| (0.8, 0.8) | zero | 2.236, 2.191, 2.411, 2.141, 2.399, 1.759 | 0, -0.045, +0.175, -0.095, +0.163, -0.477 |
| | direct | 0.938, 0.755, 1.217, 1.105, 1.053, 0.565 | 0, -0.182, +0.280, +0.168, +0.115, -0.373 |
| | selected direction 2 | 1.039, 0.849, 1.316, 1.136, 1.167, 0.128 | 0, -0.190, +0.277, +0.097, +0.128, -0.911 |
| (0.8, 1.8) | zero | 1.414, 1.301, 1.658, 1.390, 1.582, 0.939 | 0, -0.113, +0.244, -0.024, +0.168, -0.475 |
| | selected = direct | 0.860, 0.643, 1.116, 0.980, 0.929, 0.339 | 0, -0.217, +0.256, +0.120, +0.069, -0.521 |
| (2.8, 0.8) | zero | 2.236, 2.397, 2.150, 2.026, 2.304, 2.054 | 0, +0.161, -0.086, -0.210, +0.068, -0.182 |
| | direct | 0.938, 1.182, 0.750, 0.826, 0.849, 1.307 | 0, +0.244, -0.188, -0.112, -0.089, +0.369 |
| | selected random 0 | 0.740, 0.976, 1.127, 0.562, 0.761, 1.383 | 0, +0.236, +0.386, -0.178, +0.021, +0.643 |
| (2.8, 1.8) | zero | 1.414, 1.624, 1.249, 1.206, 1.434, 1.417 | 0, +0.210, -0.166, -0.208, +0.020, +0.003 |
| | direct | 0.860, 1.086, 0.572, 0.756, 0.708, 1.372 | 0, +0.226, -0.288, -0.103, -0.152, +0.513 |
| | selected random 1 | 0.380, 0.621, 0.109, 0.292, 0.315, 1.404 | 0, +0.241, -0.272, -0.089, -0.066, +1.023 |

These are the same-action interventions, not independently optimized
suffixes. All previously selected prefixes had immediate physical position
progress, and one equals direct, so this panel does not establish the
required non-improving selected-chain mechanism.

## Candidate implementation boundary

A new standalone fixed/seeded-random action-bank generator and fail-closed
selection helper reject declared privileged sources and unexpected outcome
fields. The generator itself has no simulator or model input, but caller
supplied provenance labels are not cryptographic source attestation. It is
not yet wired into the full RGB optimization/execution pipeline, does not
yet include a model-generated spectral candidate or model-greedy arm, and
has no independent performance result. Existing 12-candidate results remain
diagnostic because they included privileged direct.

The full [history receipt](../le-wm-main/results/pc_geometry_history_interface_20261103_codex.json),
[physical traces](../le-wm-main/results/pc_geometry_rgb_fixed_bank_physical_20261103_codex.json)
and [common-suffix interventions](../le-wm-main/results/pc_native_velocity_intervention_20261103_codex.json)
preserve all results. No selector tuning or gate modification was done.
