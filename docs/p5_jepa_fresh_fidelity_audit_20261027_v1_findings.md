# Fresh JEPA v11 independent saved-array findings

Primary: actual host-CUDA v11 `cuda_main_v1/report.json`; v10 is contrast only.
Independent audit status: `saved_fresh_fidelity_arithmetic_consistent`, 284/284
checks passed. Numeric/I/O contract tests: 13 passed in 0.16s. This audit made
zero neural calls and zero physics calls; all 568 sealed file hashes matched
before/after. No pre-existing source, report or manifest was edited.

## Reconstructed MSE against actual factual frame15

| Arm | Independent float64 MSE | Actual minus arm |
| --- | ---: | ---: |
| actual | 0.03550688368692746 | 0 |
| normalized_zero | 0.038879372452745094 | -0.0033724887658176314 |
| negative_normalized | 0.04624835273261704 | -0.01074146904568958 |
| reverse_steps | 0.034722140376820775 | +0.0007847433101066878 |

Maximum absolute MSE receipt discrepancy: 6.938893903907228e-18, consistent
with reduction arithmetic. Rank: reverse_steps < actual < normalized_zero <
negative_normalized. No invented scientific pass threshold or tie tolerance.
v10/v11 all five saved array files have identical SHA256 and numeric values;
costs/deltas/order, input hashes, control packets and model receipts agree.
The v11 failure-retention/attempt-count repair is NOT a reversal explanation.

## Inputs and runtime receipts

Both roles' 16 frames, all32 state/proprio/RGB/full-public-state/transition
asset receipts, journal hash-chain intents/returns/terminal binding and
302 source-native dt=.01 substep receipts check out. Command15 targets the
pre-step agent position plus 100*u, not a canonical LeWM velocity command.
Full-public-state checks do not authenticate unsaved private contact caches.

Frame0/5/10/15 input hashes and all nine consumer arrays per role match the
native float32 dataset projection exactly. The controls preserve both prior
history packets/proprio and alter only the last five-step packet.
Normalized zero denotes native action mean, NOT raw zero. On this fixed
schedule the direct-u32 versus dataset-projection normalized-packet maximum
absolute difference is exactly zero (not an observed rounding-error cause).

v11 attempted/completed encoder1/proprio1/predictor4 receipts agree; all four
arms completed with no source/finalization errors. Strict-load and unchanged
CUDA/frozen/eval model receipts are consistent and SHA-bound. These are
authenticated runtime receipts, not independent neural reexecution here.

## Limits

One fixed fresh calibration seed, same seed as zero pilot; not a held-out
cohort. Every successor is compared with the ONE actually executed factual
target, not separately executed counterfactual targets. Lower reverse error
does not establish physical benefit, intervention accuracy or control quality.
No prospective planning, observed-target fitting, effectiveness or P5 admission
claim. Source-native official encoder with mapped cached weights retains the
previous weight-provenance boundary; no official/HF equivalence claim.

## Frozen artifacts

Manifest: `le-wm-main/results/p5_jepa_fresh_fidelity_audit_20261027_v1/manifest_frozen.json`

SHA256: `439ae1a916760a0e7660c078269d1ffd8cec456b98cd379a790bddc8b1353615`

Actual saved audit: `le-wm-main/results/p5_jepa_fresh_fidelity_audit_20261027_v1/v11_primary_saved_arrays_audit_v1.json`

SHA256: `890638df8144c8513037c56e1a9dcbf4a9db8b95eaff9671a921c313adc4f855`

Source, tests and protocol doc are frozen in the manifest. No further edits
to these files are planned; independent rerun uses the doc's array-only command.
