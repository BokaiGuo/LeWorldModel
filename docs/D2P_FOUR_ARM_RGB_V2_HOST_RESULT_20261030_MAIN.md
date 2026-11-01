# Four-arm RGB feedback: action packet repair and host run

The first known-root exploratory run failed before any control action: the
planner returns `selected_raw` with shape `(3, 5, 2)`, while the launcher
used `[:5]`, retaining all three packets and failing the `(5, 2)` check.
The ten warmup steps are not control actions. The original failure and
artifacts remain unchanged.

The v2 launcher requires a finite, bounded `(3, 5, 2)` tensor and dispatches
only `selected_raw[0]`. Its new shape/bounds regression tests, the original
launcher tests and planner tests passed: 15 tests. This is a wiring fix, not
a changed scientific gate.

Prepared SHA-256:
`c01f0905b5e4ccf838dec7814f6b3c9a9bfadfa46374379dac250e6fb199870b`.
Host CUDA run report:
`le-wm-main/results/codex_four_arm_rgb_physical_20261030_v2_RUN_MAIN01/report.json`;
SHA-256 `b16315fae44cb8f9d46165d8bed4c76adcb77521a8f9b8f284ec54c5704245a9`.
GPU process presence was observed with `nvidia-smi`.

All four arms completed 10 warmup and 20 control steps, with no reported
episode errors. The report records 16 completed encoder and 16 projector calls,
and 16 sealed nodes with ledger status `closed_unverified`. Native success was
false for all four. Final object position errors for structured, native,
random and immediate-progress were respectively 20.0270, 6.5301, 28.1174
and 34.2928; these are descriptive single-root outputs, not evidence of
relative performance.

A separate saved-data check recomputed all 16 candidate-trace file hashes and
confirmed the selected index and first raw packet in each trace equal both the
sealed node commands and the corresponding five actions recorded as executed.
This validates that specific saved-data linkage, not the full physical journal
or the completeness of cost accounting.

The arms do not share the same optimizer or established equal total budget.
The native arm is a CEM variant, not an authenticated native strong baseline.
Ledger closure is not independent full cost validation. The root is known,
and this run does not test the three-prefix common-suffix mechanism, establish
P1, or support independent generalization. A separately implemented common
optimizer and matched-budget comparison remain necessary.
