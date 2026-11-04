# Fixed-bank cp5 refresh: same-action development result

The prewritten `PC_POINTMAZE_FIXED_BANK_CP5_REFRESH_PROTOCOL_20261103.md`
compares 384 already-sealed plans on four exposed roots. Both the cp10 RGB
images and complete legacy endpoint arrays match the previous replay exactly.
The open-loop score recomputation agrees with the saved scores under the
predeclared tolerance. Frozen-model inference ran on host CUDA; model
parameter hashes before and after agree.

| Root | Bank | Open vs actual RGB reversed | Refresh vs actual RGB reversed | Open masked MSE | Refresh masked MSE | Open-choice distance | Refresh-oracle-choice distance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1884 | random | 270/1128 | 300/1128 | 0.4550 | 0.8119 | 0.3215 | 0.3332 |
| 1884 | spectral | 238/1128 | 366/1128 | 0.3843 | 0.7337 | 0.3029 | 0.4691 |
| 1885 | random | 528/1128 | 357/1128 | 0.5762 | 0.4073 | 0.2819 | 0.4141 |
| 1885 | spectral | 551/1128 | 384/1128 | 0.5654 | 0.4467 | 0.2042 | 0.2042 |
| 1886 | random | 169/1128 | 478/1128 | 0.1371 | 0.8300 | 0.2522 | 0.4090 |
| 1886 | spectral | 173/1128 | 495/1128 | 0.1657 | 0.7633 | 0.0463 | 0.6916 |
| 1887 | random | 723/1128 | 588/1128 | 0.7633 | 0.8184 | 0.7038 | 0.6639 |
| 1887 | spectral | 692/1128 | 466/1128 | 0.8399 | 0.8122 | 0.6755 | 0.6917 |

Refreshing reduces mean masked prediction MSE in three of eight banks and
reduces model/actual-RGB ranking reversals in four. The diagnostic's
refresh-score argmin has better terminal distance in one bank, ties in one,
and is worse in six. For root 1887 it does not restore distance below 0.5.
These are descriptive, dependent candidate-bank comparisons, not eight
independent experiments or evidence of a statistical effect.

**Important confounds and scope:** refreshed context uses RGB-derived
position with velocity set to zero, not the actual hidden velocity; its
context differs from an open prediction in both horizon and input. Thus this
does not isolate drift or prove feedback intrinsically harms control.
Obtaining each candidate's actual cp5 requires executing its own prefix:
the refresh-score selection over the complete bank is a physical oracle, not
a deployable planner. Neither selection nor those diagnostic counts establish
P1, independent generalization, or equal-budget planning advantage. The
checkpoint's training-data provenance remains unresolved.

During implementation, the first replay failed the exact-match gate:
384 endpoint rows differed by at most `1.34e-5` and six frames differed by
at most three intensity levels. The new driver had omitted the source
renderer’s float32 state conversion. Restoring it yielded exact matches; the
failed attempt was an input-contract error, not a model or theory result.
Two subsequent implementation attempts exposed and corrected encoding tensor
dimension errors before any result was emitted.

Artifacts:

- JSON `le-wm-main/results/a0_pointmaze_fixed_bank_cp5_refresh_20261103_DEV01.json`,
  SHA-256 `36c814bbe1123a64dcccc77c6b0a0e818d96df87f08044b6d8b345589508ef0c`.
- NPZ `le-wm-main/results/a0_pointmaze_fixed_bank_cp5_refresh_20261103_DEV01.npz`,
  SHA-256 `9e53d5623da3ae10e1a2bd23f520bced90d94ad8135c0bcd1ce104ac1ec80e1c`.
- Saved-data and masked-arithmetic tests: `2 passed`; they do not rerun
  neural inference or the physical renderer.
- The receipt records 3,840 high-level physical steps, 384 cp5 and 384 cp10
  RGB encodes, 384 open and 384 refreshed prediction samples, and 1,152
  predicted packet transitions. This diagnostic's costs are not a fair
  performance comparison between deployable policies.
