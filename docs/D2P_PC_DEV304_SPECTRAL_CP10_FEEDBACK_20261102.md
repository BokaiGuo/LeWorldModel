# Dev304 spectral cp10 real-RGB feedback development experiment

For each of the two model-selected goal-ranked and two Q-goal-ranked
spectral candidates, this experiment replays the original five-step
prefix and the first five steps of the selected spectral tail. At that
observed cp10 state, a common single-population CEM is run using real RGB
history, actual two-packet action history and the same goal. It evaluates a
two-packet terminal latent score with 10 predictor calls and 3,200 predictor
samples per arm, and then executes the first five steps of its selected tail.
The upstream spectral selection and downstream replan costs are not fully
certified or equalized against a baseline.

All four branches replayed their ten executed actions against the saved
states and cp10 RGB checkpoint, and completed the five-step feedback tail.
The model remained frozen; the saved verifier checks plan/physical bindings,
tail actions, recorded ledgers, and 24 saved frames. All four had native
success false.

| Spectral prefix | Replanned reward | Position error | Angle error |
| --- | ---: | ---: | ---: |
| goal / original | -301.061 | 204.623 | 2.5038 |
| goal / coherent | -340.990 | 171.486 | 2.5836 |
| Q goal / original | -272.557 | 202.244 | 2.2716 |
| Q goal / coherent | -297.600 | 187.523 | 2.6681 |

The old open-loop candidates and these replanned continuations share their
first ten executed actions, but select different final five actions. This
one-root development comparison is not an independent trial or proof that
feedback generally helps. It does not prove reconfiguration, same-budget
baseline advantage or P1 admission. It also does not rule out additional
interface or metric problems. Source code for this physical runner was bound
before dispatch in its separate physical contract.

Artifacts relative to `le-wm-main/results/`:

- Replan contract SHA-256 `c5b14f21d2f4ad83a4e0e6c69bb3a824f79890a69fdfae1b0e71a8416592a008`.
- CUDA plan report `p1_dev304_spectral_cp10_replan_20261102_PLAN_DEV01/report.json`,
  SHA-256 `dd97f800c0d9443d112551eaedc9e999e27245a69db563024a467ff34e0f2c90`.
- Physical contract SHA-256 `07b315954b1cd951cce758a3802b527f8b2116f5c26f365cbcc26e8983d0d9a0`.
- Physical report `p1_dev304_spectral_cp10_physical_20261102_RUN_DEV01/report.json`,
  SHA-256 `9060a4909c7e02a985528d0015cc0bf399491d079180e16c47ef5be930945c7f`.
- Saved verification `p1_dev304_spectral_cp10_saved_20261102_DEV01.json`,
  SHA-256 `9037e8c699dc1f54df8ce1971ab733d0896c1281cf8e31b407315a0ecca83fb1`.
