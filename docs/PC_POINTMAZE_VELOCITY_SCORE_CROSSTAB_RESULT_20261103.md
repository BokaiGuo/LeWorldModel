# Fixed-action velocity and score cross-tab result

This diagnostic reuses 384 outcome-exposed two-packet candidate plans in eight
groups. Each score is compared with other candidates in the same group only;
there are 9,024 dependent candidate pairs across four roots. Numbers are
ranking reversals, not independent observations; lower means greater ordering
agreement.

| cp5 velocity input | Predicted score | Versus real RGB cost | Versus physical position distance |
| --- | --- | ---: | ---: |
| Zero | Visual | 3434 | 3383 |
| Zero | Proprio | 3797 | 2994 |
| Causal RGB | Visual | 1026 | 2229 |
| Causal RGB | Proprio | 3677 | 2894 |
| Simulator qvel, diagnostic only | Visual | 567 | 2318 |
| Simulator qvel, diagnostic only | Proprio | 3636 | 3045 |
| Actual terminal RGB cost | — | — | 2167 |

Visual scores and MSE reproduced the saved receipts exactly; actual RGB
costs differed by at most 3.58e-7. Frozen model parameter hashes matched
before and after. The host-CUDA run re-encoded 384 actual frames, encoded
1,152 cp5 inputs and evaluated 1,152 fixed predictions; it did not rerun
physical trajectories. The initial sandbox attempt could not see CUDA and
failed before inference; its failure journal was retained.

Compared with zero velocity, causal RGB velocity improves agreement for
visual scores on both references. Simulator qvel further improves visual
agreement with actual RGB but worsens agreement with physical distance
relative to causal RGB in this panel. Actual RGB and physical distance still
disagree. Neither predicted proprio ranking nor lower RGB prediction error
proves better real planning. All candidate-specific cp5 RGB images came from
physical oracle replay, and qvel is privileged; neither constitutes a
deployable action selection result.

Protocol: `docs/PC_POINTMAZE_VELOCITY_SCORE_CROSSTAB_CUDA_20261103_CODEXEXCLUSIVE.md`.
Run receipt:
`le-wm-main/results/pc_pointmaze_velocity_score_crosstab_RUN_20261103_codexexclusive_02/report.json`.
Independent saved-output audit:
`le-wm-main/results/pc_pointmaze_velocity_score_crosstab_SAVED_AUDIT_20261103_codexexclusive_01.json`.
The referenced source dependencies are not a certified complete dependency
closure, RGB-derived velocity is a quantized temporal average, and source
roots are not independent. No scientific gate is opened.
