# Atlas and PointMaze diagnostic synthesis

This note consolidates existing development results; it does not introduce
new physical runs, alter historical contracts, or establish a PC gate.

## Atlas strong-chain injection and pruning

| Setting | Candidate / scoring / pruning observation | Interpretation |
| --- | --- | --- |
| Exact unicycle, depth 2, 8 exposed goals | Gradient-generated chains were accepted, retained and selected in all 8; direct and split context hashes agreed. | Consistent with a coverage gap in this particular setup; does not certify other objectives or depths. |
| Exact unicycle, depth 3, 8 exposed goals | Constructed delayed chains had the same terminal state as early chains, but all 8 delayed intermediate nodes were pruned at width 2. | Current-cost pruning can discard a useful continuation; this is a finite-beam limitation, not proof of a coding bug. |
| Wall unicycle, 3 exposed search seeds | A successful external waypoint chain's second node ranked 6th and was pruned in all 3 width-2 runs. | The successful continuation did not reach the later candidate bank; oracle injection is not Atlas performance. |

Sources: `PC_ATLAS_ORACLE_INJECTION_RESULT_20261103.md`,
`PC_ATLAS_DEPTH3_INJECTION_PRUNING_RESULT_20261103_CODEX.md`,
`PC_WALL_UNICYCLE_INJECTION_RESULT_20261103.md`. The injected solutions are
privileged and their generation costs are not a fair Atlas comparison.

The matched root-source test had identical selected plans for spectral,
random and ordinary sources across 8 exposed goals and 3 seeds; this panel
does not demonstrate incremental spectral benefit. Its call counts were
matched by padding and are not natural equal-total-compute evidence. See
`PC_ATLAS_DEPTH3_MATCHED_ROOT_SOURCE_RESULT_20261103.md`. A non-oracle wall
diversity rule failed to recover the detour on all 3 seeds; see
`PC_WALL_UNICYCLE_DIVERSE_BEAM_RESULT_20261103.md`.

## PointMaze fixed-action velocity and score

The existing frozen-checkpoint CUDA analysis compares 384 outcome-exposed
plans in 8 groups (4 roots), with 9,024 dependent within-group pairs. Values
are pairwise reversals, not independent samples.

| cp5 velocity input | Prediction score | Reversals vs actual RGB cost | Reversals vs physical distance |
| --- | --- | ---: | ---: |
| Zero | Visual | 3434 | 3383 |
| Causal RGB | Visual | 1026 | 2229 |
| Simulator qvel (offline diagnostic only) | Visual | 567 | 2318 |
| Zero | Proprio | 3797 | 2994 |
| Causal RGB | Proprio | 3677 | 2894 |
| Simulator qvel (offline diagnostic only) | Proprio | 3636 | 3045 |
| Actual terminal RGB cost | — | — | 2167 |

Source: `PC_POINTMAZE_VELOCITY_SCORE_CROSSTAB_RESULT_20261103.md`.
Candidate-specific cp5 frames are post-execution oracle inputs and qvel
must not enter the RGB planner. RGB velocity is a quantized temporal average;
its alignment with instantaneous qvel is not certified. Actual RGB cost
and physical distance disagree, so reduced visual prediction error is not
evidence of better physical selection.

The later render-clock audit found legacy rendering one integration tick
behind current qpos in its measured panel. Adding in-place forward corrected
the geometry but failed the predeclared exact trajectory-invariance check
(258 trajectory/tick pairs differed, maximum position component difference
4.45e-16). A separate shadow-rendering path remains to be certified; neither
the old frame nor the proposed change should be silently substituted. See
`PC_POINTMAZE_RGB_FORWARD_CLOCK_RESULT_20261103_CODEX.md`.

## Current verification and next discriminating work

On 2026-11-03 the existing Atlas injection/pruning regression set passed
16 tests, and the velocity-score plus matched-root saved-audit set passed
7 tests. These are code and saved-artifact checks, not new model or physical
experiments.

The next useful comparisons are (1) non-oracle search on a predeclared task
with independently verified need for non-greedy action, comparing ordinary
lookahead and spectral sources with natural, separately metered cost, and
(2) fixed-action score/timing checks after a shadow renderer is validated.
Keep the renderer validation separate from policy selection and retain the
historical results. No change to admission follows from this synthesis.
