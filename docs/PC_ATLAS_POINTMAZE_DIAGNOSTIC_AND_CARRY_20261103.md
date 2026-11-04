# Atlas / PointMaze diagnostics and joint-tail follow-up

This is a development synthesis of existing, already exposed results plus a
new offline saved-array audit. It is not a prospective protocol or a gate
decision. The two PointMaze diagnostics below have different action banks,
timing and scoring interfaces and must not be pooled.

## Atlas strong-chain injection and pruning

| Test | Input and evaluation | Result | Interpretation |
| --- | --- | --- | --- |
| Depth-2 strong-chain injection | Eight exposed exact-unicycle goals, gradient chain injected into Atlas banks | 8/8 retained and selected; split/direct context hashes agree | In this setting an available strong solution receives a favorable terminal score; this does not establish learned-model scoring validity. |
| Depth-3 intermediate pruning | Two deliberately constructed chains per goal with nearly identical terminal costs, width 2 | All 8 delayed middle nodes pruned; early middle nodes retained | Current-score beam pruning can discard a useful continuation; the construction uses privileged future actions and is not an empirical pruning rate. |
| Matched-call candidate ablation | Two extra random, spectral, or fixed-turn candidates; 8 goals × 3 seeds | Mean terminal errors 0.3043, 0.3567, 0.0099 respectively | Spectral candidates do not outperform these alternatives here. Calls are matched, not runtime or full computational cost; fixed-turn is task-specific. |
| Depth-3 pruning variant | Ordinary lookahead vs lookahead plus diversity, same banks and padded backend calls | Same final actions in all 24 runs | The development gain over v2 cannot be attributed to diversity; ordinary lookahead explains it. |

Sources: `PC_ATLAS_ORACLE_INJECTION_RESULT_20261103.md`,
`PC_ATLAS_DEPTH3_INJECTION_PRUNING_RESULT_20261103_CODEX.md`,
`PC_ATLAS_MATCHED_EXACT_RESULT_20261103_EXCLUSIVE.md`, and
`PC_ATLAS_DEPTH3_PRUNING_ALTERNATIVE_RESULT_20261103_CODEX.md`.
Injection adds privileged actions and unequal work; it is a diagnostic, not
Atlas performance. A genuinely matched spectral-versus-random-versus-lookahead
comparison of the full method remains open.

## PointMaze fixed-action velocity × score

Previously saved cp5 diagnosis, 384 fixed plans and 9,024 dependent within-group
pairs; entries are ranking reversals, lower is better:

| Initial velocity | Predicted score | Against actual RGB cost | Against physical distance |
| --- | --- | ---: | ---: |
| zero | visual | 3434 | 3383 |
| causal RGB estimate | visual | 1026 | 2229 |
| simulator qvel, diagnostic only | visual | 567 | 2318 |
| zero | proprio | 3797 | 2994 |
| causal RGB estimate | proprio | 3677 | 2894 |
| simulator qvel, diagnostic only | proprio | 3636 | 3045 |
| actual RGB cost | — | — | 2167 |

Candidate-specific cp5 frames are retrospective diagnostic inputs; simulator
qvel is privileged and must not enter the RGB-only selector. Separate cp2
tests on 36 fixed prefixes gave predicted-versus-physical ranking reversals
20, 21 and 15 for causal, zero and offline-qvel inputs respectively, across
36 dependent within-root pairs. These counts are not independent samples.
See `PC_POINTMAZE_VELOCITY_SCORE_CROSSTAB_RESULT_20261103.md` and
`PC_CAUSAL_CP2_VELOCITY_CROSSTAB_RESULT_20261103.md`.

## Joint-plan carry: physical follow-up on one exposed root

The saved root 8349 experiment used each prefix's own original joint tail and
the same 50-sample CEM configuration per feedback initialization, using real
cp7 RGB. Lower cp17 position distance is better:

| Prefix | Original tail | Zero-initialized feedback CEM | Own-tail-initialized feedback CEM |
| --- | ---: | ---: | ---: |
| zero | 0.39825 | 0.70954 | 0.22165 |
| greedy | 0.30865 | 0.83021 | 0.14061 |
| raw qualified | 0.29007 | 0.96898 | 0.18806 |

The qualified prefix changed distance from 1.65537 at cp2 to 1.66743 at
cp7. Under this particular position metric it did not make immediate
progress, but its carry-init result remains worse than greedy carry-init.
This is not proof of independent mechanism or method superiority. The
initializations have different upstream information and costs; 50 equal
feedback samples alone do not certify total-budget fairness.

The runner records 300 two-packet samples, four encodes, 153 RGB backend
steps and 234 position-replay steps, plus upstream joint-search cost.
Despite the pre-run protocol's wording about an additional old-only score,
the actual runner reuses the old prior score already included in carry CEM's
50 samples; no extra old-only score call is recorded.

New offline audit `experiments/pc_joint_carry_saved_audit_20261103.py`
checks source and archive hashes, joint-plan/prefix binding, six final CEM
score decisions and decoded actions, RGB prefix/source replay, and recorded
position distances for all nine branches. It uses the separately saved
normalizer receipt; it does not establish the historical runtime's
normalizer provenance or reproduce neural predictions and physical execution.
Two tests detect changed actions and scores; both passed. Existing relevant
Atlas, velocity and joint experiment tests: 14 passed.
