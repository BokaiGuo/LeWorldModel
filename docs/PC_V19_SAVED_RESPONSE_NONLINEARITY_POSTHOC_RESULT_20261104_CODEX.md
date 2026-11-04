# v19 actual-candidate response error: saved-data result

The stored small-step finite-difference response was compared with the
already-scored nonlinear response of the actual generated candidate
actions. The combined feature contains two positions and the task score.

| Search depth | Nonzero candidates | Median absolute error | Median relative error |
| --- | ---: | ---: | ---: |
| 0 | 64 | 0.263 | 0.553 |
| 1 | 252 | 0.224 | 0.654 |
| 2 | 220 | 0.228 | 0.570 |

Relative-error denominators below `1e-8` were excluded: relative counts
are 64, 233 and 216 respectively. The value is in the stated combined
feature metric; it is not a physical distance or an error percentage for
each component. This retrospective result indicates substantial local
linearization mismatch on these saved candidate actions. It does not
identify its cause, prove the selected directions are useless, or establish
that the mismatch caused the planning failures. No new dynamics evaluations
or search were performed.

Protocol: `PC_V19_SAVED_RESPONSE_NONLINEARITY_POSTHOC_PROTOCOL_20261104_CODEX.md`.
Individual candidate records and source hash:
`le-wm-main/results/pc_v19_e0_same_chain_mechanism_dev_20261104_codex/response_nonlinearity_posthoc.json`.
