# Paired shadow-RGB spectral initialization: development result

Four real simulated-physics branches completed: the baseline and spectral
arms each reset independently on the two exposed roots, replayed the same
sealed 12-step prefix, and performed five action/re-observation/replanning
cycles using their own shadow RGB stream. Their final RGB-derived projected
scores (lower is better) were:

| Exposed root | Baseline | Spectral |
| --- | ---: | ---: |
| r8343_a0_p2 | 0.771494 | 0.704153 |
| r8345_a0_p2 | 1.723802 | 1.619192 |

The separate saved-artifact audit matched all 72 reset/command attempts to
their 72 returned RGB artifacts, verified hashes and input RGB/action bindings,
and checked 20 selected first actions against the saved model plans. All 80
prediction attempts and returns and the ten structure computations were
recorded; model parameter hashes matched before and after. The two arms each
scored 500 CEM samples. Spectral additionally incurred ten response forwards,
20 reverse-vector calls and ten SVDs, so this is **not equal total compute**.

This is a real simulator execution but the table is a model/RGB proxy score,
not independently measured physical goal distance or native success. The
prefixes and roots were already used during development; there are only two
paired roots, no random-initialization control at matched resources, no
demonstrated reconfiguration mechanism and no independent generalization.
These results do not establish P1 or Planning Compiler admission.

The test suite passed 16 tests when launched with `PYTHONPATH=.`. Without that
environment setting, 14 passed and two subprocess crash tests failed because
the subprocess could not import `experiments`; that failure is a test launch
configuration issue, not evidence of a physical-run failure.

The [journal](../le-wm-main/results/pc_shadow_spectral_paired_dev_20261103_CODEX/events.jsonl)
and [artifact audit](../le-wm-main/results/pc_shadow_spectral_paired_audit_20261103_CODEX.json)
preserve the branch results and bindings. The `experiment-code` workflow was
used for the frozen development contract and explicit accounting.
