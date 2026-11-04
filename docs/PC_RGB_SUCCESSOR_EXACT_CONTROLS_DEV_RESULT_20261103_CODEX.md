# Exact successor policy: additional development controls

Following `PC_RGB_SUCCESSOR_EXACT_CONTROLS_DEV_PROTOCOL_20261103_CODEX.md`,
four controls were run after the successor results were already observed on
the same 24 exposed wall roots. They use the exact same action limits and a
480 prediction-token cap; this is a post-outcome development comparison, not
an independent confirmation or an optimally tuned strong-baseline evaluation.

| Method | Mean terminal distance | Success `<0.05` | Tokens/root | Successor wins/ties/losses |
|---|---:|---:|---:|---:|
| Successor (sealed earlier) | 0.52813 | 0/24 | 480 | — |
| Same-bank greedy | 0.51611 | 0/24 | 479 | 5/0/19 |
| Random bridge | 0.52655 | 0/24 | 480 | 10/0/14 |
| One-step prefix lookahead | 0.51613 | 0/24 | 479 | 8/0/16 |
| Direct full CEM | 0.52728 | 0/24 | 480 | 10/0/14 |

Lower distance is better. Greedy and lookahead retained one prefix and spent
more optimization budget on its suffix; random bridge retained four. The
successor performs worse in mean than each control on this small panel, but
these configurations are not general search upper bounds. A structured
arm used one SVD per root, successor used three, and full CEM used none;
prediction-token matching is not equal total compute. No control selected its
zero-prefix fallback.

All 96 control decisions and their prediction traces were sealed and
recomputed. The first response bank was regenerated and exactly matched the
sealed successor bank for every structured control; additional tests
independently reconstruct same-bank greedy, the seeded random bridge bank,
lookahead selection, final comparison and token charges. The direct-CEM
no-SVD test also guards a concrete implementation error found on the first
attempt: the accounting dictionary had no `svd` entry when no SVD was called.
It was changed to record zero; the failed attempt produced no result file.
Six relevant tests passed.

Raw records are under
`le-wm-main/results/pc_rgb_successor_exact_controls_dev_20261103_codex_*`.
This is a planning comparison only. The new random-bridge and lookahead
controls have not yet been incorporated into the selected-chain common
suffix and matched replan mechanism intervention. PC2 and PC3 remain unpassed.
