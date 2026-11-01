# Known root0 zero/greedy fixed-tail result

Report SHA-256 `1ebd4ba7fb15f11cd16bb76eeed49b6c4a737b951d4c2caaae798ec7e4e8f7e0`.
The four precommitted physical branches used the earlier cp10 root0 common
tail. No neural calls were made. Saved verification checks sources, journal,
actions, state/goal, root and cp5 snapshot/RGB, frame hashes and accounting.
This is not an independent simulator replay.

| Prefix | Tail | Last actual step | jointXY | Native success |
| --- | --- | ---: | ---: | --- |
| zero | common | 15 | 87.9792369 | no |
| zero | zero | 15 | 175.1768184 | no |
| greedy | common | 11 | 16.1372087 | yes; native stop |
| greedy | zero | 15 | 70.9856353 | no |

The early-success branch is not padded and has no step-15 endpoint; its
step-11 metric is not a same-time comparison with the other branches. No
out-of-domain censoring, truncation or infrastructure errors were reported.
The greedy prefix is a **cross-method model-greedy control**: it comes from
the historical `p1_three_prefix_own_host` DLS-mix optimized H3 candidate
bank, not from the new cp10 spectral8 structured node's own candidate bank.
The three-prefix implementation scores the first packet of its eight optimized
plans and selects the first minimum; that selection was sealed before its
original dispatch. Its selected and greedy first packets were identical on
root0 according to `D2P_P1_THREE_PREFIX_H10_RESULT_20261101_MAIN.md`.
No byte-identical candidate-bank equivalence with spectral8 was established.
Consequently this result must not be described as a spectral8 same-bank
greedy comparison or as evidence of spectral8-versus-own-greedy separation.

Outcomes of earlier work were
already known when the extension was designed. This is a known-root,
post-outcome development intervention, not independent confirmation or a P1
admission. Replayed physical snapshots are newly collected receipts.
