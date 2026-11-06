# v25 fixed-bank common suffix development result

All four preregistered roots completed. Each root had 18 automatically
generated first-layer prefixes; each prefix received two 121-sample
full-action CEM restarts with shared restart seeds. The frozen retention
selection, all attempted and returned actions, recorded scores, token
totals, optimizer proposals, and alternate same-environment geometry
replay passed their audits.

| Root | all-bank best | response | diverse | greedy | random | lookahead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.500126 | 0.500126 | 0.501078 | 0.500246 | 0.501078 | 0.501078 |
| 1 | 0.500262 | 0.500894 | 0.500894 | 0.500262 | 0.500894 | 0.500894 |
| 2 | 0.500201 | 0.500583 | 0.500583 | 0.501484 | 0.500201 | 0.501484 |
| 3 | 0.207494 | 0.207494 | 0.207494 | 0.502650 | 0.500444 | 0.207494 |

Distances below 0.05 were defined as success; no bank or retention rule
achieved one. Response and state diversity differed only on root 0:
response added candidate 16, whose best observed suffix score was
0.500126, and displaced candidate 12, whose score was 0.668933. In that
root the response-retained set had a better best score than the other
four retained sets, but its absolute score remained far from success.
On root 3, diversity, response and lookahead all retained candidate 7
and improved on the greedy and random retained sets; this is not an
increment attributable to response because its selected set was identical
to diversity.

Per root, generation cost 1,920 prediction tokens, additional lookahead
1,350, and common all-candidate suffix optimization 13,068. Generation
already includes v25's response calculations and complete-chain
refinement. These diagnostic expenditures exceed the full-method budget:
this is not an equal-total-cost planner comparison. The finite suffix
search is not an optimality certificate, and the roots have unknown
broader provenance. The fixed bank comes from the first layer only.
These results neither prove the PC2 selected-chain mechanism nor PC3
advantage. The panel is stopped and must not be tuned or reopened.

Protocol: `PC_V25_FIXED_BANK_SUFFIX_E0_DEV_PROTOCOL_20261104_CODEX.md`.
Raw records: `le-wm-main/results/pc_v25_fixed_bank_suffix_e0_dev_20261104_codex/`;
`audit.json` checks trajectory and optimizer replay, and
`generation_audit.json` binds candidates, response probes and retention
to recorded generation calls.
