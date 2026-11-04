# v18 successor-reachability retention: E0 development result

All 24 prespecified task/arm runs completed. The original nonlinear
terminal-distance success threshold was `<0.05`.

| Arm | Success | Tokens per task |
| --- | ---: | ---: |
| v17 | 1/8 | 4,093–4,095 |
| v18 | 1/8 | 4,093–4,095 |
| Four-start CEM | 2/8 | 4,095 |

v18 evaluated eight successor finite-difference probes per task, four at
each nonterminal layer. Both response arms and CEM succeeded on root 2;
CEM alone succeeded on root 0. This panel does not show a v18 success
advantage. Equal token counts do not imply equal total compute, and eight
exposed development tasks do not establish generality or PC3.

Every recorded endpoint was checked against alternate same-environment
geometry, with shared proposal banks, legal actions, packet-level token
accounting and best complete-chain selection also audited. These are
consistency checks, not independent replication. The successor value is
an unconstrained local linear heuristic; a negative result here does not
prove that reconfiguration or a different search allocation is impossible.
No visual or PC2 same-chain mechanism claim follows.

Post-run code inspection found that when the lowest-value candidate was
already retained by an immediate-score slot, v18 additionally reserved the
next-lowest candidate rather than filling that slot by diversity. The
declared best candidate was still retained, but this extra reservation was
not explicit in the protocol. The best candidate was already in an
immediate slot in 11 of 16 nonterminal layers on this panel. Preserve these
results as v18 evidence with this implementation qualification; the
corrected interpretation is implemented separately as v19 and has not
been evaluated on this stopped panel.

The eight-task stopping rule is reached; do not retry, replace roots or
tune the panel. Protocol:
`PC_V18_REACHABILITY_E0_DEV_PROTOCOL_20261104_CODEX.md`.
Plans, full action traces, manifest and audit:
`le-wm-main/results/pc_v18_reachability_e0_dev_20261104_codex/`.
