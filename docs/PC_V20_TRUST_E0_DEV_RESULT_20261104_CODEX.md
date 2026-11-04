# v20 candidate trust filtering: exact-Wall development result

All 32 prespecified task/arm runs completed. The original nonlinear
terminal-distance threshold was `<0.05`.

| Arm | Success |
| --- | ---: |
| v17 | 0/8 |
| v19 | 0/8 |
| v20 | 0/8 |
| Four-start CEM | 3/8 |

Every saved endpoint, budget and complete-chain selection passed the
alternate same-environment geometry audit. A separate saved-record audit
recomputed the trust rule and confirmed that every retained intermediate
candidate was admitted. It admitted 48/80, 184/320 and 191/292 candidates
at the three respective depths. The rule was therefore active, but the
panel provides no evidence of a success advantage. Prediction tokens do
not establish equal total hardware cost.

The threshold is an exploratory feature-space heuristic; this result
does not prove response-guided planning or trust-region methods generally
ineffective. The panel has reached its eight-task stop: no retries,
substitutions, threshold changes or expansion. It is not a PC2 mechanism,
independent PC3, visual or generalization result.

Protocol: `PC_V20_TRUST_E0_DEV_PROTOCOL_20261104_CODEX.md`.
Manifest, plans, raw traces, geometry and trust audits:
`le-wm-main/results/pc_v20_trust_e0_dev_20261104_codex/`.
