# v15 response/global portfolio: E0 development result

The predeclared four-task, three-arm exact-Wall panel completed and its
saved action batches were audited against the alternate same-environment
geometry. The shared global banks, raw action bounds, scores, token counts,
and best scored complete-chain incumbents were checked.

| Root | v13 response | Full-action global CEM | v15 portfolio |
| --- | ---: | ---: | ---: |
| 0 | 0.500228 | 0.503258 | 0.500228 |
| 1 | 0.500671 | 0.500795 | 0.500671 |
| 2 | 0.003902 | 0.022954 | 0.019749 |
| 3 | 0.500736 | 0.501197 | 0.501738 |

At the fixed `<0.05` success threshold each arm succeeded on 1/4 tasks.
The portfolio chose the global CEM chain on root 2 and the v13 child on
the other three. It did not outperform the 2,409-token v13 arm on any root.
For root 2 its 1,920-token child scored 0.031181, and the reserved CEM
improved that child to 0.019749, but the larger-budget v13 achieved
0.003902. This finite example does not establish a general allocation
optimum or exclude other mechanisms.

All arms consumed 2,409 prediction tokens except the response and
portfolio arms on root 1, which consumed 2,407 because a legal remainder
could not pay for another full-chain evaluation. Equal prediction tokens
do not imply equal response, SVD or total computation; portfolio records
18 SVD calls per root. Wall times in the raw records are exploratory
single-process timing, not a reliable hardware cost curve.

The panel is exposed development data with four tasks, one dynamics
family and unknown broader provenance. Successful chains witness
feasibility only for their own tasks. Other task failures do not prove
infeasibility. The planned task count has been reached: do not rerun,
replace roots, or tune this panel. No same-chain mechanism intervention,
visual control, independent confirmation, or full-cost match is implied;
PC2 and PC3 remain unestablished.

Protocol: `PC_V15_PORTFOLIO_E0_DEV_PROTOCOL_20261104_CODEX.md`.
Sealed manifest, traces, per-arm plans and audit:
`le-wm-main/results/pc_v15_portfolio_e0_dev_20261104_codex/`.
