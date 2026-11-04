# v21 exact-Wall prospective same-chain mechanism result

All eight prespecified plans and interventions completed. No final
selected chain reached distance `<0.05`, and both the base zero/greedy
intersection and the full zero/greedy/lookahead/random intersection were
0/8. This panel therefore does not establish the same-chain mechanism for
v21 under the fixed protocol.

Plans were sealed before intervention. Saved complete-chain selection and
intervention reconstruction were checked, and 68,520 saved control
trajectories were verified with alternate same-environment geometry.
These are internal consistency checks, not independent replication or
proof that finite control searches reached their optima.

The eight-task stopping rule is reached: retain this bounded negative
result and do not retry roots, relax conditions, or infer that the general
theory is impossible. Broader provenance is unknown. This does not itself
establish a PC3, PC4 or PC5 conclusion.

Post-run budget inspection found an important implementation limitation:
the 1,920-token allocation left only 1–10 full-chain refinement samples
per task, split across one to four starts, with 1–3 samples per start.
Every start therefore had only one sampling batch; the updated CEM
distribution was never sampled. This concrete resource-allocation
deficiency limits the negative result's interpretation. It does not by
itself prove that a corrected optimizer will succeed. The original
execution remains unchanged; a separately versioned implementation
reserves budget for at least two sampling generations per active start.

Protocol: `PC_V21_E0_SAME_CHAIN_MECHANISM_DEV_PROTOCOL_20261104_CODEX.md`.
Sealed plans, predictions, intervention actions and audits:
`le-wm-main/results/pc_v21_e0_same_chain_mechanism_dev_20261104_codex/`.
