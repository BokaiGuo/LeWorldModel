# v19 retention fix: exact-Wall development result

All 32 prespecified task/arm runs completed, and saved trajectories
passed the same-environment geometry, common-bank, token and selected-chain
audit.

| Arm | Success |
| --- | ---: |
| v17 | 1/8 |
| v18 | 1/8 |
| v19 | 1/8 |
| Four-start CEM | 1/8 |

All four succeeded on the same root. v18 and v19 returned exactly the
same final score on all eight tasks despite differing retained-index
records on six tasks; all their final incumbents were recorded as coming
from full-chain refinement. This observation does not prove retention had
no causal effect, nor does it establish a PC2 mechanism or PC3 advantage.
Each v18/v19 task evaluated eight successor probes. Equal prediction tokens
do not establish equal total hardware cost.

The prespecified stop is reached: do not rerun or tune this panel. It is
an exposed small development panel with unknown broader source
independence. Protocol:
`PC_V19_RETENTION_FIX_E0_DEV_PROTOCOL_20261104_CODEX.md`.
Manifest, action traces, plans and geometry audit:
`le-wm-main/results/pc_v19_retention_fix_e0_dev_20261104_codex/`.
