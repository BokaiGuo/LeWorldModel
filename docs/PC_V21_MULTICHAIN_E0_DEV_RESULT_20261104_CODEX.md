# v21 multi-chain refinement: exact-Wall development result

All 32 prespecified runs completed and passed saved action, alternate
same-environment geometry, shared-bank and prediction-token checks.

| Arm | Success |
| --- | ---: |
| v17 | 2/8 |
| v19 | 2/8 |
| v21 | 2/8 |
| Four-start CEM | 3/8 |

v21 used four starts on every task. Its per-start sample allocations
were balanced and independently reconstructed from saved actions and
scores. On roots 0 and 2, a retained-beam start reached terminal distances
0.0767 and 0.1286, versus 0.5007 and 0.5003 for v19, but neither crossed
the prespecified success threshold; CEM reached 0.0201 and 0.0601 on those
roots. These are development observations, not independent evidence of
mechanism or an overall advantage.

The panel has reached its stop: do not retry, replace roots or change the
threshold. Equal prediction tokens do not imply equal total cost. Protocol:
`PC_V21_MULTICHAIN_E0_DEV_PROTOCOL_20261104_CODEX.md`. Plans, raw actions,
geometry audit and restart audit:
`le-wm-main/results/pc_v21_multichain_e0_dev_20261104_codex/`.
