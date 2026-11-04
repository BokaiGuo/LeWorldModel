# v14 carry-conditioned nominal: E0 development result

The predeclared five-arm comparison completed 40/40 plans and its audit
passed. Every arm/root used 1920 five-step prediction tokens, except root 4
of the three response arms, which used 1918.

| Arm | Success / attempts | Mean terminal distance | Mean wall seconds |
| --- | ---: | ---: | ---: |
| v14 carry-conditioned response | 1/8 | 0.39284 | 0.0231 |
| v13 global + response | 1/8 | 0.38868 | 0.0229 |
| v12 response | 0/8 | 0.50052 | 0.0249 |
| Same global bank + full-action CEM | 2/8 | 0.37884 | 0.0364 |
| Three-start direct CEM | 0/8 | 0.44476 | 0.0320 |

Timings are small CPU-side E0 measurements, not total-cost parity or RGB/GPU
cost curves. On the eight paired tasks v14 tied v13 twice and had larger
terminal distance on six. Both succeeded on root 6; matched global CEM
succeeded on roots 6 and 7. These observations do not establish population
ranking on a small development panel. They do not show that the v14 nominal
replacement helped or provide evidence of a spectral advantage.

On root 6, the best global initial candidate had distance 0.4176. Both
v14 and v13 found their successful final chains in full-chain refinement,
with distances 0.03964 and 0.01961 respectively. This does not distinguish
the causal contributions of the global initialization, recursive search,
and optimizer; the matched no-response control is an essential boundary.

## Verification and scope

The audit verified all initial banks were identical across v14, v13, and
global CEM; on each root, the carry-conditioned nominal was actually used
at each response depth and all three layers differed from v13's candidate
records. All saved actions, endpoints and scores were checked with two
same-environment geometry implementations. Complete-chain incumbents,
dependency hashes and token counts were checked. The v12–v14 unit tests
passed 26/26. These tests and replays are not independent replication.

Protocol: `PC_E0_V14_CARRY_NOMINAL_DEV_PROTOCOL_20261104_CODEX.md`.
Manifest, raw traces, result and audit:
`le-wm-main/results/pc_e0_v14_carry_nominal_dev_20261104_codex/`.

The fixed 40-attempt stop was reached. Do not tune or extend this panel.
The v14 candidate is a development E0 implementation and is not yet
connected as the formal RGB feedback method. Neither the selected root 6
chain nor any other chain was subjected here to a prospectively specified
same-chain mechanism intervention. Therefore PC2 and PC3 remain
unestablished; PC0, PC1 and PC4–PC5 are not certified by this experiment.
