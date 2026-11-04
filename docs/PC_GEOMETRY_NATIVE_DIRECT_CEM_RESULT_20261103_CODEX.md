# Direct full-horizon CEM versus the exposed prefix bank

The independently fixed [protocol](PC_GEOMETRY_NATIVE_DIRECT_CEM_PROTOCOL_20261103_CODEX.md)
uses the same four starts, official MuJoCo 2.1.2 XML, persistent action
clock, 15-action horizon and terminal position distance as the earlier
privileged prefix-bank diagnostic. This is a development comparison
performed after observing the earlier bank outcomes, not a prespecified
independent statistical test.

| Start | Direct CEM 100 | 300 | 600 | 1200 | Best of all 12 prefix CEM searches |
| --- | ---: | ---: | ---: | ---: | ---: |
| (0.8, 0.8) | 0.05633 | 0.05633 | 0.05633 | 0.05633 | 0.16163 |
| (0.8, 1.8) | 0.43157 | 0.25447 | 0.08441 | 0.03116 | 0.02761 |
| (2.8, 0.8) | 0.31830 | 0.17647 | 0.17647 | 0.17647 | 0.08726 |
| (2.8, 1.8) | 0.13267 | 0.04445 | 0.04445 | 0.02205 | 0.01912 |

Each bank point above consumes 12 × 100 suffix evaluations, excluding
prefix generation and common-suffix diagnostics; direct CEM at 1200
consumes 1200 full-chain evaluations. These numbers are a useful
physical-evaluation comparison, not a certified equal wall-time or model
compute comparison. The two methods use different search dimensions and
random streams. The best bank directions on the middle two non-progress
examples differ from direct CEM by only 0.00355 and 0.00293 at the final
budget point. The other two bank winners do not qualify as non-improving
and the first is worse than direct CEM.

All results use privileged simulator position in the optimizer and
post-outcome selection of the bank's best prefix. None is an RGB-only
model-selected outcome, an efficiency claim for spectral programs, or
P1/PC2/PC3 admission. The direct search's budget curves are informative:
at (0.8, 1.8) the score drops from 0.43157 at 100 calls to 0.03116 at
1200, so comparing only the first budget would be misleading.

The [receipt](../le-wm-main/results/pc_geometry_native_direct_cem_20261103_codex.json)
stores all 4,800 evaluated action chains and curve points. Each of the 16
curve-winning chains was independently replayed with the native evaluator.
Six combined targeted tests passed for the backend, fixed prefix bank and
direct CEM diagnostics.
