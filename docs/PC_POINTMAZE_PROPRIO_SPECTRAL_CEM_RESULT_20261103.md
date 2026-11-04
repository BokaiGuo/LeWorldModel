# Proprio-Jacobian spectral versus random feedback CEM

The [fixed protocol](PC_POINTMAZE_PROPRIO_SPECTRAL_CEM_PROTOCOL_20261103.md)
executed both arms on all eight exposed development roots with host CUDA,
real RGB feedback at step five, and ten physical steps per arm. Both arms
used the same predicted-proprio goal score and 48 initial plus 48 tail
samples. Spectral replaced four initial candidates using the top right
singular vector of the predicted proprio action Jacobian. The model parameter
hash remained unchanged, and all terminal images and positions matched
separate replay.

| Root | Random distance | Spectral distance |
| ---: | ---: | ---: |
| 1888 | 0.064630 | 0.064628 |
| 1889 | 0.503851 | 0.503851 |
| 1890 | 0.566931 | 0.542632 |
| 1891 | 0.246634 | 0.238115 |
| 1892 | 0.125917 | 0.125731 |
| 1893 | 0.289715 | 0.319644 |
| 1894 | 0.273890 | 0.244703 |
| 1895 | 0.418107 | 0.370462 |

Spectral terminal distance is lower on six roots, higher on one and equal
on one; the difference on 1888 is approximately 0.000002. The strict
offline endpoint threshold `<0.5` is **6/8 for both arms**. These results
are descriptive on exposed roots, not independent confidence estimates or
native online success. Root 1889 remains above threshold in both arms.

Each spectral arm used two extra JVPs and one SVD. Recorded structure times
range approximately 0.038–0.067 seconds; identical prediction sample counts
do not make this an equal-total-cost comparison. Goal encoding, rendering,
and the full low-level operation ledger were not separately metered. No
common-suffix intervention, non-greedy mechanism, program composition,
certified fresh roots, strong baselines, or P1 admission is established.

- Report:
  `le-wm-main/results/a1_pointmaze_proprio_spectral_cem_20261103_DEV01.json`,
  SHA-256 `b39bc04c3bceb2f54cc982b32d5517f5ae6be10d8f9ede598ac7496ab3ecda71`.
- Arrays: same basename `.npz`, SHA-256
  `0d02d85e0d650d6a21bdd6fe65fa2af9a580dc7874901b24c337261a9f7dd440`.
- Saved CEM traces, Jacobian singular direction, actions, endpoints,
  score and accounting checks: **3 passed**. Tests do not rerun CUDA or
  physics.
