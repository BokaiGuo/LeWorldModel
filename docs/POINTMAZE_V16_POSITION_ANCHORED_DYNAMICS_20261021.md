# PointMaze V16: position-anchored dynamics repair

V15 revealed a representation-objective mismatch: V14 history improves velocity but worsens RGB position. V16 keeps direct RGB position and injects only frozen V14 velocity into local visual dynamics.

`direct RGB position + frozen V14 velocity + local visual dynamics`

The 32-root design stage used only privileged-oracle versus RGB-greedy headroom to choose the goal. It did not read full-V14 or anchored performance. It froze `[0.8, 1.8]`, then retained the goal, checkpoint, horizon, calibration roots, and code.

| Evaluation panel | Roots | Anchored gain over RGB greedy | 95% paired bootstrap CI | Anchored / RGB success |
| --- | ---: | ---: | --- | ---: |
| Primary | 32 (`7322–7353`) | +0.13183 | [+0.00595, +0.28476] | 30/32 / 28/32 |
| Independent replication | 62 (`7354–7415`) | +0.02252 | [-0.02018, +0.07329] | 58/62 / 55/62 |
| Fixed-design pooled estimate | 94 | +0.05973 | [+0.00671, +0.12116] | — |

The replication has a positive point estimate but crosses zero; this is not two individually significant replications. The pooled fixed-design estimate is positive. The privileged oracle retains positive headroom in both panels.

This is real RTX 5060 Ti CUDA evidence with RGB-only deployable inputs. It is narrow same-dynamics PointMaze evidence for observation-anchored fusion, not frozen-latent-only planning, cross-environment transfer, or a universal world model.

Artifacts:

- `le-wm-main/results/pointmaze_v16_position_anchored_design_20261021_cuda/design.json`
- `le-wm-main/results/pointmaze_v16_position_anchored_primary32_20261021_cuda/report.json`
- `le-wm-main/results/pointmaze_v16_position_anchored_replication62_20261021_cuda/report.json`
