# Confirmatory covariance--predictor diagnostic audit

This audit treats each source checkpoint/environment artifact as one cluster. Split seeds are nested stability probes and are averaged within source before inference.

## Predeclared protocol

- Primary endpoint: `h4_k65` (horizon 4, 65 candidates).
- Secondary endpoints: horizon 2 with 65 candidates and horizon 4 with 17 candidates, when supplied.
- Estimand: equal-weight source-cluster mean of alignment regret minus validation-error regret.
- Test: one-sided exact sign permutation over source clusters; intervals: 95% source-cluster bootstrap.
- Split seeds, episodes, candidates, and horizons are not independent replication units.

## Results

| Endpoint | Horizon | Candidates | Source clusters | Mean delta | 95% cluster bootstrap CI | Exact p | Holm p | Reject |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| h4_k65 | 4 | 65 | 8 | -1.20842 | [-2.46621, -0.198932] | 0.0155642 | 0.0466926 | True |
| h2_k65 | 2 | 65 | 8 | -0.0290662 | [-0.0705325, 0.0157823] | 0.0894942 | 0.124514 | False |
| h4_k17 | 4 | 17 | 8 | -0.836009 | [-1.90727, -0.0680118] | 0.0622568 | 0.124514 | False |

## Source-level effects

| Endpoint | Source | Delta |
| --- | --- | ---: |
| h4_k65 | DINO-WM-PointMaze-base | -0.0546481 |
| h4_k65 | DINO-WM-PointMaze-s0 | -0.0609706 |
| h4_k65 | DINO-WM-PointMaze-s1 | -0.0777301 |
| h4_k65 | DINO-WM-PushT | -0.0167046 |
| h4_k65 | JEPA-WM-PointMaze-s0 | -3.99154 |
| h4_k65 | JEPA-WM-PointMaze-s1 | -1.62516 |
| h4_k65 | JEPA-WM-PointMaze-s42 | -3.88364 |
| h4_k65 | JEPA-WM-PushT | 0.0430476 |
| h2_k65 | DINO-WM-PointMaze-base | 0.00718592 |
| h2_k65 | DINO-WM-PointMaze-s0 | -0.00381699 |
| h2_k65 | DINO-WM-PointMaze-s1 | -0.00608084 |
| h2_k65 | DINO-WM-PushT | 0.080942 |
| h2_k65 | JEPA-WM-PointMaze-s0 | -0.0835966 |
| h2_k65 | JEPA-WM-PointMaze-s1 | -0.0973693 |
| h2_k65 | JEPA-WM-PointMaze-s42 | -0.119331 |
| h2_k65 | JEPA-WM-PushT | -0.0104631 |
| h4_k17 | DINO-WM-PointMaze-base | -0.0554543 |
| h4_k17 | DINO-WM-PointMaze-s0 | -0.0647453 |
| h4_k17 | DINO-WM-PointMaze-s1 | -0.0786916 |
| h4_k17 | DINO-WM-PushT | 0.0176287 |
| h4_k17 | JEPA-WM-PointMaze-s0 | -3.98354 |
| h4_k17 | JEPA-WM-PointMaze-s1 | -1.741 |
| h4_k17 | JEPA-WM-PointMaze-s42 | -0.923295 |
| h4_k17 | JEPA-WM-PushT | 0.141036 |

## Boundary

- The primary result is a source-cluster stability analysis, not an episode-level generalization claim.
- A positive result would support conditional model-selection utility for the declared fixed-spectrum probe family only.
- The audit does not establish end-to-end neural training gains, physical causality, or online planning.
