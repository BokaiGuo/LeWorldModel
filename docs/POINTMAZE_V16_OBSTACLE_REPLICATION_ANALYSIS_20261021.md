# PointMaze V16 obstacle challenge: independent replication analysis

## Question and fixed controller

Can direct RGB position, combined with a frozen V14 velocity estimate and a
calibration-local dynamics map, improve closed-loop PointMaze control over RGB
greedy navigation when a static obstacle blocks the direct route to the fixed
RGB target `[1.8, 2.8]`?

The controller, target, candidate bank, calibration roots, 24-block horizon,
and design artifact were frozen before all outcome panels. GPU preflight used
the host RTX 5060 Ti and recorded a CUDA allocation plus cuDNN recurrent
probe in every score report.

## What the independent panels show

The original 32-root primary panel was positive: terminal-distance gain over
RGB greedy was `+0.13747`, 95% paired root-bootstrap CI
`[+0.03925, +0.25449]`. Its 62-root independent replication was not itself
decisive: `+0.00810`, CI `[-0.04443, +0.06306]`. This is not a reason to
silently retry or to tune the controller on those roots.

The root-level audit found a concrete protocol cause: roots had been sampled
iid across four *static*, generator-recorded start-cell strata. The
replication happened to include 24/62 roots from `(0.8,0.8)`, versus 8/32 in
the primary panel. In that stratum RGB greedy is already near saturation and
the anchored controller has mean gain `-0.04056`; the other three strata are
positive (`+0.03228` to `+0.15741`). Thus the panels estimate different
mixtures of geometries, even though neither was outcome-selected.

To test this explanation without reusing a root, a third 24-root panel was
sealed from unused IDs using exactly six roots per static stratum. Selection
read only the original generator labels and a fixed seed; it never read an
outcome report, policy score, or oracle rollout. It gave a positive point
estimate `+0.08038` and success `23/24` versus RGB greedy `19/24`, but its
small-panel paired CI `[-0.00478, +0.17697]` crosses zero. It is therefore a
supporting mechanism check, not a standalone success claim.

Across all three disjoint panels (118 roots), root-level—not time-block—paired
bootstrap gives a position-anchored terminal-distance gain of `+0.05788`,
95% CI `[+0.01446, +0.10545]`, with a strict-success increase of `+16.10pp`.
The privileged one-step oracle retains headroom (`+0.09462`, CI
`[+0.05787, +0.13404]`).

## Mechanistic boundary

The first action agrees with the privileged oracle on 94.9% of roots for the
anchored selector, yet its post-divergence trace-aligned candidate-ID agreement
is only 39.3%. The latter is explicitly a trajectory-divergence diagnostic,
not common-state action-ranking accuracy. The evidence supports an early
geometry/velocity correction in this declared obstacle challenge; it does not
establish stable action-effect ranking after trajectories separate.

## Permanent protocol repair

`generate_pointmaze_obstacle_challenge_extension.py` now supports
`--balanced-start-cells`: labels are allocated equally and permuted with a
seed *before* continuous jitter and rollout construction. This removes the
avoidable mixture drift in future primary/replication designs. The separate
root selector and aggregator refuse root reuse, mismatched design artifacts,
and block-level pooling.

## Evidence boundary

This is a same-dynamics, one-maze, fixed-goal, RGB-position-plus-frozen-V14
velocity intervention. It is not frozen-latent-only evidence, a GDD-WM
teacher result, cross-environment validation, or evidence for a universal
world model.

## Artifacts

- Design: `le-wm-main/results/pointmaze_v16_obstacle_challenge_design128_20261021_cuda/design.json`
- Primary: `le-wm-main/results/pointmaze_v16_obstacle_challenge_primary32_20261021_cuda/report.json`
- IID replication: `le-wm-main/results/pointmaze_v16_obstacle_challenge_replication62_20261021_cuda/report.json`
- Equal-stratum panel: `le-wm-main/results/pointmaze_v16_obstacle_challenge_stratified_replication24_20261021/panel.json`
- Equal-stratum score: `le-wm-main/results/pointmaze_v16_obstacle_challenge_stratified_replication24_20261021_cuda/report.json`
- Combined root-level audit: `le-wm-main/results/pointmaze_v16_obstacle_challenge_aggregate118_20261021/aggregate.json`
