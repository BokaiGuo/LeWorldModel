# PointMaze P0: oracle-headroom repair

## Why the original controller comparison was not always identifiable

A controller can only demonstrate action-selection value when the shared
candidate bank and planning horizon leave measurable improvement over the
baseline.  In a fresh PointMaze extension, a design procedure selected the
goal with the largest *mean* privileged one-step-oracle minus RGB-greedy
terminal-distance gap.  Its fresh primary panel nevertheless had an oracle
gain of only `+0.00224`, with a paired 95% bootstrap interval
`[-0.03662, +0.03563]`.  The anchored controller's negative result in that
panel is therefore classified as `planner_limited`, not as evidence against
the representation or dynamics repair.

The failure was a design-contract issue: maximizing a noisy mean does not
guarantee that a testable action-selection gap exists.

## Repair

`experiments/run_pointmaze_v16_position_anchored_dynamics.py` now uses a
versioned design artifact with an oracle-headroom gate.  A candidate goal is
eligible only if the paired bootstrap lower confidence bound of

\[
  d_{\rm RGB\ greedy} - d_{\rm privileged\ oracle}
\]

on disjoint design roots exceeds a declared non-negative lower bound.  It
then selects by this lower bound before its mean gain.  A failed gate creates
a design artifact with status `rejected_insufficient_oracle_headroom`; the
evaluator rejects it rather than scoring learned controllers.

This is a benchmark/planner identifiability gate.  The privileged oracle is
used only for design-time headroom assessment and never enters the RGB policy.

## Fresh P0-qualified panel

The second independent trajectory extension used a new generator seed and
unread roots.  Its design roots `7736-7799` selected `[1.8, 2.8]` only after
passing the gate:

| Quantity | Value |
| --- | ---: |
| Design oracle-minus-greedy gain | `+0.13043` |
| Design paired 95% CI | `[+0.05296, +0.21703]` |
| Primary roots | `7800-7831` |
| Replication roots | `7832-7893` |

The frozen algorithm, goal, calibration roots, candidate budget and
replanning horizon were then unchanged.

| Evaluation panel | Anchored minus RGB-greedy terminal-distance gain | 95% paired bootstrap CI |
| --- | ---: | --- |
| Primary (32 roots) | `+0.00420` | `[-0.04239, +0.06470]` |
| Replication (62 roots) | `+0.02365` | `[-0.03283, +0.09370]` |
| Fixed-policy pooled (94 roots) | `+0.01703` | `[-0.02357, +0.06626]` |

This does **not** establish a closed-loop gain.  The oracle does retain a
positive replication interval (`+0.06401`, `[+0.01459,+0.12621]`), but that is
not a benefit of the learned controller.

Trace-level analysis identifies the remaining mechanism: RGB-greedy matched
the privileged oracle action on `85.3%` of primary and `88.4%` of replication
blocks, whereas the position-anchored V14 controller matched it on `48.7%`
and `45.0%`.  Thus the current random-initial-state regime still leaves too
little persistent action-choice ambiguity for the learned dynamics component;
when it intervenes, it commonly moves away from the oracle action.

## Consequence

The next valid PointMaze experiment is a predeclared geometry-stratified
obstacle-intersection challenge pool.  Initial conditions must be selected
from static maze geometry before any learned-policy evaluation, with separate
design and test roots.  It must again pass the same P0 oracle-headroom gate.
This is necessary before interpreting a controller comparison as a
representation, dynamics, or GDD-WM result.

## Artifacts

- `le-wm-main/results/pointmaze_v16_position_anchored_fresh_design_20261021_cuda/design.json`
- `le-wm-main/results/pointmaze_v16_position_anchored_fresh_primary32_20261021_cuda/report.json`
- `le-wm-main/results/pointmaze_v16_position_anchored_fresh2_design_20261021_cuda/design.json`
- `le-wm-main/results/pointmaze_v16_position_anchored_fresh2_primary32_20261021_cuda/report.json`
- `le-wm-main/results/pointmaze_v16_position_anchored_fresh2_replication62_20261021_cuda/report.json`
