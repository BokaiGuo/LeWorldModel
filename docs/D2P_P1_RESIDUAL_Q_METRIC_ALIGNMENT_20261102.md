# Residual-prefix RGB metric alignment

This is a post-outcome development diagnostic on one previously used dev303
root. The frozen CUDA model rebuilt the existing goal-RGB tangent metric Q
without training and verified the goal embedding. The same eight sealed
model predictions, actual RGB latent encodings and physical outcomes were
compared. Q was not used to reselect actions or change any earlier gate.

Within each of the two four-arm groups, six pairs were compared. Lower latent
distance, position error, angle error and negative reward are treated as
better. Pairwise counts are descriptive, not independent statistical samples.

| Planning group | Distance | Predicted/actual latent inversions | Actual distance/reward inversions | Actual distance/position inversions |
|---|---|---:|---:|---:|
| latent | unweighted | 0 | 4 | 5 |
| latent | goal Q | 2 | 4 | 1 |
| goal-tangent | unweighted | 4 | 4 | 2 |
| goal-tangent | goal Q | 1 | 1 | 4 |

Each position comparison contains one physical-position tie; tied pairs
cannot be counted as inversions. The corresponding angle-error comparisons
also contain one physical-angle tie. For reward there are no ties. Q may
therefore change which mismatch is exposed, but this one-root analysis
does not support Q as a generally superior task metric. Physical reward,
position and orientation are distinct; no post-hoc preference between them
was introduced.

The saved result contains all metric values, per-pair differences and tie
flags, rebuilt Q metadata, and model before/after hashes:
`le-wm-main/results/p1_residual_metric_alignment_20261102_run01.json`,
SHA256 `1d79d546f43d927fa80e62fdcf69fdbd3db9692c084513ee6d1446ecab07f4ee`.
The numeric direction and nonfinite-input tests pass. The previous saved
trajectory verification is rerun before analysis. No independent outcome,
new physical action, planner change or P1 admission is implied.

Following the experiment-code workflow, this run was a targeted evaluation
of the existing metric, rather than a new task-specific scorer fit to the
same outcomes. Any new metric must be frozen before evaluation on a separate
development panel, and must not use simulator state inside an RGB planner.
