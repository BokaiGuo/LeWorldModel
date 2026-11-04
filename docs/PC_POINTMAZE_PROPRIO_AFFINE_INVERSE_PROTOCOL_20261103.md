# Fixed-bank proprio affine-inverse diagnostic

The current frozen PointMaze checkpoint config declares a one-step 4-to-16
affine proprio encoder (`prop_mlp=false`, `tubelet_size=1`) whose feature is
repeated spatially. Prior goal-token Euclidean costs blend position,
velocity and encoder geometry. This development test asks whether the fixed
encoder can be analytically inverted to produce a position-only candidate
score without training a readout or using physical outcomes to select actions.
It is a new, retrospective diagnostic, not a modification of prior scores.

Inputs are exactly the eight 48-candidate groups in the sealed
`pc_pointmaze_velocity_score_crosstab_RUN_20261103_codexexclusive_02` result,
and its saved three sets of predicted proprio tokens and public goal RGB
state. Use the checkpoint's 16x4 affine weights/bias and the same stored
archive-wide normalization mean/std used by the model. These statistics'
relationship to training remains unverified. Require full column rank and
finite numbers; compare reconstructed encoded goal with the saved goal
token before scoring. For each candidate and each of the three velocity
conditions, average its 256 predicted tokens, subtract the goal token,
project with the Moore-Penrose inverse, multiply by the two position
normalizer standard deviations, and score by the resulting 2-D Euclidean
norm. Record residual to the encoder affine subspace. This is a projection,
not a claim that off-manifold predicted tokens decode to true states.

Compute scores before loading physical distances. Then evaluate paired
within-bank ranking reversals and physical distance at the deterministic
argmin, alongside previously recorded visual and proprio latent scores.
Save all groups, ties and failures without outcome-based filtering. The
real-qvel input is diagnostic only, and candidate-specific actual cp5 images
are oracle replays; no planner may use those inputs in deployment. No model
inference, physical execution, or parameter training is needed: this reads
previous host-CUDA predictions and checkpoint weights. Strong source SHA and
shape binding, unchanged source/model weights, and a source/result receipt
are required. Previously exposed roots and candidates are dependent;
none of these outcomes is an independent or gate-passing result.
