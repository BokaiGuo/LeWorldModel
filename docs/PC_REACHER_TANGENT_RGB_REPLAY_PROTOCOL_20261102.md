# Tangent common-tail same-action RGB replay protocol

Bind the model-only report SHA
`d85a12443483e2268a0849ad72b6afccf3b68804a536df06bb0f780620f41589`,
the complete 70-branch physical report SHA
`2c3956da23c504f20b4b05d113e30c3f3055169c021e10e63b6732c963c90f74`,
and the model report's CPU preflight. Replay every fixed prefix/tail pair on
both exposed roots, without outcome-based filtering. Restore and check the
same root and goal, and check qpos distances at steps 5, 10 and 15 against
the corresponding saved physical branch. Bind raw and normalized actions
through the frozen normalizer.

At every five-step boundary compare (a) the open-loop prediction from the
root, (b) the one-packet prediction from immediately preceding real RGB and
executed action history, and (c) the encoding of the actual successor RGB.
At step 15 verify the predicted planning coordinate against the previously
sealed model endpoint. Save latent vectors, RGB hashes, qpos distance,
latent prediction errors and goal-Q costs. The simulator state is evaluation
only and must not influence actions or model inference.

All 70 branches are outcome-exposed, share two roots and use the same fixed
actions. This diagnostic cannot show that feedback improves control, nor
does prediction or task-distance agreement establish P1, independent
generalization, native success or equal-budget performance. Preserve every
branch and any errors or early stops; do not alter actions or infer
post-termination values. Account for encoder and predictor work separately.
