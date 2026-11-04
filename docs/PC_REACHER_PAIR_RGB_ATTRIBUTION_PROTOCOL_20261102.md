# Fixed-pair real RGB latent attribution

Encode all 52 verified replayed branches using the frozen Reacher checkpoint,
host CUDA, and actual cp5/cp10/cp15 RGB plus the corresponding executed
five-step action packets. Use the already saved cp5 RGB and goal for the
direction reference. Do not select branches or change metrics after
inspection. Compare within each fixed pair: saved model L2 and direction
scores, true final-RGB encoded scores, and wrapped physical final distance.
Report each root separately, including ties; the 13 pairs within each root
are dependent and the roots were previously exposed.

This analysis separates prediction-to-observed-latent and
observed-latent-to-physical rank disagreement, but does not establish a
causal mechanism, planning advantage, or P1 admission. Replayed RGB and
state are used only for post hoc evaluation, not for selecting actions.
