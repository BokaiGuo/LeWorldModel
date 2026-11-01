# Exploratory RGB-derived goal-pusher metric

This diagnostic was specified after prior physical/model results and is not an
independent confirmation or a preregistered experiment. It reuses all eight
known-task groups and all four fixed 15-step candidate action sequences per
group. The frozen model replays the same actions and first checks the
original latent mean-square scores against the saved model report.

For each original goal image, four synthetic variants translate the
blue-dominant pusher sprite by eight pixels left, right, up and down.
Non-white destination overlap and image clipping are rejected. Erased sprite
pixels are replaced with white: this is not an exact physical render and may
remove occluded content. The goal and four variants are encoded by the same
frozen model on host CUDA. Two central-difference latent vectors span a
two-dimensional basis when rank two is verified. For each original model
rollout endpoint, the exploratory score is
`mean(residual^2) + 16 * ||basis.T @ residual||^2 / 192`, with first-declared
ties. No physical state enters this score, and no model or readout is trained.

The fixed multiplier is exploratory, not calibrated, and the extra encoding
cost is accounted separately. This only tests whether a goal-image-derived
metric changes the model-side selector. It does not show improved physical
outcome, equal-budget performance, true capability change, or P1 admission.
The preliminary first attempt stopped before CUDA inference because the exact
color mask collided with antialiased sprite pixels. The revised generator
includes blue-dominant antialiasing and continues to reject other non-white
overlaps; it requires the full candidate replay and original-score check.

## Preflight outcome

Both `HOST01` and `HOST02` attempts stopped before CUDA inference and before
creating their requested result directories. `HOST02` still rejected a
non-white target. A read-only pixel inspection found non-blue block-colored
pixels including `[119,136,153]` in translated destination regions of the
second goal; overlapping pusher/object imagery makes naive white erasure
and relocation an unreliable reconstruction. Some non-white pixels also
come from original anti-aliasing, so a general collision label alone is
insufficient to assign the cause in each direction. No metric scores or
candidate selections were produced. These are image-construction/preflight
failures, not scientific evidence for or against the metric.

Do not simply relax the non-white guard or drop the obstructed root.
The next version needs a separately fixed intervention that handles
occlusion and preserves the original scene, or a compatible set of genuine
same-object counterfactual images; its scientific and compute scope must be
documented before running.
