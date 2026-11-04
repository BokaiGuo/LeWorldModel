# FD fixed physical actions: saved-RGB model replay protocol

Purpose: for all 20 previously executed development branches, compare the
same-action open predictions and one-packet predictions refreshed with
recorded RGB/action history to the actual encoded RGB at steps 5, 10 and 15.
Compare goal-Q ranking and wrapped joint-distance ranking for all five
selected/parameter-zero matched suffix pairs per root. This is exploratory
analysis of already-exposed roots; no action is selected and no result can
change an existing gate.

First verify the pinned physical report, plan, model report and saved-model
audit with the independent consumer, including current source hashes. Bind
the preflight, checkpoint, action normalizer and goal metric. Reconstruct each
root's warmup exactly to recover its causal three-frame history and require
the original RGB/goal/history identities. For each saved branch, decode all
three recorded checkpoint RGBs after checking their hashes, and shift only the
saved prior frames and executed raw actions into the causal history. No
simulator state is passed to the model. Recompute normalized actions from
the actual saved raw actions and compare with the corresponding model actions.
The step-15 open prediction must agree with the sealed model endpoint at
`atol=rtol=1e-5`. Save the three predictions, observed latent, Q costs, MSE,
RGB hashes and physical evaluation distance. A failed branch remains an error
and is not dropped from denominators.

Use host CUDA only; model weights remain frozen and hashes are checked.
Allocate a fresh result directory, record errors, timing, attempted calls,
and all identity checks. A result is not evidence of improved feedback control,
independence, equal-budget advantage, or P1 admission. Record explicit costs
and do not treat shared-root branches as independent samples.
