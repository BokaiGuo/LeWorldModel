# CP5 pooled vs spatial objective development comparison

Use all 16 previously sealed prefixes and the same eight candidate tails,
actual RGB frames, and physical endpoints from the fixed-tail oracle panel.
Do not change candidate generation or use physical outcomes to select.
Recompute the frozen model's one-step prediction from each real CP5 RGB
context and the same normalized candidates. Compare three terminal scores:

1. prior pooled visual MSE, to reproduce the saved source selector;
2. full per-spatial-token visual MSE;
3. full per-spatial-token visual MSE plus 0.1 times predicted proprio-token
   MSE to the encoded goal's raw-zero proprio.

The third uses the arithmetic of the released PointMaze L2 objective and
its `alpha: 0.1` evaluation configuration, but this experiment retains
single-frame contexts, fixed-zero proprio, frame-10 goal, a small fixed bank
and other development choices; it is **not** full official baseline parity.
Encode all 128 already executed real successor RGB frames and calculate the
corresponding full visual MSE as an evaluation-only oracle. Report actual
position distance for each model-selected index from the same sealed
physical candidate panel. Confirm saved pooled scores, goal/CP5 RGB hashes,
candidate images and model identity; store full spatial predictions, real
encodings and scalar scores for reproducibility.

The candidate outcomes have been seen, so this is an exploratory method
comparison on four dependent roots, not an independent or preregistered
confirmatory test. No simulator state or real future outcome enters scoring;
physical outcomes are looked up only after the new selectors are computed.
Full-archive normalization, fixed raw-zero proprio, development camera and
unknown checkpoint training membership limit any deployment or P1 claim.
