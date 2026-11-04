# Dev304 Q-weighted model-to-RGB and physical decomposition

This post-outcome development diagnostic encodes the 18 fixed physical
candidate endpoints from the Q-weighted spectral experiment. It applies the
same saved goal metric to model-predicted and actual RGB latents; no outcome
is used to reselect actions. Frozen model hashes agree, and CUDA accounting
records 19 encoder and 19 projector samples, without prediction calls.
Candidate/action/frame bindings are checked by the physical saved verifier;
the analysis independently reproduces each saved predicted Q score.

For each of the two feedback contexts, the model-selected spectral candidate
has lower predicted Q distance and lower actual RGB Q distance than its
anchor, but worse object position error, angle error, and native reward.
This directly demonstrates metric-to-physical disagreement on these two
examples, not a proof that model prediction errors are absent.

Across 72 candidate pairs within the same feedback context, predicted versus
actual Q rankings disagreed in 13; actual Q versus position disagreed in 50;
versus angle in 28; and versus negative reward in 29. No exact ties were
recorded. Pairs share one known development root and trajectories; they are
not independent trials, and very small nonzero margins count. This is not
an estimate of a population failure rate or proof of a unique root cause.
No branch achieved native success; this result does not establish P1 or any
independent generalization.

Artifacts relative to `le-wm-main/results/`:

- Encoding contract SHA-256
  `a71bca0efa55352977bd3284650d49420ac5ac20060f9381e59bf86886e650d4`.
- CUDA encoding report `p1_dev304_qgoal_real_rgb_20261102_RUN_DEV01/report.json`,
  SHA-256 `bc6342f59b8b07b952ea388ab4f1b2ce50850b9156a7c2e01a3edf19ca18bd0d`.
- Analysis `p1_dev304_qgoal_real_rgb_saved_20261102_DEV01.json`, SHA-256
  `53bdcbb85bbe7609a4d696432e578360e374d29e61ee6a402cab945a99c3f090`.
