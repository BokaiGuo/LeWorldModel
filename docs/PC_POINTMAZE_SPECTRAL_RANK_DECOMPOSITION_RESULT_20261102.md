# Fixed spectral-candidate ranking decomposition

Following the fixed post-outcome protocol
`PC_POINTMAZE_SPECTRAL_RANK_DECOMPOSITION_PROTOCOL_20261102.md`, all
60 candidate slots from the prior A1 development run were executed in the
native RGB environment and encoded on host CUDA. The original model scores
were not recomputed or used to change the selected actions. The 12
previously selected RGB hashes and separately replayed physical endpoints
reproduced the sealed source exactly.

Across the 12 root-by-method sets, each containing ten within-set pairs:

| Comparison | Exact-sign reversed pairs | Exact ties |
| --- | ---: | ---: |
| Model predicted latent goal cost vs. actual RGB-encoded latent goal cost | 59/120 | 0 |
| Actual RGB-encoded latent goal cost vs. position-only goal distance | 45/120 | 0 |

These are descriptive counts, not 120 independent trials: there are only
four exposed roots, candidate slots within a set share a start, and zero
actions recur across sets. Exact equality, not a preregistered numerical
tolerance, defines a tie; small margins must not be interpreted as robust
ranking errors without a separate margin analysis.

The fixed root-1880 spectral set gives a concrete prediction failure. The
model selected index 3 with predicted cost 0.01834, but its actual encoded
cost was 0.02308 and physical distance 1.02121. Index 4 had worse predicted
cost 0.03525 but actual encoded cost 0.00700 and physical distance 0.26567;
it was the best of that fixed set on both latter measures. This is a
same-action, same-time exploratory comparison, not permission to
retroactively replace the selected action.

There is also a separate scoring-to-task failure: on root 1882's spectral
set the predicted-cost minimizer and actual encoded-cost minimizer are both
index 1, whereas the physical-distance minimizer is index 4. Thus
prediction/actual-latent agreement alone does not ensure the physical
objective is optimized. The physical position labels come from the audited
MuJoCo-2.2 replay backend, not directly from the online RGB bridge's hidden
state. The chosen goal is archive frame 10 while these actions finish at
step 5; this experiment preserves the original contract and cannot by
itself separate time-horizon choice from metric choice.

The protocol uses constant raw zero proprio, full-archive normalization,
a development-selected camera and known roots. The result saves individual
RGB hashes and encoded scalar costs, not the images or latent vectors; the
encoded distances cannot be independently recomputed from this report alone
without repeating execution and encoding. Four targeted tests recomputed
source bindings, candidate-to-endpoint distances and all saved pair margins;
they do not independently verify each saved model encoding. There is no
strong-baseline, equal-resource, independent-root, reconfiguration or P1
claim.

- Runner: `le-wm-main/experiments/a1_pointmaze_spectral_rank_decomposition_20261102.py`
- Result: `le-wm-main/results/a1_pointmaze_spectral_rank_decomposition_20261102_DEV01.json`
- Result SHA-256:
  `a2e8ffea229396075b6208837f6607268fe9496bbd79f9fb9bb1cafb82fc09e2`

Next compare a predeclared task-aligned RGB-based evaluation and shorter
feedback planning against the same saved action candidates, with raw
latent vectors and images sealed for independent recomputation. Any
modified selector must then be frozen and tested on separate roots.
