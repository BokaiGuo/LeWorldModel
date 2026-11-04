# Dev304: full-horizon action-response candidates after RGB feedback

## Scope and protocol

At each of two real cp5 RGB/action-history contexts from the same previously
used dev304 root, the frozen CUDA model evaluated the full two-packet
Jacobian `d z_terminal / d actions` (192 by 20) around that context's
previously planned feedback tail. Two SVD modes were ranked by damped
projection on the latent goal residual. Their positive and negative
directions at radius 0.15 were compared with two fixed random directions
using both signs and the unmodified anchor, giving nine candidates per
context. Actions were clipped to bounds and then evaluated with the nonlinear
model. The lowest terminal latent MSE was selected without physical labels.

All 18 candidates were subsequently replayed and physically executed, not
only the model winners. Each candidate starts after its own exactly replayed
prefix, so the two contexts are not independent roots and cross-context
results confound the prefix. The model objective is ordinary latent distance,
not a demonstrated pose metric. This is progress-program candidate
generation around an existing plan, not evidence of a reachability-changing
reconfiguration program or full D2P.

## Results

Finite differences along one spectral direction per context, at epsilon
0.001 before action clipping, had relative Jacobian errors 0.01267 and
0.01572. These are local checks, not evidence that the linear approximation
holds at radius 0.15 or at contact transitions.

| Context | Candidate | Model latent cost | Native reward | Object position error | Object angle error |
| --- | --- | ---: | ---: | ---: | ---: |
| single-population prefix | anchor | 1.566485 | -312.309 | 207.536 | 2.7110 |
| single-population prefix | model-selected spectral | 1.565938 | -291.355 | 207.621 | 2.7369 |
| two-population prefix | anchor | 1.612874 | -295.760 | 205.792 | 2.9933 |
| two-population prefix | model-selected random | 1.601240 | -283.754 | 205.948 | 2.8265 |

All 18 branches ran their full ten-step candidate tails, with no native
success. In the first context the spectral candidate improves native reward
but slightly worsens both object position and angle errors relative to its
anchor. In the second, the selected candidate is a random direction and
improves angle but slightly worsens position. We cannot attribute reward
changes to improved object control, infer spectral superiority, or regard
these 18 correlated branches as independent episodes.

Model accounting: 384 VJP attempts and returns, two SVDs, 16 predictor calls
on 48 samples, seven encoder calls, and identical before/after model hashes.
The separate 18 physical branches and prior anchor search add costs; there is
no end-to-end equal-budget certification. The saved verifier checks source
hashes, action bytes, prefix replay, 252 decoded RGB frames, termination,
row shards, finite-difference arithmetic and model winner indices. It does
not independently rerun the model, prove checkpoint training provenance or
establish independent confirmation.

Artifacts: `results/p1_dev304_feedback_spectral_20261102_RUN_DEV01/report.json`,
`results/p1_dev304_feedback_spectral_physical_20261102_RUN_DEV01/report.json`,
and `results/p1_dev304_feedback_spectral_saved_20261102_DEV01.json`.
