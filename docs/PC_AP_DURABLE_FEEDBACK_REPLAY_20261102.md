# AP durable feedback: known-root replay

The new durable AP development launcher ran on host CUDA using the existing
cooperative lock. Four exposed root/arm slots completed two observed
five-step planning blocks each. The run reconstructs the same memory episodes,
model, roots and decisions as
`results/pc_ap_official_kernel_physical_dev_20261102_v1/report.json`: it is
a repeat of existing development outcomes, not four independent new trials.

The saved-record consumers verify journal chains, root/goal and feedback
bindings, predictor call lifecycle, and physical step records. A separate
comparison matches the earlier reference's seeds, action bytes, root/goal and
successor RGB hashes, and every saved step's distance/reward/stop fields.
Reference images are available as hashes, not raw images for independent
pixel comparisons; historical warmup actions cannot be compared stepwise
from that reference.

| Root | AP observed-target final distance | Final-goal final distance |
| --- | ---: | ---: |
| 20263160 | 0.804990 | 0.779041 |
| 20263161 | 1.346018 | 1.019378 |

These are ten-step wrapped joint distances on two previously exposed roots.
Each arm recorded two decisions with 34 predictor blocks and three batches per
decision. Other model, memory-index, encoding, warmup and wall-clock costs are
not fully metered. No recorded step had a stop flag or nonzero reward. This
does not establish official AP full-controller fidelity, training-data
independence or an equal-resource comparison to D2P; the latter's recent
feedback experiment executes 15 planned steps, not ten.

The durable saved consumer result SHA-256 is
`7177dcd0297a0c0531fdbdd08bede5bde8b7f0e0faa900d3ebe57c72f691f055`;
the reference replay consumer result SHA-256 is
`923f7f8117d5ba2e49b8761b85c890bdef5ef9513c58fbc6c8b87fca4467331a`.
The replay, saved-consumer and launcher tests passed together (21 tests).
These verify persisted records and test fixtures, not an independent physical
execution witness or a new scientific success.
