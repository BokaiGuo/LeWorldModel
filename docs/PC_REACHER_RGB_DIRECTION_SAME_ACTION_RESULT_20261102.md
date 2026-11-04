# Same-action scoring of L2 and normalized-direction suffixes

For each of 26 fixed prefix contexts on two exposed roots, the two saved
physically executed suffixes were rescored from the same real cp5 RGB, goal,
and actual action history. The frozen model received the saved bounded
normalized action packets directly. The raw actions reconstructed from all
saved normalized packets match the saved physical action witnesses exactly
in this CPU reconstruction; this does not independently certify the
original model tensors or simulator execution.

| Root | Direction suffix physically better | L2 model prefers direction | L2 model vs physical reversed | Direction model vs physical reversed |
| --- | ---: | ---: | ---: | ---: |
| 20263160 | 4/13 | 9/13 | 9/13 | 7/13 |
| 20263161 | 0/13 | 1/13 | 1/13 | 0/13 |

There were no exact score/physical ties in these 26 comparisons. Reversed
means that the sign of the model's predicted terminal-cost difference is
opposite to the physical final wrapped joint-distance difference for the
same pair of suffixes. The physical and model metrics are different, and the
real final RGB latent for these 52 branches was not encoded here. Thus this
result **does not isolate** model prediction error from latent-to-task
scoring mismatch. Nor does it prove the new metric is better: the two roots
are exposed, prefixes share their roots, and this is post-exposure analysis.

The L2 model's re-score prefers the direction suffix in 9/13 first-root
pairs despite its optimizer originally selecting the L2 suffix. The two
CEM searches had different objectives and therefore different adaptive
candidate populations; this is not a contradiction or proof of an
optimizer defect. Batch-level numerical differences and unsaved original
candidate scores are additional limits.

The saved-only consumer validates source hashes, full physical/RGB pairing,
ordered normalized actions and physical outcomes, reconstruction residuals,
finite scores and rank counts. It cannot independently recompute model
scores: the prediction vectors were not saved. No new physical steps or
training occurred. Ten targeted tests passed, including source tampering and
an analytic equivalence check between direct bounded-action rollout and the
existing parameterized rollout.

- Plan SHA-256:
  `5e2971cc4f1ff1a49ab5e7e5a3327a3bc5e81c60ba37271c7b4b5b4e5836abdb`
- Model re-score SHA-256:
  `4be88439b65bb7222f0e796d6fa80cbb9da3fbe6eb7321e9800c1dc15ffea001`
- Saved verification v2 SHA-256:
  `552d58283e3a573e8693014393a554c783c6823555527ca111eea9a644b59937`

The next causal decomposition requires actual terminal RGB encoding of the
same suffixes or newly executed fixed actions with complete terminal frames,
plus paired latent-vs-physical ranking; it is not P1 confirmation.
