# Fixed-action prediction / RGB / physical ranking decomposition

All 52 fixed action chains from the 26 exposed-root pairs were replayed with
host EGL. The source warmup, root/goal RGB and each cp5 RGB hash matched;
all 15 actions completed, and all final physical distances matched their
source within the predeclared absolute tolerance of 1e-6. A separate
saved-data consumer checked the hash-chained journal, all actions, frames,
root states, distances and complete branch roster.

Host CUDA then encoded actual cp5/cp10/cp15 frames with the corresponding
executed action packets. Using L2 and normalized-direction scores, respectively:

| Root | Model vs real latent rank reversal | Real latent vs physical rank reversal | Real latent ties |
| --- | --- | --- | --- |
| 20263160 | 2/13, 2/13 | 6/13, 6/13 | 2/13 |
| 20263161 | 1/13, 0/13 | 0/13, 0/13 | 0/13 |

Reversal excludes ties. Both ties on root 20263160 have identical final RGB
hashes and physical distance differences around 5e-9; these should not be
interpreted as substantive failures. Exact rank signs are reported as
predefined rather than retrospectively thresholded.

These are two previously exposed roots with dependent pairs. This is
post-exposure error attribution, not P1 admission, independent generalization,
or evidence that a changed selector wins. In particular, real-latent
disagreement with wrapped joint distance is not itself proof of a bug or a
unique root cause. The saved checker validates bindings and arithmetic but
does not independently recompute neural embeddings or saved model scores.
Future work should test a deployable RGB-only task-aligned score without
simulator-state leakage, then freeze it and evaluate independently.

The sandbox EGL attempt failed its root RGB identity check before executing a
branch; the separate host EGL replay passed. The first CUDA attribution
attempt failed at its first encoding because raw history was shaped
`(3,5,2)` instead of the model's `(3,10)` packet contract; the separately
sealed v2 corrected that connection. Neither failure is a scientific
negative result.

- Replay plan: `5657983996890506536def63c9cef3114a8a200bacba7638bb97285d9c65dede`
- Physical journal: `98569bcddc661cb776d35e1c297041421bcfc5f31e58e59aa0ea8e21f09f0bdc`
- Physical saved verifier: `15bca8f27a774b4583b7db7762c913a8220b1ea2e5c9b22827601ac01446b043`
- Attribution plan: `dbd68ad9663c882d4661fab96a551e3173ef57f3f43eb75ed286b9cffd32d6fc`
- CUDA attribution: `e31965acecdf6f650fa3142a019f266cbf1bcd1c6011d200fa7b07e04121ca67`
- Targeted tests: 4 passed, including action tamper and nonfinite rejection.
