# Dev304: model, real RGB and physical endpoint comparison

This is a post-outcome development diagnostic of the 18 fixed candidates in
`D2P_PC_DEV304_FEEDBACK_SPECTRAL_20261102.md`. The two feedback contexts share
one known root. All candidate actions were already fixed and physically
executed before this analysis; no physical outcome was used to select them.
The frozen model encoded the goal and all 18 real terminal RGB images on
host CUDA. Its weights were unchanged.

Within each context, all 36 candidate pairs were compared. Across both
contexts, predicted versus real-encoded goal distance agreed for 48 pairs
and disagreed for 24; real-encoded goal distance versus object position
error agreed for 44 and disagreed for 28; versus angle error, 58 and 14;
versus negative native reward, 49 and 23. No exact ties were recorded.
These 72 pairs share trajectories and a root, and are not independent
episodes. Small nonzero margins count as directional differences; these counts
do not establish their statistical or practical significance.

For non-anchor spectral candidates, the norm of the linear prediction error
relative to the nonlinear model displacement ranges from 0.696 to 0.958;
for random candidates it ranges from 0.212 to 0.468. This is descriptive at
the actual clipped actions: small denominators and floating-point noise matter.
It is not a physical prediction error or a calibrated trust region. The
earlier finite-difference test at epsilon 0.001 therefore does not certify
the linear approximation at radius 0.15. Anchors have no action displacement,
so their relative linear error is undefined: an initial analysis mistakenly
reported 1.0 because two anchor rollouts differ at floating-point scale. The
original output is retained; DEV02 corrects the diagnostic and the test
asserts that anchor ratios are null. No experiment or selection is changed.

The data show both model-to-RGB and RGB-to-task ordering mismatches on this
panel. They do not establish the cause, prove a code interface is correct in
all settings, or validate a new objective. No candidate succeeded in the
original physical experiment. This analysis is not evidence of PC2/PC3,
legacy P1 admission, or independent generalization.

Artifacts, relative to `le-wm-main/`:

- `results/p1_dev304_spectral_real_rgb_20261102_RUN_DEV01/report.json`
  SHA-256 `38f149b6d472a134717509f1f15331d9a66e806e90dd569f770a9216438b8b89`
- Corrected analysis `results/p1_dev304_spectral_real_rgb_saved_20261102_DEV02.json`
  SHA-256 `e911211fb095c91da536023fda44ca100821793ea0977e39488f69993d0d96ef`
- Initial analysis `results/p1_dev304_spectral_real_rgb_saved_20261102_DEV01.json`
  is preserved as a superseded diagnostic.
