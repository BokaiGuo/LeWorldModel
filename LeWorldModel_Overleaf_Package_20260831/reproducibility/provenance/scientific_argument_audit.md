# Scientific Argument Audit

## Verdict

Status: human-paper-ready with an oral-level scientific argument for the current
evidence package; oral selection is not guaranteed. The independent JEPA-WM
temporal diagnostic is now complete for the released checkpoint under a
two-frame held-out protocol, while task-level and cross-checkpoint generality
remain outside the claim.

The manuscript now reads as a theory paper organized around one scientific update: an eigenvalue-only spectral-transfer analysis is under-identified because an isospectral predictor family can change covariance-predictor eigenspace alignment and transfer readouts. The added temporal controls show how this measurement behaves across released checkpoint families without promoting the result to a universal learned-world-model law.

## One-Sentence Scientific Update

Same predictor eigenvalues do not determine spectral transfer: in a finite-dimensional isospectral family, rotating predictor eigenspaces relative to latent covariance produces an exact commutator curve and changes transfer readouts.

## Value-Relevant Comparison

The comparison is not model performance or online planning. The comparison is eigenvalue-only spectral audit versus eigenvalue-plus-eigenspace-alignment audit under a fixed predictor spectrum.

## Main Evidence Chain

| Manuscript object | Scientific question | Evidence | Status |
|---|---|---|---|
| Theorem 1 | Does the commutator have a closed-form oblique peak under fixed spectrum? | Single-plane block derivation | supported |
| Proposition 1 | Can predictor eigenvalues identify covariance-predictor alignment? | Isospectral family with nonconstant commutator | supported |
| Proposition 2 | Can a transfer readout vary under fixed predictor eigenvalues? | Closed-form horizon-one coordinate-energy expression | supported |
| Figure 2 | Does the analytical prediction appear in the full sweep? | Rotation sweep with top-band lift, residual growth, and random-basis null | supported |
| Table 1 | Is the peak explained by spectrum-preserving alternatives? | Permutation-preserving and spectrum-matched random-basis nulls | supported |
| Table 2 | Are spectrum shape and basis alignment separable? | Fixed-spectrum and fixed-alignment ablations | supported |
| Figure 3 | Does the measurement remain nontrivial in a separate frozen predictive-representation family? | I-JEPA ViT-H/16 features on six standard vision benchmarks | supported as a bounded learned-representation anchor |
| Supplement | Are the boundary, robustness, and learned-anchor details visible? | Transfer-proxy, anisotropy, dimension, mechanism, heat, oblique-frame, seed/noise, frozen PushT/I-JEPA, temporal controls, action interventions, split robustness, and estimator ablations | supported within stated scope |

## Strongest Rival Explanations

1. The result is only a property of predictor eigenvalue decay. Addressed by the isospectral construction.
2. The peak is a coordinate-order artifact. Addressed by the permutation-preserving null.
3. Any random basis would produce the same effect. Addressed by the spectrum-matched random-basis null.
4. The result is a single selected metric. Addressed by transfer-proxy robustness.
5. The result ignores spectrum shape. Addressed by the fixed-alignment anisotropy scan.

## Claim Boundary

The main paper supports a finite-dimensional same-spectrum ambiguity and its measurement consequence. It does not prove learned PushT performance, online planning improvement, JEPA training quality, image-classification accuracy, or the exact single-plane identity for arbitrary nonlinear learned latents.

## Current Remaining Risk

The main remaining reviewer risk is importance, not internal consistency: a reviewer may accept the theorem but ask whether the measurement consequence is consequential enough for modern learned representations. The current manuscript answers this in two ways. First, the finite-dimensional construction is an identification counterexample, so exact knowledge of predictor eigenvalues cannot certify transfer geometry without eigenspace-alignment information. Second, the frozen PushT and frozen I-JEPA anchors show that the same covariance-relative measurements are nontrivial outside the analytical matrices, without promoting them to learned-dynamics theorems.

The added held-out JEPA operator-fit rank sweep sharpens this boundary rather
than changing the thesis. Across five fits on two datasets and three ranks, a
low-rank view map predicts held-out features ($R^2=0.97141$--$0.99840$) while
the fitted commutator remains below the spectrum-preserving random-basis null.
This separates a robust displacement statistic from an identified learned
transition operator and removes a likely reviewer shortcut.

The independent JEPA-WM temporal diagnostic adds a second, action-conditioned
anchor without changing the theorem's scope. Across 21 held-out PushT
trajectories with two-frame context, latent MSE is 0.07257, 0.06465, and
0.09167 at horizons 1, 2, and 4. The commutator/null pairs are 1.17128/1.39180,
1.59889/1.40656, and 2.09273/1.41442. These values validate the measurement
protocol for one released checkpoint; they do not establish a universal
learned-dynamics law or downstream task gain.

The cross-family DINO-WM PointMaze replay provides an external stress test.
Using a three-frame context and all 200 validation episodes from the public
archive, with membership reconstructed from the released 0.9 split, seed-42
episode splitter, and official full-archive normalization, latent MSE is
0.00201, 0.00491, and 0.01121 at horizons 1, 2, and 4, while the commutator
exceeds the spectrum-preserving null at all three horizons (37.90/1.47,
61.51/1.42, 96.48/1.42). This transfers the diagnostic across implementations;
the protocol records the split as code-reconstructed because the archive does
not ship a serialized membership file.

A second DINO-WM PushT checkpoint supplies an explicit-split replication: all 21
public validation trajectories yield latent MSE 0.01672, 0.02316, and 0.06510,
with commutators 0.62873, 0.72850, and 2.08349 against null means 1.38166,
1.38331, and 1.43895. The horizon-4 crossing while h=1/2 remain sub-null is
important negative evidence: the readout is sensitive to regime and horizon,
not a universal monotone learned-dynamics law.

Finally, a matched-environment JEPA-WM PointMaze checkpoint was evaluated on all
200 episodes from the same code-reconstructed split. Full-rank commutators are
60.48199, 43.44444, and 37.19464 against null means 1.45494, 1.57804, and
1.53736; rank-truncated fits (8--128) remain above their null at every tested
rank and horizon. This provides full validation-coverage evidence and shows
that the cross-family signal extends beyond the 21-episode PushT setting.

The final estimator-control pass fits rank-truncated ridge operators only on
training folds and evaluates held-out episodes. At rank 16 and ridge
$10^{-2}$, paired $R^2$ is 0.082/0.183/0.025 for JEPA-WM and
0.935/0.808/0.554 for DINO-WM at horizons 1/2/4, while pair-shuffled controls
are negative throughout. This closes the same-sample-overfit shortcut without
changing the claim: it is evidence for checkpoint-local paired correspondence,
not a theorem about arbitrary learned world models.

The action-order intervention supplies a stricter temporal control: within-episode
future-block derangements preserve episode identity and the action-block multiset.
The resulting MSE changes are positive on PushT but heterogeneous on PointMaze,
so the paper treats them as fixed-checkpoint replay sensitivity rather than a
universal causal dynamics result. Reconstructed PointMaze memberships likewise
preserve positive paired-minus-shuffled gaps across splitter seeds and a tail
holdout, while h=8, context, direction, and sample-size analyses expose the
estimator choices that delimit the correspondence signal.

## Pass 30 Argument Compression

The main text now follows a single reviewer-visible chain: the isospectral
family establishes identification failure; the commutator identity isolates
eigenspace alignment; analytical nulls and heterogeneous learned checkpoints
map the empirical boundary; and the Discussion converts the result into an
operator-pair reporting principle. Repeated defensive disclaimers were replaced
by protocol-first descriptions and one concentrated scope statement.

## Repairs Made In This Pass

- Rewrote the Abstract around an isospectral counterexample rather than a generic spectral-analysis motivation.
- Sharpened the Introduction so the first page states the expectation being challenged.
- Reframed the Results opening as the main counterexample to spectrum-only transfer.
- Added Proposition 2 to connect the commutator argument to an explicit transfer functional.
- Rechecked that learned-latent and online-planning claims remain outside the theorem.
- Added an Introduction paragraph explaining why a finite-dimensional isospectral counterexample changes spectral-transfer measurement even before learned-checkpoint generality is claimed.
- Rewrote the Discussion so the central consequence is a positive audit requirement rather than a defensive toy-model caveat.
- Added a frozen I-JEPA six-benchmark anchor and updated the claim ledger so learned-representation evidence is separated from theorem evidence.
