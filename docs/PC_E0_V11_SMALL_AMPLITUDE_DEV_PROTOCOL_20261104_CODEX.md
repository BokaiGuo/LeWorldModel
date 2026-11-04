# E0 v11 amplitude ablation: development protocol

## Question and fixed design

The post-hoc scale sweep found large local-response extrapolation errors on
previously optimized E0 prefixes. Test whether reducing v11's response-proposal
amplitudes improves complete-chain optimization, keeping every other compiler
setting, random seed, and maximum prediction-token budget fixed. The small
amplitudes `(0.10, 0.20)` are an exploratory design choice, not a certified
trust region. The usual v11 configuration uses `(0.25, 0.50)`. Both relinearize
at each visited successor. Neither arm detects linearization failure or
performs adaptive shrinking; joint full-chain CEM refinement is unrestricted.

Compare both v11 arms with the existing three-start direct full-action CEM
baseline. All arms use 1920 five-physical-step prediction tokens, retain the
best scored complete chain, and report final distance, `<0.05` success, actual
consumed tokens, trajectory samples, prediction calls, SVD count and measured
wall time. Equal token caps do not establish equal total compute. Record all
actual float32 actions, model outputs and scores. No invalid or failed run is
excluded from the reported denominator.

Eight new development tasks are Cartesian combinations of start
`[-0.5, y, theta]`, `y in {-0.19, 0.21}`, `theta in {-0.55, 0.9}`, and goal
`[0.5, gy]`, `gy in {-0.17, 0.13}`. Reject exact overlaps with the explicit
v5/v10, waypoint and response-waypoint panels checked by the runner. This is
not a certificate against all historical exposure, nor a confirmation set.
Each task/arm runs once, seeds specified in the runner; no post-result tuning,
task replacement or additional repetitions. This is a bounded exploratory
comparison, not a powered statistical test or PC2/PC3 confirmation.

Before interpreting results, recompute every recorded trajectory with the
exact endpoint model and an independent line-segment collision implementation;
verify hashes, actions, token accounting, score and final selection. Failures
remain failures. If all 24 runs complete, stop this panel regardless of outcome.
Future revisions require a different hypothesis, version and development panel.

This experiment does not establish mechanism, fair RGB performance, spectral
increment or broad extrapolation validity. A positive result cannot replace
same-final-chain interventions, fully costed strong baselines, or source-
independent confirmation.
