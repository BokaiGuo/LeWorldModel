# PointMaze V10: Large Fixed-Policy Confirmation

## Why this test exists

V9 selected the latent recurrent policy below using only its selection half,
before its outcome labels existed:

```text
history source = source_history_model
hidden width   = 16
contrast loss  = 1.0
temperature    = 0.75
policy         = recurrent
```

V9's independent 96-root authorization half did not permit execution, while
its subsequently opened 64-root outcome had a positive paired diagnostic. The
outcome cannot be used to turn V9 into an authorized controller. V10 therefore
does not introduce a new model, feature family, gate, or tuning search.

## Frozen origin and new evidence

- V10 verifies the V9 frozen-selection file declares the configuration above
  and records `outcome_labels_opened=false` and
  `outcome_labels_generated=false`.
- The resulting file hash is saved as V10's protocol reference.
- V10 uses new root IDs `[11832, 12920)` and includes the V9 context in its
  historical identity ledger.
- Its new panel contains 256 train, 512 calibration, and 256 outcome roots,
  plus 64 disjoint effect-screening roots required to construct the robust
  candidate bank.  It therefore uses fresh root IDs `[11832, 12920)`.
- Candidate-bank design remains the two-fold robust in-support bank and must
  pass before model fitting.

## Authorization and scoring

Because the complete policy is already fixed, all 512 fresh calibration roots
are authorization-only. No split is used to choose a feature, hyperparameter,
or routing rule. The recurrent policy is active only if its paired bootstrap
lower confidence bound over action-only is positive; otherwise V10 falls back
to action-only before opening its 256 outcome-root labels.

## Decision rule

- Positive authorization and outcome lower bounds: evidence that the V9
  positive diagnostic transfers under a larger fresh panel.
- Non-positive authorization: the controller remains fail-closed; V9's
  outcome improvement is classified as an unreplicated diagnostic direction.
- Authorization positive but outcome non-positive: the fixed-policy benefit
  did not transfer and the investigation returns to root-distribution or
  action-effect heterogeneity.

This remains renderer-backed, offline PointMaze evidence. It is not an RGB
closed-loop result, cross-environment proof, or universal-world-model claim.

## Construction correction before execution

The first V10 draft requested only the 1,024 evaluation roots.  The builder
correctly rejected it before a panel, calibration model, frozen policy, or
outcome label existed because the robust interaction-pair bank additionally
requires 64 roots disjoint from all evaluation roles.  The executable V10
contract therefore binds a new 1,088-root v30 extension.  The incomplete v29
extension is retained as an auditable, unused pre-build artifact and is not an
input to V10.
