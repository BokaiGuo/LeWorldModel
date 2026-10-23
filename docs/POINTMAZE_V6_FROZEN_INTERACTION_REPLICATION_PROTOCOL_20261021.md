# PointMaze V6 frozen-interaction replication protocol

V6.1 is the first fresh panel after repairing the V5 support-shift, outcome
sealing, candidate-ordering, and score-time-refit defects.  Its outcome labels
have now been opened and it must not be used to choose a new model family,
training budget, or calibration rule.

Before reading either remaining outcome panel, this document freezes two
independent replications of the unchanged V6 interaction protocol.

| Panel | Root seed | Held-action seed | Train-action seed |
| --- | ---: | ---: | ---: |
| V6.2 | 2026102620 | 2026102621 | 2026102622 |
| V6.3 | 2026102630 | 2026102631 | 2026102632 |

Every build must exclude contexts V1--V6.1, use the same frozen JEPA
checkpoint/PCA/renderer procedure, persist the selected recurrent and
action-only weights before scoring, and materialize outcome labels only after
the frozen-selection hash verifies.  No hyperparameter or architecture change
is allowed between panels.

Primary replication estimand: paired root-level regret gain of recurrent over
action-only.  The confirmatory aggregate reports a root-resampled bootstrap
over the concatenated independent V6.1, V6.2, and V6.3 outcome roots.  The
claim remains bounded to the renderer-backed offline matched-support setting;
privileged source position is only a shared task-cost readout and is not a
live RGB planning input.

## V6.4 candidate-in-support control

The completed V6.1--V6.3 panels match pulse magnitudes but retain fully
hash-disjoint train and held sequences.  They therefore remain a joint test of
history use and continuous candidate-sequence interpolation.  They cannot by
themselves answer whether history changes the ranking of actions that are
already in the model's intervention support.

Before building this panel, V6.4 freezes a separate, non-pooled control:

| Panel | Root seed | Held-action seed | Train-action seed | Candidate regime |
| --- | ---: | ---: | ---: | --- |
| V6.4 | 2026102640 | 2026102641 | 2026102642 | shared_candidate_in_support |

All 16 held sequences are included verbatim among the 64 source-training
sequences; source, calibration, and outcome roots remain mutually disjoint
and all roots/actions used by V1--V6.3 are excluded.  The architecture,
optimizer, epochs, calibration split, authorization criterion, frozen-weight
hash, and delayed outcome-label materialization are otherwise unchanged.

V6.4 is a mechanism control, not a rescue rerun and not part of the frozen
V6.1--V6.3 numeric-action-OOD aggregate.  A positive result supports
root/history-conditioned action ranking under supported interventions; a
negative result rules out the claim even after candidate extrapolation has
been removed.  Neither outcome alone establishes live RGB planning.

## V6.6 predeclared visual-motion selective-history control

V6.5 is closed and found a heterogeneous *exploratory* pattern: its aligned
history arm was directionally better than action-only in one low-motion
stratum and worse in its complementary stratum.  This observation is not used
as V6.5 evidence.  V6.6 tests one frozen response to that pattern on a new
panel, before any V6.6 outcomes exist.

The source model remains the same pulse-aware FiLM/action-only pair.  On the
calibration-selection half only, it may select from this predeclared policy
family using root-level visual motion energy
`||z_t - z_(t-35)||_2` computed from the eight stored visual history frames:

1. action-only;
2. recurrent for every root;
3. recurrent only below the selection-root 25%, 50%, or 75% energy quantile;
4. recurrent only above the same three quantiles.

The selected threshold and direction are frozen.  The authorization half must
have a strictly positive lower 95% paired-bootstrap regret gain over
action-only; otherwise deployment falls back to action-only.  Outcome roots,
candidate costs, and realised terminal states are inaccessible until this
artifact and its serialized model hash have been written.

V6.6 uses the in-support bank and excludes all roots and action hashes from
V1--V6.5.  It is a single predeclared selective-routing test, not evidence for
a general online selector or conditional coverage guarantee.  Failure means
the V6.5 heterogeneity does not transport under root-disjoint replication;
success still requires an independent panel before any controller claim.

## V6.8 historical preregistration: interaction-screened candidate-bank replication

V6.7 is closed. It provided the first held-out mechanism evidence that aligned
visual history can reduce action-ranking regret when the candidate bank is
constructed to require a state-action interaction. V6.8 is declared before
building its context or materializing any V6.8 outcome label.

The following items are frozen:

- candidate construction: `state_interaction_pairs`; 64 design roots, 96
  equal-energy pulse-order proposals, low mean terminal-effect eligibility,
  then high cross-root terminal-effect variation; no goal or task-cost access;
- action regime: `shared_candidate_in_support`; all 16 held action hashes must
  occur in the 64 source-training candidates;
- model, optimizer, 100 epochs, FiLM history/candidate architecture,
  selection/authorization split, outcome sealing, and authorization rule;
- protocol compatibility: V6.8 used the historical v2 matched-panel schema,
  v4 ordinary selection schema (v2 for the observability selection), and v3
  frozen-model schema. It is historical evidence only; current runners reject
  v1/v2 panels and require the v3 goal-bound panel schema plus a builder
  manifest. It requires per-label candidate provenance, an explicit support
  regime, an explicit selected policy, and a matching runner-source hash at
  score time; prior schemas remain historical artifacts and are not replay
  inputs;
- primary estimand: paired root-level regret gain of aligned visual recurrent
  over action-only on 32 new outcome roots;
- secondary estimands: aligned visual recurrent over cyclic history mismatch,
  and diagnostic privileged-state effect over action-only;
- all V1--V6.7 roots and candidate hashes are excluded from V6.8. Design roots
  must be separate from its train, calibration, and outcome roots.

V6.8 uses root/action seeds `202610281`, `202610282`, and `202610283`. The
outcome labels remain unavailable until the selected model, authorization
result, candidate/action identities, and exact model hash are persisted. The
aggregation tool accepts only V6.7/V6.8 interaction-screened reports,
verifies outcome-root and held-action-hash disjointness, then bootstraps pooled
root-level paired gains. The V6.7 screening objective and V6 runner settings
may not change after its outcome was inspected.

## V6.8 result and resulting boundary

V6.8 was built and scored exactly under the predeclared v2 contract. Its
authorization gate did not permit recurrent deployment and its independent
primary held-out gain over action-only crossed zero. The released policy stays
action-only. Because V6.7 and V6.8 together retain a positive but
non-significant pooled action-only gain, no additional panel may be selected,
stopped, or redesigned based on this pooled number without a new registered
replication protocol. The supported conclusion is limited to the aligned
history-versus-mismatch mechanism contrast, not controller utility.

## Goal geometry correction for new panels

The historical V6.1--V6.8 artifacts used the legacy target `[1.2, 1.2]`.

## Current V6.9 protocol gate

The current confirmation path is the v3 goal-bound panel plus
`temporal_ood_v6_manifest.json`. A formal panel must use the free-region goal
`[0.8, 0.8]` unless legacy reproduction is explicitly requested, and must
record the goal in metadata and every source/context row. The builder refuses
to overwrite a non-empty output directory and excludes both prior evaluation
roots and metadata-only interaction-design roots.

Before fitting, calibration and scoring now verify the manifest paths and
hashes, exact root counts (64/48/32), exact candidate counts (64 source and
16 held per root), candidate generator identities, root-to-instance
one-to-one mapping, causal history ledger, support regime, and goal
provenance. Scoring validates the frozen policy, calibration partition,
authorization contract, and model provenance before it creates any outcome
label file. The CLI defaults to CUDA and does not expose a CPU formal-run
fallback.

V1--V6.8 artifacts are therefore historical/debug evidence and are expected
to fail closed under the current runner. A V6.9 result is not confirmatory
until a fresh panel is built and the complete GPU
`build -> calibrate -> score` chain finishes under this gate.
That target is wall-sensitive in the legacy PointMaze XML and is retained only
for exact historical reproduction. New V6 builds now default to the free
region target `[0.8, 0.8]`; requesting `[1.2, 1.2]` requires the explicit
`--allow-wall-sensitive-goal` acknowledgement. The selected goal is stored in
every context row and build report, so planner/evaluator goal drift becomes a
protocol error rather than an unreported source of negative results.

An isolated, non-confirmatory replay on the already-opened V6.8 panel changed
only this goal to `[0.8, 0.8]`. Recurrent regret changed from `0.01189` to
`0.00832`, while action-only changed from `0.01404` to `0.01383`; the paired
outcome gain was `+0.00551`, but the 32-root bootstrap interval was
`[-0.00420, 0.01887]`. Thus the legacy target amplified the negative result,
but goal geometry alone does not explain it or establish a stable gain. The
replay is recorded at
`results/pointmaze_temporal_ood_v6_8_goal_sensitivity_20261021_cuda_retry/`
and is explicitly non-confirmatory because its outcome roots were already
opened.

## V6.9 result: protocol repair does not rescue the interaction gain

The fresh V6.9 panel completed the full CUDA chain on an RTX 5060 Ti after
two launcher defects were repaired. The launcher now invokes all Python
stages as repository modules and applies the bundled-cuDNN wrapper to build,
calibration, and score; direct script invocation and mixed cuDNN environments
are rejected as runtime errors. The panel contains 64 train, 48 calibration,
and 32 outcome roots, uses the free-region goal `[0.8, 0.8]`, has zero prior
evaluation/action identity overlap, and keeps outcome labels unmaterialized
until the frozen selection artifact exists.

The independent authorization half rejected the selected recurrent policy:
its gain over action-only was `-0.00168`, with 95% paired-bootstrap CI
`[-0.00780, +0.00276]`. On held-out outcome roots, recurrent regret was
`0.01713` versus `0.01374` for action-only, so the active policy correctly
fell back to action-only. This panel therefore does not support a stable
temporal-memory deployment gain, but it is valid evidence: the earlier V5
action-support, legacy-goal, outcome-sealing, root-split, and launcher/cuDNN
confounds are not present here.

The updated boundary is that state-interaction screening can expose a
history-dependent signal in some panels, but the signal does not yet
transport to an authorized policy under a fresh matched-support replication.
The V6.9 artifact is stored under
`results/pointmaze_temporal_ood_v6_9_protocol_repair_20261021_cuda/` and must
not be used to tune another selector or claim live RGB planning.
