# PointMaze negative-result root-cause audit (2026-10-14)

## Outcome

The PointMaze learned-transfer failure is not explained by one selector bug.
The corrected audit identifies three separate factors:

1. the original external intervention was out of action support and was not
   aligned with the predictor's four-block horizon clock;
2. observed-trajectory prediction does not identify counterfactual action
   effects under the available action excitation;
3. the RGB source representation, when evaluated from a single source frame,
   does not reliably expose the hidden velocity needed for short-horizon
   effect prediction.

The first factor was fixed in code. The latter two remain a genuine learned-
representation limitation under the current frozen protocol.

## Protocol repair

The v1 external evaluator populated only the first of four 5-step action
blocks and simulated only five simulator steps. The training adapter consumed
all four blocks and predicted horizons `{1, 2, 4}`. The corrected v2 evaluator
therefore:

- repeats each diagonal `+a/-a/zero` intervention over all four blocks;
- avoids an artificial exact-zero action coordinate with two diagonal action
  directions;
- records physical effects after 5, 10, and 20 simulator steps;
- evaluates the predictor at the same block horizons.

The old v1 artifacts are retained. The v2 artifacts are:

- `results/pointmaze_external_evidence_panel_20261014_v2/`;
- `results/pointmaze_external_records_20261014_v2/`;
- `results/pointmaze_external_strict_benchmark_20261014_v2/`.

## Evidence

| condition | external action-inverse risk | interpretation |
|---|---:|---|
| v1 protocol, 50 checkpoints | mean `1.150` | optimistic protocol; action-support/horizon mismatch |
| v2 protocol, 50 checkpoints | mean `1.897` | corrected independent intervention |
| strict v2 matched benchmark | `2/5` identifiable stages; `Delta R=0.0046`, 95% CI `[0, 0.0092]` | positive point estimate but gate fails; one environment only |
| intervention adapter, frozen latent | `1.496` | direct counterfactual supervision does not remove the representation bottleneck |
| intervention adapter + latent history `z_t-z_(t-5)` | `1.282` | temporal history provides partial information |
| same intervention adapter + privileged simulator state | `0.257` | upper bound isolates hidden-state/representation observability as the dominant bottleneck |

The direct intervention adapter results are in:

- `results/pointmaze_intervention_effect_adapter_20261014.json`;
- `results/pointmaze_intervention_effect_adapter_history_20261014.json`;
- `results/pointmaze_intervention_effect_adapter_privileged_upper_bound_20261014.json`.

All intervention labels use the legacy PointMaze XML replayed with modern
MuJoCo offline. Simulator state is not used as a runtime model input.

## Scientific conclusion

The supported conclusion is now:

> Lower observed-transition prediction error does not identify
> action-conditioned task transfer. A corrected action-balanced intervention
> exposes substantial transfer error; short latent history reduces it only
> partially, while privileged state removes most of the error. Under the
> current frozen RGB source protocol, geometry cannot be certified as an
> independent transfer-risk selector.

This is a representation/observability boundary, not permission to tune the
selector or relax the matching tolerances. The next valid method change is a
temporal-observation or intervention-supervised workstream with independently
held-out roots. It must not be merged into the original matched-quality
geometry estimand.

## Verification status

- full repository regression before this audit: `616 passed, 3 skipped`;
- focused external-protocol regression after the repair: `11 passed`;
- v2 external panel: `50/50` reports, `0` failures;
- v2 strict record build: `50/50` records, `0` failures;
- v2 strict benchmark: completed and fail-closed as reported above.

## Decision-bank follow-up (same-day protocol repair)

The first hard candidate bank exposed an additional benchmark defect: its
eight outcome roots contained zero positive task labels.  The bank therefore
could not support a success-rate or success-selection claim.  This was caused
by the combination of distant/high-velocity source roots and a legacy
round-robin split that placed the few reachable roots in train/calibration.

The repaired bank is:

`le-wm-main/results/decision_benchmark_candidate_bank_20261014/jepa_readout_repaired_task_effect_mlp_seed8411_stage005_near_goal_v3_balanced_source/`

It uses a pre-outcome nearest-goal source rule and
`distance_balanced_v1` root assignment.  It contains 40 roots, 16 candidates
per root, 20 simulator steps, and 24/8/8 train/calibration/outcome roots.  The
outcome split contains 3 successful candidates across 1 root.  D0 schema and
provenance pass; D1 remains blocked because realized RGB latents are absent.

On this repaired bank, intervention-supervised selection gives:

| input context | effect RMSE | selected regret | random regret | paired regret gain |
|---|---:|---:|---:|---:|
| frozen source latent + action | `0.404` | `0.083` | `0.136` | `+0.053` |
| privileged source state + action | `0.107` | `0.019` | `0.136` | `+0.117` |
| latent + privileged source state + action | `0.105` | `0.012` | `0.136` | `+0.124` |

These are intervention-supervised diagnostics, not evidence that frozen RGB
planning is solved.  They isolate a residual observability gap: the frozen
latent can provide useful ranking on a support-corrected bank, but it is still
far from the simulator-state upper bound.

Finally, a causal four-window latent history was evaluated at source window
15 using only windows 0, 5, 10, and 15.  On its independently held-out
outcome roots, frozen single-frame context achieved regret `0.073` versus
random `0.191` (paired gain `+0.118`), while causal history achieved regret
`0.064` (paired gain `+0.127`) and success gain `+0.15`.  This is a promising
mechanism result, but it is one environment/checkpoint panel and still lacks
realized RGB successor verification; it cannot yet support a universal or
closed-loop claim.

The D0 validator now warns whenever an outcome split has zero positive labels,
and the adapter report records paired regret/success gains plus whether its
input is frozen-latent, causal-history, or privileged-state.  This prevents
the original support failure from being silently interpreted as a model
failure in later runs.

## D1 RGB successor verification (temporal-interface repair)

The first RGB verifier attempt exposed a protocol failure rather than a
learned-model result.  It encoded each source or terminal image as an
independent one-frame input.  JEPA-WM's visual encoder uses temporal tubelets,
so this changed the representation interface: the released source latent is
the `source_frame` entry of a temporal context sequence, not the encoding of
an isolated RGB frame.  The verifier also initially used frame zero instead
of the declared `source_frame=5` and therefore produced a false source-latent
discrepancy.

`experiments/decision_benchmark/verify_rgb_successor.py` now uses the exact
temporal contract:

```text
observed frame 0 -> observed source frame 5
                 -> realized candidate frame 10
                 -> realized candidate frame 15
                 -> realized candidate frame 20
                 -> realized candidate frame 25
```

The realized candidate frames are obtained after each five-step action block;
the frozen encoder reads the terminal latent at temporal index `1+h` (with
`h=4` for the current bank).  The source preprocessing is also reproduced
exactly from the original export, including its declared PointMazeDataset /
EncPredWM transform path.

The full artifact is:

`results/decision_benchmark_candidate_bank_20261014/rgb_successor_v2_temporal/`

It contains 40 root-disjoint roots and 16 candidates per root.  All 40
renderer-consistency gates and all 40 source-latent exact-replay checks pass.
In the 640 candidate pairs, action-conditioned prediction versus realized RGB
successor has PCA-space L2 RMSE `0.1631`, versus `0.2011` for the frozen
source-latent baseline: relative improvement `17.8%`, with `69.5%` of pairs
improved.  On the held-out outcome roots, the improvement remains `13.9%`
(`0.1785` versus `0.2121` RMSE; `64.8%` of pairs improved).

This repairs the previous D1 negative result caused by a mismatched temporal
protocol and establishes realized RGB successor-latent availability.  It does
not by itself establish closed-loop online planning, physical causality, or a
universal world model: those still require a policy that selects actions from
the verified RGB interface and an independent held-out planning consequence.

## Causal-history repair and remaining boundary

The first history candidate bank had a real implementation defect: `history_windows=4`
was stored in the records, but the predictor still received only the current
8-dimensional source projection. The panel and bank interfaces now carry a
causal context explicitly, record `history_windows` in the checkpoint, and
select the corresponding context at candidate-generation time. Focused
protocol tests pass, and the repaired history panel contains 10 checkpoints
(linear/MLP, five seeds) with episode-disjoint train/calibration/outcome
support `54/14/32`.

The paired artifacts are:

- `results/pointmaze_history_panel_20261014/`;
- `results/decision_benchmark_candidate_bank_20261014/jepa_history4_conditioned_source/`;
- `results/decision_benchmark_candidate_bank_20261014/paired_history_conditioning_outcome8.json`;
- `results/decision_benchmark_candidate_bank_20261014/paired_history_effect_adapter.json`.

The causal history did not improve the realized-RGB successor predictor on
the eight predeclared outcome roots: history-minus-single-frame RMSE was
`+0.00834` (root bootstrap 95% CI `[+0.00175,+0.01505]`; exact sign-flip
`p=0.0547`). This is not evidence that history is useless. On the same
history-conditioned candidate bank, a task-effect adapter using the four
causal latent windows reduced held-out action-selection regret from `0.1897`
to `0.0798`; the seed-bootstrap CI for history-minus-single regret was
`[-0.1224,-0.0973]`. Selected success was unchanged in this small panel.

The resulting diagnosis is narrower and actionable:

> Direct history concatenation helps the intervention-supervised task-effect
> readout, but does not transfer reliably to RGB successor prediction under
> the current small-sample/counterfactual action-support shift.

Accordingly, history is now permitted as a task-effect/context feature only
after an independent action-effect gate. It is not promoted to a universal
latent-dynamics improvement. The verifier also supports `--root-split` so
that calibration and outcome roots cannot be confused by root ordering.

## Candidate-conditioned successor collapse

The next audit found a failure hidden by aggregate successor RMSE. The
task-effect checkpoint projects candidate successors through one task-readout
direction. Across the 40-root RGB candidate bank, predicted candidate spreads
were only about `0.03--0.08` of realized RGB spreads and pairwise variation
correlation was generally near zero. Different action candidates were thus
mapped to nearly the same predicted successor.

The old support gate also accepted every root because it checked only
coordinatewise min/max ranges. The implementation now fits a train-only
standardized joint nearest-neighbour radius over complete action sequences;
coordinate bounds remain for diagnostics and cannot release the bank alone.
The task-effect checkpoint still fails the new collapse gate on all outcome
roots, so the stricter gate exposes the bottleneck rather than hiding it.

Matched interventions separate the failure modes:

- ordinary MLP stage 005: physical inversion `0.239`, outcome spread ratio
  about `0.175`, collapse gate `0/8`;
- ordinary MLP stage 075: outcome spread ratio about `0.889` and `5/8` collapse
  gates pass, but RGB successor error is `6.2%` worse than the source baseline;
- standardized MLP stage 005: physical inversion `0.217`, but outcome spread
  ratio about `0.054` and collapse gate `0/8`.

Thus candidate variation preservation and candidate prediction correctness are
distinct conditions. Increasing late-stage variation is not a repair if it
increases successor error. A train-root-only five-seed RGB candidate-successor
adapter was also implemented as an intervention-supervised control; it reached
held-out projection RMSE `0.2065` but physical inversion `0.442`. RGB
counterfactual supervision alone did not recover decision ranking.

The operational repair is fail-closed: a candidate bank must pass joint action
support, candidate variation preservation, calibrated task contrast, and an
independent held-out physical consequence before release. No current
checkpoint passes this conjunction. The next legitimate method change is a
mode/velocity-aware action-effect representation or short-pulse re-observation;
changing selector weights or support thresholds would not address the measured
collapse.

## History-4 temporal contract repair and remaining transfer failure

The verifier had a second protocol defect: it used only two RGB context frames
for a bank whose predictor was trained with four causal latent windows. This
has been repaired in
le-wm-main/experiments/decision_benchmark/verify_rgb_successor.py and
generate_candidate_bank.py. The repaired contract exposes
source_frame-15,-10,-5,source_frame for history-4 and reads the realized
terminal at context_length + predicted_horizon - 1; tests cover both legacy
and history-4 contracts.

The full GPU rerun (RTX 5060 Ti, 40 roots, 640 candidates) passes all
renderer/source gates. Outcome-root RGB successor RMSE is 0.14927 versus
0.19879 for the source baseline. A train-root-only five-seed adapter was
then trained with the complete causal RGB history. Its outcome RMSE is
0.1810, compared with 0.1915 for the same-bank current-source control.
The adapter-level RGB candidate inversion improves to 0.4473 from 0.4656,
and collapse-gate coverage improves to 32/40 from 27/40.

The repair does not solve the decision problem: physical inversion becomes
0.5178 for history-4 versus 0.4801 for the single-source control, and the
calibrated release coverage remains 0/8. Therefore the correct conclusion is
not that history-4 is a stable planning solution. It is that temporal
interface consistency removes one confound and yields a real visual
successor gain, while the remaining failure is an RGB/task/action-effect
geometry mismatch. Any next method must improve within-mode action
identifiability or use a short-pulse re-observation; relaxing the collapse or
contrast gate would be invalid.
