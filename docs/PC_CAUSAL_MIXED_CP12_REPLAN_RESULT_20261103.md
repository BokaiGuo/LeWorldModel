# Real-RGB cp12 replanning with matched controls

All 32 mixed-prefix branches and the eight zero/greedy control branches
replayed their saved first 12 physical actions with exact RGB-frame agreement.
Each then reoptimized its original last five actions from real cp12 RGB using
the same 50-sample CEM and root-specific seed. Every branch selected the
refined tail and improved its physical cp17 position distance relative to
its own old tail; the 32 mixed branches had a mean change of −0.1932.

| Root | Mixed old mean | Mixed new mean | Zero new | Greedy new | Outcome-exposed mixed branches better than both |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8343 | 0.9737 | 0.8303 | 0.8569 | 0.0553 | none |
| 8345 | 1.7442 | 1.5842 | 1.5044 | 0.6612 | none |
| 8349 | 0.8014 | 0.6112 | 0.5125 | 0.7076 | raw 4: 0.2660; spectral 3: 0.4619 |
| 8351 | 1.8408 | 1.5614 | 1.5780 | 0.6533 | none |

The root-8349 winners were identified after observing physical outcomes and
had already improved position during their five-step prefixes. No
physical-nonprogress mixed prefix beat both equally refreshed controls on
these roots. The original model-selected raw and spectral winner in each
root was the shared greedy prefix; these oracle candidates must not be
reported as selected or as independent results.

The additional cp12 opportunity and computation were not available to the
old tail, so the within-branch improvement cannot distinguish real
observation, shorter planning horizon, and extra optimization. Matching
zero/greedy at cp12 removes that particular feedback asymmetry in the table,
but prefix generation and prior search costs remain unequal. No common-suffix
mechanism, native success, independent roots or full strong-baseline budget
certificate is established.

The 32-branch run added 65 encodes, 1,600 model samples, 544 RGB-backend
steps and 544 separate position-backend steps. Matching controls added 17
encodes, 400 samples and 136 steps on each backend. These are in addition to
upstream search and earlier experiments. All 32 candidate and eight control
saved bindings, first-12-action/RGB histories, chosen-tail decoding and
physical distances passed independent saved-output checks. The historical
normalizer receipt is not complete training-provenance certification.
Six focused tests passed.

- Candidate protocol and report:
  `docs/PC_CAUSAL_MIXED_CP12_REPLAN_PROTOCOL_20261103.md`,
  `le-wm-main/results/pc_causal_mixed_cp12_replan_20261103_DEV01.json`.
- Control protocol and report:
  `docs/PC_CAUSAL_MIXED_CP12_CONTROLS_PROTOCOL_20261103.md`,
  `le-wm-main/results/pc_causal_mixed_cp12_controls_20261103_DEV01.json`.
- Saved audits:
  `le-wm-main/results/pc_causal_mixed_cp12_replan_saved_audit_20261103_DEV01.json`,
  `le-wm-main/results/pc_causal_mixed_cp12_controls_saved_audit_20261103_DEV01.json`.
