# Matched imagined versus real-RGB cp12 replanning

All 40 saved physical prefixes were replayed through cp12 with exact RGB
agreement. Both arms then used the same original tail, root-specific random
seed, 50-sample one-packet CEM and task score. The real arm encoded actual
cp12 RGB/state; the imagined arm encoded cp7 RGB/state and predicted cp12
with the same five physical actions.

| Root | Imagined-context mean distance | Real-RGB-context mean distance | Real better / imagined better / tie | Identical final tails |
| ---: | ---: | ---: | ---: | ---: |
| 8343 | 0.7652 | 0.7554 | 5 / 3 / 2 | 0/10 |
| 8345 | 1.4869 | 1.4839 | 1 / 0 / 9 | 4/10 |
| 8349 | 0.6124 | 0.6110 | 6 / 4 / 0 | 0/10 |
| 8351 | 1.4708 | 1.4723 | 2 / 2 / 6 | 4/10 |

The table averages ten branches per root, including zero and greedy.
Distance ties use absolute tolerance `1e-8`; identical tails require exact
action-array equality. Across all branches, original-tail, imagined-context,
and real-context means were 1.2643, 1.0838 and 1.0806. The real context was
better on 14 branches, imagined on nine, and 17 were distance ties.

Thus the earlier large reduction against the old tail cannot be attributed
solely to the additional real RGB observation: similarly budgeted CEM from
an imagined context also improved these exposed branches, and the difference
between the two contexts was small and mixed. This comparison does not prove
the observation is useless in general. The branches share four previously
exposed roots, and the state estimates, task score, model and optimizer are
specific to this development setup.

Both arms used 50 optimization samples per branch; the imagined arm also
spent one cp7-to-cp12 model prediction, whereas the real arm consumed the
actual cp12 observation. Their total compute is therefore not identical.
The imagined run added 41 encodes, 40 context-prediction samples, 2,000 CEM
samples, 680 physical RGB-backend steps and 680 position-backend replay
steps, plus all prior work. Raw physical state remained evaluation-only.

All 40 saved action, RGB prefix, tail-decode, physical-distance and comparison
bindings passed audit. The audit did not re-run neural inference or certify
training provenance. The first attempt failed before physical execution due
to missing TensorDict batch metadata; the tested interface fix and cost are
documented in `PC_CAUSAL_MIXED_CP12_IMAGINED_FIRST_ATTEMPT_20261103.md`.
No common-suffix mechanism, independent root, strong-baseline claim, native
success or scientific gate is established.

- Protocol: `docs/PC_CAUSAL_MIXED_CP12_IMAGINED_PROTOCOL_20261103.md`.
- Runner: `le-wm-main/experiments/pc_causal_mixed_cp12_imagined_20261103.py`.
- Report and trajectories:
  `le-wm-main/results/pc_causal_mixed_cp12_imagined_20261103_DEV01.json`
  and same-name NPZ.
- Saved audit:
  `le-wm-main/results/pc_causal_mixed_cp12_imagined_saved_audit_20261103_DEV01.json`.
