# Fixed-candidate real-RGB feedback diagnostic

All 384 fixed initial CEM candidates were evaluated on the same two-packet
actions. The source's original two-packet CUDA predictions were recomputed:
the largest score difference was `5.960464477539062e-7`, and each of the
eight root/arm selections matched. All regenerated initial RGB images and
matching selected-prefix images passed hash/equality checks. This narrows
the possibility of a saved-score, reset, or selected-prefix binding bug;
it does not authenticate the checkpoint's original training interface.

After executing each first five-step packet in the native bridge, the model
predicted the second packet from actual cp5 RGB. Both predictions and actual
cp10 RGB were scored using the same original task mask and goal.

| Quantity | Open loop | Real cp5 refresh |
| --- | ---: | ---: |
| Mean task-mask feature MSE to actual cp10 | 0.65385 | 0.66921 |
| Goal-cost ranking reversals against actual RGB | 4,960 | 4,754 |
| Correlated within-root/arm pairs | 9,024 | 9,024 |

Refresh reduced feature MSE for 166 of 384 candidates and reduced mean MSE
for both arms of root 1884, but increased mean MSE for both arms of each of
the other three roots. It reduced ranking reversals on five of eight
root/arm groups, increased them on three. Candidate outcomes and pairs are
correlated and all roots are exposed development data. These observations do
not establish a general negative effect of feedback, nor show that refreshing
improves physical control. Open-loop and refreshed predictions have different
prediction horizons and proprio histories; this comparison does not identify
which difference causes the errors.

The run used the existing single-frame, fixed raw-zero proprio interface and
the same full-archive action normalizer as its source; checkpoint training
membership and original normalizer provenance remain unknown. No physical
outcome was fed into candidate selection. No extra model was trained. There
is no independent Planning Compiler P1 confirmation.

- Protocol:
  `docs/PC_POINTMAZE_CEM_FEEDBACK_SAME_ACTION_PROTOCOL_20261102.md`
- Runner: `le-wm-main/experiments/a0_pointmaze_cem_feedback_same_action_20261102.py`
- Result: `le-wm-main/results/a0_pointmaze_cem_feedback_same_action_20261102_DEV01.json`
  SHA-256 `3683960716f7d954e28e6d76f46ec630c91c673fde7e3072b3d59e3150584bc2`
- Arrays SHA-256:
  `091e4528c45516cb2ce8b6354566a93f5e06c6946ee26248ac97bd8ed5d9e569`
- Validation: both candidate replay and feedback diagnostic test files,
  six tests passed; they check arithmetic, arrays, action packing and masks,
  not independent reproduction of the complete model or simulator.
