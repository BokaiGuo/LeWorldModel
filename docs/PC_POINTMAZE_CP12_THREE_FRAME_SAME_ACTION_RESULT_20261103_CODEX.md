# CP12 three-frame same-action development result

The prewritten `PC_POINTMAZE_CP12_THREE_FRAME_SAME_ACTION_PROTOCOL_20261103_CODEX.md`
ran on an NVIDIA RTX 5060 Ti. It uses four exposed roots, eight already
executed branches per root, and no new physical steps or selection. The
single and three-frame arms use the exact same actions[12:17] and predict
RGB/proprio latent at time 17. Three-frame conditioning additionally uses
observations at 2 and 7 and their outgoing executed packets. Inputs at each
observation are RGB-derived causal estimates, not simulator qvel.

| Root | Visual MSE one | Visual MSE three | Three lower, visual | Proprio MSE one | Proprio MSE three | Three lower, proprio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8343 | 0.024112 | 0.023533 | 6/8 | 0.00008380 | 0.00012028 | 5/8 |
| 8345 | 0.019740 | 0.016063 | 7/8 | 0.00012903 | 0.00008376 | 7/8 |
| 8349 | 0.036529 | 0.032758 | 7/8 | 0.00036373 | 0.00026150 | 6/8 |
| 8351 | 0.016130 | 0.016512 | 3/8 | 0.00015545 | 0.00017352 | 4/8 |
| All branches | 0.024128 | 0.022216 | 23/32 | 0.00018300 | 0.00015977 | 22/32 |

Means weight branches equally, not independent tasks. These selected,
dependent, already exposed trajectories do not establish generalization,
better ranking, physical success, or better control. More history is not
free: it requires two additional observation encodes and longer predictor
context, even though each arm performs one predictor call. The exact
historical training provenance and RGB render timing remain unresolved.
Real cp17 RGB is exclusively an offline evaluation target. No privileged
simulator state was fed to either prediction.

The one-frame visual and proprio predictions match the prior saved cp12
refresh tensors exactly (maximum absolute error 0.0). Parameter hashes match
before and after. This is a reference for the single arm, not an independent
real-model implementation of the three-frame arm. The latter's action order
is exercised with a synthetic predictor; checkpoint conditioning on three
frames remains an interface hypothesis.

- Run: `le-wm-main/results/pc_pointmaze_cp12_three_frame_same_action_20261103_CODEX.json`,
  SHA-256 `e27017d25a31b7d35dc50d0ea9062d10c3541dbdfe4a12f3fc073b0a2af0aad7`.
- Array SHA-256 `2a951b25d5cbcb7180962ae63052a37fb296315717a0cc54ebcff8e6875d453e`.
- Focused tests: **6 passed**, including a saved-output recomputation of MSE,
  original action binding, normalizer packing, roster and timing. These tests
  do not rerun CUDA inference. The run records important direct input and
  implementation hashes, but does not certify the full transitive dependency
  closure or the historical checkpoint training provenance.
- Initial sandbox run could not access CUDA; the successful run used the
  authorized host GPU. No existing experiment or result was modified.
