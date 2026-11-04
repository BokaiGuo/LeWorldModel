# Successor RGB model trace replay: audit result

Under `PC_SUCCESSOR_RGB_MODEL_TRACE_REPLAY_AUDIT_PROTOCOL_20261103_CODEX.md`,
the recorded RGB contexts and action histories were reconstructed and every
saved model-input batch was rerun with the frozen PointMaze checkpoint.

- 24 decisions, 1,220 prediction batches, 7,240 trajectory samples, and
  11,516 prediction tokens were replayed.
- The maximum absolute discrepancy was **0.0** for both recorded position
  outputs and model scores on this hardware; 0 batches exceeded `1e-4`.
- Recorded action normalization, model parameter hash, goal image, observation
  image receipts, decision receipts, valid action shapes, finite outputs and
  sample/token counts were checked.
- Original files were not rewritten. The new machine-readable audit is
  `le-wm-main/results/pc_successor_rgb_execution_dev_20261103_codex_model_trace_replay.json`.

This resolves the specific concern that the saved input batches and saved
model outputs disagree under the current reconstructed environment. It does
not validate model predictions against physical outcomes, make the four roots
independent, or pass PC1–PC3. Audit-time hashes of dependencies omitted from
the original receipt cannot establish what those files contained at the
original execution time. This gap must be closed prospectively for any new
confirmation version.
