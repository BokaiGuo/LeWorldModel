# Explicit action-history unroll: implementation and CUDA validation

The current PointMaze inference loader sets the model wrapper's context window
to one. The local wrapper's multi-frame path pads missing action features
with zeros; it does not accept the actual past action packets. Consequently,
that path cannot establish actual-history conditioning merely by passing
multiple images.

The checked local dataset sampler takes observations at `start + 5k`, while
concatenating five consecutive actions beginning at each corresponding
`start + 5k`. The local training path encodes those aligned sequences and
uses one-step shifted prediction loss. The selected configuration specifies
`num_hist=3`, `num_pred=1`, and a rollout training context window of three.
This supports the outgoing-action alignment used here for the inspected code
and configuration; actual checkpoint training data, config identity and
normalizer provenance remain unverified.

Added `experiments/pointmaze_action_history_unroll_20261103.py`: it requires
exactly one normalized historical action packet per observed transition,
then binds each candidate outgoing packet to its current observation. Its
sliding window retains observed actions as future predictions are appended.
Invalid shapes, absent history, nonfinite actions and unsupported proprio
mode fail closed. No external model code was changed.

Under `PC_POINTMAZE_ACTION_HISTORY_UNROLL_VALIDATION_PROTOCOL_20261103.md`,
host CUDA comparisons on all eight previously exposed prefix groups found:

- Two-frame, one-step visual predictions exactly matched the previously
  saved direct-history predictions: maximum absolute difference `0.0`.
- One-frame, two-step visual and proprio predictions exactly matched the
  existing one-frame wrapper: maximum absolute differences `0.0` for both.
- Parameter hashes before and after matched. The comparisons use the existing
  frozen checkpoint and normalizer; equality is conditional on this panel.
- `.venv/bin/pytest -q tests/test_pointmaze_action_history_unroll_20261103.py`:
  **2 passed**, covering multi-step sliding order, gradients and fail-closed
  input checks using a synthetic predictor.

CUDA receipt `le-wm-main/results/pointmaze_action_history_unroll_validation_20261103_DEV01.json`:
SHA-256 `1fc401e6521321550e6ea8a67626970de9337ba206f1d0d1db6f474d8d5f23f3`.

This validates a reusable interface against two existing inference paths,
not a physical planning improvement or the checkpoint's training contract.
The test sequences beyond the first tail were not physically executed.
Multi-step history behavior has synthetic order/gradient tests but no
independent real-model reference for window greater than one. The new
interface is not yet wired into the real feedback planner; fixed-bank
negative results remain unchanged and P1 remains unproven.
