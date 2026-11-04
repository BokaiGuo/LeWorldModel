# Native feedback selected-chain endpoint audit

The eight selected action chains in
`a0_pointmaze_native_feedback_time_20261102_DEV01.json` were reconstructed
from the sealed eight-action bank and each saved first/tail index. No
candidate was reselected. Two independent replays were run from the same
recorded reset:

1. Official visual XML with the legacy `mujoco-py` renderer and the
   development-selected camera, capturing post-step RGB and final state.
2. The existing audited MuJoCo-2.2 position-label backend and its
   position-oriented XML.

For all eight fixed chains, the legacy replay reproduced both the saved
step-5 and step-10 RGB hashes. Its final four-component state and the
position backend's final state were bitwise identical, and the latter
also exactly reproduced the saved offline endpoint. This closes the
same-action **replay** discrepancy for this specific fixed panel. The
original live bridge did not expose or record its final physical state, so
the audit does not directly compare a live-captured state against either
replay. Pixel hashes and deterministic same-contract replay strengthen the
link but do not prove internal-state identity of the original execution.

The first audit attempt failed the exact saved-endpoint check. The cause was
identified in the input interface: the original `PointMazeDataset` casts
the float64 archive states to float32 before using them for reset and goal,
whereas the initial audit used the raw float64 archive. Replaying with the
same float32 cast reproduced the saved position backend. No sealed original
outcome was modified. Future replays must bind the actual reset and goal
dtype, not just the archive file hash.

No tolerance was preregistered. The observed zero difference concerns only
these selected trajectories, this machine, these versions and these XMLs;
it is not a general backend-equivalence certificate. The roots are exposed,
normalization provenance remains unknown, and the two planning arms used
unequal computational resources. No success claim, independent
generalization claim or P1 admission follows.

- Audit: `le-wm-main/experiments/a0_pointmaze_feedback_endpoint_audit_20261102.py`
- Result: `le-wm-main/results/a0_pointmaze_feedback_endpoint_audit_20261102_DEV01.json`
- Result SHA-256:
  `31b77198d7064eb51aa915a15cac9e62554c7de52f4e88840df797c78136c7f1`
- Seven targeted tests passed across the audit, saved planning decisions and
  live RGB bridge. These tests do not constitute independent replication.
