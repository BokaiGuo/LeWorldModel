# Same-action model / actual RGB / physical decomposition

Using the prewritten protocol, all 51 saved old, zero-init and carry-init
trajectories were re-encoded at cp17 with the frozen checkpoint and
RGB-derived causal state. Within each of 17 source roles, compare its three
tail pairs; no actions were chosen or executed in this analysis.

| Comparison | Agree | Reversed | Tie |
| --- | ---: | ---: | ---: |
| Model predicted score vs actual RGB-derived score | 46 | 5 | 0 |
| Model predicted score vs physical distance | 44 | 7 | 0 |
| Actual RGB-derived score vs physical distance | 49 | 2 | 0 |

The 51 pair records share roots and trajectories and include duplicate
roles; these are not independent observations. The physical column is the
source's separate position-backend replay, not a direct native executor
measurement. This result concerns the projected proprio metric constructed
from causal RGB state estimates, not visual-token MSE.

Among the four role records where the model preferred the carry tail but
the old tail had lower physical distance:

- root 8345 greedy and root 8349 raw-winner: the model's predicted ordering
  reversed relative to the actual RGB-derived scoring, while actual RGB
  agreed with physical ordering;
- root 8351 greedy and spectral-winner: the model agreed with actual RGB,
  but that ordering opposed physical distance. These two are duplicate
  action chains, so they are one observed pattern, not two independent
  cases.

Thus neither predictive distortion nor evaluation mismatch alone explains
every observed reversal. The outcome is exposed and diagnostic; it does not
establish prevalence on new roots or validate a selector.

The successful host CUDA run used one goal and 51 actual RGB encodes, no new
prediction rollout and no new physical commands; model parameter hashes
matched. The sandbox attempt failed before inference because CUDA was
unavailable and wrote no result. A CPU test recomputed all causal RGB states
from the saved frames, checked the analytic affine position-projection
identity to `1e-5`, and recomputed pair labels. The related tests report
**5 passed**. These checks do not independently rerun the visual encoder
or certify original training provenance, RGB timing, or physical replay.

- Protocol: `docs/PC_JOINT_CARRY_SAME_ACTION_OUTCOME_DECOMP_PROTOCOL_20261103.md`
- Runner: `le-wm-main/experiments/pc_joint_carry_outcome_decomp_20261103.py`
- Result: `le-wm-main/results/pc_joint_carry_outcome_decomp_20261103_DEV01.json`,
  SHA-256 `659466fccb29739051e398c6ee27d07aca581616b10f36c26307efedaddcd866`
