# PointMaze archive-start fixed-bank diagnostic, 2026-11-02

This is a development diagnostic, not P1 admission or an RGB-only planning
result. The previously evaluated 1900–1907 oracle starts after online probing;
its start state is not archive frame 0, and the archive RGB files for those
roots are absent. Pairing that oracle with frame-0 model inputs would therefore
be invalid.

The new diagnostic instead uses roots 1880–1883, recorded frame 0 as the start,
and recorded frame 10 as the visual goal. The physical evaluation goal is the
position in the same archive frame. The model evaluates all 8 one-packet and
64 two-packet plans per root; one packet repeats the candidate for five
archive actions. The eight fixed directions are the first magnitude group of
the previously saved root-1900 bank, chosen before running this diagnostic.
The model takes recorded RGB **and proprio**; in this dataset proprio is a copy
of simulator state. Consequently this is not an admissible RGB-only planner
and cannot support an RGB-only control claim. The physical simulator receives
state for initialization and evaluation only. No model was trained.

The visual latent goal and position-only physical goal are different metrics:
the comparison identifies a combined prediction/representation/metric
mismatch, not its unique cause. The actions are fixed off-line, the roots are
already known, and checkpoint training membership and normalization provenance
are not certified. The simulator backend is position-validated, not
full-state-equivalent.

| Root | Model-selected pair | Position-oracle pair | Physical position regret |
|---|---|---|---:|
| 1880 | 4, 4 | 4, 2 | 0.157374 |
| 1881 | 2, 3 | 3, 6 | 0.342323 |
| 1882 | 3, 6 | 3, 7 | 0.141739 |
| 1883 | 7, 4 | 1, 3 | 0.038427 |

On the recorded 10 archive actions, replayed terminal positions match the
archive frame-10 positions to at most 8.1e-7. This checks this start and action
replay, not all counterfactuals or full simulator state. Each model batch was
also checked against its first serial prediction; the saved checker recomputes
all 64 physical distances and both argmins from saved arrays. The validator
initially caught a float64/float32 loader discrepancy in its own source-state
check; it was fixed to match the dataset's explicit float32 conversion before
the above validation was produced. This was not a model or scientific failure.

Artifacts:

- `le-wm-main/experiments/a0_pointmaze_archive_bank_alignment_20261102.py`
- `le-wm-main/experiments/a0_pointmaze_archive_bank_validate_20261102.py`
- `le-wm-main/results/a0_pointmaze_archive_bank_alignment_20261102_DEV01.json`
  SHA-256 `93cdc7067ca623f1eb80439d1e41ff18b3be4468d8befdc3d9c6cecad26d3291`
- `le-wm-main/results/a0_pointmaze_archive_bank_validate_20261102_DEV01.json`
  SHA-256 `60f2bfb51cdcaa58373cf99a3bd1fd530e0e9e96d306134659f662cc6e932d14`
- Tests: `test_a0_pointmaze_archive_bank_alignment_20261102.py` plus the
  existing packet-oracle test: 3 passed.

Next, disentangle model prediction from task-metric mismatch using actual
counterfactual RGB observations at matching times and the same starting
context, while enforcing an RGB-only input contract. Do not equate this
position-only oracle regret with a real closed-loop P1 failure or success.
