# Joint carry same-action RGB/physical outcome decomposition

Predeclared post-outcome development analysis of all 51 already executed
trajectories in the sealed all-roots carry receipt: 17 source roles, each
with old, zero-init and carry-init tails. Outcomes and roots are exposed;
there is no new planner, method selection, independent inference or gate.

For each fixed saved trajectory take its saved RGB frames at commands 16
and 17, estimate the cp17 state using the same causal RGB estimator and
0.05-second interval, encode RGB/state with the same frozen checkpoint, and
evaluate the same goal-projected proprio task cost used by the source
planner. The goal is constructed from the original RGB goal and calibration.
Use the saved model score of the actual selected tail (prior if not refined,
otherwise refined), the new actual RGB-derived score and the source
evaluation-only cp17 physical distance. Compare all three fixed tail pairs
within each role, excluding exact ties and reporting per-root and aggregate
counts; also specifically compare old versus carry on the four roles where
carry physically worsened. Do not use simulator state or physical distance
to construct the RGB score, change decisions or choose any candidate.

Bind result/source NPZ, checkpoint, RGB goal, geometry metadata, model state
hashes, normalizer and code; verify all 51 source histories and action shapes.
Record one goal encode, 51 actual observations, model parameter hashes,
all three scoring columns, and pairwise ranking details. Encoding costs are
additional; there are no physical steps or fresh model predictions. The
actual score uses RGB-estimated state and an affine proprio encoder, not the
whole visual embedding; this is conditional metric diagnosis, not a pixel
or independent outcome certificate.
