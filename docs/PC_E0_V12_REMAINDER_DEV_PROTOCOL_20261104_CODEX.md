# E0 v12 remainder-batch repair: development protocol

v11 only refines in complete batches of 12, leaving up to 11 full-chain
evaluations unused even where the packet-token budget permits them. v12
preserves the existing full-batch draws and final-pair reserve, and spends
any remaining whole-chain budget on one last partial batch. The incumbent is
tracked independently; the partial batch contains new proposals rather than
duplicating it. This repairs unused search opportunity, not a verified root
cause of previous failures.

On each task compare v11 and v12 with identical response configuration, seed,
1920-token cap, scoring, and abstention. Also run the three-start CEM arm from
the preceding amplitude development protocol as a context baseline with the
same cap. Save every actual float32 input, output, score, final choice and
resource counters. For each v11/v12 pair verify every predictor batch before
the v11 final comparison is identical, the new extra batch fits in the
remaining cap, and the final chosen score does not worsen on this deterministic
backend. The actual compiled chains can differ; no mechanistic or RGB claim
follows.

Eight new development tasks are the Cartesian product of start
`[-0.5, y, theta]`, `y in {-0.23, 0.24}`, `theta in {-0.7, 1.05}`, and goal
`[0.5, gy]`, `gy in {-0.2, 0.16}`. Reject exact matches with the cited v5,
v10, waypoint, response-waypoint and small-amplitude panels. Broader historical
independence remains unknown. Seeds and the 0.05 success threshold are fixed in
the runner. Run every arm once per root and retain failures in the denominator.

After these 24 attempts, stop this panel regardless of outcome. Recompute
stored trajectories using both the exact backend and independent line-segment
collision implementation, and check action/score, budget, candidate incumbent
and source hashes. All outcomes are development evidence only; equal packet
counts are not equal total compute, and a successful synthetic test is not a
PC2 or PC3 gate.
