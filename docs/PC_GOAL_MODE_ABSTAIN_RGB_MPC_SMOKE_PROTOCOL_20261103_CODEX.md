# Goal-conditioned RGB MPC with abstention: integration smoke

One previously used PointMaze development root, fixed frozen checkpoint and
RGB goal. Execute the existing real 15-action warmup. At each of three
decision times, encode the actual RGB history at 5-action spacing and the
last ten actually executed actions; compile bounded 15-action programs using
the existing goal-conditioned response candidate generator and the same
12-sample CEM suffix optimizer. Retain the zero-prefix program unless the
model's best complete-chain score is **strictly** smaller (`minimum_gain=0`).
This is abstention from nonzero prefix, not a zero-action or safety guarantee.

Seal a separate decision file before dispatching that decision's five-action
prefix. Observe fresh RGB, append the actual actions, and recompile. The
15-action planning horizon is receding even in the final decision, where
only five actions are executed; report this horizon mismatch, do not claim
the unexecuted suffix as realized. Per decision, cap model prediction at 132
five-action token samples and separately record encoding, predictions, SVD,
suffix optimization and elapsed time. The total task trajectory is 15 actions.
No physical state enters selection. There is no matched baseline or independent
evaluation here; this smoke does not establish scientific gate passage.

The frozen input history, camera calibration, goal image, checkpoint, seed,
source hashes and action and RGB logs must be recorded. Existing results and
historical contracts are not modified. Any non-immediate mechanism claims
remain outside this protocol.
