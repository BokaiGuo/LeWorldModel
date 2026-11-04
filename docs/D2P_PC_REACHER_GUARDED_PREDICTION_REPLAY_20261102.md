# Guarded spectral candidates: same-action prediction and RGB replay

The sealed selected, zero and same-bank-greedy actions were replayed at two
development roots. For each of six action chains, the frozen model predicted
all three five-step token boundaries open-loop from the original observed
history; at each boundary a separate one-token prediction was also made from
the immediately preceding real RGB/action history. Actual RGB encodings and
evaluator-side qpos distance were obtained at steps 5, 10 and 15.

Raw action witnesses were decoded from their saved ten-dimensional normalized
tokens and matched within 1e-6; root and goal RGB hashes, first-successor RGB
hash, and first/final qpos distances matched the sealed physical run. The
rollout adapter's sequential action order and normalization binding have two
passing unit tests. Latent arrays, individual costs and hashes are saved in
the result; these checks do not certify training-data provenance.

At step 15, for selected versus zero (negative delta favors selected):

| Seed | Open predicted latent goal MSE delta | Actual RGB latent goal MSE delta | Actual qpos-distance delta |
|---|---:|---:|---:|
| 20263160 | -0.007743 | +0.004450 | +0.011860 |
| 20263161 | -0.004401 | -0.007953 | +0.021131 |

On the first root predicted and observed latent ordering disagree. On the
second they agree, but actual latent ordering disagrees with qpos-distance
ordering. These are distinct observed failure patterns, not proof that either
is the only cause of the outcome. They are two known roots and the three arms
share roots and planning history, so there is no independent-sample inference.

At step 15, refreshed one-token prediction had lower latent MSE to actual RGB
than open-loop prediction in all six arms. At step 10 this happened in four
of six; at step 5 both predictions are identical by construction. This
diagnostic does not show that refreshing improves control, because no new
actions were chosen using these diagnostic predictions.

The audit used three open and three refreshed prediction samples per arm,
plus RGB encodings and physical replay, whose full cost is not included.
There is no new task-success evaluation, common-suffix causal inference or
independent confirmation here.

Artifact:
`le-wm-main/results/reacher_pc_guarded_prediction_replay_20261102_v1/report.json`.
