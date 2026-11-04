# Dev304: old versus replanned tail at the same real cp10 context

The four old and replanned five-action tails were already physically
executed from their own exactly replayed cp10 states. This post-outcome
diagnostic uses the frozen CUDA model to score both tails from the same
three real RGB observations and actual two-packet action history. The
score is predicted goal-latent MSE after the one packet that was actually
executed. It is not the original replanner's two-packet terminal objective,
and it was not used to choose a tail. Each arm has one batched predictor
call on the two tails; the model weights remained unchanged.

| Context | Predicted old / new MSE | Native old / new reward | Position old / new | Angle old / new |
| --- | --- | --- | --- | --- |
| goal original | 1.4364 / 1.3589 | -299.061 / -301.061 | 204.425 / 204.623 | 2.7667 / 2.5038 |
| goal coherent | 1.5464 / 1.6181 | -299.909 / -340.990 | 203.127 / 171.486 | 3.0124 / 2.5836 |
| Q goal original | 1.6083 / 1.4017 | -315.422 / -272.557 | 210.565 / 202.244 | 2.8070 / 2.2716 |
| Q goal coherent | 1.6568 / 1.6480 | -324.170 / -297.600 | 222.198 / 187.523 | 3.0058 / 2.6681 |

The one-packet model score prefers the new tail in three arms; native reward
prefers it in two of those. In the remaining arm both the one-packet score
and reward prefer the old tail, although the new tail has lower position
and angle errors. Metrics cannot be collapsed into a single success claim.
All four pairs share one development root and all branches failed native
success. This result neither establishes that the original two-packet
selector would retain the old tail nor proves P1 or independent performance.
The re-scoring cost is additional and is not included in previous budgets.

Artifacts relative to `le-wm-main/results/`:

- Contract SHA-256 `73c4e4bd24164f9ecb1e287d325a5e95e09c37dd7ea3c0ad763f3a91a713e87d`.
- CUDA report `p1_dev304_cp10_tail_pair_cuda_20261102_RUN_DEV01/report.json`,
  SHA-256 `7ce671d80bdd0e00fceed6bf6b5afeebbb89d029e6a91379ee99020bb7490609`.
- Paired analysis `p1_dev304_cp10_tail_pair_saved_20261102_DEV01.json`,
  SHA-256 `331343fb555255903e8cdf416c7bb54e80a01a7e801150d8db2baa206c24535b`.
