# Reacher FD: same-action saved RGB replay

Under the fixed development protocol, the host-CUDA model replay completed all
20 previously executed branches at steps 5, 10 and 15. Actual RGB and raw
actions came from the sealed physical result; the causal warmup context was
reconstructed and checked for exact root/goal identity. The step-15 open
prediction matched the sealed model endpoint; checkpoint and parameter hashes
matched, and frozen parameters remained unchanged. This is a model analysis of
old physical outcomes, not a new physical intervention.

| Root | Step | Open MSE | Refreshed MSE | Refresh lower |
| --- | ---: | ---: | ---: | ---: |
| 20263160 | 5 | 0.004422 | 0.004422 | 0/10 |
| 20263160 | 10 | 0.007997 | 0.005997 | 10/10 |
| 20263160 | 15 | 0.014326 | 0.007068 | 10/10 |
| 20263161 | 5 | 0.004883 | 0.004883 | 0/10 |
| 20263161 | 10 | 0.010213 | 0.005066 | 10/10 |
| 20263161 | 15 | 0.018355 | 0.006670 | 10/10 |

At step 5 both predictions use the same context by construction. Ten
branches within each root share context and are not independent roots. MSE
is not a physical success or replanning result.

For each root the five fixed comparisons pair selected against
parameter-zero with the same suffix. Strict sign reversals in step-15
selected-minus-zero differences are:

| Root | Open Q vs actual-RGB Q | Refreshed Q vs actual-RGB Q | Actual-RGB Q vs physical wrapped joint distance |
| --- | ---: | ---: | ---: |
| 20263160 | 0/5 | 0/5 | 3/5 |
| 20263161 | 2/5 | 1/5 | 1/5 |

Q and joint distance are different evaluation quantities; this small
post-exposure diagnostic cannot identify a unique failure cause or establish
generalization. Physical outcomes do not enter planner input. The paired
physical outcomes still contain prefix-end initial differences and do not
prove greater controllability.

The first replay (`v1`) used the predicted-coordinate interface for observed
latent too, following an older replay. The corrected `v2` explicitly uses the
observed interface for observed latent; both runs remain preserved. The
third replay (`v3`) additionally verifies the exact warmup packets and saves
the planning coordinates and goal coordinate. The independent saved-data
consumer rebinds every checkpoint to physical RGB hash and qpos distance,
recomputes latent MSE, reproduces paired contrasts, and independently
recomputes all 180 Q values from those saved coordinates and the pinned metric.
It does not rerun the model or independently certify those coordinates.
An initial Q verification failed because it performed the coordinate
subtraction in float64, whereas the model subtracts in float32 before
converting to float64; reproducing that operation yields the matching values.
Its counterexample tests plus history/RGB tests pass (8 total).

The run counted 20 three-step open batches (60 predicted tokens), 60 one-step
refreshed batches, and 122 attempted context-encoding calls, plus goal encoding.
Reported wall time starts after setup/model loading/source verification, so it
is not total planning cost. This diagnostic does not compare methods under a
budget or establish that feedback improves real control. The replay runner
did not acquire a cooperative GPU lock; it therefore provides no exclusive
resource or contention guarantee.

- `v1` report SHA-256:
  `637978c670c509167aadc960e2e913e54e0b7c84225212fbac2d43fccd15a4bd`
- `v2` report SHA-256:
  `a482ef7e38b55a3fbcc621779aa7ca5601e85002d0b0d93aa0b6474aecc17164`
- saved-data audit SHA-256:
  `202f9ddd6d534f971999690c79845ffc4dc2277dbc49dd2f964e1d5bb22ddb91`
- `v3` report SHA-256:
  `385685b97d440616f6466f7dfac138d86ddfd82b43f4d09b48654914d32468b8`
- `v3` saved-data audit SHA-256:
  `f0cc4da576c648b023f1f1bd894105b31b7cb419300987761050600e5e3605ee`
