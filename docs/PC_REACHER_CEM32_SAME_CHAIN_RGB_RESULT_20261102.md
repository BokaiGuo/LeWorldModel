# Reacher CEM32: predicted, actual RGB and physical ranking

This is a development diagnostic on two previously exposed roots, not an
independent confirmation or a P1 admission test. The six fixed physical
chains and their original model scores were recorded before this analysis.
The physical replay used CPU-decoded tails; bitwise identity with the
CUDA-decoded actions used for model scoring remains uncertified.

The RGB runner encodes the recorded actual frames at steps 5, 10 and 15,
with the executed three action packets, and separately encodes the same
root-bound goal. A goal lookup bug would have requested unavailable
candidate-specific contexts; it was repaired to use the available neutral
root context and verify its goal image hash for each chain. Model parameters
were frozen and their before/after hashes agreed. No new physical actions
were executed.

| Root | Prefix | Model predicted latent cost | Actual RGB latent cost | Physical joint distance |
| --- | --- | ---: | ---: | ---: |
| 20263160 | selected | 0.78704214 | 0.77289522 | 0.56680170 |
| 20263160 | neutral | 0.80371284 | 0.79541469 | 0.60645225 |
| 20263160 | same-bank greedy | 0.80814236 | 0.77995008 | 0.47821287 |
| 20263161 | selected | 1.93611503 | 1.83712363 | 0.67768069 |
| 20263161 | neutral | 1.94080448 | 1.84207523 | 0.67962742 |
| 20263161 | same-bank greedy | 1.93884897 | 1.84871364 | 0.65786652 |

All three columns are lower-is-better. On both roots, selected beats greedy
in both predicted and actual encoded latent cost, but greedy has the lower
physical joint distance. Thus this paired comparison supplies evidence for
a latent/physical ranking mismatch on these particular chains, not evidence
that the model prediction is correct. On root 20263161 the predicted ranking
of neutral versus greedy also reverses in actual encoded latent cost.
Original model prediction vectors were not saved, so a vector-level
prediction error decomposition is unavailable. These are two roots, not
six independent experiments; no general failure rate can be inferred.

The saved-only consumer rechecks the physical source receipt, event hash
chain, ordered six arms, action history, frame and goal identities, scalar
bindings, and actual latent costs recomputed from the saved coordinates.
This proves consistency of saved records, not independent simulator
attestation. Four tamper tests cover frame, history, coordinate and arm
order; the relevant suite has 14 passing tests.

Artifacts:

- RGB plan SHA-256:
  `bc6133b391ec43b84a06de09ae0fc875ae0230d2159e3af6168a2e59872cdaf6`
- RGB journal SHA-256:
  `55f401c363c67b3c78f407ed82726cee397f2ebe496a5f70cba92ff2a5c87b68`
- Saved validation SHA-256:
  `a22e497bae27ad74efabe2af437fb84811b1d2f683efa31420c731df3604e76b`

These results do not establish non-greedy control-capability change, an
equal-budget advantage over greedy or strong baselines, independent roots,
or completion of the Planning Compiler research programme.
