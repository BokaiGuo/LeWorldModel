# P3 pre-outcome allocation: host CUDA development result

The frozen known-root controller ran both allocations on host CUDA and the
saved consumer verified the journal chain, ordering, fixed candidate prefixes
and adapter-dispatched counts. Both arms measured the same two forward-latent
signals, 0.027939 and 0.036196, before either allocation was frozen. The
signal allocation was `[5, 15]` and the reversed control `[15, 5]`; hence the
two policies actually used different quotas.

Each arm recorded two signal evaluations, 20 candidate evaluations, 44
action encodes and 44 predictor samples/calls. These are adapter boundary
counts, not full FLOPs, inherited root encoding, wall cost or energy.
Candidate scores are model-space terminal costs, lower is better:

| Root | Signal allocation winner | Reverse allocation winner |
| --- | ---: | ---: |
| 0 | 1.008468 | 0.972351 |
| 1 | 0.229728 | 0.229728 |

Thus this small development result does not show a model-score advantage for
the signal allocation. The winner on root 1 was the same candidate in both
arms. This signal is an H2 zero-action latent displacement, not the earlier
Jacobian signal; it should not be described as one. No physical outcome,
independent roots, complete cost equality or P3/C3 admission was established.
The saved consumer validates recorded consistency, not a fresh model replay.

A subsequent saved-data identity audit matched every selected root/action
pair to an existing physical record. On root 0, the old records give native
distances 191.686 for this run's signal allocation and 186.519 for its
reverse control; on root 1 both select the same action and the old distance
is 278.345. The mapped records report native success false. These are old
outcomes, not new independent physical executions or an independently
replayed simulator. The physical winner contract was written after these
model outcomes were opened; it cannot be treated as pre-registered.

- Prepared contract SHA-256:
  `f5f3c8caa423a413f972967f5a86c0cc9ce1e615c71139d51e74ee852ba6a11a`.
- Host report SHA-256:
  `202ffc7a6c8fb7f26301a137028862a08cd4e014ca95c4387220b4adc49d8729`.
- Saved-consumer SHA-256:
  `85b16d3875cf360059fd166d6f961588dea94c4988d8e725ee8ce07380f2bed7`.
