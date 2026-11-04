# Exact-dynamics spectral initialization diagnostic

Protocol: `PC_SPECTRAL_START_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md`. The four
roots were selected from already observed gradient witnesses. This is a
post-seal diagnostic on privileged exact dynamics, not an independent
benchmark, learned RGB result, or P1/PC2 admission.

## Injected reference chains

The four previously selected gradient chains all rank first when their full
chains are injected beside the nine restricted spectral-bank candidates
and evaluated using the same exact replay score. Their first velocities are
approximately `-0.1803`, `-0.4605`, `-0.5016`, and `-0.2600`; every bank
candidate fixes first velocity at zero. Minimum Euclidean distances in the
two-dimensional first-action space to a bank prefix are `0.18421`,
`0.46206`, `0.50808`, and `0.26084`, respectively.

Thus the original restricted bank cannot represent any of these reference
prefixes. The injected chains are not passed to the planner or counted as
spectral results. The score orders these references ahead of the restricted
bank on these roots; this is not a general proof that all scores or visual
models rank correctly. The restricted bank enumerates all nine candidates and
has no intermediate beam pruning. This says nothing about whether a separate
Atlas implementation would prune them. Fixing each gradient prefix and
optimizing its one-action suffix analytically yields errors at most
`7.66e-6`; this does not show that the old turn-only prefix optimizer could
produce the missing first action.

## Same full-action optimizer, different starts

Each arm uses four starts and bounded L-BFGS-B with the same closed-form
objective and gradient. All counts below include nine candidate-generation
scores for the spectral and no-SVD response arms; finite-difference and SVD
overhead is recorded separately. The table reports mean terminal distance
across these four selected roots:

| Objective cap | Spectral | Response, no SVD | Fixed generic | Random |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.130309 | 0.130309 | 0.122299 | 0.008953 |
| 32 | 0.000000 | 0.000000 | 0.000009 | 0.000000 |
| 64 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Rounded zeros are not assertions of symbolic exactness. Spectral and no-SVD
response starts and resulting errors agree in all 12 root-budget comparisons.
The response matrix has one column; its SVD returns a sign that cancels in the
bounded projection. Here the SVD adds nine decompositions but no additional
candidate direction or observed score improvement. This is specific to the
one-dimensional response and does not establish that a higher-rank spectral
method cannot help elsewhere.

At cap 16, selected random starts perform best on average; by 32, all methods
are near zero. Individual starts may terminate early, and unused allocations
are not reassigned under this protocol. Mean total actual objective evaluations
at caps 16/32/64 were respectively 16/30/33 for spectral, 16/31.5/47 for
fixed, and 16/30.5/40 for random. The two response arms additionally incur
45 generation dynamics transitions, including 18 for finite differences;
the spectral arm incurs nine SVDs. These are separate counters, not a common
FLOP budget. Wall times and all per-root results are in the sealed JSON; the
microsecond-scale timing differences on this small CPU task are not
generalizable efficiency evidence.

Conclusion: on these selected exact-dynamics roots, the immediate gap is
restricted prefix coverage, and the single-column SVD provides no observed
increment over identical no-SVD projection. This does not justify reopening
the stopped PointMaze mechanism construction. Whether a different, higher-rank
spectral construction helps a strong optimizer remains unproven.

Artifacts:

- `le-wm-main/results/pc_spectral_start_diagnostic_20261103_codex.json`
- `le-wm-main/experiments/spectral_reachability/pc_spectral_start_diagnostic_20261103_codex.py`
- `le-wm-main/tests/test_pc_spectral_start_diagnostic_20261103_codex.py`:
  three tests passed.

The separate PointMaze performance line still has no authoritative evidence
closing root-history, normalizer-fit and checkpoint-training membership. Its
development status does not imply training-disjoint independent confirmation.
