# Constrained common-tail search: CUDA development result

The fixed development protocol
`PC_REACHER_CONSTRAINED_CUDA_LAUNCH_20261102.md` was executed on the two
previously exposed Reacher roots with the frozen model. Each root produced
all 21 planned points and the full 21-candidate endpoint/response census.
The checkpoint parameter hash and sealed plan both passed postchecks. No
physical outcome entered the search.

| Root | Eligible / 20 non-neutral | Candidates violating at least one absolute-tail condition | Minimum squared constraint loss |
| --- | ---: | ---: | ---: |
| 20263160 | 0/20 | 20/20 | 0.00002056 |
| 20263161 | 0/20 | 20/20 | 0.00006809 |

The five selector tests, in order, are immediate-cost increase, negative
mean final delta, no worsened common tail, negative mean neutral-adjusted
delta, and no worsened adjusted active tail. Counts failing each test among
the 20 candidates were `[10, 10, 20, 8, 20]` and `[10, 10, 20, 10, 12]`
respectively. Recomputing these comparisons from the saved score rows
reproduced all 40 eligibility flags. Both selectors returned `null`;
neither a near-feasible loss nor a neutral fallback was promoted.

The fixed search's nominal cost per root was 21 prefix predictions, 210
suffix-token predictions and 16 scalar VJPs; endpoint census added 21
prefix and 210 suffix-token predictions. Search/census wall times were
0.454/0.164 s and 0.238/0.158 s respectively, excluding model loading,
root reconstruction, inherited goal-Q construction and unreported
encoding/internal backward work. The budget is greater than the prior
tangent search and is not an equal-total-compute comparison.

The plan SHA-256 is
`f8bc433ee2cd7a1810102349574f8a13f20f9231777baa6c913640aef6fb9a3b`;
the report SHA-256 is
`14ed522959434b1c936960606b99103e32145842705a63b24b6f770c66cbc1b4`.
Full scores, traces, raw and normalized actions, predicted endpoints and
non-gating response spectra are in
`le-wm-main/results/reacher_pc_constrained_cuda_20261102_v1/report.json`.

The result rejects this fixed model-only search on these development roots;
it is not a proof that the physical mechanism is impossible. There is no
selected action to claim as a P1 witness, no native physical or independent
success result, and no gate change.
