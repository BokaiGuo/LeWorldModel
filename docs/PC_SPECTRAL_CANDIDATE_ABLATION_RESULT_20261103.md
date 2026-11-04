# Fixed-bank spectral candidate ablation

On the eight previously exposed exact-unicycle development goals, all arms
used the same six first-prefix slots (four source-specific proposals plus zero
and greedy), one identical suffix optimization per prefix, and nonlinear
full-chain replay. Terminal position errors:

| Goal | Spectral | Random | Coordinate lookahead |
| ---: | ---: | ---: | ---: |
| 0 | 0.0310 | 0.0204 | 0.0310 |
| 1 | 0.1288 | 0.0502 | 0.1288 |
| 2 | 0.2254 | 0.1987 | 0.2254 |
| 3 | 0.3198 | 0.3198 | 0.3198 |
| 4 | 0.4111 | 0.2933 | 0.4111 |
| 5 | 0.4602 | 0.2150 | 0.4602 |
| 6 | 0.3710 | 0.2324 | 0.3710 |
| 7 | 0.2782 | 0.2488 | 0.2782 |
| Mean | 0.2782 | 0.1973 | 0.2782 |

Random had lower error on 7/8 and tied on one. In this axis-aligned initial
condition the full one-step response basis generates the same candidate set
as coordinate lookahead, so equality is expected and cannot establish a
distinct spectral advantage. The positive finding from the prior injection
diagnostic is only that those supplied strong actions were correctly evaluated
on these roots; these ablations show the tested discrete spectral bank still
does not reliably generate them.

The sampled random prefixes can mix forward velocity and turn in one step,
whereas the four spectral/coordinate proposals are single basis directions.
This is a concrete difference in candidate coverage and prevents attributing
the result solely to the presence or absence of spectral geometry. It also
does not establish that mixed actions alone explain all observed differences.

Each arm used 40 rollout calls. The spectral arm additionally incurred 8
Jacobian calls versus 7 in the other two; these counts include suffix and
common greedy construction, while wall time is logged separately. Fixed
candidate count does not equalize all compute. These are analytical
development cases, not independent roots, real RGB feedback, learned-model
physics or a PC gate result.

Protocol: `docs/PC_SPECTRAL_CANDIDATE_ABLATION_PROTOCOL_20261103.md`.
Saved full actions, scores and counts:
`le-wm-main/results/pc_spectral_candidate_ablation_20261103.json`.
