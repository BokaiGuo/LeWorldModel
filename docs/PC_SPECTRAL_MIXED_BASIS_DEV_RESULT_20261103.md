# Mixed-action spectral basis development result

The protocol evaluated the same eight exposed unicycle goals with four
predeclared sampling seeds per goal. Every arm used 16 mixed-action proposals,
zero and greedy controls, the same one-packet suffix optimizer, and direct
nonlinear full-chain replay. The table averages position error over the four
seeds for each goal; these are repeated samples on the same roots, not
independent tasks.

| Goal | Raw isotropic | Full spectral isotropic | Spectral, second coefficient × 0.25 |
| ---: | ---: | ---: | ---: |
| 0 | 0.0209 | 0.0209 | 0.0163 |
| 1 | 0.0429 | 0.0429 | 0.0324 |
| 2 | 0.1001 | 0.1001 | 0.1236 |
| 3 | 0.0849 | 0.0849 | 0.2145 |
| 4 | 0.0708 | 0.0708 | 0.3054 |
| 5 | 0.1076 | 0.1076 | 0.3935 |
| 6 | 0.1779 | 0.1779 | 0.4785 |
| 7 | 0.2456 | 0.2456 | 0.5602 |
| Mean | 0.10635 | 0.10635 | 0.26557 |

For these axis-aligned root contexts, the returned full spectral basis is
exactly the raw-coordinate identity. With matched random coefficients, the
first two arms generated identical candidate actions and outcomes on every
case. Their equality does not test a case with nontrivial spectral rotation.
Shrinking the second coefficient deliberately changes the proposal
distribution and performed worse on six of eight root averages; this is a
negative result for that specified heuristic, not for all possible
task-aware spectral distributions. It is consistent with needing turns in
these tasks, without establishing a non-greedy mechanism.

Each arm made 112 rollout calls; raw used 19 Jacobian calls and each
spectral arm 20, plus its SVD. Equal samples therefore did not equalize total
compute. Wall time is saved per arm. No neural model or real physics was
invoked and no gate is opened.

Protocol: `docs/PC_SPECTRAL_MIXED_BASIS_DEV_PROTOCOL_20261103.md`.
Full candidate actions, costs, timing and counts:
`le-wm-main/results/pc_spectral_mixed_basis_20261103.json`.
