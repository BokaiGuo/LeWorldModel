# H25 fixed CUDA winners: native physical replay

All four fixed jobs, in base order, executed the joint and library_chain
arms. Each used the actual 15-step prefix and the corresponding ten-step
saved CUDA winner, with no padding or retries. Four independently invoked
saved-consumer processes accepted the reports, journal chains, actual action
ACKs, task/root/goal identities, 106-event cp15 feedback guards, PNG bytes
and pixels, and terminal state. The consumer is a separate process but shares
the lifecycle validation implementation; it is not an independently authored
implementation.

All eight arms completed 25 plan actions without native termination or
truncation. Native success was false in all eight. At the actual final state:

| Base | Arm | Object position error | Wrapped object angle error (rad) |
|---|---|---:|---:|
| 0 | joint | 69.1390 | 0.5593 |
| 0 | library_chain | 69.1390 | 0.5593 |
| 1 | joint | 25.4755 | 0.0880 |
| 1 | library_chain | 26.1752 | 0.1029 |
| 2 | joint | 61.9876 | 0.0813 |
| 2 | library_chain | 33.3175 | 0.2326 |
| 3 | joint | 61.9876 | 0.0813 |
| 3 | library_chain | 33.3175 | 0.2326 |

Object success is unavailable, not false. Native success is the original
native task metric, not an object-only success measure.

Across the four jobs, attempted and completed physical operations were each:
8 constructors, 8 closes, 16 resets, 32 set-state operations, 24 goal
setters, 16 goal renders, 8 goal binds, 280 public steps (80 warmup plus
200 plan), 2,832 physics steps, and 888 explicit renders. The sum of inclusive
job wall times was 29.133 s, excluding gaps between jobs; this is not total
CUDA plus physical compute. Model planning and physical resource axes remain
separate and cannot be summed as interchangeable calls.

The joined saved receipt is
`results/codex_p2_program_h25_physical_20261030_v1_JOIN_01.json`, SHA256
`51fe16a15e2e83b255e1e5a4b7b644b82a263c17db94ca38c29d9ea1ce5d1283`.
Each B0–B3 run and its separate prepare and receipt are retained under
`results/codex_p2_program_h25_physical_20261030_v1_B*`.

This is known-root exploratory evidence, not a fresh independent confirmation,
equal-total-compute claim, proof of arbitrary program composition, or P1/P2
gate passage. Bases 2 and 3 have the same context and results. No GPU was used
for this physical replay, and no scientific gate was changed.
