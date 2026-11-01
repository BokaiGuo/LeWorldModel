# P2 prefix-recompile fixed-winner physical replay

All eight fixed CUDA winners were replayed in four base-major jobs, joint
then prefix_chain. Each action sequence was the source-native 15-step
prefix plus the two literal five-step CUDA winner packets. Four independent
process invocations of the saved reader verified source and winner binding,
the 106-event cp15 RGB/state guard, action ACKs, journal chains, saved RGB,
and stopping; the four receipts were joined at fixed denominator eight.
The saved reader shares lifecycle code with the producer and is not an
independently authored implementation.

All eight arms completed the 25-step horizon, with no native termination,
truncation or engineering error and no padding or retry. Native success was
false for all arms. Object-only success is unavailable, not false.

| Base | Arm | Final object position error (native units) | Final wrapped angle error (rad) |
|---:|---|---:|---:|
| 0 | joint | 69.1390 | 0.5593 |
| 0 | prefix_chain | 13.2343 | 0.0810 |
| 1 | joint | 25.4755 | 0.0880 |
| 1 | prefix_chain | 26.1752 | 0.1029 |
| 2 | joint | 61.9876 | 0.0813 |
| 2 | prefix_chain | 61.6974 | 0.4317 |
| 3 | joint | 61.9876 | 0.0813 |
| 3 | prefix_chain | 61.6974 | 0.4317 |

Across all four physical jobs, attempted and completed operations were each:
8 constructors and closes, 16 resets, 32 set-state operations, 24 goal
setters, 16 goal renders, 8 goal binds, 280 public steps (80 warmup and 200
plan), 2,832 physics steps and 888 explicit renders. Summed whole-job wall
time was 38.220 seconds, excluding between-job gaps and prior CUDA planning.
The previously saved CUDA ledger has, per joint arm, 36 packet calls,
1 Jacobian, 1 SVD and 17 scores; per prefix_chain arm it has 28 packet calls,
5 Jacobians, 5 SVDs and 17 scores. Nested model-entry counts are not additive
FLOPs; model loading, energy, total FLOPs and inherited acquisition are
separate or unknown.

The joined receipt is
`results/codex_p2_prefix_recompile_physical_20261030_v1_JOIN_01.json`,
SHA256 `e2b6eec3e1de5616f38222059c8b0cb20b377c4345bbefeba7d38a5a0dcab3f9`.
All B0–B3 prepares, run directories, journals, frames and saved receipts
remain in their respective exclusive result paths. Tests: 12 passed,
including literal winner identity, resealed tampering, prefix guard failure,
early native stop and saved PNG tampering.

These are known-root exploratory observations. Bases 2 and 3 share context
and outcome, and the joint winner actions have previously been replayed in
H25. No independence, equal total compute, causal composition mechanism,
fairness advantage or P1/P2 gate passage is claimed. The CUDA contract was
created after prior results were known. No model was loaded or GPU used for
this physical stage.
