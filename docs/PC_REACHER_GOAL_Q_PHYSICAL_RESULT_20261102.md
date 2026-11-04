# Goal-Q selection: real-feedback development outcome

This follows `PC_REACHER_GOAL_Q_PHYSICAL_PROTOCOL_20261102.md`. Two previously
exposed Reacher roots each have five reported arms. All ten prefix executions
returned five steps and all ten suffix executions returned ten steps, with
`terminated=false` and `truncated=false`; each suffix receipt reports 32
forward rollout evaluations. The checkpoint parameter hashes before and after
match. The report does not contain a separate per-step native-success field:
do not infer one from completion or from joint-distance values.

| Root | Arm | Prefix index | Initial qpos distance | After prefix | Final qpos distance |
| --- | --- | ---: | ---: | ---: | ---: |
| 20263160 | selected | 3 | 1.003500 | 0.984010 | 0.537083 |
| 20263160 | zero | 0 | 1.003500 | 1.010031 | 0.546408 |
| 20263160 | same-bank Q-greedy | 4 | 1.003500 | 0.982156 | 0.497367 |
| 20263160 | refined 0 | 6 | 1.003500 | 1.019245 | 0.557469 |
| 20263160 | refined 1 | 7 | 1.003500 | 0.997244 | 0.545422 |
| 20263161 | selected, **ineligible zero fallback** | 0 | 1.182600 | 1.245444 | 0.598310 |
| 20263161 | zero | 0 | 1.182600 | 1.245444 | 0.598310 |
| 20263161 | same-bank Q-greedy | 4 | 1.182600 | 1.148837 | 0.760234 |
| 20263161 | refined 0 | 8 | 1.182600 | 1.269222 | 0.711966 |
| 20263161 | refined 1 | 9 | 1.182600 | 1.245755 | 0.597585 |

For root 20263160 the selected prefix already improves qpos distance against
both the initial state and the separately executed zero prefix; its final
distance is lower than zero's but worse than the Q-greedy arm's. This does
not establish non-immediate-progress reconfiguration. On root 20263161 the
model selector returned no eligible prefix; selected is the required zero
fallback, not a second independent reconfiguration witness. The physical
runner labels that arm `selected`, so consumers must retain this distinction.
For that root the saved selected and zero full raw-action arrays, normalized
arrays, and observed-successor context records are identical.
Duplicate arms are retained in the denominator, not treated as independent
roots.

Each arm replans from its own real RGB and action history. The suffix CEM
still optimizes the original Euclidean latent objective rather than Q.
Selection, encoder-derived Q, and original upstream candidate generation
are not fully charged in a common total resource budget. No common-suffix
mechanism intervention or independent confirmation occurred in this run;
it does not open either the Planning Compiler P1 gate or the separate legacy
ATM/MSE gate.

Saved physical report SHA-256:
`9a529f5cdd59c0ac1098f3f52327b6d4b52c4841e8a992d4a01b556e14d4457f`.
Model-only selector input SHA-256:
`2b128d35b9f0bceb23247a8b8b16086f97d142ee5a0ce18aeeb9c7f9f6714fb8`.
Root source SHA-256:
`647b93058349f8f58f4a0947d9440149d1b98ecdd6b22899483d7e341d33124a`.
Runner SHA-256:
`53177057e79269d1e1c754de18fa1948b6c94e8da42c274cd97c5c9545258dc0`.
