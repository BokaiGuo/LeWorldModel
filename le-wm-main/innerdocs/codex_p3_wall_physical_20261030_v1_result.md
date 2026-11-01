# Wall-delivery winners: physical replay result

Both fixed six-branch batches completed, with separate saved consumers verifying
the original prepare, journal, root/goal/warmup identity, action ACKs, stop
flags, metrics, frames and operation census. The consumers do not independently
rerun the simulator. Both consumer summaries have the same bytes and SHA
`706d73e50d7c1882e2cb17622577e78e7f644fbaf4b9195f663746124e5876ec`;
their filenames alone do not authenticate which source run they consumed.
The separate completed consumer invocations used these explicit bindings:

| Point | Original prepare SHA256 | Physical report SHA256 |
|---|---|---|
| 20 | `2a0fe5fdeedce10980b55a58795e230b4da919d99bf9bf3413eff509252afd6e` | `426e666910e862f18a1923b2c562ec04fae9fc98fad66a6ee67bca8cb7259fb3` |
| 30 | `7b3cfea2dcb1318aebdffa732e1a387122ca00851b09cc1e6ad09ba29a9a132b` | `b7a084c3aa82210cf670f006a8713719b33ec1709e67b8abb4d86e6899140eb8` |

All 12 branches returned ten planned steps, none stopped early, and all had
native success false. Final object position error / wrapped angle error:

| Point | Root | uniform | signal | reverse |
|---|---:|---:|---:|---:|
| 20 | 0 | 71.522 / 0.500 | 65.096 / 0.441 | 70.787 / 0.476 |
| 20 | 1 | 76.403 / 0.082 | 76.403 / 0.082 | 76.403 / 0.082 |
| 30 | 0 | 70.787 / 0.476 | 70.787 / 0.476 | 70.787 / 0.476 |
| 30 | 1 | 76.403 / 0.082 | 76.403 / 0.082 | 76.403 / 0.082 |

Per point the actual attempted and completed operations were: 6 constructors
and closes, 12 resets, 24 set-state, 12 goal renders, 6 goal binds, 120 public
steps, 1224 internal physics steps and 132 explicit renders. The measured
physical execution scopes were 6.080 s and 6.157 s, respectively; these are
batch durations, not per-strategy inference or end-to-end latency. No neural
inference was performed during physical replay. The original planning limit was
30 s per process, and excludes this physical execution and saved consumer.

All 12 selected raw float32 action hashes match the earlier online-budget
physical batch at the same point/root/strategy. Accordingly this is a
known-root repeated physical replay, not an independent sample or independent
confirmation. Position/angle are descriptive, not a new scientific pass.
Equal wall ceilings and matched physical procedure do not imply equal total
compute; the three policies differ in signal costs and candidate allocations.
Existing assets are preserved and scientific gates are unchanged.
