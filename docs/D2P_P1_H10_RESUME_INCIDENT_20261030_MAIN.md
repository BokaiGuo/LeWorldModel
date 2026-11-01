# H10 continuation: incident and evidence boundary

The original fixed 1008-branch H10 run stopped without an aggregate report.
Anchors 0–7 had complete reports; anchor 8 also had an interrupted journal.
The incomplete material is preserved under RUN01.

The first continuation incorrectly passed a constructed Gym environment as
the callable environment factory. RUN01 `anchor_08/report.json` records
`TypeError: 'OrderEnforcing' object is not callable` on branch 0. This is a
continuation implementation error, not a physical or scientific negative.

RUN02 retains byte copies of anchors 0–7 and the original binding and runs the
remaining fixed roster. The continuation first invokes the original shard
checker on the existing anchors, then uses the original `execute_shard` for
the remaining anchors. Its factory now returns a callable; the focused
regression tests cover the callable and exclusive report write.

A second issue was found after RUN02 started: the version loaded in its
process calls `old.base` during the final directory fsync, although that
attribute does not exist. The on-disk source has been corrected to
`runner.base`; the already running process does not acquire this correction.
If all shards complete, it may write the aggregate file and then exit with
that exception. Preserve that file and independently pin its hash and run the
original full saved consumer. Do not overwrite it or infer success from its
presence. A repeated resume must not replace the aggregate.

The updated continuation validates the entire binding against the original
prepare, and compares an existing aggregate's immutable facts rather than
requiring invocation-specific `created_anchors` to be identical after a
restart. These source changes are not loaded by the already-running process;
they are for a subsequent recovery only. Focused regression and executor
tests passed (10 tests), but do not replace full saved consumption.

The original prepare does not bind the separate continuation entry point.
Successful original saved consumption would verify saved actions, frames,
state, receipts, source bindings and the full roster, but does not alone prove
the continuation process's runtime authenticity. RUN02 remains exploratory
known-root evidence, not independent confirmation or P1 admission.

After the factory correction, RUN02 anchor 8 was completed and independently
checked with the original fixed prepare and `check_shard`: all 42 branches
were `probe_completed`. Its report SHA-256 is
`5a3e485c3ab28ebf560bcb8557151a0b5a6c847b17d206016cd030547de0f96a`.
This is a shard-level verification, not an all-24 aggregate or mechanism result.

Subsequent independent shard checks also accepted anchor 9 (41 completed, one
native early stop; SHA-256
`edea4caa5f56a3425de0b8fdcd573baeb9dd5a5ded7c92fa3e8a01c62d3145f2`),
anchor 10 (42 completed; SHA-256
`7d7d6fb26313e73201a7be1dcc71f0898cce2f79da9b0fe0bccdb68497d70d01`),
and anchor 11 (42 completed; SHA-256
`2d6231a16131d0225f3043d945bb797eb57cc5d740aa625dbedcce743fede19e`).
The early stop remains a missing full-horizon endpoint.

Anchor 12 subsequently completed and the original `check_shard` accepted all
42 branches as `probe_completed`. Its report SHA-256 is
`0dea41cde1515b0880568654c4e8fb61884af900552d30dee79dc35b5a8d77e1`.
This is an integrity check only; the all-24 saved consumer and the
pre-specified full-panel analysis have not yet run.

Anchor 13 was also accepted by the original `check_shard`, with 42
`probe_completed` branches and report SHA-256
`23193d7ec45c76ac1e16ac4b6538e462bb6403df0a5a10b8af0ba6ef4156ac53`.
The aggregate and full-panel conclusions still require all 24 anchors.

Anchor 20 has also been checked with the original `check_shard`, against the
fixed prepare and its 42-probe bank. Its report SHA-256 is
`e949357a598fbed86c130f6d6729a8f566c67feaa5b236e9553ac78a9e57d89f`.
The check covers saved branch identity, action bytes, frames and the recorded
resource ledger. This is a shard integrity check, not an outcome analysis or
proof of continuation runtime provenance; the full aggregate remains pending.
