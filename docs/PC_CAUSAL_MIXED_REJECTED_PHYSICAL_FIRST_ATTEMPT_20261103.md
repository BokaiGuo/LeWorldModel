# Failed first attempt: rank-count shape contract

The initial development run completed all 32 physical branches and both
position-backend replays, but failed before writing arrays or a report.
The borrowed `rank_counts` function enforces exactly nine outcomes while the
new fixed bank has eight per root. The exception was
`ValueError: nine finite paired outcomes required`. Because the completed
trajectories were only in process memory and were not persisted before the
exception, none can be independently audited; they are not experimental
evidence. The original run performed 544 RGB-backend and 832
position-backend steps, plus the model calls specified in the protocol.

The runner was corrected to use an explicitly tested eight-outcome rank
counter. The predeclared candidate bank, planning, action execution and
analysis definitions remain unchanged. A full rerun is required; all costs
of this failed engineering attempt remain additional to the rerun.
