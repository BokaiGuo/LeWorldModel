# Q-consistent physical development: paired restore repair

The first execution at `reacher_pc_q_consistent_physical_20261102_v1`
preserves six error records. All six failed before a prefix action:
`ObserveSteps._goal` was read-only, while snapshot restoration assigns the
goal. These records are engineering failures, not physical outcomes.

Before running v2, the runner was repaired to restore and verify the same
initial snapshot before **every** arm and to forward goal assignment to the
underlying goal wrapper. It records the stage reached, prefix actions and
suffix execution receipt even when an arm fails. Tests cover repeated paired
restore, identity failure, and goal forwarding. The existing experimental
protocol, source candidates, selection, seeds and physical metrics remain
unchanged; v2 is a separate development engineering retry, not independent
confirmation. No outcome from v1 is used to choose a candidate.
