# RGB feedback tail retention: integration protocol

This is a development integration of the existing PointMaze RGB executor,
not a new PC gate or an independent comparison. Reuse the sealed root plans,
four development roots, two root budgets and four methods from
`PC_BUDGET_ALLOCATION_DEV_PROTOCOL_20261103_CODEX.md`; do not regenerate or
overwrite their original results.

Replay warmup and the first 10 selected actions in the original shadow RGB
backend. Confirm the saved RGB hashes, frozen checkpoint, normalization and
root-plan dependency hashes. At the actual refreshed RGB and action history,
score the original five-step tail with the same model, target and prediction
meter used for all new candidates. Allocate one of 96 five-step prediction
samples to that score and the remaining 95 to the existing CEM optimizer.
All methods use the same rule; ties retain the old tail. Seal the old and
new scores, both action sequences, selected sequence, costs and input hashes
before executing the selected tail. Replay the same warmup and first ten
actions, verify the feedback RGB hash, and execute only the selected tail.

The `smoke` mode uses only the first sealed row to validate wiring; `full`
uses every sealed row. Modes write distinct new files and refuse overwrites.
Neither mode changes the already stopped non-immediate mechanism protocol.
The resulting images and action trace demonstrate integration, not physical
performance. Physical evaluation must be a separate post-seal replay; full
cost certification and independence remain unresolved.
