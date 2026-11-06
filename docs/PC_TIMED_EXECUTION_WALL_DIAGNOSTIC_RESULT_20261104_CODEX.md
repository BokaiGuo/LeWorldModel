# Timed execution: exact-Wall wiring diagnostic result

The single prespecified diagnostic task completed for both v17 and the
shared-bank multistart CEM through the same execution entry point. The
actual prediction ledgers record 42 calls / 1,919 tokens and 50 calls /
1,920 tokens, respectively. The one-token remainder for v17 reflects the
planner's legal batch allocation; it was not filled with unneeded calls.

Both runs used identical first proposal banks. Every recorded endpoint and
score was checked against an alternate implementation of the same
environment; the selected action chain was among those actually scored.
Trace compression happened after the planning clock stopped. CPU planning
time includes trace capture and does not cover visual encoding or GPU work.

The separate regression tests exercise v17 and CEM with carry at remaining
horizons 15, 10 and 5. These tests and the exact-Wall check verify the
execution interface, not task performance, PC2/PC3, or independent
reproducibility. This was a one-task wiring check and is stopped.

Protocol: `PC_TIMED_EXECUTION_WALL_DIAGNOSTIC_PROTOCOL_20261104_CODEX.md`.
Raw traces, call ledgers, manifest and results:
`le-wm-main/results/pc_timed_execution_wall_diagnostic_20261104_codex/`.
