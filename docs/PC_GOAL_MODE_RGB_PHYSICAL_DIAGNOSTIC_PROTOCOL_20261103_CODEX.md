# Goal-mode RGB smoke: sealed-bank physical diagnostic

Replay every complete candidate already saved in the goal-mode RGB smoke plan,
using the same warmup, root, action domain, legacy MuJoCo backend and fixed
goal `(1.8, 2.8)` as the existing PointMaze physical evaluation. Verify
plan/execution/source hashes and that the selected, dispatched and reconstructed
action chains are identical before replay. Check backend identity, scene hash,
trace length, and each action's bounds. Save full traces and distances after
warmup, after the five-action prefix and after the full 15-action plan.

The selected chain is the actual pre-outcome model decision. Other candidates
are fixed-bank post-selection diagnostics, including the same-bank greedy
candidate if present; their physical outcomes are not fed to the planner.
No new plan or candidate is generated, and the old stopped mechanism protocol
is not reopened. This single reused development root cannot prove a mechanism,
fair performance advantage, success rate, generalization, or PC gate passage.
