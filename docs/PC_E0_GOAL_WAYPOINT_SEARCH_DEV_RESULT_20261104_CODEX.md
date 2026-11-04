# E0 goal-relative waypoint search: development result

The preregistered eight-root, three-arm run completed 24 plans. The new
waypoint-CEM baseline uses a hand-designed goal-relative geometric prior,
informed by this exposed wall family. It does not use a privileged witness as
an action proposal, but it is **not** an automatic spectral response program
and its outcome cannot be attributed to v11.

| Arm | Mean terminal distance | Success `<0.05` | Prediction tokens per root | Mean wall time |
|---|---:|---:|---:|---:|
| Goal-relative waypoint CEM | 0.287868 | 2/8 | 1919 | 0.044 s |
| v11 response | 0.501222 | 0/8 | 1893–1917 | 0.024 s |
| Three-start direct CEM | 0.500795 | 0/8 | 1920 | 0.031 s |

On roots 0 and 2, the waypoint arm's final scores were 0.012610 and 0.025895;
the **actual selected** first-five-action prefix moved farther from the goal
than the initial state on both. Root 7 reached 0.055807, above the unchanged
success threshold. These are descriptive development observations: prefix
distance alone does not establish a change in subsequent action response or
that a suffix exploited it. Both successful scores arose from a bounded search
using the hand-designed waypoint prior; no PC2 or PC3 gate has passed.

Every prediction, action and terminal score was recomputed using the original
exact backend: 24 plans, 17,860 trajectories. A separate line-intersection
implementation replayed all 229,600 actual action steps and matched saved
endpoints. It is a second implementation of the same abstract wall, not an
independent physical environment. The waypoint search generated 903
wall-crossing complete-chain evaluations across the eight roots, compared
with 3 for v11 and 10 for direct CEM; repeated CEM samples are included and
must not be interpreted as independent successes. Four selected waypoint
chains crossed the wall; two of those reached the success threshold.

The waypoint geometry, scale and model family are informed by earlier wall
diagnostics; the roots are new only relative to the cited v5 and v10 panels,
and broader historical independence is unknown. Same token budget does not
equate total computation. These results support a narrower hypothesis:
allocating search to lateral intermediate targets can improve complete-chain
coverage on this family. They do not establish that spectral selection causes
the improvement or that the prior transfers to RGB.

[Protocol](PC_E0_GOAL_WAYPOINT_SEARCH_DEV_PROTOCOL_20261104_CODEX.md) ·
[full traces, arithmetic and structural audits, independent geometry replay](../le-wm-main/results/pc_e0_goal_waypoint_search_dev_20261104_codex/).
The fixed panel is now stopped. A separate, explicitly post-hoc common-suffix
and equal-budget intervention can examine these final chains, but cannot turn
this hand-designed baseline into the frozen v11 method or independent proof.
