# PC2 bounded atlas development smoke

The newly added search expands actual action-witnessed model successor nodes
and recomputes response geometry at each expanded node. It retains a
fixed-width beam per first-prefix branch, selects complete plans using their
nonlinear terminal model cost, and replays each retained complete witness from
the root. It reports a separate model-only nonprogress qualification against
both zero and same-bank greedy; qualification does not establish a physical
mechanism. The selected plan and qualified candidate need not be identical.

The predeclared eight previously exposed unicycle goals produced mean terminal
position error 0.1419533942 for the atlas and 0.00000365461 for the existing
four-start analytic gradient control. The control had lower error on all eight
cases. Two roots had a model-qualified nonprogress branch. This analytical
task is the model itself, not a visual model or a real physical intervention.

Each atlas run used 76 rollout calls and 7 Jacobian calls; the baseline uses
up to 32 closed-form objective and gradient evaluations. These units are not
interchangeable and do not establish equal resource budgets. Atlas branch
searches share depth and beam policy but can have different candidate counts
when response rank differs; no equal per-prefix compute certification is
claimed. Counts do not cover visual encoding or real neural compute.

Artifact: `le-wm-main/results/pc_atlas_unicycle_smoke_20261103_codex.json`.
Protocol: `docs/PC_ATLAS_SEARCH_DEV_PROTOCOL_20261103_CODEX.md`.
Fourteen compiler/search tests pass, including a multistep finite-difference
check for the smoke backend. Tests and this development smoke do not pass PC2,
PC3, PC4, PC5, or any legacy gate.
