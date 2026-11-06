# New feasible-Wall panel: saved coverage diagnosis

The sealed four-root development plans and traces were analyzed without
new planning or candidate injection. Counts below are scored full-chain
samples, including repeated samples, not independent tasks.

| Root | v13: crossings / samples | v13 successes | Global CEM: crossings / samples | CEM successes |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2 / 484 | 0 | 2 / 640 | 0 |
| 1 | 3 / 484 | 0 | 495 / 640 | 50 |
| 2 | 211 / 484 | 0 | 571 / 640 | 39 |
| 3 | 2 / 484 | 0 | 25 / 640 | 0 |

Crossing means a saved state ever has `x > 0`, not necessarily that the
terminal state is on the right. Both planners initially score the same
global bank, but subsequent sampling and budgets differ. Thus root 1
shows that v13 scored no successful chain while the CEM control did;
the source audit also checked v13 selected the best complete-chain score
it actually evaluated. It does not establish which change in allocation,
initialization or search caused the difference.

Root 2 is different: v13 did generate 211 crossing chains, and its best
terminal distance was 0.06517, above the fixed 0.05 success threshold.
Crossing coverage alone is therefore insufficient for success. The
same-bank CEM scored 39 successful samples there.

On the two CEM-success roots, the post-hoc nearest v13 first-layer action
prefix to the CEM selected chain had RMS 0.556 and 0.794; the nearest
retained values were 0.570 and 0.794. These raw-action distances have no
predefined effectiveness threshold and need not preserve behavior through
collision branches. Their use as an oracle or an input to v13 is prohibited.

The original plans were audited with alternate same-environment geometry;
this analysis verified audit and trace hashes and recomputed scores. It is
neither independent replication nor proof of a pruning, optimizer or
theoretical failure. The 4/4 geometric feasibility witnesses belong only
to the separate privileged diagnostic and do not alter these method scores.
PC2/PC3 remain unestablished and the source panel remains stopped.

Protocol: `PC_V13_NEW_FEASIBLE_WALL_COVERAGE_POSTHOC_PROTOCOL_20261104_CODEX.md`.
Per-root data and hashes:
`le-wm-main/results/pc_v13_new_feasible_wall_coverage_posthoc_20261104_codex.json`.
