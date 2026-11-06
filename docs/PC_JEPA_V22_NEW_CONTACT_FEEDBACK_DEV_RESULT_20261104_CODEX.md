# v22 native JEPA contact feedback: development result

The fixed-seed privileged witness moved the object by 129.935 pixels and
rotated it by 1.148 radians; its actions, RGB, proprioception and physical
states replayed exactly at all 26 boundaries. Witness actions and state
were evaluator-only, not planner inputs. This establishes feasibility of
this constructed target, not a general task-distribution feasibility
claim.

Both isolated planner arms completed three RGB-plus-proprio feedback
decisions with 1,920 prediction tokens per decision. Both failed the
predeclared joint-position and angle success criterion at the fixed
15-action endpoint:

| Arm | Final object error, pixels | Final angle error, radians | Final joint XY error |
| --- | ---: | ---: | ---: |
| v22 response | 29.638 | 0.587 | 91.149 |
| Shared-bank multistart CEM | 76.196 | 1.473 | 132.463 |

Saved-data auditing verified actions, input histories, source hashes,
candidate scoring and cost accounting. The additional chronology audit
verified the first proposal bank was identical across arms, carry was
rescored in later contexts, and each decision was sealed before the
corresponding five physical actions. Same-backend replay reproduced both
trajectories exactly at all 26 boundaries. These checks are not a complete
information-flow proof or independent replication.

The physical witness and each planner arm were attempted once on the
predeclared seed. A witness runtime preflight initially rejected a missing
bytecode-setting before any artifact or physical execution; it passed
after that setting was supplied. The previous v13 failed panel was not
reopened. The isolated worker used the v22 timed planner shared with the
Reacher diagnostic.

This is one development task with unknown source independence, not a
credible advantage or PC4/PC5 passage. Prediction tokens do not include
all GPU, encoding or wall-clock costs. Records and audits:
`le-wm-main/results/pc_jepa_v22_contact_feedback_dev_20261104_codex/`.
