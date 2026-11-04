# Reacher task-weighted response SVD: development result

This fixed model-only search ran on two previously exposed development roots.
It differentiates predicted, task-weighted terminal coordinates for four
common suffixes relative to a parameter-zero suffix, with respect to the
prefix. It projects that response Jacobian off the immediate Q-gradient and
generates signed right-singular directions. This is a prefix-to-suffix-response
Jacobian, not a proof that physical controllability changed.

The first attempt failed on both roots before producing candidates because
the CUDA efficient-attention backward has no derivative for the JVP's
reverse-over-reverse calculation. The separately sealed retry scoped SDPA to
the math backend and completed; both root/goal identities, frozen model hash
and the plan postcheck matched. The 12 non-neutral proposals per root were
all evaluated and retained.

| Root | Leading singular values | JVP versus finite difference relative error | Strict selected |
| --- | --- | ---: | --- |
| 20263160 | 27.3673, 7.9377 | 0.0000911 | none |
| 20263161 | 30.4940, 9.6718 | 0.0000983 | none |

An arithmetic recount of the saved rows gives separate pass counts for
immediate Q non-improvement, mean final Q improvement, every final Q
non-worsening, mean adjusted improvement and every adjusted non-worsening:
`[12, 5, 0, 9, 0]` and `[12, 4, 0, 9, 3]`, respectively. These subsets
overlap; no candidate satisfies the conjunction. The first test here is
defined against the parameter-zero prefix in model Q, not physical task
error. No physical result or independent root was used or produced.

Both searches used ten JVPs plus proposal evaluations; total backward work,
encoding, prior Q construction and complete wall-resource accounting were
not independently measured. Search wall times in the report omit model load.
No fair-baseline or P1 claim follows. A low finite-difference error verifies
one leading local derivative direction, not all derivatives or nonlinear
prediction accuracy. The saved report has not been independently physically
replayed, and task-coordinate/physical alignment remains open.
The launcher records individual root exceptions in the report but can exit
zero even when every root fails, as the first attempt did. Consumers must
check each root's status and the postchecks, not just the process exit code.

- Failed run report SHA-256:
  `9a403b49d21099136683007057a05b44a707cc93872f9ee7e8dcaad33c089095`.
- Successful retry report SHA-256:
  `b3f4e0cab1c63c0c1f3d7c6cc43309a2d81e6644ea8748885569f38b097479ca`.
- Retry plan SHA-256:
  `8c0d1a3decc124e12a29068289e49fe7597fe3286fc3e8b3a69cbcac31ff24d1`.
