# FD common-tail physical attempt: root identity mismatch

The fixed two-root, 20-branch physical development plan was prepared with
SHA-256 `6de6f2ed64c80af6dd66eb27a8db920cffeeaf8d0c3bc82c3274dff6e305456a`.
The initial execution returned two `root_error` records and executed **zero
planned branches**. Report SHA-256
`fd6f76134762d0db88ecd4af043594d342f4c003749120dc4c345053abd61d77`.
The top-level `finished` means the loop ended; it does not mean the branch
experiment completed.

The original check combined root RGB, goal RGB and warmup-action hashes into
one exception. A separately sealed diagnostic version recorded each hash
before the unchanged guard and ran with no CUDA visibility override. Again
zero branches were dispatched. On both roots the warmup action hashes matched
exactly, but **both root and goal RGB hashes differed** from the model
preflight. Diagnostic report SHA-256
`a6011263f2ef6740bec6350141d956a0716ab1913fdb27350af3feb396b9cb1e`.
The full expected/actual hashes are in that report.

This is a reconstruction/observation identity failure before planned prefix
execution. It does not establish that the selected action, world model or P1
mechanism failed. Different images could reflect environment, rendering,
state or dependency changes; the precise cause has not been established.
Do not remove or loosen the identity guard or relabel this failed attempt as
20 completed branches.

## Subsequent host execution

A separately saved diagnostic on the host reproduced both roots' goal RGB,
root RGB and warmup action hashes exactly. Subsequently the unchanged sealed
plan was executed in a new host output directory: 20/20 branches returned
all 15 steps. The separate saved-output consumer checked the plan, model,
action, per-step RGB and state bindings. See
`PC_REACHER_FD_COMMON_PAIRED_EFFECTS_20261102.md` for the paired result.
The exact difference between the sandbox and host remains unproven; recovering
identity on the host is not evidence that CUDA or any one rendering component
alone caused the mismatch. The host result does not overwrite this failure.
