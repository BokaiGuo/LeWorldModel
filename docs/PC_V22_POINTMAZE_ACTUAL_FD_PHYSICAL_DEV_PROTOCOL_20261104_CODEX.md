# v22 actual PointMaze response probes: physical development diagnostic

Freeze this protocol before execution. Use only the response arm's already
sealed third decision and its actual two nominal 21-sample finite-difference
batches. Replay the saved 25-action warmup and feedback prefix from the
same setup, check the RGB boundary, and branch into every saved five-action
probe. Independently replay the same actions in the native backend and check
the first 26 states against the existing sealed native trajectory.

Compare model-projected, causal RGB-estimated, and native position response
matrices at ±0.05. Report absolute and relative Frobenius errors and cosine
when defined, and non-tie pairwise score-versus-distance ordering. Save
all actions, branch images, physical traces, hashes and failure artifacts;
do not feed any physical result back into planning or modify the already
stopped live panel. This is one exposed boundary and one scale, not PC1
certification or independent generalization.
