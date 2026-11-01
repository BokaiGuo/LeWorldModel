# Fixed cross-goal development panel

Before new outcomes, fix both off-diagonal pairs of the two existing authenticated
roots and the two saved cp10 RGB targets: root0→target1 and root1→target0. Retain
all four arms and the entire roster, not whichever target yields a desired result.
Original diagonal experiments remain preserved and are not superseded.

Use the prefix-protected, response-diverse, first-feedback-carry planner unchanged:
same checkpoint, optimizer, feedback times, 20-step horizon and cost accounting.
No physical state enters the planner: it receives authenticated target RGB latent
and actual RGB history. Simulator goal state is used only by native evaluation,
as in the original runner.

Bind target pixels, saved CUDA latent, task identity, physical root-goal hash and
native goal to the same other-root target. Keep the original initial root/history.
Never transplant the other root's reachability witness: it does not establish
reachability of this pair within the fixed horizon. If initially successful,
reject before execution and report the condition rather than replace the root.

This is a task-coverage diagnostic, not independent confirmation. Both roots and
targets have appeared in development. Do not assert training independence.
Record prefix physical changes separately from success; a non-improving prefix
alone is not a capability mechanism. If such a prefix is selected, the next test
is shared H10 suffix intervention and independently optimized suffixes.
