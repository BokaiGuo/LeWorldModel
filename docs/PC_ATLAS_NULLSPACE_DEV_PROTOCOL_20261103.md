# Action-response nullspace candidate development protocol

The previous atlas selects only the nonzero right-singular directions exposed
by `local_geometry`; directions in the nullspace of the task-weighted
one-packet response are absent from the spectral bank. This may omit actions
that do not immediately change the task coordinates but change subsequent
dynamics. It is a candidate-coverage hypothesis, not evidence that those
directions provide useful reconfiguration.

Keep the v2 implementation and result unchanged. Add a separate v3 candidate
bank that appends both signs of at most two task-weighted response-nullspace
directions, obtained from an actual one-packet Jacobian at zero action. Use the
same whitening and rank tolerance as the local geometry and respect raw action
bounds. Evaluate all branches using the existing v2 nonlinear search and direct
replay. Do not alter acceptance criteria or outcome labels.

Development comparison: rerun the same eight exposed angular unicycle goals,
with the same algorithm configuration and random seeds as the v2 smoke.
Record terminal errors, selected and qualified first-prefix kinds, attempted
and returned calls, and the existing four-start 32-evaluation gradient control.
Nullspace expansion adds calls and candidates; this is not an equal-budget
comparison or independent test. Preserve negative results. Test the unicycle
turn direction as well as wide, tall, and zero-rank matrices and boundedness.
No neural model is used in this diagnostic.
