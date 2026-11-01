# P1 total-budget host

`p1_total_budget_host_20261101_codex.py` is a new opt-in host. It does not
modify sealed runners or saved results.

The host freezes a complete per-root, per-arm reservation before model load.
It binds source hashes and root/task/goal/model/normalizer/hardware/method/
controller identities, then routes model load, RGB encoding, action encoding,
prediction, backward, Jacobian, SVD, solve, candidate evaluation, replay,
feedback and physical callbacks through durable fail-stop journals. Attempts
are charged before callbacks and are never refunded. Physical resources use a
second source-bound vector and journal.

Structured protocols must reserve positive model, encoding, prediction,
backward, Jacobian, SVD, solve, candidate and physical resources. CEM protocols
may reserve zero backward/Jacobian/SVD/solve, but must reserve candidate and
model/physical resources. Any omitted required axis, source drift, unknown
operation, cap violation, missing neural module hook or lifecycle error fails
closed.

The returned status `ready_receipt_unverified_cost` means only that every
non-conditional positive reservation axis was observed and both journals
closed without a detected violation. It does not certify FLOPs, energy,
wall-time fairness, root independence, or P1 admission. Conditional axes must
be declared before execution to support native early termination; unused
budget remains visible in the receipt.

Known uncovered bypass classes are emitted in every protocol and receipt:

- callbacks can execute uninstrumented Python or CUDA outside the host API;
- functional autograd and custom CUDA kernels are not globally intercepted;
- environment calls outside the physical guard are not process-sandboxed;
- wall-time, FLOPs, energy and GPU contention are not certified;
- external precommitment timing and independent-root status are not authenticated.

CPU contract tests exercise complete structured and CEM routes, pre-dispatch
budget rejection, missing-hook rejection, source drift, unknown operations,
incomplete coverage and explicit early-stop conditional axes. They do not run
the frozen model, CUDA, or the physical environment.
