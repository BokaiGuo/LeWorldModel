# v22 fixed-terminal Reacher RGB wiring diagnostic

Before execution, freeze this protocol and the runner dependencies. Run one
attempt per arm, v22 and the shared-bank multistart CEM, on Reacher seed
20263519, target delta `(0.45, -0.50)`, using the frozen checkpoint and
normalizer and executed dataset warmup from the existing adapter preflight.
This is a development integration diagnostic, not a PC3/PC4 performance
test; development, normalizer and checkpoint-training independence are
unknown. Do not substitute a root or retry an arm after a failure.

Require the declared CUDA GPU and EGL renderer before creating artifacts.
Both arms start from the same reset, action warmup and image context, and
use only RGB, goal RGB and action history. Replan with 1,920 prediction
tokens at each remaining horizon 15/10/5; seal the complete decision and
all prediction batches before dispatching five actions. Re-score the
previous selected tail under each new observation and retain the best
complete scored incumbent. Save observations, actions, traces, resource
ledgers, timestamps, physical evaluation and dependency hashes. Separate
encoding, prediction and synchronized planning time, and report that
serialization is excluded and that prediction tokens are not total cost.

Audit saved context, action bounds, packet timing, carry, selection,
decision-before-dispatch and shared-backend physical replay. Physical
state is evaluator-only; shared-process separation is not an independent
information-isolation certification. Stop after both attempts and report
failures in the denominator. No success, significance or gate conclusion
follows from this single root.
