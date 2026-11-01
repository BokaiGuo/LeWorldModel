# P3 fixed-wall delivery exploration

Before this new run, fix the six slots in order 20/uniform, 20/signal,
20/reverse, 30/uniform, 30/signal, 30/reverse. The same 30-second monotonic
deadline starts before launching each fresh host process. It includes process
startup, source checks, cooperative lock queueing, CUDA initialization, model
load, signal calculation, search, journal and child report publication. A
completed process must return by the deadline and then pass the existing
separate saved consumer. Consumer time is outside the delivery budget.

The budget is a rounded engineering choice informed by the preceding known-root
batch (maximum observed process wall approximately 22.72 seconds); it is not
preregistered or an independent confirmation. The policies, candidate banks and
per-axis caps remain those of the sealed host prepare. In particular uniform
evaluates four more candidates, while signal/reverse independently pay the
actual signal cost. No claim of equal total FLOPs, energy, or scientific gate
pass follows from a common wall cap.

Each slot has one attempt; do not retry failure or reuse previous results.
Deadline failures terminate the entire child process group, retain partial files,
and are recorded as not delivered. Deadline enforcement is best effort at OS
scheduling granularity, and reports measure observed wall, not invented values.
The remaining slots run in order even if one fails. No physical execution is
authorized by this contract. Lock wait is included and the existing lock governs
only cooperating local processes. Cache, external contention and complete
dynamic source closure remain uncontrolled or unknown.
