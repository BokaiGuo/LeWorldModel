# P3 queued standalone-cost exploration

This new batch retains the earlier failed batch and its no-retry rule unchanged.
It measures all six slots afresh, in this fixed order: 20/uniform, 20/signal,
20/reverse, 30/uniform, 30/signal, 30/reverse. Known roots and previously
observed outcomes make this exploratory, not preregistration or independent
confirmation. A failure terminates this batch without retry.

Each slot is a separate host process with its own model load and CUDA context.
It blocks on the existing single-inode cooperative lock before CUDA initialization,
holds it through child cleanup, and records lock wait separately. A queued attempt
is not a result. Process wall includes queue time, imports and verification;
child and phase durations have different boundaries and must not be summed as
independent costs. The lock governs cooperating processes on this host only.

Per slot, the four forward-operation axes each have cap 2*point+8; metric cap
is point+4; Jacobian and SVD caps are 2 each; VJP directions, gradient API and
backward hooks each have cap 384. Signal and reverse independently measure the
online structural signal; uniform does not perform artificial shadow work.
Actual nested counts are reported separately and cannot be added as FLOPs.
Wall time is measured, not forced equal. Model loading I/O, context and model
loading times and CUDA memory peaks are separate observations. Cache state,
complete dynamic source closure, external interference, FLOPs and energy remain
uncontrolled or unknown. None of this changes the scientific gate or proves
equal total compute.
