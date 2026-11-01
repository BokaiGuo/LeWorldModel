# P3 rank allocation fixed-horizon exploration

Post-observation engineering contrast, not independent confirmation. Fixed
two known roots, common root-specific candidate banks in increasing index,
points 20 and 30 and arms uniform, signal, reverse. Each arm is separately
executed and independently accounted. Uniform spends no signal cost and
searches 12/12 or 17/17 candidates. Signal and reverse independently compute
both root signals. For unequal signals, signal allocates ceil(3*point/4)
to the higher-ranked root and the remainder to the other; reverse swaps.
Exact equality gives equal allocation and is explicitly indistinguishable.
No signal magnitude threshold or outcome selection is used.

This contrast deliberately defines fixed-horizon search; it does not use
the separate patience-5 early-stop algorithm, does not execute dummy calls,
and does not impute absent work. Signal/reverse must differ in candidate
sets when their signals are unequal and both complete; winners can still
coincide. All attempted candidates have actual model cost evaluated. Separate
neural axes, SVD, Jacobian, VJP and scheduler costs retain the existing per-arm
vector caps. Do not sum nested axes or claim equal total cost. Failures and
timeouts remain in the fixed denominator; no retry.

Six independent slots; 30 seconds per process including launch and lock wait,
consumer outside deadline. Frozen model, host CUDA, project cooperative lock;
no training, no physical execution, and no gate changes. Known roots and
earlier outcomes were observed before this protocol; dynamic source closure
and global GPU isolation are not certified.
