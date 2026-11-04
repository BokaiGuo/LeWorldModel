# Post-seal moving-history physical diagnostic

After all model choices and selected RGB executions were sealed, replay
the same 15-action warmup and every one of the 44 model-optimized programs
on the official native MuJoCo 2.1.2 evaluator. Each of these is exactly
five prefix plus ten model suffix actions. The model selector must never
consume this report. Separately, for each root apply six identical
ten-action suffixes (zero and five uniform random sequences, fixed NumPy
seed `2026110350`) after each of selected, zero and `random_0` prefixes;
all prefixes come from the sealed action bank. If selected equals a
reference, retain the duplicate role. Record prefix state, terminal state,
absolute position distance, relative change against own zero suffix,
all physical transitions, and success only as the pre-existing <=0.25
position-distance diagnostic.

Compare the two fixed history modes' saved costs against the same 44
physical endpoints: mean absolute proxy-distance difference, 55 within-root
pair ordering reversals with ties explicitly excluded, and selected index.
This compares fixed candidate chains; it does not reoptimize either arm.
The common-suffix physical probes and 44 evaluator replays are diagnostic
costs and are not planning costs or available to the controller.

Do not report this as native-success or mechanism confirmation: this panel
is exposed, prefixes are non-spectral, the executed selected suffix is
open-loop, the same-budget independent cp5 replanning and strong closed-loop
baselines are not yet run, and simulator training membership is unknown.
Report exact call counts, elapsed time and whether a full cost audit exists.
