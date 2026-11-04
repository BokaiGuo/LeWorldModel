# Task-region spectral CEM initialization: physical development comparison

On previously exposed development roots 1884–1887, compare a common bounded
CEM optimizer with (A) Gaussian initial samples and (B) the same samples
except that four first-generation two-packet proposals are replaced with
task-region spectral prefixes. Compute the task-region Jacobian and mask from
initial and goal RGB exactly as in
`PC_POINTMAZE_RGB_TASK_SPECTRAL_DEV_PROTOCOL_20261102.md`. In arm B inject
prefixes `+0.3v`, `-0.3v`, `+0.6v`, `-0.6v`, each with zero second packet.
This tests initialization of a common optimizer, not reconfiguration programs.

Both arms optimize a two-packet, ten-action horizon for three generations of
16 samples each. The initial Gaussian has zero mean and 0.5 standard
deviation in physical action units; clip samples to [-1,1]. Update mean and
population standard deviation from the best four clipped samples, with
standard deviation floored at 0.1; select the best evaluated sample across
all generations, deterministic lowest-index ties. Use a fixed per-root
NumPy generator to pre-generate identical Gaussian innovations for both
arms. Score the terminal full spatial latent MSE restricted to the
RGB-derived task-region mask. Execute the selected first packet, capture
real RGB at step 5, recompute the mask and use the same optimizer for a
one-packet, 3-by-16 tail search. Both arms use the same separate tail random
innovations, although their real feedback contexts can differ. Execute each
tail to step 10 and evaluate offline position distance after both decisions.

Retain per-iteration physical samples, scores, selected plans, RGB hashes and
the action Jacobian. Use host CUDA for model calls and frozen parameters;
do not use physical labels to select. Each arm scores 48 two-block plans and
48 one-block tails (144 predicted block transitions), and gets one feedback
encoding and ten physical steps. Only the spectral arm incurs the two JVPs
and one SVD; report those and timing, and do not call the total computational
budget equal. This local small CEM is not an official implementation nor a
certified strong baseline. No zero/greedy/GRASP arms or formal mechanism
interventions are included here.

This is a development test on roots previously observed during method
development; history/normalizer/training provenance is not independently
certified, raw proprio is zero and the model gets one RGB frame. Position
distance is from offline replay, not native success. No reconfiguration,
independent generalization, fair-total-budget advantage or P1 admission can
be claimed from this run.
