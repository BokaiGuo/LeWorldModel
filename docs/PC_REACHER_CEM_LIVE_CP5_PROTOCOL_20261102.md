# Live RGB cp5 replanning on exposed Reacher development roots

Use the previously selected prefix 12 and neutral prefix 0 on each of the
two already opened roots, with the frozen checkpoint and the existing
two-stage executor. Each arm executes its fixed five-step prefix, captures
actual successor RGB and actual action history, runs CEM with 32 samples,
three iterations, four elites and seed `root + 5000`, and executes the
ten-step tail. The executor additionally runs a five-step neutral-prefix
diagnostic for every arm before compiling the suffix; this cost is recorded
and must not be hidden or mistaken for the main action chain.

Run on host CUDA with the cooperative lock and host EGL. Save the model hash,
actual RGB contexts, executed action intents and returns, selected suffix,
planner counts and physical metrics. Validate those fields against the
sealed setup and compare with the corresponding previously saved 32-by-3
observed-context CEM and physical replay. The earlier outcomes were already
opened, so this is a development integration and replay check, not a blind
confirmation or evidence of superiority. Planned four arms, no outcome-based
selection. No additional policy, critic or model training; simulator state
is only used for restoration and evaluation.

The CEM receipt reports 96 candidate rollout evaluations but the full
encoding, model predictor, control and wall-time costs remain separately
uncertified. Do not claim equal total compute, independent generalization,
or P1 admission. Keep errors and non-completion in the journal.
