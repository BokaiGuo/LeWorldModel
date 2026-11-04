# Score-augmented response: Reacher RGB feedback development protocol

This is a new score-geometry ablation, not a retry or replacement of the
stopped v13 Reacher panel. It compares the same frozen model and original
v13 generator with and without the additional legal model-score response,
plus matched global-bank full-action CEM. It is one exposed development
task, not independent PC3/PC4 confirmation.

Before any manifest or physical task context, check the declared CUDA
device and EGL Reacher render. If checks fail, stop without creating a
run. Use environment seed `20263289`, goal delta `(0.45,-0.50)`, legal raw
actions `[-1,1]^2`, common dataset-derived executed warmup, and a 15-step
fixed endpoint. Broader historical, normalizer and training provenance
are unknown. Feasibility within this horizon is not assumed.

Run the three arms once each from identical reset and warmup: `v13`,
`v17`, and `global_cem`. All receive the same goal image, legal RGB/history
interface and per-decision 1,920-token cap. They execute 5 actions and
replan at horizons 15/10/5 using their own real RGB history and their own
old-tail carry. Their global bank generation and seeds match. CEM uses
per-step optimization and paired zero-prefix comparison as in the prior
Reacher runner. For both response arms use the same two-dimensional chart
scaled by `1/sqrt(latent dimension)`; v17 additionally includes the
frozen-model full latent MSE as a response feature with scale 1 and target
0. Final selection in every arm always uses full latent MSE. Score scale
is a prespecified heuristic, not a universally calibrated value.

Seal source/config/model/normalizer dependencies and each decision before
dispatch. Save raw predictions, scores, selected chains, carry, actual
dispatch, RGB/history contexts, model parameter hashes, evaluator-only
physical distances and resource counts/times. Check identical initial
contexts, legal actions, tokens, incumbent and carry bindings; replay
execution with the same backend where feasible. The physical evaluator is
in the same process, so this is not an isolation proof. Separate encoding,
prediction and planning times are not complete matched-cost certification.

Stop after one attempt per arm, including errors; no retries, replacements
or target tuning. A single task and model do not establish general
improvement or mechanism. No ground-truth physical state or witness is
available to the planner.
