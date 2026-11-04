# v22 fixed-terminal Reacher RGB wiring diagnostic

The preregistered development task completed three real RGB feedback
decisions for each arm. v22 and shared-bank multistart CEM used remaining
horizons 15/10/5 and executed five actions per decision. Their measured
physical goal distances were:

| Arm | Initial | After 5 | After 10 | After 15 |
| --- | ---: | ---: | ---: | ---: |
| v22 | 1.297960 | 0.725441 | 0.013082 | 0.011497 |
| multistart CEM | 1.297960 | 1.044913 | 0.761594 | 0.048464 |

Saved-data audits verified decision-before-dispatch binding, action
bounds, packet accounting, old-tail rescoring, shared initial observation,
and final paired choice. The largest selected-score minus lowest score
over all saved full-horizon batches was about `5.59e-9`; final-pair
selection was checked separately. Same-backend replay reproduced the
saved images, action history and physical distances for both arms.
An additional saved-trace reconstruction checked all four v22 refinement
starts at each decision: they used respectively 3, 12 and 38 generations
at horizons 15, 10 and 5, with updated elite distributions entering the
next generation. Each decision spent 1,920 prediction tokens.

CUDA preflight initially rejected an unspecified `CUDA_VISIBLE_DEVICES`
before any output artifacts or task execution. Binding the declared GPU
UUID allowed the single permitted attempt per arm. Artifacts are in
`le-wm-main/results/pc_v22_timed_feedback_reacher_wiring_20261104_codex/`.

This is one development root with unknown provenance, not independent
replication, isolation certification, or a PC3/PC4 performance result.
Physical state was evaluator-only, but planner and evaluator shared a
process. Prediction tokens do not measure total compute; encoding and
synchronized planning timing are recorded separately, excluding storage.
