# Timed feedback entry point: Reacher RGB wiring diagnostic

Run the new common timed feedback entry point once per arm, v17 and
four-start shared-bank CEM, on one new development Reacher task. This is
an implementation integration diagnostic, not a PC3/PC4 performance
test. Use environment seed 20263371, target delta `(0.45,-0.50)`, the
same frozen checkpoint and normalizer as the existing v17 Reacher
preflight, executed dataset warmup, and a fixed 15-action endpoint.
Broader task, normalizer and checkpoint training provenance remain unknown.

Before creating artifacts check the declared CUDA device and EGL renderer.
Each arm starts from the same reset, warmup and image context, receives
only RGB, goal RGB and action history, and replans with 1,920 prediction
tokens at remaining horizons 15/10/5. At each decision save the current
context, every prediction batch and a sealed decision before dispatching
the first five legal raw actions. Record and re-score the previously
selected tail under the new image context. Planning timing includes trace
capture but excludes serialization and visual encoding. Encoding,
prediction and planning timings are separate; GPU synchronization is
required. Evaluator-only physical measurements must not feed the planner.

Stop after one attempt per arm, including errors; do not retry or replace
the task. Seal dependencies, checkpoint and normalizer hashes, record
actual execution, and audit the saved observations, call ledger, carry and
dispatch chronology. Evaluation and planning share a process, so this is
not an isolation or source-independence certification.
