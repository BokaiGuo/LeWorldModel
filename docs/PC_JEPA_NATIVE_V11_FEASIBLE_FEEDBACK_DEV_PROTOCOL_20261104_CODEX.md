# JEPA v11 feasible-goal feedback interface diagnostic

This is one constructed, exposed development task, not the stopped random-goal
root and not a benchmark sample or confirmation. Its only hypothesis is that
the same v11 compiler can process real native observations, execute actions,
and recompile with an intact prior tail and reconciled packet ledger. It cannot
establish the success rate or difficulty of the native random-goal task.

Before planning, reset native PushT with seed 2026110402, execute the original
fixed ten-command warmup, then execute exactly 15 commands sampled once from
`default_rng(2026110403).uniform(-0.45, 0.45, (15, 2))`, rounded to float32.
The terminal RGB and physical state define the target. Record the full witness,
the warmup and its hashes, and replay the witness from reset before planning;
every boundary's RGB, proprio and state must match exactly and terminal success
must hold. This construction is an evaluation-only existence witness, not a
planner candidate. The model worker sees only input context and goal RGB; its
allowed code does not read the witness or physical evaluator output.

Run v11 response and the existing zero/carry multistart CEM, each from a fresh
identical reset and warmup. Three decisions per arm use 15/10/5 remaining
physical commands, execute five per decision, and a cap of 1920 prediction
tokens per decision. No additions, retries, adaptation or seed selection based
on outcome. Failure, exception or timeout is recorded and not replaced.
Source identities and all other frozen model/physics settings follow the
existing v7 development collector. This constructed task is not evidence of
the random goal's feasibility, independent checkpoint provenance, fair strong
baselines, or any PC gate. Unknown source independence stays unknown.

Exclusive results record goal/witness separately from planner input; every
decision is sealed before dispatch and its carry is passed at the next real
observation. The evaluator audits history, decision and command bindings,
packet counts, actual physical metrics, and original source hashes. GPU uses
the existing cooperative lock and isolated worker. A saved-data audit is not
an independent physical or neural reproduction.
