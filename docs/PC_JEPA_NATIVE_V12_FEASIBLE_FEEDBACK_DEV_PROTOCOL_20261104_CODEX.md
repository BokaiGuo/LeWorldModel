# JEPA v12 constructed-feasible feedback: development protocol

Test v12's same compiler and remainder-batch repair through three decisions
of native PushT physical feedback, not only E0. This is one newly constructed
development root, not the stopped random-goal experiment or an independent
benchmark. It cannot establish a success rate, PC2 mechanism or PC3–PC5.

Using the existing v11 native runtime, checkpoint, normalizers, ten-command
warmup, GPU lock and isolated worker input rules, reset with seed `2026110412`.
Construct the target by executing exactly 15 legal actions from
`default_rng(2026110413).uniform(-.45,.45,(15,2))` after the warmup. Replay
the entire witness from reset and require exact RGB, proprioception and state
agreement at all boundaries. Witness actions and evaluation-only physical
state must never enter the planner input; only the goal RGB may enter.

Run v12 response and the same zero/carry multistart CEM baseline once each
from identical fresh reset and warmup. Execute five actions after each sealed
15/10/5-step decision, then update legal RGB, proprioception, history and
carry. Each decision has at most 1920 five-step prediction tokens. Fix the
seed, do not replace failures or repeat based on outcomes, and retain every
partial run. Time, encoding and SVD costs are reported separately; equal token
caps do not imply equal computation.

Audit the witness, model/worker/compiled-version binding, hashes, action and
RGB history, normalized packet inputs, carry, incumbent and final pair,
dispatch and per-decision ledger against saved data. Independently replay
physical commands only if the dedicated replay checks are available; neither
saved-data checks nor same-backend replay establishes independent scientific
replication. Record actual physical distance and success; do not infer that
the planner sees physical truth or that random goals are feasible.
