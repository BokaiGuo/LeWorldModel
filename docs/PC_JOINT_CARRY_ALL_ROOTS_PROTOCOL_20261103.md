# Joint-plan carry on all exposed development branches

This extends the previously run root-8349-only carry/zero-tail follow-up,
using all existing joint-gradient source branches on the four exposed roots.
The roster is the complete saved roster, in order: 4+4+5+4 = 17 branches.
Retain duplicate actions and the source decision's `qualified=None` for
other roots; do not manufacture a nonprogress candidate. Roots and outcomes
have already been exposed, so this is development, not independent
confirmation or a gate.

For every saved branch bind its source `r{root}_plans[iteration,candidate]`
to its saved executed five-action prefix: controls use candidate 0/1 with
their saved `control_iterations`; raw/spectral winners and raw/spectral
qualified use their respective stored decision indices. Fail on any mismatch
or missing/unexpected role. Reconstruct the actual cp7 RGB and causal state
from saved history. Before planning, independently replay the seven-command
prelude/prefix on the active native renderer and require all eight RGB frames
to match the source history byte-for-byte; if not, stop the complete run
without substituting any input or opening partial results.

Each branch gets the same 50-sample packet CEM, fixed seed `2026110380+root`,
bounds and task score, with two separate initializations: raw zero and that
branch's own old joint-program tail. They use identical seeded noise and
real cp7 RGB context. Compare prior and final mean by the model score as in
the previously used frozen kernel; save all scored candidates and scores,
initial scores, selected tail and decisions. Physically execute all three
fixed variants per branch: source old joint tail, zero-init CEM and
carry-init CEM. Recheck the whole RGB prefix on every execution. The
zero-init trajectory must exactly reproduce its saved source trajectory,
otherwise fail. Save RGB, actions, and evaluation-only position replays.

Report source roles and per-root cp2/cp7/cp17 position distances, model
scores and actual cost counters, including all encodes, CEM samples, RGB
commands and position replays. Include duplicate actions, and separate
qualified from unconditional results. Position-only replay is not a direct
native RGB executor measurement and model training provenance is not
certified. With differing joint-plan prior information and inherited search
cost, equal 50-sample suffix budgets do not establish equal total compute.
This experiment evaluates feedback initialization, not program mechanism,
and cannot establish P1 without further independent mechanism and baselines.
