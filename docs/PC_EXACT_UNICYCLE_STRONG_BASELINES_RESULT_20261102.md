# Stronger exact-unicycle development controls

These controls reuse the original 64 deterministic angular goals after the
pilot results were observed. The angle grid is not 64 independent physical
tasks. The system is analytical, not a learned visual world model.

| Method | Mean terminal position error | Comparison against fixed five bridges |
| --- | ---: | --- |
| Fixed five turn-only bridges | 0.193885 | — |
| Five uniform random turns, same response/transition counts | 0.255768 | fixed wins 37, random wins 27 |
| Multistart bounded gradient | 0.004245 | gradient wins 62, fixed wins 2 |
| Task-specific analytical witness | effectively 0 | reaches every goal |

The analytical witness uses first velocity equal to goal x, turns by a
goal-sign-dependent right angle, then moves by the absolute goal y. Its
maximum observed error was 6.25e-17. Unlike the tested reconfiguration
bridge, this witness generally makes immediate position progress; its
purpose is an exact task-specific solvability ceiling, not evidence of a
non-greedy mechanism or a generic planner.

The original one-step greedy implementation breaks equally good angular
choices by always taking zero turn. This specific tie-break discards future
control choices even though turning simultaneously with immediate progress
is allowed by the dynamics. Thus the earlier five-bridge victory over this
greedy must not be read as evidence that non-greedy turning is necessary.
The earlier common-suffix mechanism witness remains mathematically valid for
its specified action pair; the new controls show that it does not establish
a planning advantage on this task.

The random bridge and fixed bridge each use five responses, ten transitions
and five objective evaluations. The gradient planner uses 29–32 closed-form
objective and analytical gradient evaluations per goal, four fixed starts
and at most eight evaluations per start. These different operation classes
are not converted to a common cost and wall time was not certified. The
analytic oracle has task-specific privileged dynamics and is not a fair
performance baseline for a general learned-model planner.

Seven tests passed, including finite-difference verification of the
gradient, action feasibility and exact oracle witnesses. These findings do
not prove the visual-model reconfiguration mechanism false. They do show
that this particular exact-dynamics pilot is too easy to support the claimed
advantage over competent full-action planning; a task with genuinely
necessary reconfiguration and strong matched baselines is still needed.

Artifacts:

- `le-wm-main/results/pc_exact_unicycle_strong_baseline_20261102_v1/report.json`
- `le-wm-main/results/pc_exact_unicycle_random_bridge_20261102_v1/report.json`
- `docs/PC_EXACT_UNICYCLE_STRONG_BASELINE_PROTOCOL_20261102.md`
- `docs/PC_EXACT_UNICYCLE_RANDOM_BRIDGE_PROTOCOL_20261102.md`
