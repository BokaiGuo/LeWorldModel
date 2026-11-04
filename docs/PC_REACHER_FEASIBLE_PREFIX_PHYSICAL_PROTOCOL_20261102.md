# Physical evaluation of all fixed feasible prefixes: development protocol

Use the frozen two-root model-only report
`reacher_pc_feasible_prefix_development_20261102_v1/report.json` as the sole
candidate source, bound by SHA-256. For each root execute every appended
candidate in bank order, plus selected, zero, and the same-bank immediate
greedy control. No candidate is chosen using physical outcomes. Reconstruct
the exact root and goal RGB and warmup history from the report's root source;
verify model, stats, and parameter hashes, selected parameters, candidate
index rows, history index, and root and goal image hashes before execution.

Use the existing `_two_stage` execution path for all arms: five physical
steps of the fixed prefix, obtain that arm's actual RGB and executed action
history, then use identical 16-sample, two-iteration, four-elite CEM to plan
two further five-step packets and execute them. Record all physical state
and stop/censoring information emitted by that path, plus the selected
prefix, suffix, and invocation costs. Physical state is evaluator-only.
All roots are already exposed and selected/zero/greedy arms repeat earlier
actions; those repeats do not create independent samples.

This is a candidate/selector development diagnosis, not a common-suffix
mechanism test or equal-total-cost method comparison. No outcome-ranked
candidate may be described as the prior model-selected method. Incomplete or
failed runs must remain failures, with no replacement roots or omitted arms.
