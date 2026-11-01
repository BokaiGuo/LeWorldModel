# Common-budget CEM root1: fixed development run

Before opening this run's outcome, reuse
`p1_common_budget_cem_host_20261101_leibniz` unchanged for its predefined
root1, its cp10 goal, seeds, horizon, early-stop rule and budget. Execute
one run only with new exclusive prepare and result paths. Do not replace the
root or alter settings based on success. Persist the prepare digest and all
host, model, node, event and physical receipts.

This root and goal were previously observed in development. The run tests the
new CEM replay/selection accounting on a second existing task; it is not
fresh-root confirmation or a total-compute comparison. A structured arm must
be separately bound to the exact same source and goal before a paired
comparison can be made. Model inference must use host CUDA, never CPU.

## Recorded run

The fixed run completed on host CUDA (RTX 5060 Ti) with unchanged model
parameter and buffer hashes. It executed six control steps and stopped on
native success; nodes at steps 10 and 15 were not started. Its report has
SHA-256 `b9f2d1a5a5b7653ec0ddaaebe2608bd72efd1a886d72441d4a4cb1b9de503479`.
The two completed node ledgers are `closed_unverified`: actual usage,
coverage and fairness are not certified, and upstream shared cost is unknown.
No matched structured run on this exact root and goal is established here.
This result is development evidence only, not independent confirmation or
P1 admission.

Independent saved-artifact checks recomputed the report hash and both the
physical and CEM journal hashes. All six executed actions exactly match the
prefix of the two sealed node command packets. The remaining four actions of
the second packet are censored by native early stopping; no later node was run.

## Subsequent structured development run

The separately prepared structured run
`le-wm-main/results/codex_p1_structured_root1_cp10_20261101_exclusive_RUN_CODEXROOT1DEV01/report.json`
completed on host CUDA and stopped on native success at control step 8. Its
saved action seal consumer passed. Direct comparison of the two reports
confirms equal root/goal binding, model hash, initial state and initial RGB
pixel hash. The CEM run stopped at step 6. These are two exploratory outcomes
on one known root, not a statistically meaningful ranking: the structured
prepare was constructed after the CEM result was visible, and neither total
compute nor ledger coverage is certified. The structured run's source, goal
and full receipt should be used for subsequent same-action mechanism work,
without treating the difference in stop steps as a P1 result.

Direct recomputation from the saved cp0 and cp5 physical states, using the
same goal, gives:

| Arm and step | Pusher XY error | Object XY error | Wrapped angle error | Joint XY error |
|---|---:|---:|---:|---:|
| Both, 0 | 55.6552 | 14.0746 | 0.1355 | 57.4073 |
| Structured, 5 | 28.9679 | 13.9500 | 0.1343 | 32.1519 |
| CEM, 5 | 26.9503 | 0.7543 | 0.0002 | 26.9609 |

Thus the structured prefix has a substantial immediate pusher and joint-XY
improvement, even though object errors change only slightly. Its small object
change does not establish absence of immediate improvement under the joint
or pusher metrics. These are descriptive calculations on one known root.
