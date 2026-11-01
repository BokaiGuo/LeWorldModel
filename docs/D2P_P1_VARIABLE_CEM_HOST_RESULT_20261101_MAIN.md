# Variable CEM host development results

The fixed, post-development design in
`D2P_P1_VARIABLE_CEM_HOST_20261101_EXCLUSIVE.md` was executed on host CUDA
and physically on the two known cp10-goal roots. At each planning boundary
its predictor/action-encoder dispatch cap was at most 195, with candidate
counts dependent on the remaining horizon. Structured's observed calls at
boundaries 0/5/10 were 174/192/183 in the existing development reports.
This is a common forward-call limit for the observed nodes, **not** a common
total-compute budget or a prospective confirmation.

| Known root | Variable CEM native success | Control steps | Predictor/action-encoder dispatch calls by boundary |
|---|---|---:|---|
| 0 | yes | 17 | 195, 194, 193, 193 |
| 1 | yes | 6 | 195, 194 |

The earlier structured arm also succeeded on these roots after 15 and 14
steps respectively. Different early stops, backward/Jacobian costs, selector
accounting and setup costs preclude an equal-compute ranking. Neither this
observed success nor these two non-independent tasks open P1.

Root 0 report:
`le-wm-main/results/codex_p1_variable_cem_host_20261101_exclusive_root0_RUN_MAIN01/report.json`;
SHA-256 `ec644cb8961d3e673edb2baa877bbe9faa530921fd9ce4f5889a197214333c68`.
Root 1 report:
`le-wm-main/results/codex_p1_variable_cem_host_20261101_exclusive_root1_RUN_MAIN01/report.json`;
SHA-256 `1c919ae38784d03822b948479593700f161d4ae7f71126d9e11daade7a4959d7`.
The runner checked candidate and rollout events before dispatch. The separate
saved-event audit `D2P_P1_VARIABLE_CEM_TWO_ROOT_SAVED_AUDIT_20261101_EXCLUSIVE.md`
recomputed saved action dispatch and the first native success at steps 17 and
6. The CEM event chain is internally consistent but its digest is not anchored
directly in the report. Node ledgers remain `closed_unverified`; their counts
are reserved upper bounds, not certified actual GPU costs. The audit did not
rerun model scores or the physical environment.
