# Successor-response retention: fixed-bank development result

Four exact-Wall tasks and all ten v13-generated first-layer candidates per
task were evaluated. Each prefix received three 120-sample full-action
suffix CEM starts; five selection rules were evaluated on the same bank.

| Root | All-bank best | Greedy | State diversity | Response/ridge | Lookahead | Random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.500144 | 0.500402 | 0.500402 | 0.500144 | 0.500402 | 0.500779 |
| 1 | 0.500277 | 0.500403 | 0.500403 | 0.500403 | 0.500403 | 0.500403 |
| 2 | 0.500306 | 0.500306 | 0.500306 | 0.500306 | 0.500723 | 0.500723 |
| 3 | 0.501865 | 0.502914 | 0.502914 | 0.502914 | 0.502914 | 0.502733 |

No rule, nor the diagnostic all-bank best, reached the fixed `<0.05`
success threshold on any root. On root 0 the response/ridge rule kept
the best sampled continuation, while greedy, state and lookahead did not;
the numerical distance differences are small and no practically
meaningful threshold was predeclared. These results do not show that
successful continuations do not exist or that stronger optimization would
not find them. They do not establish a response-retention advantage.

Per root the compiler used 1,920 prediction tokens; all-prefix successor
response probes used 1,230, random-lookahead probes 720, and common suffix
search 10,800. This diagnostic evaluates every candidate and is not an
equal-total-cost method comparison. Raw actions, endpoint scores, selected
indices and optimization traces were retained. The audit replayed saved
trajectories with alternate same-environment geometry, recomputed responses,
selectors and incumbent scores, and checked prediction counts; this is not
independent physical replication. The separate explicit probe-input audit,
if present, is recorded alongside the data.

After the plans and audits were sealed, the separately prescribed
privileged feasibility diagnostic found two legal witnesses per task
(8/8), each within the original 15-step horizon. Both exact dynamics and
the alternate same-environment geometry agreed. This establishes
feasibility for these four tasks, not that any selector or planner found
the witnesses. Witness actions were not used in candidate generation.
See `PC_RESPONSE_RETENTION_FEASIBILITY_POSTHOC_PROTOCOL_20261104_CODEX.md`
and `le-wm-main/results/pc_response_retention_feasibility_posthoc_20261104_codex.json`.

The four-root stopping rule has been reached. Do not repeat, replace or
tune the panel. Broader provenance remains unknown. PC2, PC3 and visual
control are not established.

Protocol: `PC_RESPONSE_RETENTION_FIXEDBANK_E0_DEV_PROTOCOL_20261104_CODEX.md`.
Results, traces and audit:
`le-wm-main/results/pc_response_retention_fixedbank_e0_dev_20261104_codex/`.
