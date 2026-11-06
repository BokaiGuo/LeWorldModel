# v25 retention E0 development result

The preregistered eight-root development panel completed 40/40 runs and its
recorded-action, prediction, token and same-environment geometry audit
passed. All roots have unknown broader provenance; this is not independent
replication or a confirmation panel.

| Root | response | diverse | greedy | random | shared CEM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.500510 | 0.500510 | 0.500510 | 0.500510 | 0.500689 |
| 1 | 0.509434 | 0.506904 | 0.502736 | 0.510553 | 0.501889 |
| 2 | 0.507466 | 0.507466 | 0.502144 | 0.507466 | 0.503704 |
| 3 | 0.511662 | 0.511662 | 0.502573 | 0.511662 | 0.501394 |
| 4 | 0.501036 | 0.501036 | 0.500673 | 0.501036 | 0.500614 |
| 5 | 0.505249 | 0.505249 | 0.505249 | 0.505249 | 0.501830 |
| 6 | 0.502736 | 0.502736 | 0.502736 | 0.502736 | 0.024117 |
| 7 | 0.504788 | 0.504788 | 0.504788 | 0.504788 | 0.501401 |

Values are final distances; the prespecified success threshold is below
0.05. Each v25 arm succeeded on 0/8 roots and shared CEM on 1/8. Each v25
run used six to eight successor probes. In the response arm, response and
diversity retention sets differed on at least one same-bank layer for 3/8
roots. Response and diversity final distances differed on only one root,
where response was worse. This does not establish a causal effect across
the two full runs: their later candidate banks can diverge.

All four retention arms pay for response probes, including the alternatives
that do not use them; shared CEM does not. Hence the former isolate the
retention rule under this setup but are not a no-response-cost ablation.
Probes use packet-constant reduced action coordinates, not the full
per-step Jacobian. These data do not establish useful reconfiguration,
PC2 mechanism, or PC3 advantage. The fixed-bank equal-suffix diagnostic
was subsequently reported separately in
`PC_V25_FIXED_BANK_SUFFIX_E0_DEV_RESULT_20261104_CODEX.md`.

Post-hoc paired comparison at the same 1,920-token cap: response has a
lower final distance on 1/8 roots and a higher distance on 7/8 than shared
CEM; the median response-minus-CEM distance is 0.003591. These exposed
roots are not independent statistical confirmation. Response made 460–468
full-horizon sample evaluations and spent 516–540 tokens on shorter
horizons; shared CEM made 640 full-horizon evaluations. Full-horizon
counts include response probes and proposals, not just optimization.
Response also performed 18 SVDs per root. Prediction tokens do not account
for SVD, non-prediction computation, or GPU/encoding/feedback cost.
This panel provides no evidence of a cost-adjusted response advantage;
it does not isolate response's increment relative to the same search
with probes removed.

The panel is stopped; do not tune or rerun it.

Protocol: `PC_V25_RETENTION_E0_DEV_PROTOCOL_20261104_CODEX.md`.
Raw plans and audit: `le-wm-main/results/pc_v25_retention_e0_dev_20261104_codex/`.
