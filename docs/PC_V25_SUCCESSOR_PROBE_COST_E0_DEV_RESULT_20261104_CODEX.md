# v25 successor-probe cost ablation: E0 development result

All 16 paired runs on eight preregistered development roots completed.
Recorded prediction calls, actual actions, token totals, selection,
alternate same-environment geometry, and deterministic optimizer replay
passed their audits.

| Root | v25 response | No successor probes | Response minus no probes |
| --- | ---: | ---: | ---: |
| 0 | 0.403809 | 0.403809 | 0.000000 |
| 1 | 0.512470 | 0.509677 | 0.002793 |
| 2 | 0.316826 | 0.268531 | 0.048295 |
| 3 | 0.506817 | 0.500647 | 0.006170 |
| 4 | 0.235892 | 0.235892 | 0.000000 |
| 5 | 0.503016 | 0.504491 | -0.001475 |
| 6 | 0.183213 | 0.145554 | 0.037659 |
| 7 | 0.502348 | 0.507003 | -0.004655 |

The response arm had lower distance on two roots, equal distance on two
and higher distance on four; the median paired difference was +0.001396.
Both arms were 0/8 under the existing `<0.05` success rule. Seven pairs
used exactly 1,920 prediction tokens in both arms; on root 1 the no-probe
arm used 1,918. The response arm made six successor probes per root;
the ablation made none. The common optimizer used 104–120 refinement
samples with probes and 150–166 without. Both retained local response
generation and made 18 SVDs per root.

This compares the combined effect of successor probes, their retention
policy and their use of budget, including complete-chain probe incumbents;
it is not a pure directional or all-response ablation. Its exposed roots
have unknown broader provenance, and 1,920 prediction tokens do not
equate to matched visual, GPU, memory or overall runtime. The evidence
does not establish a positive net benefit from successor probes in this
protocol, but does not establish that response-based planning is
impossible. Neither PC2 nor PC3 is passed.

Protocol: `PC_V25_SUCCESSOR_PROBE_COST_E0_DEV_PROTOCOL_20261104_CODEX.md`.
Raw results, call journals, `audit.json`, and same-code
`optimizer_replay_audit.json` are in
`le-wm-main/results/pc_v25_successor_probe_cost_e0_dev_20261104_codex/`.
The panel is stopped; do not tune or rerun it.
