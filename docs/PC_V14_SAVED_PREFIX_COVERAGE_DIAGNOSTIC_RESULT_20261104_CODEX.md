# Saved v14-panel prefix coverage: post-hoc diagnostic

The fixed analysis compared the saved same-bank CEM selected actions with
the saved v13/v14 response candidates and retention indices. It performed no
planning, prediction, or environment rollout. The source result, manifest,
and prior audit hashes were checked; all eight root/task pairs and all three
layers per response arm were present. The numerical record contains 48
root/arm/layer rows.

| Response arm | Prefix steps | Mean nearest generated RMS | Mean nearest retained RMS | Nearest candidate retained |
| --- | ---: | ---: | ---: | ---: |
| v13 | 5 | 0.352 | 0.366 | 5/8 |
| v13 | 10 | 0.442 | 0.497 | 1/8 |
| v13 | 15 | 0.492 | 0.515 | 4/8 |
| v14 | 5 | 0.161 | 0.223 | 0/8 |
| v14 | 10 | 0.206 | 0.261 | 0/8 |
| v14 | 15 | 0.248 | 0.334 | 1/8 |

RMS is over raw action coordinates and time, not terminal-state distance.
The table does not imply that v14 improves task performance: its
carry-conditioned nominal is derived from the shared global bank and its
recorded task outcome did not improve on v13. The CEM final solution is
post-hoc and is not an input to either method.

Among the two CEM-success roots, v14's nearest generated/retained RMS was
respectively 0.179/0.200 and 0.095/0.192 at five steps, and 0.226/0.229
and 0.154/0.231 at ten steps. None of these four nearest candidates was
retained. The retained minimum can still be close to the generated minimum;
selection may remove one near-tie without materially affecting coverage.
At 15 steps the layers are full chains, not prefixes with a remaining
optimization opportunity.

This identifies an observable retention/coverage question worth testing
under a new fixed-bank, common-suffix protocol. It does **not** establish
that any of the discarded candidates has a useful continuation, that
Euclidean action proximity preserves physical behavior across collisions,
or that pruning caused either method's failure. No scientific gate changes.
The eight-root panel remains stopped; this analysis neither extends nor
repeats it.

Protocol: `PC_V14_SAVED_PREFIX_COVERAGE_DIAGNOSTIC_PROTOCOL_20261104_CODEX.md`.
Script: `le-wm-main/experiments/pc_v14_saved_prefix_coverage_20261104_codex.py`.
Raw output and dependency hashes:
`le-wm-main/results/pc_v14_saved_prefix_coverage_20261104_codex.json`.
