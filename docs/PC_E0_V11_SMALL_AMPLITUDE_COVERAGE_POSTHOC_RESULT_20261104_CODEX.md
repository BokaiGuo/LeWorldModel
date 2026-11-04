# E0 v11 amplitude panel: post-hoc scored-chain coverage

All previously saved actions were independently replayed after the fixed panel
stopped. The table counts repeated scored **15-step trajectories**, not
independent roots or unique plans.

| Arm | Full chains scored | Ever x > 0 | Roots with crossing | Best distance among crossings | Successful chains |
|---|---:|---:|---:|---:|---:|
| v11 small amplitudes | 3,836 | 1 | 1/8 | 1.218 | 0 |
| v11 original amplitudes | 3,836 | 4 | 3/8 | 1.213 | 0 |
| Three-start CEM | 5,120 | 27 | 7/8 | 0.727 | 0 |

None of the 24 ultimately selected chains crossed. The selected chain was
nevertheless checked against every scored full-chain score in the source
audit; no scored crossing chain reached the success threshold. Thus, on this
particular panel and scored bank, evidence favors inadequate generation or
optimization of successful complete chains over a final selection discarding
an already scored successful chain. It does not isolate candidate generation
from pruning or suffix optimization, and does not establish why the searches
failed. Counting crossings alone does not establish task success or mechanism.

Source trace hashes, terminal positions and scores were verified in the
independent line-segment replay. The analysis remains post-hoc, development
only, and same-environment: no gate or independent replication is claimed.

Protocol:
`docs/PC_E0_V11_SMALL_AMPLITUDE_COVERAGE_POSTHOC_PROTOCOL_20261104_CODEX.md`.
Full root- and horizon-level counts:
`le-wm-main/results/pc_e0_v11_small_amplitude_dev_20261104_codex/coverage_posthoc.json`.
