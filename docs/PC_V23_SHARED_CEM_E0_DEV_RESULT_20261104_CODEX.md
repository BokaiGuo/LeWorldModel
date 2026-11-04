# v23 shared-optimizer exact-Wall development result

All 16 planned runs completed, and the saved-action audit passed for the
common bank, token accounting, selected complete chain and alternate
same-environment geometry. The response compiler succeeded on **0/8**
tasks and the shared-optimizer CEM baseline on **1/8**, at the prespecified
terminal-distance threshold of 0.05. Compiler token use was 1918–1920;
baseline use was 1920 per root.

| Root | v23 distance | shared CEM distance |
| --- | ---: | ---: |
| 0 | 0.500105 | 0.501719 |
| 1 | 0.509190 | 0.504342 |
| 2 | 0.503988 | 0.053321 |
| 3 | 0.503234 | 0.350739 |
| 4 | 0.327006 | 0.046084 |
| 5 | 0.507132 | 0.501052 |
| 6 | 0.511613 | 0.081377 |
| 7 | 0.500105 | 0.500627 |

The response compiler evaluated one successor-reachability probe per root.
This panel does not show a response-compiler advantage. The comparison
controls optimizer implementation and total prediction tokens, but not
how the two methods allocate them; it cannot identify the effect of spectral
directions. The geometry check uses the same environment, and source
independence remains unknown. No PC2/PC3 gate is passed by this diagnostic.
Do not repeat or tune on this stopped panel.

Raw traces, manifest and audit are in
`le-wm-main/results/pc_v23_shared_cem_e0_dev_20261104_codex/`.
