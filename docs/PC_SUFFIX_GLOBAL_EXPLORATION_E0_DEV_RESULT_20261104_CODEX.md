# Global-exploration suffix optimizer: E0 development result

All four preregistered exact-Wall roots had legal privileged witnesses,
and each optimizer received the same witness prefix and three 120-sample
full-action suffix starts. The existing suffix CEM found a successful
continuation on 1/4 roots; the new 25%-uniform exploration optimizer did
so on 0/4. Per-root best distances were:

| Root | Existing CEM | Global exploration |
| --- | ---: | ---: |
| 0 | 0.026289 | 0.061133 |
| 1 | 0.509116 | 0.522288 |
| 2 | 0.340147 | 0.206104 |
| 3 | 0.502302 | 0.508797 |

Each arm spent 1,080 prediction tokens per eligible root, plus one
shared three-token zero-suffix evaluation. The new optimizer does not show
a success advantage in this small development diagnostic. The implementations
also differ in whether elite updates use pre-clipped parameters or executed
bounded actions, so this is not a single-factor uniform-sampling ablation.

All action batches were replayed in alternate same-environment geometry.
A separate audit reconstructed all 120 seeded new-optimizer generations,
including actual global and local samples and elite distribution updates.
The panel has reached its stopping rule; no roots were replaced. Witness
prefixes are privileged and these results cannot establish PC2/PC3
performance. The new optimizer remains an experimental component and is
not promoted into v22.

Artifacts: `le-wm-main/results/pc_suffix_global_exploration_e0_dev_20261104_codex/`.
