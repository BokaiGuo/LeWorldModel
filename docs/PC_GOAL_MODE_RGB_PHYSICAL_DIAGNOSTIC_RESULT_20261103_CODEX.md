# Goal-mode RGB smoke: post-seal physical diagnostic

All three saved programs were replayed using MuJoCo 2.1.2.14, the fixed
previously used scene and goal `(1.8, 2.8)`. Hashes and selected action
binding passed. The selected program was executed by the RGB smoke runner
before this physical replay; the other two are post-selection diagnostics.

| Fixed candidate | Selected | Same-bank greedy | Initial distance | After prefix | Terminal |
| --- | --- | --- | ---: | ---: | ---: |
| zero | no | no | 1.745588 | 1.681042 | 1.175822 |
| goal_-1 | yes | yes | 1.745588 | 1.471325 | 0.708339 |
| goal_+1 | no | no | 1.745588 | 1.909632 | 0.896799 |

The selected chain has lower terminal distance than the same-bank zero-prefix
diagnostic, but it is also the greedy candidate and its physical prefix
improves distance immediately. The post-hoc `goal_+1` prefix worsens distance
but its terminal distance does not beat the actually selected chain. No common
suffix intervention, resource-matched strong-baseline test, independent
confirmation, or physical RGB frame-to-replay state match is established by
this table. It is not a PC2 mechanism or PC3 performance result.

Full state traces and binding hashes are in
`le-wm-main/results/pc_goal_mode_rgb_smoke_20261103_codex_physical.json`.
