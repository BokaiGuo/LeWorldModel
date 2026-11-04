# Goal-conditioned RGB compiler: integration smoke result

One reused PointMaze development root was replayed with real RGB/action
history. The frozen model generated goal-ranked response candidates,
evaluated complete chains, sealed the selection, and the RGB bridge executed
the selected fifteen actions. This was open-loop after selection, not a
multi-round feedback evaluation or an independent performance comparison.

The selected mode was singular mode 0; its goal contributions were about
2.1875 and 0.6687. The selected candidate was `goal_-1`. The model's proxy
score was 1.7586 initially, 1.4758 after the prefix and 0.7810 at the
predicted full-chain endpoint. These are model proxy scores, not physical
terminal distances. The prefix already improves the proxy and is not a
non-immediate-progress witness under that proxy.

The call meter recorded 60 trajectory samples, corresponding to 132
five-action token samples, 27 prediction calls, one SVD, and three suffix
optimizations. Four encoder calls were recorded. These counters are not a
complete hardware-cost certificate, and no resource-matched baselines or
physical outcome comparison were run. The zero prefix is a candidate, not
a complete abstention contract. No PC gate has passed as a result.

Artifacts:

* `le-wm-main/results/pc_goal_mode_rgb_smoke_20261103_codex_plan.json`
* `le-wm-main/results/pc_goal_mode_rgb_smoke_20261103_codex_execution.json`
* `le-wm-main/results/pc_goal_mode_rgb_smoke_20261103_codex_images.npz`
