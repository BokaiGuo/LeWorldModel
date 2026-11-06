# 本地 PushT CEM：三 seed 预算曲线开发结果

按[追加协议](PC_NATIVE_CEM_SEED_BUDGET_EXPANSION_20261105_CODEX.md)，在已经观察原预算结果后补充 seed 43、44 低中预算；因此是探索性而非确认性结果。

| seed | 64×3，elite 6 | 300×10，elite 30 | 300×30，elite 30 | 中档独有成功／高档独有成功 |
|---|---:|---:|---:|---:|
| 42 | 28/50 | 47/50 | 47/50 | 1 / 1 |
| 43 | 26/50 | 42/50 | 45/50 | 0 / 3 |
| 44 | 23/50 | 39/50 | 42/50 | 1 / 4 |

本地 `world.evaluate` 所记秒数分别为：

| seed | 低 | 中 | 高 |
|---|---:|---:|---:|
| 42 | 13.76 | 28.62 | 61.91 |
| 43 | 13.79 | 28.86 | 60.08 |
| 44 | 13.55 | 30.44 | 62.76 |

时间包括评价和视频保存，不含全部初始化，不是完整的同步 GPU／内存／系统成本；不同任务成功时刻也会改变后续执行量。每个 seed 内任务抽样一致，但不同预算随机候选不嵌套，低预算 elite 比例略有不同。seed 同时影响任务抽样和求解器随机流，面板可能重叠。

结果限制了“10 轮和 30 轮同样有效”的解释：seed 42 总数相等但有一对任务互换；43 和 44 分别净少 3 个成功。它只提供质量—成本开发曲线，没有建立预定非劣效界、独立来源、统计等价，也没有测试 Planning Compiler。

原始逐任务记录：seed 42 [低](../le-wm-main/results/pc_standard_pusht_lowbudget50_seed42_20261105_codex/eval_metrics.json)、[中](../le-wm-main/results/pc_standard_pusht_midbudget50_seed42_20261105_codex/eval_metrics.json)、[高](../le-wm-main/results/pc_standard_pusht_native50_seed42_20261105_codex/eval_metrics.json)；seed 43 [低](../le-wm-main/results/pc_native_cem_low_seed43_20261105_codex/eval_metrics.json)、[中](../le-wm-main/results/pc_native_cem_mid_seed43_20261105_codex/eval_metrics.json)、[高](../le-wm-main/results/pc_standard_pusht_native50_seed43_20261105_codex/eval_metrics.json)；seed 44 [低](../le-wm-main/results/pc_native_cem_low_seed44_20261105_codex/eval_metrics.json)、[中](../le-wm-main/results/pc_native_cem_mid_seed44_20261105_codex/eval_metrics.json)、[高](../le-wm-main/results/pc_standard_pusht_native50_seed44_20261105_codex/eval_metrics.json)。
