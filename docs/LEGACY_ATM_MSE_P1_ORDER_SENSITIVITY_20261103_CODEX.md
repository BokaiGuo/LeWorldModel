# 旧 ATM/MSE P1：输入顺序敏感性

这是对既有结果的探索性复算，不是 D2P P1、新实验、独立复制或替代 gate。
固定原 100 条 provisional 和 140 条 external 记录，先验证原顺序的
逐 stratum 匹配集合、可识别状态、两种选择及 ΔR 和原 primary 统计均重现；
然后每组用种子 20261103 产生 32 次全记录随机排列，调用未修改的原
benchmark、原匹配阈值、原统计及原 bootstrap seed。记录文件与历史
报告在运行前后哈希一致。排列为有限样本，不是所有可能顺序。

| panel | 原 primary ΔR 均值 / 单侧 p | 有变化的排列 | 排列后 ΔR 均值范围 | 排列后 p 范围 |
|---|---:|---:|---:|---:|
| provisional | 0.0533573 / 0.0620969 | 28/32 | -0.0261619 至 0.0533573 | 0.0607470 至 0.625969 |
| external | 0.00307572 / 0.493525 | 20/32 | 0.00307572 至 0.0139198 | 0.245538 至 0.493525 |

全部排列的原 claim gate 仍为 false。这里 p 和区间依旧沿用原 stratum
独立性假设，且同一个固定种子的 Monte Carlo p 会随输入值变化；数值范围
仅用于描述排序敏感性，不能作为新检验或挑选有利顺序的依据。

在这 64 次排列中，匹配集合、可识别状态、失败原因和 MSE selector 均未变；
变化仅见 full selector 和相应 ΔR。provisional 受影响的是 PointMaze
stage 5、10、50、75，external 是 PushT stage 75。前者四组的原匹配池
中 `task_metric_alignment` 均有精确相同值，原 `_rank` 使用稳定排序
分配不同名次；stage 50 原顺序下 full 综合分数的最小值本身也并列。
external PushT stage 75 的六项输入指标均无精确并列，但原顺序下两个
候选的综合名次均值并列，`argmin` 取排列中第一个。因而不能把所有
不稳定性仅归因于单指标并列；综合分数并列也足以产生顺序依赖。
这些是对当前保存记录和实现的机制检查，不证明每次变化的唯一原因。

历史 provisional 的 ATM 字段不是独立动作干预；external 的来源和根
身份也有既有边界。不能用这次审计提升因果性、统计独立性或可推广性。

完整逐排列、原索引、逐 stratum 选择与统计：
`le-wm-main/results/legacy_p1_order_sensitivity_20261103_codex_RUN01/report.json`，
SHA-256 `dceee981a2a2b6e7a243627132079a1dd9f739b35fe663dff668591050ba3ec8`。
脚本 `le-wm-main/experiments/empirical_spectral_audit/legacy_p1_order_sensitivity_20261103_codex.py`
使用独占的新输出目录，存在即报错；测试
`le-wm-main/tests/test_legacy_p1_order_sensitivity_20261103_codex.py`。
这不是对旧实现或 gate 的修改。
