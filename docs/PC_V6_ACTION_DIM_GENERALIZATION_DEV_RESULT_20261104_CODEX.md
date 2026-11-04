# V6 动作维度泛化：接口与二维回归

Cube 的合法原始动作为每步五维；v5 在前缀、有限差分、CEM 和 carry 中写死二维。新增独立 [v6 开发核心](../le-wm-main/experiments/spectral_reachability/pc_target_reach_action_dim_v6.py)，将动作维数和两条名义动作显式输入，保持 5 步 packet、二维目标投影、最多 15 步固定终点及 v5 的生成、保留、优化和弃权结构。它不修改 v5，也不宣称任意 packet 长度或任意目标维度已支持。

[接口回归](../le-wm-main/tests/test_pc_target_reach_action_dim_v6.py)为 **10 passed**：在合成预测器上，二维的三种剩余时域、两种保留策略逐批动作和决策与 v5 相同；五维在三个时域验证合法形状、25 个 packet 坐标的正负探针及动作域。合成检查不是 Cube 实验或任务性能。

[E0 精确动力学等价记录](../le-wm-main/results/pc_v6_action_dim_v5_e0_equivalence_20261104_codex.json)在 v5 已封存的 18 根逐批重跑并对照 **13,266 条**轨迹、请求动作、float32 动作、预测、候选、保留、最终完整链及预算；全部一致，耗时不要求相等。它共享动力学和已暴露任务，只覆盖首次 15 步决策；不能以此宣称五维模型接口、Cube 真实控制或不同模型复现已完成。

下一步需固定 Cube 非特权 nominal、绑定原冻结 checkpoint 与五维动作统计，先检查真实模型 packet 时钟和动作转换，再另行协议运行视觉闭环与公平基线。旧 Cube 后训练修复 checkpoint 不能替代原冻结模型主结果；v5 正负结果也不自动升级为 v6 gate 证据。
