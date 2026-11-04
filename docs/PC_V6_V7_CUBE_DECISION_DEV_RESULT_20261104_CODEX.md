# Cube 冻结模型中的 v6 决策和 v7 弃权修复

按 [v6 协议](PC_V6_CUBE_DECISION_DEV_PROTOCOL_20261104_CODEX.md)，在同一已暴露的 Cube 训练源 RGB 上下文、原冻结模型、15 步固定时域上，v6 运行了完整一次模型内决策：[记录及轨迹](../le-wm-main/results/pc_v6_cube_decision_dev_20261104_codex.json)。43 次预测、867 条轨迹、1920 packet-token；生成与优化调用被计入该轴，但编码、分解、审计和其余墙钟成本不等价于 token。

首个 51 样本响应批独立分批重算，批量与逐条 chart 最大差 `9.54e-7`，得分最大差 `2.98e-8`，中心差分矩阵绝对 Frobenius 差 `3.77e-5`，相对差 `4.14e-6`。三个探针和最终链的逐条原参考 rollout 在本次记录中与缓存预测一致。均使用关闭 TF32 的同一冻结模型，不是模型—物理响应验证。

发现可复核的 v6 选择语义错误：最终 incumbent 与清零前缀比较链的实际 float32 动作完全相同，前 5 步均为零，但不同批次评分相差 `1.21e-8`，记录却为“未弃权”。不能把此标签解释为非零前缀机制。这个问题由逐元素检查及保存的全部评分轨迹定位，不证明其他负结果均由代码造成。

按新的 [v7 事前协议](PC_V7_PAIRED_ABSTENTION_DEV_PROTOCOL_20261104_CODEX.md)，[v7 编译器](../le-wm-main/experiments/spectral_reachability/pc_target_reach_paired_abstention_v7.py)从 1920 总 token 中预留最终两条完整链评分；同批次比较，实际 float32 动作完全相同时确定性选择零前缀。未引入近似相等阈值。v7 合成回归包括真实 v6 搜索在 15/10/5 时域的预算接口，以及同动作、不同动作选择。

[v7 Cube 开发结果](../le-wm-main/results/pc_v7_cube_decision_dev_20261104_codex.json)使用同一根与目标：44 次预测、867 条轨迹、1920 token，其中配对评分占 6 token；两条实际动作相同且同批分数均为 `0.0149083836004138`，最终选择 `zero_prefix` 并标记弃权。v6 和 v7 获得的搜索预算不同，不能据此作方法优劣比较。v7 首批 51 探针的独立重算与上述数值一致。`tests/test_pc_target_reach_paired_abstention_v7.py` 为 **12 passed**，其中 3 项运行真实 v6 搜索路径。

**边界：**这是单根源内模型诊断，没有派发动作、对照强基线或产生真实控制成绩。v7 尚未在 E0 机制、Reacher、导航和接触任务中复验；PC0–PC5 原状态保持不变。相似但不完全相同的候选在数值误差附近可能仍不稳定，待预先设计的精度与排序敏感性实验判断。
