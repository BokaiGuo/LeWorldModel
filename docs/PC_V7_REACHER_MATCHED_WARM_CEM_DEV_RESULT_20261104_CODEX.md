# v7 与双起点 CEM：Reacher 单根真实反馈开发比较

按[开发协议](PC_V7_REACHER_MATCHED_WARM_CEM_DEV_PROTOCOL_20261104_CODEX.md)，双起点 smooth CEM 与 v7 使用同一已暴露根、原冻结模型、目标、warmup、TF32 设置与固定终点时域，分别完成三轮真实 RGB 闭环；CEM 使用自己的旧尾和观测。[配对绑定检查](../le-wm-main/results/pc_v7_reacher_matched_warm_cem_dev_20261104_codex_paired_comparison.json)确认两臂初始历史帧、根帧、目标图像和动作历史逐项一致。

| 方法 | 第 0 步距离 | 第 5 步 | 第 10 步 | 第 15 步 |
|---|---:|---:|---:|---:|
| v7 response | 1.003500 | 0.983396 | 0.971309 | 0.560807 |
| 双起点 smooth CEM | 1.003500 | 1.091327 | 0.992984 | 0.555675 |

本根最终 v7 距离比 CEM 大 `0.005132`。每轮 CEM 使用 1920 prediction-token；v7 为 1920、1919、1920。后两轮 CEM 给 zero 与本臂 carry 独立优化机会；每轮的真实状态不同，不能将模型评分直接跨臂比较。

[CEM 记录审计](../le-wm-main/results/pc_v7_reacher_matched_warm_cem_dev_20261104_codex_audit.json)检查了旧尾、重启分配、历史动作和派发。[参考模型重算](../le-wm-main/results/pc_v7_reacher_matched_warm_cem_dev_20261104_codex_prediction_replay.json)覆盖 **297 批、3520 条轨迹**，最大位置差 `4.30e-6`、评分差 `4.77e-7`；[同后端物理重放](../le-wm-main/results/pc_v7_reacher_matched_warm_cem_dev_20261104_codex_physical_replay.json)核验初始 RGB 及四个边界的 qpos、目标位置，未核验所有中间状态或 qvel。模型重算与物理重放均非独立模型或后端复现。

这是单个已暴露根上的开发对照：不能据差距断言总体优劣，CEM 并非充分调优的强搜索上界；预测 token 相同不等于响应提取、编码、分解和总成本相同。训练来源独立性、关键消融、统计确认和其他环境仍未闭合，PC3/PC4 状态不改变。
