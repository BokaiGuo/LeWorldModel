# v6 Cube packet 适配：开发诊断

依据[开发协议](PC_V6_CUBE_PACKET_ADAPTER_DEV_PROTOCOL_20261104_CODEX.md)，对原冻结 Cube checkpoint、已暴露源数据上下文和五维动作开展接口诊断。checkpoint、训练统计收据及 95 GB 数据文件当前 SHA-256 与历史 P0 记录一致；数据源是训练来源，未来目标图像仅用于离线诊断，不具备训练外独立性。

[五维适配器](../le-wm-main/experiments/spectral_reachability/pc_v6_cube_packet_adapter.py)将 5/10/15 个五维原始动作按时间优先排列、每 5 步归一化为 25 维 token，并使用根观测 latent 和合法目标 latent 构建二维 chart；完整 latent MSE 单独用于评分。配合 v6 既有回归运行 `tests/test_pc_v6_cube_packet_adapter.py` 和 `tests/test_pc_target_reach_action_dim_v6.py`，共 **17 passed**。这是合成接口测试，不是 Cube 控制证据。

真实冻结模型对照的动作均相同，分别测试零动作和确定性正弦动作；参考使用既有 root-anchored rollout，同一模型与 RGB，非独立模型验证：

| 运行 | 批量与逐条 chart 最大误差 | 缓存与参考完整 latent 最大误差 | 结论 |
|---|---:|---:|---|
| [v2 默认精度](../le-wm-main/results/pc_v6_cube_packet_reference_v2_dev_20261104_codex.json) | 0.0015482 | 0.00114094 | 超过原预定容差；逐条缓存与逐条参考最大误差 3.58e-7 |
| [v3 关闭 matmul 和 cuDNN TF32](../le-wm-main/results/pc_v6_cube_packet_reference_v3_dev_20261104_codex.json) | 4.77e-7 | 0（本次记录） | 两条参考动作、三个时域满足预定 chart 和评分容差 |

默认环境中 matmul TF32 已关闭、cuDNN TF32 开启；本对照共同关闭两者，支持精度设置解释批量差异，但未隔离具体算子。首次运行的[原始记录](../le-wm-main/results/pc_v6_cube_packet_reference_dev_20261104_codex.json)已写出并保留，随后仅在终端打印时因错误传递 `flush` 退出；打印问题已修复。实验脚本后续添加了 v2/v3 诊断及新的输出路径，因此脚本当前版本并非首次运行时的封存副本。

**边界：**以上只验证一个源上下文、两条动作、三个时域的模型内接口，不验证大批量差分响应、真实物理、v6 Cube 完整决策、动作派发或成功率。关闭 TF32 的设置尚未集成到正式 Cube 反馈路径；默认精度结果不可视为已通过。PC2/PC3 负结果与来源限制保持不变，PC0–PC5 未因本诊断自动通过。
