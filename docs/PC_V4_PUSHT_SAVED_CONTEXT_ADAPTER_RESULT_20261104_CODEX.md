# V4 在 PushT 保存上下文中的模型接口诊断

按[协议](PC_V4_PUSHT_SAVED_CONTEXT_ADAPTER_PROTOCOL_20261104_CODEX.md)，原样 v4 response 已在四个历史暴露 PushT 校准 latent 上下文各完成一次 15 步模型内规划；未派发物理动作。每根使用 1920 预测 token、59 批、737 条不同长度轨迹样本。冻结 checkpoint 的参数与 buffer 哈希在运行前后相同。

| 历史根 | 模型选中完整链 latent MSE | 弃权 |
| --- | ---: | --- |
| 0 | 1.498843 | 否 |
| 1 | 1.256008 | 否 |
| 2 | 1.456984 | 否 |
| 3 | 1.012027 | 否 |

这些分数只是对历史 goal0 latent 的模型评分，不是物理姿态误差或成功率，也没有与其他模型评分混作性能比较。

执行时对每根的 5、10、15 步各三种固定动作，比较 v4 批量路径与既有 PushT 逐 packet CUDA 重放；36 组最大差为零。独立审计对所有 **236 批、2948 条**已记录样本重新以显式 packet 展开执行，预测位置与评分最大差均为零，并核验动作、计量与完整链选择。[原始结果](../le-wm-main/results/pc_v4_pusht_saved_context_adapter_20261104_codex_result.json)绑定各根记录、模型与依赖；[审计](../le-wm-main/results/pc_v4_pusht_saved_context_adapter_20261104_codex_audit.json)使用同一 checkpoint，不是独立模型或数据复现。

本次接线直接标准化五个相对动作，不额外除以 100；历史动作与 latent 均由旧报告提供，没有重新采集。公开示例动作统计不同于本地 HDF5 全量统计，checkpoint 训练统计及成员的溯源仍未知；历史 goal0 和输入观测的部署合法性也不能由此结果认证。没有完成真实反馈、接触重配置、姿态成功、均衡强基线或完整资源计量，因此这只是推进 E3 前的模型接口证据，不能宣称 PC0/PC1/PC3/PC4 或 E3 通过。
