# v7 Reacher 真实 RGB 反馈：单根开发结果

按[开发协议](PC_V7_REACHER_RGB_LIVE_DEV_PROTOCOL_20261104_CODEX.md)，v7 使用原冻结 Reacher 模型及同一已暴露根完成三轮真实 RGB 反馈；固定终点的剩余规划时域为 15/10/5 步，每轮执行 5 步。每轮的实际动作历史、carry、RGB 和目标都由本臂自己的运行绑定。

| 执行步数 | 真实关节目标距离 | 该轮模型 latent MSE | 预测 token |
|---:|---:|---:|---:|
| 0 | 1.003500 | 0.786575 | 1920 |
| 5 | 0.983396 | 0.782288 | 1919 |
| 10 | 0.971309 | 0.772629 | 1920 |
| 15 | 0.560807 | — | — |

三轮均未弃权；在每轮配对比较中 incumbent 与零前缀链实际动作不同。[记录审计](../le-wm-main/results/pc_v7_reacher_rgb_live_dev_20261104_codex_audit.json)核对了全部请求、预算、同批选择、动作派发、carry 及因果 RGB/动作历史。三轮最终 float32 完整链与该根历史 v5 逐元素相同；模型分数有微小浮点差异，因此这是版本回归，不是独立数据复现或 v7 优势。

[参考预测重算](../le-wm-main/results/pc_v7_reacher_rgb_live_dev_20261104_codex_prediction_replay.json)覆盖 **305 批、3642 条轨迹**，最大 chart 位置误差 `2.87e-6`、MSE 误差 `5.97e-7`，共享同一模型和 GPU。[同后端物理重放](../le-wm-main/results/pc_v7_reacher_rgb_live_dev_20261104_codex_physical_replay.json)的初始 RGB 一致，四个决策边界的 qpos 和目标位置最大误差为零；未逐步核验全部中间状态或 qvel。

运行器及物理重放控制台沿用旧版“两臂”文案，manifest 和实际记录为 v7 单臂。关节距离与 latent MSE 不是同一指标；单根执行没有强基线配对、独立训练来源认证、总算力认证或成功率统计，不构成 PC3、PC4 或 PC5 通过。
