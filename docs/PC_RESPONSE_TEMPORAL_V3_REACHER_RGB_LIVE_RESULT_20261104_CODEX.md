# v3 Reacher 真实 RGB 固定终点开发结果

按[事前协议](PC_RESPONSE_TEMPORAL_V3_REACHER_RGB_LIVE_PROTOCOL_20261104_CODEX.md)，
在已暴露根 `20263160` 上，response 和 smooth CEM 分别用真实 RGB
完成三次规划及各 5 步执行；规划剩余时域分别是 15、10、5 步。
两臂使用相同冻结模型、目标和每次 1920 token 上限。

| 臂 | 四个决策边界的真实 qpos 距离 | 三次预测 token |
| --- | --- | --- |
| response | 1.0035004 → 1.2671933 → 0.9593270 → 0.5533907 | 1920、1919、1920 |
| smooth CEM | 1.0035004 → 1.0913271 → 1.0524107 → 0.5683202 | 1920、1920、1920 |

[账本审计](../le-wm-main/results/pc_response_temporal_v3_reacher_rgb_live_20261104_codex_audit.json)
核对了六次决策的历史、选择、计量、carry、派发及时序；
[同后端物理重放](../le-wm-main/results/pc_response_temporal_v3_reacher_rgb_live_20261104_codex_physical_replay.json)
在两臂的四个边界得到完全相同的 qpos 和目标，根 RGB 一致。
物理重放不覆盖中间状态或 qvel。

按独立写入的[重算协议](PC_RESPONSE_TEMPORAL_V3_REACHER_PREDICTION_REPLAY_PROTOCOL_20261104_CODEX.md)，
使用原参考 rollout 路径重算全部 **603 批、7127 条**已封存预测。
[重算记录](../le-wm-main/results/pc_response_temporal_v3_reacher_rgb_live_20261104_codex_prediction_replay.json)
中 chart 和目标误差为零；归一化动作、二维投影、latent MSE 的
最大绝对误差分别为 `2.39e-7`、`4.30e-6`、`4.77e-7`，均在预定
阈值内。参考路径仍共享冻结模型、预处理和 GPU；这不是独立模型
或数据复现，也不证明模型目标与物理指标一致。

这是一根已暴露任务上的开发诊断，不能证明 PC3/PC4/PC5。token
不等于完整计算量；模型训练来源及 normalizer 来源仍需认证。
此外反馈时 smooth CEM 将旧尾部纳入最终比较但以零链初始化搜索，
response 则可以从旧尾部进行精修；因此现有两臂结果不支持将差异
归因于程序编译。必须在新协议下补充使用相同预算和可获得信息的
旧尾部热启动强基线，不能改写已封存结果。
