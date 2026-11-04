# v7 PointMaze：同根热启动 CEM 开发结果

按[协议](PC_V7_POINTMAZE_MATCHED_WARM_CEM_DEV_PROTOCOL_20261104_CODEX.md)，在 v7 已暴露根 0，另运行 smooth CEM 的三轮真实 RGB 反馈；后两轮分别以零链和本臂自身 carry 为 CEM 起点。两臂使用同一初始暖身、目标、冻结模型、动作统计及 15／10／5 固定终点时域；后续各自使用真实反馈。

| 臂 | 暖身后 native 重放距离 | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---:|---:|---:|---:|
| v7 response | 1.745588 | 1.681042 | 1.118418 | 0.405924 |
| 热启动 smooth CEM | 1.745588 | 1.593437 | 1.218770 | 0.109452 |

该根最终 CEM 距离较小约 0.296472；这是已暴露单根的开发比较，不代表总体性能、成功率或强搜索上界。两臂后续观测和物理状态不同，模型代理评分不可跨臂直接比较。CEM 每轮 1920 prediction-token，v7 为 1920、1919、1920；token 不是全部计算成本。

## 审计和边界

- [配对输入哈希审计](../le-wm-main/results/pc_v7_pointmaze_matched_warm_cem_dev_20261104_codex_pair_binding.json)核对两臂初始 16 帧、目标、动作历史、模型及归一化绑定；[CEM 记录审计](../le-wm-main/results/pc_v7_pointmaze_matched_warm_cem_dev_20261104_codex_audit.json)核对自身 carry、双起点分配、最优评分、时钟及 token。
- [native 动作重放](../le-wm-main/results/pc_v7_pointmaze_matched_warm_cem_dev_20261104_codex_physical.json)产生上表基线距离，但未逐状态对照原 shadow 物理状态；[shadow RGB 重放](../le-wm-main/results/pc_v7_pointmaze_matched_warm_cem_dev_20261104_codex_rgb_frame_replay.json)为 31/31 帧一致。
- [同 checkpoint 显式 packet 重算](../le-wm-main/results/pc_v7_pointmaze_matched_warm_cem_dev_20261104_codex_prediction_replay.json)覆盖 297 批、3520 条轨迹，位置与评分最大误差均约 `1.43e-6`。它使用同一模型，不是独立复现。

执行器的 `response` 是旧单臂接口标签；manifest 的方法是 CEM，未运行响应程序。该基线尚未充分调优，也不覆盖 AP-CEM 或梯度方法。根已暴露，normalizer 与 checkpoint 来源独立性仍未知，故本结果不构成 PC3／PC4 通过。
