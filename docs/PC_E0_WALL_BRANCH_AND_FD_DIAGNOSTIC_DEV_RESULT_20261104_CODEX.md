# E0 墙体分支与有限差分：事后诊断

按[墙体重放协议](PC_E0_WALL_INDEPENDENT_BRANCH_REPLAY_DEV_PROTOCOL_20261104_CODEX.md)，
独立线段交点实现重算 v8、v7 和两种 CEM 的 72 条已封存计划；
所有已保存端点和评分一致。进一步[逐状态与阻挡事件审计](../le-wm-main/results/pc_e0_wall_independent_branch_replay_dev_20261104_codex_audit.json)
覆盖 **49,572 条轨迹、691,200 个动作步**，与原解析后端一致。这是独立
几何实现，不是独立物理环境；未发现这些轨迹上的动力学接线错误。

| 臂 | 已评分完整链 | 曾到墙右侧 | 最终选中链到过右侧 | 最终链受阻 |
|---|---:|---:|---:|---:|
| v8 | 8,946 | 29 | 0/18 | 18/18 |
| v7 | 8,946 | 22 | 0/18 | 18/18 |
| smooth CEM | 11,520 | 62 | 0/18 | 18/18 |
| action CEM | 11,520 | 13 | 0/18 | 17/18 |

重复评分计入上述链数，不能视为独立样本。v8 的跨墙链来自 10 根，
最优跨墙终距约 0.713；在每个相关根上，最佳跨墙链均比该根最终链差。
因此这些记录支持“搜索很少产生既绕墙又接近目标的完整链”，而不是
“已评分的高质量跨墙链被最终选择错误丢弃”；不证明其他合法链不存在。
[完整记录](../le-wm-main/results/pc_e0_wall_independent_branch_replay_dev_20261104_codex_result.json)
保存逐根计数。

按[实际探针协议](PC_V8_V7_E0_FD_BRANCH_STABILITY_DEV_PROTOCOL_20261104_CODEX.md)，
两个版本的 36 个首层响应批次均无 ±0.05 阻挡分支变化；后继响应中，
v8 为 **12/162 批、41/1620 坐标**跨分支，v7 为 **18/162 批、
108/1620 坐标**。连接实际保留索引后，v8 有 8/72 个保留候选受此影响，
v7 有 18/72 个；每个版本分别涉及 8、18 个根。
[分支记录](../le-wm-main/results/pc_v8_v7_e0_fd_branch_stability_dev_20261104_codex.json)
和[保留索引审计](../le-wm-main/results/pc_v8_v7_e0_fd_branch_stability_dev_20261104_codex_retained_audit.json)
绑定原轨迹。跨分支说明此尺度的中心差分混合阻挡事件，不自动说明
计算错误或所有尺度都不可微。

在[事后代理协议](PC_V8_V7_E0_FD_CURVATURE_PROXY_DEV_PROTOCOL_20261104_CODEX.md)
下，仅用模型输出构造的中点残差比值在这些已暴露后继坐标上，
对 v8 的 41 个变化坐标及 v7 的 108 个变化坐标，以 0.5 阈值均无漏报，
对各自其余坐标无误报；坐标层面 AUC 均为 1.0。该结果受同一解析墙
结构和相关坐标影响，v8/v7 分别有 247/320 个后继坐标的差分分母小于
`1e-8`，不可直接外推到 RGB 或用于声称独立校准。
[代理明细](../le-wm-main/results/pc_v8_v7_e0_fd_curvature_proxy_dev_20261104_codex.json)
保存每坐标残差、分母与标签。
