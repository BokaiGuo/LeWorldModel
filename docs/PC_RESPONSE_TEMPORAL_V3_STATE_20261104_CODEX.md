# Planning Compiler v3：当前状态与证据边界

本表是 2026-11-04 本轮工作范围的保守状态，不替代原方案 PC0–PC5
合同，也不把其他版本的组件结果拼为 v3 通过。

| 项目 | 状态 | 本轮证据与剩余工作 |
| --- | --- | --- |
| 方法入口 | 开发中 | [v3](../le-wm-main/experiments/spectral_reachability/pc_response_temporal_feedback_v3.py) 在 15/10/5 步保留完整链 incumbent，真实 RGB 接线已运行；未完成完整响应、谱增量和成本合同 |
| PC0 | 证据不足 | v3 Reacher 单根同后端四边界物理重放一致；尚需逐状态一致性、原始基线及声明条件可解性 |
| PC1 | 证据不足 | Wall 事后诊断显示响应依赖名义动作与阻挡尺度；尚需正式方法实际名义、动作方向及模型—物理响应全范围认证 |
| PC2 | 证据不足 | 原[零名义网格](PC_RESPONSE_TEMPORAL_V3_PROSPECTIVE_MECHANISM_RESULT_20261104_CODEX.md) 0/24 保留；新[尾部局部事前协议](PC_RESPONSE_TEMPORAL_V3_TAIL_LOCAL_MECHANISM_RESULT_20261104_CODEX.md) 出现 1/24 有界开发见证，但来源独立性和充分性仍有限 |
| PC3 | 证据不足 | [多起点 CEM 开发比较](PC_RESPONSE_TEMPORAL_V3_MULTISTART_CEM_E0_RESULT_20261104_CODEX.md) 有信号，但无独立确认、充分调优与完整成本等价；反馈 CEM 旧尾部未用作优化初始化 |
| PC4 | 证据不足 | [Reacher 单根三轮真实 RGB](PC_RESPONSE_TEMPORAL_V3_REACHER_RGB_LIVE_RESULT_20261104_CODEX.md) 已执行并重算预测、重放四边界；未完成长时域与独立任务真实控制认证 |
| PC5 | 未开始 | 尚无固定 v3 核心算法的独立 checkpoint 与第二架构完整复现 |

| 实验组 | v3 覆盖状态 |
| --- | --- |
| E0 | 精确动力学有事前 0/24 和另一合同下 1/24 开发见证；更广机制与独立确认未完成 |
| E1 Reacher | v3 两臂单根真实 RGB 三轮、预测重算和四边界重放完成；正式认证及公平强基线未完成 |
| E2 PointMaze | 一根真实 RGB 接线与原生动作重放；独立性及广泛真实性能未知 |
| E3 PushT、E4 Cube | 未完成 v3 接触及长时域视觉操作闭环 |
| E5 跨模型 | 未完成 v3 的独立 checkpoint 和另一架构复现 |

根的开发外、normalizer 拟合外与 checkpoint 训练外是不同来源命题。
PointMaze 当前可核对的拟合集及未知训练来源见
[normalizer 边界](PC_POINTMAZE_NORMALIZER_CONFIRMATION_BOUNDARY_20261103.md)；
没有权威来源时保持 unknown。开放环、闭环、RGB、事前机制和事后
归因各自使用本版本自己的链与对应协议，不互相补成同一见证。

当前 v3 响应方向生成尚未实现原方案目标加权可达价值与响应多样性
剪枝；不能将现有响应臂称作完整原方案。反馈 CEM 未从旧尾部
初始化优化是额外的基线公平性缺口，应另立新协议和版本评估。
