# Planning Compiler v5：当前开发状态

这是 2026-11-04 的工作快照，按 Planning Compiler 原方案 PC0–PC5 记录；不替代历史 ATM/MSE P1、早期 D2P 准入或 GDD-OPD 训练阶段。旧版本结果只作背景，不拼入 v5 机制或性能通过证据。[v5 核心](../le-wm-main/experiments/spectral_reachability/pc_target_reach_canonical_v5.py)当前仍是开发版本，不是已经确认的最终研究方法。

| 项目 | 当前入口或证据 | 未完成与下一步 |
|---|---|---|
| 生成与保留 | v5 调用 v4 目标加权响应与后继局部可达价值、响应多样性保留；多名义动作、两符号候选 | 验证投影有效性、更多方向和动作空间覆盖；目标价值与响应多样性的独立增量尚未确认 |
| 优化与选择 | 多代后缀 CEM、完整链 incumbent 和联合精修、零前缀比较弃权；固定终点 15/10/5 | 更强共同优化与跨成本曲线；最终机制必须在实际精修和弃权后选中链上验证 |
| 延迟收益 | [E0 同库机制](PC_V5_E0_PROSPECTIVE_MECHANISM_DEV_RESULT_20261104_CODEX.md)与[特权注入](PC_V5_E0_ORACLE_PATH_INJECTION_DIAG_RESULT_20261104_CODEX.md) | 见证前缀仅 3/18 被扩展库保留且原库不含这些前缀；不能据此证明唯一失败原因 |
| 反馈基线 | Reacher 和 PushT 同根已有双起点 CEM 作开发背景 | v5 配对强基线尚未在来源合格任务上固定并确认；相同 token 不等于相同总成本 |
| 来源与成本 | 轨迹、逐批预测、部分计量和依赖哈希已保存 | 环境依赖闭包、硬件成本和训练/normalizer 拟合集成员尚未完整认证 |

| Gate | 当前状态 | v5 证据及缺口 |
|---|---|---|
| PC0 | 证据不足 | E0 有解析可行性与完整链注入；[PointMaze](PC_V5_POINTMAZE_RGB_LIVE_DEV_RESULT_20261104_CODEX.md) 31 帧同后端重放；仍缺正式基线、所有任务合同和完整逐状态认证 |
| PC1 | 证据不足 | 有 E0 局部响应诊断及各视觉路径同模型参考重算；这些不是 v5 实际方向跨尺度的模型—物理响应认证 |
| PC2 | 证据不足 | [18 根开发机制](PC_V5_E0_PROSPECTIVE_MECHANISM_DEV_RESULT_20261104_CODEX.md)交集 0/18；[旧 24 根事后诊断](PC_V5_CANONICAL_RESPONSE_SAME_BANK_MECHANISM_RESULT_20261104_CODEX.md)交集 0/24。不能嫁接 v3 的有限见证，也不能将有限面板升级为理论否定 |
| PC3 | 证据不足 | [18 根同 token CEM](PC_V5_E0_CEM_BASELINES_DEV_RESULT_20261104_CODEX.md)无明显优势；基线仍未充分调优，独立性、效应及统计合同缺失 |
| PC4 | 证据不足 | 同一 v5 核心完成 Reacher、PointMaze、PushT 各一已暴露根三轮真实 RGB；未建立来源合格多根、长时域真实控制改善 |
| PC5 | 未开始 | 无同一 v5 核心在独立 checkpoint 和第二架构上合格复现 |

| 实验组 | v5 覆盖与下一步 |
|---|---|
| E0 | 自动程序搜索、同链干预、特权可行链与部分基线已跑；开发机制交集为零，需在不重调旧面板前提下明确新假设与公平验证 |
| E1 Reacher | [单根真实 RGB](PC_V5_REACHER_RGB_LIVE_RESULT_20261104_CODEX.md)和参考预测、四边界物理重放已完成；仍缺独立多根、实际方向响应和强基线认证 |
| E2 PointMaze | [单根真实 RGB](PC_V5_POINTMAZE_RGB_LIVE_DEV_RESULT_20261104_CODEX.md)、模型重算、同后端逐帧和 native 动作重放已完成；物理逐状态跨后端一致性与来源未知 |
| E3 PushT | [单根真实 RGB](PC_V5_PUSHT_RGB_FEEDBACK_DEV_RESULT_20261104_CODEX.md)未成功并有后验预测诊断；未建立接触重配置机制或严格姿态成功优势 |
| E4 Cube | 未完成同一 v5 核心的长时域真实闭环；需核验合法冻结模型、接口及数据后再确定可行实验 |
| E5 跨模型 | 未完成同一核心的独立 checkpoint、架构及来源合格验证 |

开发外、normalizer 拟合外和 checkpoint 训练外是三个独立命题；[PointMaze 来源边界](PC_POINTMAZE_NORMALIZER_CONFIRMATION_BOUNDARY_20261103.md)中的未知项不能以新随机种子替代认证。v5 由运行时签名适配器接入视觉执行器，后者仍为开发集成，不能仅凭上述单根结果冻结为已通过的正式方法。
