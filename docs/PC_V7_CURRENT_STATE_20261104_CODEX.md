# Planning Compiler v7：开发状态快照

本表只按原方案 PC0–PC5 记录，不能与历史 ATM/MSE P1 或早期 D2P 准入混用。v7 是开发方法，不是已确认的最终研究方法；旧版本实验只作背景，不拼成 v7 正面证据。

| 项目 | 当前入口和证据 | 未完成与下一步 |
|---|---|---|
| 方法入口 | [v7 配对弃权](../le-wm-main/experiments/spectral_reachability/pc_target_reach_paired_abstention_v7.py)调用 [v6 可配置动作维度搜索](../le-wm-main/experiments/spectral_reachability/pc_target_reach_action_dim_v6.py)：目标加权方向、后继响应与局部价值、候选保留、多代后缀 CEM、完整链精修、配对弃权 | 响应仍为二维 chart；候选覆盖、局部价值和响应多样性的实际增量均未确认 |
| 已证实修复 | [Cube v6/v7 决策诊断](PC_V6_V7_CUBE_DECISION_DEV_RESULT_20261104_CODEX.md)：修复相同 float32 动作跨批比较导致错误弃权标签 | 非完全相同动作的近并列数值稳定性仍需检验；不把该问题视为其他负结果的原因 |
| 后缀与特权诊断 | [固定特权前缀四结点／逐步 CEM](PC_V7_E0_ORACLE_SUFFIX_ACTION_CEM_DEV_RESULT_20261104_CODEX.md)，相同预算下两臂各 4/18 根至少一流成功 | 已暴露、特权输入；不再在此基础上追逐阳性，需正式方法生成与来源合格的对照 |
| 成本与来源 | 逐批预测 token、动作、部分分解与时间记录 | 编码、导数、SVD、缓存与同步总成本未认证；开发外、normalizer 拟合外、checkpoint 训练外各需独立来源证据 |

| Gate | 状态 | v7 证据及缺口 |
|---|---|---|
| PC0 | 证据不足 | E0 有同后端轨迹审计；Reacher 和 PushT 有 RGB 反馈与回放。仍需完整正式任务合同、原基线复现和全部来源认证 |
| PC1 | 证据不足 | [Reacher 实际响应探针](PC_V7_REACHER_ACTUAL_FD_PHYSICAL_DEV_RESULT_20261104_CODEX.md)仅一根、一时域、一尺度；不能覆盖全部方向、物理任务排序与局部有效范围 |
| PC2 | 证据不足 | [E0 最终链干预](PC_V7_E0_EXPOSED_INTERVENTION_DEV_RESULT_20261104_CODEX.md)机制交集 0/18；需新假设下自动生成并事前选择的同链见证与归因 |
| PC3 | 证据不足 | [E0 多初始化 CEM](PC_V7_E0_NONZERO_CEM_DEV_RESULT_20261104_CODEX.md)、[Reacher](PC_V7_REACHER_MATCHED_WARM_CEM_DEV_RESULT_20261104_CODEX.md)、[PointMaze](PC_V7_POINTMAZE_MATCHED_WARM_CEM_DEV_RESULT_20261104_CODEX.md)及 [PushT](PC_V7_PUSHT_MATCHED_WARM_CEM_DEV_RESULT_20261104_CODEX.md)配对均为开发结果；强基线、来源合格面板及事前统计尚未完成 |
| PC4 | 证据不足 | 同一 v7 核心已运行 Reacher、PointMaze 与 PushT 单根真实 RGB 三轮反馈；均不是来源合格的多根长时域控制收益证明 |
| PC5 | 未完成 | 未建立同一 v7 核心在独立 checkpoint 和第二架构的来源合格复现 |

| 组别 | v7 当前覆盖 |
|---|---|
| E0 | 18 个暴露根的规划、同链干预与多初始化 CEM；机制交集 0/18，特权诊断不能替代自动发现 |
| E1 Reacher | 单根真实 RGB、匹配 CEM 和实际响应探针；尚无合格的多根确认 |
| E2 TwoRoom／PointMaze | [v7 单根真实 RGB](PC_V7_POINTMAZE_RGB_LIVE_DEV_RESULT_20261104_CODEX.md)及[同根热启动 CEM](PC_V7_POINTMAZE_MATCHED_WARM_CEM_DEV_RESULT_20261104_CODEX.md)已重放；v7 与同根旧 v5 动作一致，来源和强基线仍未确认 |
| E3 PushT | 单根真实 RGB、同根 CEM 和物理排序诊断；两臂均未官方成功，未建立接触机制 |
| E4 Cube | v7 有冻结模型内一次决策和弃权回归；没有据此声称真实动作执行或长时域闭环 |
| E5 跨模型 | 未核验到同一核心的独立 checkpoint 与另一架构完整复现 |

[PointMaze 来源边界](PC_POINTMAZE_NORMALIZER_CONFIRMATION_BOUNDARY_20261103.md)和[性能线准入边界](PC_PERFORMANCE_LINE_ADMISSION_BOUNDARY_20261103_CODEX.md)仍适用：新 seed 不证明未参与预处理拟合或训练。此快照记录的是已核验证据及缺口，不代表全仓逐行审计，也不宣告研究闭环完成。
