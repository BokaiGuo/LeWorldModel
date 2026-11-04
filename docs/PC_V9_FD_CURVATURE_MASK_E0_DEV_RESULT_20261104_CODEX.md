# v9 中点残差掩码：E0 开发结果

按[开发协议](PC_V9_FD_CURVATURE_MASK_E0_DEV_PROTOCOL_20261104_CODEX.md)，
独立 v9 在原预测批次中计算中点残差，屏蔽比值大于 0.5 的响应列，
不读取墙体标签，也不额外调用预测；其余搜索、token 与最终配对规则沿用 v7。
该规则是在观察同一暴露面板的分支诊断之后确定的，不是独立确认。

| 方法 | 平均终距 | 中位终距 | `<0.05` | v9 对其胜／平／负 |
|---|---:|---:|---:|---:|
| v9 | 0.500930 | 0.500631 | 0/18 | — |
| v7 | 0.501111 | 0.500535 | 0/18 | 5／10／3 |
| smooth CEM | 0.501265 | — | 0/18 | 8／0／10 |
| action CEM | 0.501926 | — | 0/18 | 12／0／6 |

胜平负容差 `1e-10`，每根 1920 token。18 个后继响应批次含掩码，
其中 3/72 个实际保留候选含掩码；10 根最终 float32 链与 v7 完全相同。
此设置没有在该开发面板找到成功链，不证明分支掩码无价值，也不支持
把历史失败唯一归因于数值响应。没有据此再次调整阈值。

[运行记录](../le-wm-main/results/pc_v9_fd_curvature_mask_e0_dev_20261104_codex_result.json)
保存原始／掩码响应、动作和得分；
[逐批审计](../le-wm-main/results/pc_v9_fd_curvature_mask_e0_dev_20261104_codex_audit.json)
在同一精确后端重算 13,266 条轨迹，并从预测重新计算每列掩码。
线性预测器逐批与 v7 等价、非平滑列屏蔽、零秩回退等合成测试，
连同 v8 测试共 **12 项通过**；测试不代表真实模型效果。

按[事后干预协议](PC_V9_E0_EXPOSED_INTERVENTION_DEV_PROTOCOL_20261104_CODEX.md)，
对 v9 最终链进行共同尾、三流各 480 样本重规划和局部响应检查：
3/18 前缀不改善初始距离，其中 2/18 为非零前缀；
zero／same-bank greedy 基础机制交集及加入 lookahead／random 的全对照
交集均为 **0/18**。
[干预记录](../le-wm-main/results/pc_v9_e0_exposed_intervention_dev_20261104_codex_result.json)
和[重建审计](../le-wm-main/results/pc_v9_e0_exposed_intervention_dev_20261104_codex_audit.json)
使用同一实现和解析后端。该方法尚未完成真实 RGB、独立来源、
强基线确认或 PC2–PC5；不把 v7 或 v8 正结果拼为 v9 证据。
