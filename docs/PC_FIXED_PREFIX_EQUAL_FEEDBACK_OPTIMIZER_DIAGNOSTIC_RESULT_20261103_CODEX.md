# 固定前缀等资源反馈后缀优化结果

按[预先协议](PC_FIXED_PREFIX_EQUAL_FEEDBACK_OPTIMIZER_DIAGNOSTIC_PROTOCOL_20261103_CODEX.md)，
24 个已暴露任务的五类封存前缀各执行两次同配置、零初始化的 floor-CEM
后缀规划，分别在剩余 10、5 步各计 480 个五步预测 token。第二次决策
包含一次 carry 重评分；总计 120 条固定前缀反馈轨迹。没有注入历史实际
尾部或其他前缀的解。

| 前缀 | 最终平均末距 | 成功 `<0.05` | 第二步保留 carry |
|---|---:|---:|---:|
| selected | 0.50045 | 0/24 | 4/24 |
| zero | 0.50097 | 0/24 | 2/24 |
| 同库 greedy | 0.50049 | 0/24 | 6/24 |
| random bridge | 0.50048 | 0/24 | 7/24 |
| lookahead | 0.50082 | 0/24 | 2/24 |

selected 最终末距逐根严格小于 zero、greedy、random bridge、lookahead
的根数分别为 11、10、12、11。4 个非即时根中，没有任何根同时小于 zero
和 greedy，因此也不能满足原诊断的共同交集条件。数值差异很小，不能据
此推断规划优势。

与相同 selected 前缀的历史反馈实际尾部比较，新结果在 16 根末距更低、
8 根更高；历史轨迹来自不同规划策略，不是等条件性能对照。根 13 上，
历史反馈末距为 0.001087，而本次零初始化等资源 CEM 为 0.500216：这
提供了该固定前缀存在可行尾部且本配置没有找到它的直接证据。它不证明
代码错误，也不能将诊断中的历史动作当作正式方法可用初始化。

结论是，增加到两轮各 480 token 的这一组零初始化 CEM 并未解决搜索
覆盖，也未证实 selected 的非即时可规划优势。尚不能排除初始化、多峰
搜索、动作参数化或预算不足等解释；不应把负结果直接归因于理论。
该面板是暴露开发数据，五类前缀中两类为历史事后控制；PC2 事前同链、
PC3 独立确认和 RGB 控制均未由此通过。后续需要预先固定、各前缀公平
获得的多起点或结构化初始化以及完整成本，再做新诊断；不能追调本协议。

[完整计划与预测](../le-wm-main/results/pc_fixed_prefix_equal_feedback_optimizer_20261103_codex_plans.json)
SHA-256：`1861a3afd24eabffd28854f095d4f8aeb33ca9bae65f6fc0ad6e4c12901b7f6a`；
[逐批重放审计](../le-wm-main/results/pc_fixed_prefix_equal_feedback_optimizer_20261103_codex_audit.json)
SHA-256：`3c92de2b233f8787684c8f04f565cb108fff1be99aa2b4da95b4e6b4c2276d70`。
审计检查每批评分、token、所选链和整链执行；独立测试重构随机采样及
精英更新，验证两轮初始化和采样轨迹。以下命令从 `le-wm-main` 运行，
结果 **3 passed**：

```bash
.venv/bin/pytest -q -p no:cacheprovider \
 tests/test_pc_fixed_prefix_equal_feedback_optimizer_20261103_codex.py \
 tests/test_pc_response_feedback_executed_tail_intervention_20261103_codex.py \
 tests/test_pc_response_diverse_beam_same_chain_diag_20261103_codex.py
```
