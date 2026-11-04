# 多方向最终前缀：等资源反馈规划诊断

按照[预先协议](PC_MULTIDIRECTION_EQUAL_FEEDBACK_MECHANISM_DIAG_PROTOCOL_20261103_CODEX.md)，
对 24 个已暴露任务的五类封存前缀，使用同一零初始化 CEM、共同随机数、
两次反馈和每轮 480 token 规划后缀；所有条件均未注入历史好尾部。

| 前缀 | 平均反馈终距 | 成功 `<0.05` | 根 20 终距 |
|---|---:|---:|---:|
| selected | 0.45979 | 2/24 | 0.500856 |
| zero | 0.50097 | 0/24 | 0.503159 |
| 同库 greedy | 0.50049 | 0/24 | 0.502386 |
| random bridge | 0.50048 | 0/24 | 0.500221 |
| lookahead | 0.50082 | 0/24 | 0.502985 |

按前次事后共同尾部变化及非即时条件，加入本次等资源反馈重规划比较后，
同时优于 zero 和 greedy 的交集仍为 **1/24（根 20）**；同时优于四控制
仍为 **0/24**。根 20 的 selected 不优于 random bridge，且与控制差距
很小；不得把这当成强基线优势。

selected 的两个成功根为 6 和 17，均不满足预设非即时条件。它们证明
在这些固定前缀、精确状态和额外反馈预算下，该后缀优化配置找到可执行
轨迹；**不是原 480-token 首次编译器的成功**，也不是同一事前链上的
非即时机制见证。前次 120 完整链搜索和此次两轮反馈使用不同计算量和
反馈机会，不能作为等预算优化器对比。

这是一项事后敏感性诊断：保留了 selected、zero、greedy 的有限交集，
但历史控制、暴露根和已知搜索不足仍限制结论。不能据此宣告 PC2、
PC3、真实 RGB 或独立泛化通过，也不能认定失败由理论或代码必然造成。
原编译器及原干预结果不变。

[轨迹与预测](../le-wm-main/results/pc_multidirection_equal_feedback_mechanism_diag_20261103_codex_plans.json)
SHA-256：`4177ccce5e43c63a52fd392ca4cfc131da77f593649cba6beff59f23c50a275a`；
[逐批审计](../le-wm-main/results/pc_multidirection_equal_feedback_mechanism_diag_20261103_codex_audit.json)
SHA-256：`628dbf750435e129f63cc57949add5c375f0ea3d109c0d15c23d4c9be7bf1203`。
已重放 120 条反馈轨迹、重建 CEM 采样并检查前缀和预算。以下测试从
`le-wm-main` 运行，结果 **3 passed**：

```bash
.venv/bin/pytest -q -p no:cacheprovider \
 tests/test_pc_multidirection_equal_feedback_mechanism_diag_20261103_codex.py \
 tests/test_pc_multidirection_successor_same_chain_diag_20261103_codex.py \
 tests/test_pc_multidirection_successor_exact_dev_20261103_codex.py
```
