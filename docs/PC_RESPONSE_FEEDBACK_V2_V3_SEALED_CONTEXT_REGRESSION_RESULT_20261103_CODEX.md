# V2/V3 固定上下文回归结果与更正

## 范围与结论

在此前 24 个已暴露任务的封存第十步状态和 carry 上，仅重算最后五步的规划；
没有重新执行前十步，也没有增加任务或调参。每根规划的预测预算均为 480
token。各批预测以精确动力学重放，检查动作、评分、选中链、预算及产物哈希。
这里的物理状态输入仅适用于精确动力学开发，不代表合法 RGB 输入。

| 方法 | 相对 V1 的末步结果 | 平均终态评分 | 成功 |
|---|---|---:|---:|
| V2 | 18/24 条链相同；5 根评分更低、1 根更高 | 0.4814442782 | 1/24 |
| V3 | 24/24 条链与评分相同 | 0.4816535267 | 1/24 |

V3 每根最终比较前的 66 个预测批次与 V1 的动作、评分逐项一致。V3 的
合成反例测试证明：当已评分的完整探针优于旧选择时，新选择保留该探针；
在这 24 个固定上下文中没有发现此漏洞实际触发。测试与该面板均不能说明
其他上下文不会触发。

## 对 V2 先前说明的补充更正

[原修复说明](PC_RESPONSE_FEEDBACK_FULL_HORIZON_INCUMBENT_FIX_20261103_CODEX.md)
描述了 V2 维护全局 incumbent，但未充分说明：V2 在联合精修前就把 carry
及探针最优值用于精修初始化；V1 则在精修之后才比较 carry。实际有 11 根
从联合精修开始的第 56 批采样已发生变化。因此 V2 不只是选择阶段修复，
其六条最终链的变化不能单独归因于探针保留。原报告及结果保留不覆盖。

V3 按[预先协议](PC_RESPONSE_FEEDBACK_V3_SELECTION_ONLY_PROTOCOL_20261103_CODEX.md)
保留 V1 的生成和精修顺序，仅在末步最终比较时纳入所有已评分完整链。
第 0、1 阶段及 full-CEM 臂仍委托 V1，不宣称这些阶段也已做全局 incumbent
修复。

## 复现与审计

V2 的 [计划](../le-wm-main/results/pc_response_feedback_v2_sealed_context_regression_20261103_codex_plans.json)
与 [审计](../le-wm-main/results/pc_response_feedback_v2_sealed_context_regression_20261103_codex_audit.json)、
V3 的 [计划](../le-wm-main/results/pc_response_feedback_v3_selection_only_regression_20261103_codex_plans.json)
与 [审计](../le-wm-main/results/pc_response_feedback_v3_selection_only_regression_20261103_codex_audit.json)
均已封存。审计记录的计划 SHA-256 分别为
`1213be7b9a7a91d32d7d5ad6727808fdf3c926faf89d6cc7c1f61ce48b86a85a` 和
`f74e7cbdce693794e7af9a10938f1ac01470c2c01b8de95c22357932a06caa48`。

从 `le-wm-main` 运行以下回归，2026-11-03 结果为 9 passed：

```bash
.venv/bin/pytest -q -p no:cacheprovider \
 tests/test_pc_response_feedback_v2_v3_sealed_regressions_20261103_codex.py \
 tests/test_pc_response_diverse_feedback_v3.py \
 tests/test_pc_response_diverse_feedback_v2.py \
 tests/test_pc_response_diverse_feedback_v1.py \
 tests/test_pc_response_diverse_fixed_terminal_exact_dev_20261103_codex.py
```

该结果是同机器、已暴露根的固定上下文工程回归，不是 V3 三轮完整反馈
实验、独立复现或 RGB 闭环，也不证明 PC2–PC5 通过。无法据此把 V1
三轮结果称为 V3 的完整实测，或把 V2 的数值差异解释成方法优势。
