# v2 封存上下文回归结果

依照[事前协议](PC_RESPONSE_TEMPORAL_V2_CONTEXT_REGRESSION_PROTOCOL_20261104_CODEX.md)，
对已暴露的 24 根、两臂、后两次决策共 96 个 v1 封存上下文运行 v2。
独立审计重新计算所有批次的精确动力学端点及分数，并检查原 carry、
float32 转换、实际 token、最终完整链全局最优和零前缀弃权：96/96
通过；每次实际消耗 1919 或 1920 token。

- 95/96 选中链与原 v1 相同；96 次中没有一次在 v2 内部 v1 选择后
  被全局 incumbent 检查改写，弃权 4 次。
- 唯一不同为根 3、response、第 1 次后续决策：v1 分数
  0.5006632544，v2 分数 0.5014638910。v2 没有评分该 v1 链；
  v2 的预算预留改变了采样，不能将这一差异归因于探针保留修复。
- 反例和两个方法版本的测试共 16 项通过。这证明所测代码行为，
  不证明该方法在真实任务中获得科学优势。

本实验复用旧上下文，不执行新的闭环轨迹，也不是独立确认或 PC2、
PC3 机制和性能证据。v2 仍未确保在联合精修前采用全部探针中的最佳
incumbent；旧 v1 的 RGB 和 E0 效果不能归给 v2。

复现：

```bash
.venv/bin/python -m experiments.pc_response_temporal_v2_context_regression_20261104_codex plan
.venv/bin/python -m experiments.pc_response_temporal_v2_context_regression_20261104_codex audit
```

入口采用排他创建；已有产物时不会覆盖。复现时应复制封存依赖到干净
环境和新的输出路径，不能直接重运行覆盖原结果。

- [逐批计划](../le-wm-main/results/pc_response_temporal_v2_context_regression_20261104_codex_plans.json)，SHA-256 `1b889af61e2a647f267f1c0a0729fdcafc17cb5dfe74f6ad15ebc6ed7a561de2`
- [独立审计](../le-wm-main/results/pc_response_temporal_v2_context_regression_20261104_codex_audit.json)，SHA-256 `524a7d6d27aca04d5b315a05b4acf2925ea937bab1abdb57f1eae2a48a46467b`
- [运行与审计入口](../le-wm-main/experiments/pc_response_temporal_v2_context_regression_20261104_codex.py)
