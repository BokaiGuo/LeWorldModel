# PC2 与等槽位成本证据重建（2026-11-04）

## PC2 同链重建

运行：

```text
PYTHONPATH=. .venv/bin/python \
  experiments/pc2_retention_response_reconstruction_audit_20261104_codex.py
```

产物：

[reconstructed_audit_20261104.json](/home/gbk/code/LeWorldModel/le-wm-main/results/pc2_retention_response_same_chain_mechanism_20261104_codex/reconstructed_audit_20261104.json)

结果：

- 16/16 条记录的来源 response 计划与机制审计中的 `selected` 链逐元素一致；
- greedy 控制和 ridge-value 控制均与其来源候选逐元素一致；
- `comparisons`、分支稳定性、共同尾部收益、重规划收益和最终 smooth 判据均可由保存记录重算；
- 重算仍为 1/16，唯一通过的是 root 8；
- 当前 `lookahead` 不是普通 suffix lookahead，而是
  `local_ridge_reachability_value_minimum`；
- root 8 的 ridge-value 控制与 greedy 控制重复，不能作为独立普通 lookahead 对照；
- 该重建是保存结果的一致性审计，不是重新执行模型或物理，因此不升级 PC2。

## 成本字段重建

运行：

```text
PYTHONPATH=. .venv/bin/python \
  experiments/pc_retention_cost_reconciliation_20261104_codex.py
```

产物：

[cost_ledger_reconciled_20261104.json](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_retention_equal_slot_cost_dev_20261104_codex/cost_ledger_reconciled_20261104.json)

完整链样本数按 prediction journal 中动作形状为 `(N, 15, 2)` 的实际调用重新计数，
包含最终配对评分。相对于保存的 `resource.full_samples`：

- response、greedy、diverse、random 合计 64/80 条记录存在低计数；
- global CEM 从 journal 计算，没有发现同类保存字段缺口；
- 因此旧账本中的 `full_samples` 不能直接用于跨臂精细比较；
- 统一的 outer elapsed 已保留，但递归臂 inner planner wall time 与 global CEM
  的外层耗时边界不同，故 `timing_comparable=false`；
- GPU、编码、同步、内存和 FLOPs 仍未测量；
- `full_cost_certified=false` 保持不变。

## 处理决定

本轮只修复了证据口径和重建能力，没有改写旧结果、删除失败样本、替换控制定义或
重新挑选 root。当前 response retention 仍是开发负结果：2/16，global CEM 为
6/16；它不能作为正式优势方法。下一次方法开发若继续，必须另立版本，并在修复后的
journal/计时接口上预注册普通 lookahead、完整成本和停止规则。
