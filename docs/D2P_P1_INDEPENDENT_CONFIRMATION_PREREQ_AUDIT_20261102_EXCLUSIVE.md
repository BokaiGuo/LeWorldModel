# P1 独立确认前置：fail-closed 认证合同

日期：2026-11-02  
状态：**前置未认证；本文件不授权 GPU、物理、训练或 gate 修改**

本次只新增本文件、审计器和测试。审计器
`le-wm-main/experiments/spectral_reachability/p1_independent_confirmation_prereq_audit_20261102_exclusive.py`
只消费显式 JSON 元数据，不导入模型、数据集、模拟器或 CUDA。它输出
`PASS` 或 `FAIL_CLOSED`；任何 `unknown`、不完整声明或负控制违规都阻断。

## 当前 fail-closed 缺口

1. **root identity 未达独立权威。** 现有 dev303 sidecar 只能说明在已封存
   DEV301/DEV302 bounded ledger 中未发现 seed `301000003` 的执行；不能证明全局历史、
   未记录重启或开发者盲性。
2. **source receipt 不构成完整来源闭包。** dev303 准备资产有 source receipt/hash
   绑定，但其合同仍保留 `historical_coverage=unknown`，且不能替代完整 reset/event
   ledger、producer run、源文件闭包和穷尽尝试上界。
3. **历史排重不完整。** 现有历史排重索引明确保留 unknown scope，不能由“未找到记录”
   推出 fresh；精确 hash 计数也不能替代 state/goal/action-history 的语义或近似排重。
4. **checkpoint training/split provenance 未认证。** 现有 checkpoint 配置审计明确
   `training_identity.verified=false`；缺 checkpoint→training run、训练数据字节、
   split manifest/member identity、normalizer fit/export 和训练代码 revision 的权威绑定。
5. **dev303 不得转证。** dev303 是 development-only holdout。任何 dev303 result、
   selector outcome、物理 outcome 或由其选择 baseline 的结果进入独立确认，均直接阻断。
6. **本次没有 gate 变更。** `gate_modified` 必须作为负控制保持 `false`；审计工具不具备
   修改 gate 的路径。

## 可执行认证合同

候选独立确认只能提交一个全新的 `candidate_id` 和 source namespace；不得复用
`dev303` 命名空间。提交方必须提供以下单一 manifest，字段集合不可扩展：

| 证据面 | 必须满足 |
|---|---|
| root identity | `status=verified`、`authority=independent_authority`；root/reset/goal/action-history SHA 全部绑定；权威范围覆盖所有已知运行、重启和替代 producer |
| source receipt | `status=verified`、`complete=true`；queue、reset event、producer run、每个普通文件及 SHA 全部绑定；不得用 symlink 或缺失文件声明替代 |
| historical dedup | `status=verified`、`complete=true`；索引覆盖声明范围；同时核查 root、goal、action-history，且 `approximate_duplicate_check=true` |
| checkpoint provenance | `status=verified`、`authoritative=true`；checkpoint/config、training run、training dataset、training split manifest、normalizer、training code revision 全部 SHA 绑定；`split_membership=training_disjoint` |
| negative controls | `dev303_used=false`、`outcome_used_for_selection=false`、`gate_modified=false` |

认证顺序固定为：先冻结并 hash manifest 与全部源 receipt，再运行只读审计器；只有
审计器返回 `PASS` 才能进入后续执行授权。审计器 `PASS` 仍只是来源/独立性前置，不是
P1 科学结果或 gate 通过。任何运行中的 CUDA、物理、训练和 outcome 选择都不属于本次
前置，并且本合同不授权它们。

推荐命令：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  experiments/spectral_reachability/p1_independent_confirmation_prereq_audit_20261102_exclusive.py \
  < independent_confirmation_manifest.json
```

退出码 `0` 仅表示合同字段全部满足；退出码 `2` 表示 `FAIL_CLOSED` 或输入非法。
审计器不会创建、覆盖或修改结果文件。

## 测试

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider \
  tests/test_p1_independent_confirmation_prereq_audit_20261102_exclusive.py
```
