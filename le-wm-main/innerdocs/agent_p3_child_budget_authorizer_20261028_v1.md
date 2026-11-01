# P3 child fixed/adaptive 实际预算 authorizer

这是独占的 RAM/saved-only 预算合同入口，不是神经执行器、physics 执行器或科学准入器。
它不改 common budget、child compiler、saved physical bridge 或任何封存结果。

## 取整与节点规则

每个 child/method 保留原始 policy 的 `horizon ∈ {1,3}` 与 `candidates ∈ {4,8}`。
搜索预测预算严格为 `2 × horizon × candidates`，并按整节点判断：

| 原始节点 | search predict | authorizer |
|---|---:|---|
| H1×4 | 8 | 可授权其完整节点 |
| H1×8 | 16 | 可授权其完整节点 |
| H3×4 | 24 | 可授权其完整节点 |
| H3×8 | 48 | 原子拒绝，不能取整到4或缩候选 |

旧 `agent_p3_common_budget_20261028_v1` 的 `24/(2H)` 是另一个 root comparison
wrapper 的候选分配（H1→12、H3→4），不能移植到 child policy，也不能改写原始4/8。
本 authorizer 的 `quota()` 明确返回 `legal/cost/refusal`，所有非法节点保留拒绝原因。

## fixed vs adaptive 合同

fixed 与 adaptive 使用完全相同的预注册逐轴上限：

- signal：8 predict/sample/context、2 J、384 VJP/grad/raw、2 array-SVD、1 schedule；
- compiler：按原始 H×候选的完整 `2H×N+1` predict/replay/metric、1 compile、1 array-SVD、1 solve；
- `common_node_cap = signal + 最大合法 compiler envelope`；
- `common_method_cap = 24 × common_node_cap`。

这是 equal-cap 合同，不是 equal-actual 合同。实际 ledger 逐轴记录 attempted/completed；只有 fixed
与 adaptive 所有资源轴逐项相等时才可输出 `matched_actual_vector=true`。本实现不把 391/634、不同 H、
不同候选数或不同 wall/FLOPs 折合成等算力。

## 预授权与不可退款账

按固定 48 槽顺序打开 `(child, fixed)`、`(child, adaptive)`。打开前预留 signal + 最大合法 compiler
envelope；输入必须是同一 child 的 context、task binding 和 observed ancestry。未到达槽直接保留
`not_reached`，不填零。

signal 完整返回后，调用原冻结 chooser 的精确结果；不改阈值、不重分配、不 round。H3×8 或
unsupported reconfiguration 在 signal 已付的前提下整节点拒绝，signal attempted/completed 永不退款。
每次实际 hook 先落 `attempted` 再调用，返回才落 `completed`；失败保留 pending、fail-stop，禁止 retry。
所有事件带链式 digest，saved `consume-gate` 可 RAM 重放 receipt。

运行时依赖故意 fail-closed：没有 MAIN 审核的 native signal/compiler adapter、完整数组/seal consumer、
source/prepare 闭包和 HOST CUDA 授权时，机器可读返回 `precondition_refused`，`dispatch_allowed=false`，
并列出 `missing`；本 stem 不导入 torch/gymnasium，不运行模型或 physics。

## saved authority

authorizer 的 `load_saved()` 先绑定 compiler source、实际 compiler report
`885f69f6de5ce9b78b8f3acdc84f14ef8891ba3ecd79912fc9af4e1920fc7ffb`、完整 compiler receipt
`d6322717231d72bb18e26eab1acbf6afe2dbc65eaa601af1fface21083a7e91c` 与已完成 child signal receipt，
再做 24 child/48 method projection。saved projection 保留 41 pending、5 original whole-node refused、
2 not_reached；不把历史摘要当新执行授权。

## 测试和 CLI

测试只读取已保存 JSON 或 RAM gate，覆盖取整、H3×8 原子拒绝、fixed/adaptive 同 cap 但不同 actual
向量、context/order/attempt 失败、链式 receipt 和机器可读依赖拒绝。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_budget_authorizer_20261028_v1_test.py
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_authorizer_20261028_v1 inputs --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_authorizer_20261028_v1_limits_MAIN01.json
```

把 inputs 的实际 SHA 作为 `--input-sha256` 后生成独占 prepare：

```bash
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_authorizer_20261028_v1 prepare --input /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_authorizer_20261028_v1_limits_MAIN01.json --input-sha256 ACTUAL_LIMITS_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_authorizer_20261028_v1_prepare_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_authorizer_20261028_v1 plan --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_authorizer_20261028_v1_prepare_MAIN01.json --prepare-sha256 ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_authorizer_20261028_v1_plan_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_authorizer_20261028_v1 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_authorizer_20261028_v1_prepare_MAIN01.json --prepare-sha256 ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_authorizer_20261028_v1_consume_MAIN01.json
```

上述 CLI 全部是 saved metadata；没有 `run` 子命令，不会启动 CUDA。测试不是科学证据，P1/P3/C3 和
scientific admission 始终 false。
