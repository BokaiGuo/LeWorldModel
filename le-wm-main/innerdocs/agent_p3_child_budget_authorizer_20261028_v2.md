# P3 child budget authorizer v2

这是 `agent_p3_child_budget_authorizer_20261028_v1` 的独占 fail-closed 版本化修复；v1、producer、signal、compiler、physical bridge 和所有封存结果均不修改。

## 修复的两个边界

1. `limits()` 对完整 canonical limits 对象做深拷贝等价校验，并固定
   `common_node_cap` 与 `common_method_cap` 的 canonical cap-vector SHA：
   `4de9887ccc5b63b3b4ac82c6847877bd3ece96b73f81307e647f08fbf3b7e0eb`。
   任一资源轴、allocation、signal sharing、slot order 或科学门字段变更都会拒绝；不会通过减少某一轴后重新解释为合法预算。
2. 失败 attempt 生成持久 `failure_ticket`。attempted 已记账且 pending 不清除；失败后下一固定槽即使请求 adaptive 也只产生 `fail_stop` refusal，不会重新 open、换策略、退款或 retry。receipt 中保留失败事件、ticket、pending 和原分母。

整节点仍按 `2*h*c <= 24` 原子授权：H1×4=8、H1×8=16、H3×4=24；H3×8=48 原子拒绝，绝不取整或缩候选。fixed/adaptive 只共享同一预注册 cap，不宣称实际 ledger 相等。48 method 槽与 24 child 分母、`41 pending/5 refused/2 not_reached` 语义保留。

## 限制

这是 saved/RAM 预算合同，不是神经、CUDA、physics 或 scientific admission。`runtime_preconditions` 继续返回机器可读 `dispatch_allowed=false`；测试和 saved plan 不是科学证据。

## 测试

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m py_compile /home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/agent_p3_child_budget_authorizer_20261028_v2.py
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_budget_authorizer_20261028_v2_test.py
```

## CLI

所有输入/输出路径都必须是绝对路径；输出必须是不存在的新文件，且只写 bounded metadata：

```bash
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_authorizer_20261028_v2 inputs --output /ABS/limits.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_authorizer_20261028_v2 prepare --input /ABS/limits.json --input-sha256 ACTUAL_LIMITS_SHA --output /ABS/prepare.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_authorizer_20261028_v2 plan --prepared /ABS/prepare.json --prepare-sha256 ACTUAL_PREPARE_SHA --output /ABS/plan.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_authorizer_20261028_v2 consume --prepared /ABS/prepare.json --prepare-sha256 ACTUAL_PREPARE_SHA --output /ABS/consume.json
```

CLI 不包含 `run`，不会启动神经、CPU fallback、CUDA、physics 或网络。
