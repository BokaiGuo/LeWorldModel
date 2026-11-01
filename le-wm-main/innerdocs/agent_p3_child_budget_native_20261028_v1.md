# P3 budget-instrumented native child adapter v1

这是独占的预算接线层，不修改冻结的 budget authorizer v1/v2、child signal/compiler、physical bridge 或任何结果。它把冻结 v2 `WholeNodeGate` 接到原有 native `core.Ledger`：每个 `resource_attempt` 事件先调用 gate 的 `begin`，只有预授权成功才允许原 Ledger 的实际 callable；`resource_return` 才记 completed。异常、sink 写失败和未返回 attempt 保留 attempted/pending，并使 gate 永久 fail-stop。

## 合同

- 固定 `(child, fixed)`、`(child, adaptive)` 48 槽和 24 child 分母，顺序不可跳过、重入或换 sibling。
- 每个 method 独立付 signal：原 native signal 的 H1/H3 nominal+AD、真实 shape、VJP/raw callback、array-SVD 和 schedule 由原 Ledger 逐项计账；不将 fixed/adaptive 的 cap 当 actual 相等。
- chooser 使用原冻结 `core.choose`，seal 前不得使用 outcome、Q 或物理状态。H3×8 等原 policy 预算拒绝保留 paid signal，不缩候选、不退款。
- compiler 使用原冻结 `execute_branch`/`NativeProgram`，其 program compile、whole/segmented replay、metric、selected raw5 prefix 由同一 child/mode gate 预授权。
- 所有原 child context、task binding、ancestry、真实 `[1,3,192]` 等 shape 和 source pins 保留。未到达、失败、拒绝均不是零。

`NativeFactories` 的构造要求 `MAIN_APPROVED_BUDGET_CHILD_CUDA`、CUDA device 和已审 frozen model；本 CLI 没有 run 子命令，因此不会加载模型或运行神经/physics。当前实现是接线与 RAM/saved replay，不是新 CUDA 结果、strict fullarray consumer，也不宣称 C3/P3 scientific admission。

预算上界仍是冻结 gate 的逐轴 vector：每个 method-node 最多33 predict API、2J、384 scalar VJP，24 child/method 上限792 predict，两个方法合计上限1584 predict、96J、18432 scalar VJP；这是预授权上限而非已付实际量，非FLOPs换算。原始41/5/2 census只作历史参照，新付signal必须在原chooser下重新决策，不能照抄旧decision跳过付费。新HOST整作业 wall/output/memory guard 尚未接入，故不提供 run，也不称已经形成可放行CUDA任务。

## Saved prepare / verify

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_native_20261028_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_native_20261028_v1_prepare_MAIN01.json

/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_child_budget_native_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_native_20261028_v1_prepare_MAIN01.json --prepare-sha256 ACTUAL_PREPARE_SHA
```

`prepare` 只读取已保存的完整 producer consumer/source closure，输出必须是 `results/` 下不存在的新独占文件；`verify` 只做 SHA/source/24-child/48-method 重建。实际 HOST CUDA 接线不由 CLI 启动，需 MAIN 审核 `NativeFactories`、外部 lifecycle/capacity guard 和新的完整 strict consumer 后另行授权。

## 测试边界

```bash
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_budget_native_20261028_v1_test.py
```

测试仅 RAM callback、saved ledger shape/census、attempt 异常、journal sink 失败和授权拒绝；测试不是科学证据。

实际联合回归：本stem11tests与冻结budget v2的6tests，17 passed in12.27s；未运行模型、CUDA、physics、网络或真实MAIN结果写入。saved prepare CLI留给MAIN执行；测试中的material消费边使用mock检验调用顺序，不冒称本轮已重新完成完整producer.consume。
