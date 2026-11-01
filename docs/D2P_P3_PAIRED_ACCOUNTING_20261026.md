# P3 固定预算与自适应调度配对核算（2026-10-26）

状态：仅新增纯逻辑验证器与合同测试。**P1 尚未通过，P3 未准入；没有效益、
compute advantage 或 walltime advantage 结论。** 不改变既有门禁，不启动
GPU/神经计算，不修改封存 runner、协议或现有预算模块。

## 输入与使用

实现：`le-wm-main/experiments/spectral_reachability/d2p_p3_paired_accounting.py`。
读取并沿用 `D2P_ADAPTIVE_BUDGET_V2_20261026.md` 和
`d2p_adaptive_budget_v2.py` 的 attempted predict 语义，复用
`d2p_planning_compute_ledger.py` 的完整 `Resources`、`BudgetContract`、
`Binding` 与 `compare_pair`；不将候选数或调度估计视为实际收据。

调用入口为 `compare_paired(plan, fixed_receipts, adaptive_receipts)`：

- `PairedPlan(source_seal, roots, fixed_method, adaptive_method)` 必须在 outcome
  打开前独立封存。每个 `RootPlan` 提供完整 per-root `BudgetContract` 与两方法
  共同的预定 branch ID 清单。不可从成功行、两组交集或事后结果构造该清单。
- root ID 必须绑定实际初态、RGB/history 与 goal，而不是列表位置。协议、任务、
  frozen model 必须全组一致；方法不同。每个 root 可有不同但对两方法共享的
  冻结向量上限，不能看完结果再改上限。
- 每个 `RunReceipt` 提供绑定、方法、全部 `BranchReceipt`、完整 overhead
  `Resources`、原 `PredictionBudget.receipt()` 字典和可选实测 walltime。
  即使分支失败也必须提供所有资源字段；未知不能默认成零。
- branch 是预定的共同任务/执行比较槽，不是 imagined candidate。调度可以
  搜索不同数量候选，其开销由资源字段记录，不能把成功候选列表当 branch 清单。
  若方法提前终止，保留后续槽并标为 blocked/aborted，真实未执行的工作才可填零。

按 root identity 建索引、按冻结清单顺序输出；拒绝重复、多余、缺失、混绑 root，
两组同时删除同一失败 root 也拒绝。branch 乱序允许，但集合必须恰好等于预定清单；
两组同时删除失败 branch 也拒绝。状态包括 completed/failed/blocked/aborted；
pending 不构成最终核算。只有 completed 可标 task_success；completed 本身不代表成功。

## 分轴核算与失败保留

每个 root 的总资源 = overhead + 所有 branch 的 attempted resources。
其中 `max_planning_horizon_tokens` 取最大值，其余字段相加；不存在资源换算权重，
不允许预测省一点来抵消反向/执行超支。overhead 与 branch 必须不重叠。
谱信号生成、setup、encoder、重试、验证、evaluation 的所有独占成本必须纳入；
共享成本/排除项政策须事先绑定在 protocol_id 中，不可事后免单。

| 独立资源轴 | 语义 |
| --- | --- |
| predict_calls / predict_samples | 全部实际调用次数 / 每次真实 batch 逐 token 累加，异常不退款 |
| encoder_forwards / encoder_samples | 独立编码成本，不能由 predict 推断 |
| action_vjp_rows / action_backward_calls | action 逻辑导数行 / 实际反向调用次数，分别计 |
| input_vjp_rows / input_backward_calls | 输入导数另计，不并入 action |
| rollout_samples / imagined_action_tokens / max_planning_horizon_tokens | 想象轨迹数 / B×H / 最大时域，不能当真实执行 |
| physical_action_tokens / physical_env_steps / physical_branches | 真实动作 token、env.step 与恢复执行分支独立计，不从想象成本推算 |

验证器核对 PredictionBudget v2 的 schema、完整绑定、方法与冻结预测上限，
将所有 planning/evaluation phase（含失败、不完整 phase）的 predict_calls、
predict_samples 累加，对账 attempted_predict_samples 及总资源向量。
保留 failed_predict_calls、blocked_calls、incomplete_phases；拒绝负数、bool
计数、缺失计数、成功子集退款、假完成标志和超预测上限收据。调用前 blocked
不算 predict 工作，但 blocked 事件保留；先前失败委托的资源不退款。

每对分别输出 baseline/candidate 相对上限、candidate 相对 baseline，以及
各轴 adaptive-minus-fixed 差值。`both_within_budget` 不等于 `equal_resources`；
相等但双双超限仍不合规。反向/真实执行超支保留在报告中并将合规标志置 false；
预测超限或收据对账失败则直接拒绝输入。失败分支保留状态、资源与固定分母，
successes 仅为描述性计数，绝不据此筛选比较对象。

## 明确拒绝的解释

`comparison_scope` 只允许 `all_planned_branches`，拒绝 successful-only/过滤比较。
`walltime_advantage_claim=True` 总是拒绝；未测量 walltime 保留 None，不用零替代。
即便全部提供 walltime，也只返回实测完整性标志与原数值，不推断速度优势。
调用数、样本数、VJP、执行 token 都不等于 FLOPs 或墙钟。

输出始终保留 `p1_dependency="not_passed"`、`p3_admitted=False`、
`compute_advantage=False`、`walltime_advantage=False`、`efficacy_claim=False`。
没有 P1/P2 后独立真实的同预算任务后果、完整失败轨迹、噪声/打乱信号对照和
质量—成本曲线，本验证器不能授权后续科学阶段或声明收益。

## 信任边界

本模块不采集 backward、执行计数或 walltime，不接入 runner，不分配设备。
完整 `Resources` 是调用方提供的收据断言；不能证明未漏掉模型路径、额外分支、
共享成本或真实失败，也不能证明 source seal 前瞻有效。外层必须独立审计
冻结 cohort/branch 清单、root 身份、完整日志、非重叠成本与测量覆盖。
未记录工作不得冒充零。`RunReceipt` 内 prediction_receipt 为原始可变字典；
应在 runner 结束后冻结/持久化并校验 seal，不支持并发修改输入。

神经实验仍必须主机 CUDA，无 CPU fallback。本次唯一 guard API 测试使用
无 torch、无张量、无神经执行的 CUDA 元数据假体，**不能证明 CUDA 可用**。

## 主线程 CUDA 搜索现状背景（只读核对）

已核对 `D2P_PROGRAM_SEARCH_CUDA_INTEGRATION_20261026.md` 与
`le-wm-main/results/pusht_program_search4_v2_20261026_cuda/report.json`。
报告为 `integration_completed`，8 项完整性门均 true；P1/P2/P3 admission 均 false。
四根各有 80 个 planning predict samples；完整动作串 verification 使用已有
`evaluation` phase，分别额外 2/2/1/2，故实际预测总量为 **82/82/81/82**，
合计 327，不是仅规划的 320。根 2 停在深度 1，不能补零伪装等长比较。
四根收据的 predict sample cap 均为 82；根 2 under cap 不等于与其他根相同使用量。
后续绑定应使用报告中的 `root_sha256` / budget.binding.root_id，而非 root_index。

这只是已有真实主机 CUDA 搜索的预测核算背景，不是本任务新跑实验，亦不是
固定与自适应调度的独立 paired 证据。报告只有 predict guard 收据，不足以提供本模块
要求的全部独立资源轴、预封存配对 cohort/branch 清单与两组完整 RunReceipt。
不能把未知 encoder/backward/真实执行成本填零以强行通过验证，更不能据 327
预测样本声称总资源匹配、walltime 优势或 P3 效益。未编辑上述报告或主线程文档。

## 验证证据

Python cwd：`/home/gbk/code/LeWorldModel/le-wm-main`。
使用普通解释器执行纯逻辑合同测试：

```bash
python -m pytest -q tests/test_d2p_p3_paired_accounting.py
python -m pytest -q tests/test_d2p_p3_paired_accounting.py tests/test_d2p_adaptive_budget_v2.py tests/test_d2p_model_call_instrumentation.py tests/test_d2p_planning_compute_ledger.py
```

本次执行结果：新增测试 **57 passed**；连同三个现有模块的相关回归共
**189 passed**。均使用普通解释器，仅验证逻辑/收据合同，没有启动 GPU。

覆盖 root/branch 乱序、单侧/双侧丢失败、重复/多余/混绑、完整失败成本和分母、
各独立资源上限、预测精确上限/超限、收据对账、成功子集拒绝、未知资源不能默认零、
walltime 声称拒绝与原 guard 的元数据假体 API 集成。所有结果仅为逻辑合同证据。
新增测试另用搜索形状的合成计数覆盖 80 planning + 2/2/1/2 evaluation，拒绝
只报 planning 的账目；不依赖可变的真实报告作为 fixture，不重建其未知资源。
