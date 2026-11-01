# P2 预算 guard 与原生搜索 RAM 接线验证

本轮不新增预算框架、不修改任何 producer/旧封存；仅新增
`le-wm-main/tests/chat_p2_20261027_test_native_budget_integration.py` 和独占说明/
结果。继续使用 experiment-code 技能组织固定配置验证。没有子智能体、模型训练、
CPU 神经 fallback、CUDA allocation、物理、下载、删除或推送。

## 复用的真实实现

直接使用 `p2_supported_resource_guard_20261027_v1.BudgetLedger`，其 backend SHA
由原 guard 检查。沿用前轮同输入 fixture 和五个原生搜索入口，不改算法、候选
规则、winner 选择或科学阈值。

guard 在 RAM 回调进入前 reserve，成功后 finish；失败保持 reserved/pending，
不以 finish(0) 假装实际完成零工作。回调前 cap 拒绝不会进入底层回调。unknown
轴依旧 null，不能因为没有测量就填0。该集成仅发生在测试回调，不声称原生 CUDA
producer 已采用此 supported-axis guard；实际 CUDA producer 仍用自身封存计数器。

只在明确 RAM 范围内固定 callback 上界：joint80/conditional40 为164，SRA40/
CEM40/random40 为84。这些是原搜索/replay与 mock suffix 的调用量，不含共享
compile、真实 AD、physics 或端到端 RGB acquisition，不能充当完整 pipeline cap。
suffix mock 的 nominal+ADforward 两次回调用一个两单位 reservation；此处没有
真实 Jacobian，不把 mock 算成一次实际数学导数。

## 实际执行验证

新增17项 RAM 原生接线测试：

- 五个 arm 在精确 callback cap 下完整完成，没有 padding。
- 各 cap 减1时，下一个回调在进入前被原 guard 拒绝，禁止发布完整候选。
- 五个 arm 的首回调失败后 reservation/pending 保留，完整候选 census 不缩分母。
- 组合首包花费1后，cap2不足以 reserve 两次 suffix 回调：两次都未执行。
- suffix 分组失败保留两单位 reservation，但不伪称两次均已完成；已完成下界
  只有先前成功首包，reserved3/completed1/pending1。

最初失败注入按全局packet2选择，与两单位 suffix 分组边界不一致，导致一个新
测试未触发；改为五臂统一首回调失败，并补显式 suffix 分组失败测试。未修改
任何旧测试、配置或门槛。

新增测试连同原 guard/resource arithmetic 测试：128 passed in 2.75s。
文末完整联合回归：557 passed in 27.83s。
实际生成 `le-wm-main/results/chat_p2_20261027_budget_integration_ram.json`，保存
五臂×正常/少1预算/回调失败共15场景的完整 reservation receipt、错误事件和
census摘要。它是 fixture_only 的 RAM 证据，不是实际神经成本或控制收益。

生成前重新执行前轮 native bundle verify，原三套真实 producer 的输入/封存与
100个源码绑定仍通过；未重封旧资产。bundle canonical SHA保持
`0d332bb659c03d4ed947445e57c7da81361e00ac664390748716e6b9def70974`。

## 明确的剩余阻断

现有 supported guard 不能执行以下轴的有限 cap，即使填0也拒绝：
backward_engine_api_calls、cuda_kernel_calls、flops、synchronized_device_time_ns、
peak_gpu_bytes、cpu_validation_time_ns、inherited_context_compute_units、
physics_internal_substeps。不得绕过它，不能将 supported-axis reservation 改称
全资源有效对照。补齐这些轴需要主线程批准的新执行/测量合同及真实数据。

仍缺真实 P1/freshness 外部授权、组合 v2 完整运行报告和全资源共同预算。
不启动确认性队列，不接管 P1 或 Nash/Bohr moving_overlay_capability_probe。
本轮结果不改变 P1/P2 admission；完整个人目标保持未完成。

回归（cwd `/home/gbk/code/LeWorldModel/le-wm-main`）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
 tests/chat_p2_20261027_test_native_budget_integration.py \
 tests/chat_p2_20261027_test_native_search_integration.py \
 tests/chat_p2_20261027_test_producer_bundle.py \
 tests/chat_p2_20261027_test_lifecycle.py \
 tests/test_moving_p2_saved_program_graph_20261027_v1.py \
 tests/test_moving_p2_composition_prospective_queue_20261027_v1.py \
 tests/test_p2_composition_predict_frontier_cuda_20261027_v2.py \
 tests/test_p2_no_composition_sra_20261027_v2.py \
 tests/test_p2_full20_sampling_20261027_v2.py \
 tests/test_p2_supported_resource_guard_20261027_v1.py \
 tests/test_p2_full_resource_matching_20261027_v1.py
```
