# P3 budget HOST → native physical saved handoff

仅新增本stem源码/tests/doc，不改正在执行或封存的任何依赖。无run子命令，不加载神经/physics，不重新执行host.consume。

## 权威边界

prepare必须显式提供actual host report与MAIN持久化完整strict receipt的绝对路径/SHA。缺文件、错误SHA、partial、summary、少row、gate或actual不匹配都拒绝。验证完整冻结consumer ABI（包括全部full_rows）、report/prepare/源闭包、同24child48slot与原paid chooser，不用errors0摘要冒充strict。此层依赖MAIN给定receipt SHA作为完整消费权威；它不是独立重新证明host数组/AD正确性的validator。

所有编译成功槽只交接完整strict selected raw5，绑定child/context/goal/task/source、selection digest、typed float32原始位串与command SHA；paid拒绝仍保留付费signal与原decision；未到达仍unknown。不得挑结果、缩分母、变quota/chooser或借root J。

## 原生恢复与复用

完整调用冻结physical prepare.verify，保留P4原source/native snapshot/PNG、seed/placement/warmup、executed prefix、goal与actual context闭包。

`--reuse-policy strict_saved` 还必须完整调用冻结saved-v2 physical.consume，且complete=true后才能按固定同槽判定。对全48逐条按source manifest/P4/task/child、全部source frames/body float64 bits、实际boundary、goal/RGB/context及raw5 float32位串核对；不跨槽找相似轨迹、不按reward/success/earlystop挑复用。

- `reused_saved`：继承对应旧完整物理row与原早停，不称fresh或新支付；重复轨迹仍不是独立根。
- `pending_new_physics`：raw5不同或无匹配旧命令，必须列为待新执行；不能拼接旧结论。
- `paid_refused` / `not_reached`：完整保留，不产生命令或填0。

`--reuse-policy none` 不继承旧物理结果；所有可编译槽列pending，不自动执行。strict_saved若旧consumer失败，整个prepare拒绝，不按结果静默切换策略。

新physics预算只计pending槽：每槽warmup10+executed prefix b+至多own5 public，原生早停不padding；setup实际reset2/set_state4/goal_render2/setter3与4次setup physics，所有public各10次nested physics。actual新费保持null直到真实执行；历史993public/10094physics/1075PNG不能当本次新支付。全24child/48method来自2个已知根，P1/P3/C3/eqcompute/freshness/scientific gates均false。

## runner接线范围

每槽`runner_plan`保留原`agent_p3_child_native_physical_20261028_v1.execute`参数ABI；只有pending槽的plan.route为execute，selected来自新HOST完整strict。reuse/拒绝/未到达不会被当新执行。输出同时给原source_manifest与全源hash。

冻结旧CLI的verify绑定旧报告，**不能把本handoff冒充旧schema运行**。本任务交付prepare/verify和可复用原execute的payload，不增加物理run。MAIN下一执行版本需认证本新schema，逐槽使用原execute/refs/source_plan并实现适配新路由的saved consumer；不得直接绕过旧verify调用未授权run_verified。没有新执行器或新结果时，pending不等于已验证物理结果。

## MAIN可执行saved命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_p3_budget_physical_handoff_20261028_v1 prepare --host-report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_cuda_MAIN01/report.json --report-sha256 066b2f7f6e37fc9c4ed2795a07387c473da7e2e4418414e05ce0e6080ddc2978 --strict-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_consume_MAIN01.json --strict-sha256 14c87cd5c69562a53cacc7d3082a4938fcb776b814c9fc2b00dacd99df20bd0d --reuse-policy strict_saved --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_budget_physical_handoff_20261028_v1_prepare_MAIN01.json

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_p3_budget_physical_handoff_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_budget_physical_handoff_20261028_v1_prepare_MAIN01.json --prepare-sha256 ACTUAL_HANDOFF_PREPARE_SHA

OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_budget_physical_handoff_20261028_v1_test.py
```

输出须NEW、独占本stem、metadata≤64MiB且余量200MiB；验证源/raw/hash与路由可重复重建但不覆写。tests含RAM receipt fixture、真实saved report与MAIN完整strict输入回归、缺失/篡改/原始float位差拒绝；不把RAM mock old-consumer结果当旧物理实际复验，也不把测试当科学证据。
