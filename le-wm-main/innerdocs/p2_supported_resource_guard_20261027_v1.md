# P2有限计费轴静默忽略：独立保护与P3边界

本交付只新增同stem模块、tests和本文，不改sealed源码、旧结果、共享总文档；无模型/AD/物理调用、无网络。模块CLI仅输出stdout，无结果目录、缓存或临时文件。

## 具体发现，不是缺项清单

`p2_full_resource_matching_20261027_v1.py`的`validate`允许所有AXES为非负整数或None；但`allocate`（111–122行）、`BudgetLedger`（132–155行）只对静态KNOWN集合做cap/完成量检查。UNKNOWN中的轴若被调用方明确赋为有限数值，既不拒绝，也不执行其约束。

内存反例使用真实saved common cap和conditional recipe的副本，仅将`backward_engine_api_calls`改为cap=0、fixed=0、unit=1：旧allocate仍接受40候选，usage缺此轴。另一个纯算术ledger反例cap=0、reserve=1、finish=2仍成功，event保留1/2，汇总却为None。

这是**未来接入合同静默丢失**，不是历史实验发生了未付费AD的证明。当前封存报告这8轴均为None，历史数学VJP计数也不能猜换成backward-engine API调用数。旧模块已明确限定measured-axis scope；修复针对其允许接受超出scope的有限输入，而非推翻该scope。

## 已实现的修复

NEW `p2_supported_resource_guard_20261027_v1.allocate/BudgetLedger`显式拒绝任何UNKNOWN轴的有限值，包括0，且在分配/状态变更前拒绝。其余KNOWN算术委托给SHA绑定的旧后端，保留AD维度、部分失败不退款、唯一work_id、禁止超额完成、deep-copy receipt等规则。新ledger以组合封装后端，不暴露可直接写的cap/reserved/events属性。

不是把unknown补0，也不是擅自新增时间/峰值内存的累加规则。若将来确有实测有限值，需要NEW版本实现对应聚合和预调度语义（例如peak bytes应取max），或另用能落实这些cap的后端；本保护拒绝而不是忽略。

**接线边界：**这是已可调用、已回归的纯RAM adapter，不是durable journal，不是CUDA guard。旧sealed调用点没有改，故保护不会自动覆盖旧runner。main可在NEW runner中显式导入本模块的allocate/BudgetLedger；不能声称旧执行已被追溯修复。哈希检查只绑定磁盘源码，不是对Python进程内猴子补丁或恶意绕过私有属性的安全认证。

## 真实saved证据与P3解释

CLI核验以下SHA，检查saved分配与新adapter逐字段一致；不重跑旧audit_current，也不宣称重验整个递归source closure：

- 后端源码 `experiments/spectral_reachability/p2_full_resource_matching_20261027_v1.py`：`fe1b6a3833d86918789d1227f773c56b298c30a6830f3bc64c22c3f15bd93018`。
- `results/p2_full_resource_matching_20261027_v1_audit01.json`：`9d4a56ec78afc3c1704d29e92fb9ff6b52f5c00c258a01f045210dccdc8d884f`。原80 joint / 40 conditional有用候选分配完全不变。这是prospective cap allocation，不是重新执行80/40，也不是equal actual vector。
- `results/p3_adaptive_budget_saved_audit_20261027_v1_saved01/report.json`：`29aa5458496a4d8118e78a173182e45d652739521d9664b6402668a94e602a2e`。joint机制为root signal预分配后枚举整个assigned pool；fixed-H3改变保留parent数量但仍固定144+3；runtime有causal layer feedback，但该saved证据未建立signal带来的预算节省。三者budget_saving_evidence均false，不能把全库/全allocated pool枚举说成adaptive budget advantage。

joint的AD forward已在shared中，backward-engine API次数仍unknown；本轮不据此声称新发现历史AD漏计。joint saved source_causality只支持signal arrays先于search，allocation无独立timestamped journal；runtime未保存所有child latent，不能从summary独立重算所有SVD。历史cache acquisition unknown不等于免费。本轮未建立通用cache/lookahead无泄露认证，不改目标、不借未来physical outcome选候选。

## 命令与结果边界

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_supported_resource_guard_20261027_v1.py
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_supported_resource_guard_20261027_v1

实际执行：新增测试82 passed；与旧模块纯算术测试联合110 passed、1 deselected（明确排除旧`actual_source_bound_resource_audit`，未声称重验完整source closure）。CLI退出0，status为`saved_allocation_and_synthetic_regression_verified`，原80/40分配不变，反例被拒绝。

联合回归命令：

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_supported_resource_guard_20261027_v1.py tests/test_p2_full_resource_matching_20261027_v1.py -k 'not actual_source_bound_resource_audit'

测试不使用tmp_path。覆盖8轴×3个allocation位置×0/1、cap/reserve/finish拒绝前状态不变、原legacy反例、受支持数学VJP cap仍约束、非退款/重复work_id/重复完成、畸形计数、source/report漂移、真实saved分配及P3非准入边界。文件SHA见交付消息。

CLI report分开列出actual_saved_evidence与synthetic_counterexample；`new_neural_calls=0`、`new_physical_steps=0`、`runner_integration=false`，P1/P2/P3 admission与compute_advantage全部false。纯算术回归验证不绕过P1 direct plateau/两段reach/独立根等科学前置，也不认证强基线配对或任何理论失败。
