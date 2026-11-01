# 四臂共同优化器：前瞻预算合同与接线边界

新方向是 structured D2P、native、random、immediate-progress 四种初始化，
接入同一强优化器和共同后缀；不是对已有不同规划算法的事后公平排名。
每个预算点使用同一 root/task/model/history/goal/action domain，四臂同一个
optimizer 与参数，初始化自身产生的 Jacobian、筛选和试算成本均计入本臂。
完整前瞻曲线要求运行前确定预算点、规则和全部身份；独立启动每个预算点，
不能按实际 outcome 选预算或只报告完成的臂。本文件没有设定任何实际数值、
任务、根、优化器或来源时间承诺。

## 已有资产及缺口

- `p2_grasp_reference_core_20261027_v1.py` 是官方 GRASP H1 机制子集，
  不是已证实的完整官方复现；其 transition/backward receipt 不含编码和全流程成本，
  H1 与现有 H3 不可隐式混用。有关固定代码版本和缺口见
  `../docs/p2_grasp_official_contract_20261027_v1.md`。
- `chat_p2_20261027_candidate_census.py` 保留候选分母和失败，但不是模型入口
  instrumentation，不覆盖编译/AD/solve 等全部费用；见外层
  `docs/chat_p2_20261027_candidate_census_delivery.md`。
- `d2p_planning_compute_ledger.py` 做完整向量的声明比较，不拦截执行；
  `p2_preregistered_resource_vector_20261030_codex.py` 校验预先声明的向量和
  收据但不核验来源、提交时间或 hook 覆盖；H15 durable/observable bridge
  提供特定 backend 接线，不能自动涵盖新四臂初始化和共同优化器。

## 新接口

`experiments.spectral_reachability.codex_four_arm_full_cost_20261030` 仅依赖
标准库，不载入模型。`preflight(contract, sha)` 接受精确四臂、共享优化器和参数、
统一身份哈希、外部阶段策略哈希、至少两个逐轴不递减且至少一轴严格递增的共同
cap 向量；canonical JSON 使用排序 key、紧凑分隔、UTF-8、禁止 NaN。
各臂初始化代码/配置须各自以 `initializer_sha256` 绑定，其他身份字段绑定 root、
task、model、action domain、history、goal、optimizer、optimizer_config、meter、
hardware。哈希是声明，不检查相应文件，也不能证明事前提交。

`Ledger(contract, sha, arm, level, run_id)` 创建每臂每预算点独立内存账本。
`dispatch(phase, kind, reservation, callback)` 在 callback 之前检查全向量 cap，
并不可退款地预留声明的上界；成功/异常分别保留 completed/failed，超 cap 不调用
callback，之后 fail-stop。调用者必须显式提供每个轴的非负整数，上界而非实测值；
失败不退款。`close()` 返回快照，不写文件、不覆盖运行文件、不授予执行许可。
默认没有自动发现调用、持久化或重放校验，空账本也只能是 `closed_unverified`。

轴分别为 encode API/sample、action encode API、predict API/sample、backward API、
VJP rows、Jacobian API、筛选 evaluation、solve、SVD、replay 和 physical step。
事件通过 phase 区分 setup、initialization、jacobian、screening、trial、
common_suffix、verification、physical。计数是多维度而非可相加的成本：
一个实际操作若跨多个维度，须以事先固定的计费规则计入，不得重复或漏记。
特别是 batch 的 sample 数、Jacobian 内部的 VJP/反向、试算的预测、
验证 replay 和所有臂的共同 suffix 不能隐形免费。

接线时先在外部独立固定完整 contract 和所有被引用的源码、候选规则、预算点与
meter，按对应预算点及臂实例化 Ledger；在实际 encoder、action encoder、
predict、autograd、筛选、线性代数、replay、physical 的**叶子入口**调用 dispatch，
然后将 close 快照与原始持久日志独占保存。Jacobian 等外层操作不可在其 callback
内嵌套 dispatch：应在独立、不嵌套的事件记外层调用，同时逐个拦截内部叶子调用；
这种接线必须在执行前完成并验证，不能以事后常数代替。
同一份共享准备须按预先固定的 standalone 规则合理归属各臂，不能当作零。

## 限制与前置

此代码是可运行的 **离散资源合同和合作式预留 guard**，不是完整实际全成本计量器。
外部必须证明上界有效且每条调用路径均经过 hook；调用者可绕过、错误声明或篡改
内存状态，事件快照也不具持久、防篡改或崩溃恢复保证。没有实际 hook 和独立
持久 consumer 时不得用于生产放行；现有 H15 持久桥也不能自动证明这套新接口
的覆盖。异步 CUDA 同步、实际 wall、峰值显存、FLOPs、CPU/数据准备和外部物理
代价均未计量或硬限制，不能因为离散 cap 相等宣称全成本相等；不允许将未知填零。
还需要确定四臂具体初始化、共同强优化器和 H1/H3、实际动作域及最终评估规则，
统一失败与预算点分母，外部预注册时间证明，真实源码闭包及持久账本验证。
超 cap 回调以外的不可抢占工作与测量开销也需单独约束或如实报告。

现有测试只是合成 callback 的接口回归，不是资源实测或性能数据；不变更 P1 门、
不授权 GPU/物理、不证明预算公平、不证明因果收益或独立确认。

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider \
 tests/test_codex_four_arm_full_cost_20261030.py \
 tests/test_p2_preregistered_resource_vector_20261030_codex.py \
 tests/test_p2_h15_observable_budget_bridge_20261030_codex.py \
 tests/test_p2_grasp_reference_core_20261027_v1.py
```

在 `le-wm-main` 中运行：67 passed in 2.55s；没有 GPU 或物理生产。
