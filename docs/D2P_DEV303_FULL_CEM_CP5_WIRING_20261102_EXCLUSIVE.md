# dev303 完整时域 CEM：CP5 接线与执行前置

## 结论

本轮新增 CPU 可测的 CEM 搜索、真实模型 leaf 调用适配及 host callback 调度
接线；未运行 CUDA、神经推理或物理环境。它不是独立原生 host runner：
真实 RGB 与动作回执的认证、环境/模型装载、按尝试计费的持久预算驱动、
数据和源码冻结、锁、停止规则及实际硬件计量，仍须由经过审查的 native host
接入。接口不自行授予 CUDA 或物理授权。不得将本轮测试称作本地基线结果。

## 输入与适用范围

`results/task_p1_dev303_goal_blind_20261101_codex.json.gz` SHA256 为
`3787aba5dfe49c97b9c738e347e228c8089449af0df84a5f91788ceb769b2b12`；
`load_task()` 校验该任务字节、seed `301000003`、blind source SHA、非准入与
freshness unknown。此任务已有开发后果，不能称未触碰 holdout、独立根或确认。
此验证不证明模型运行时加载了同一任务；host 必须实际绑定 task、模型、goal、
normalizer、目标矩阵与源码，运行前后验证。

## 可复用的现存实现

完整原生执行路径为
`le-wm-main/experiments/spectral_reachability/codex_p1_full_cem_host_v2_20261101_exclusive.py`，
底层控制器为
`le-wm-main/experiments/spectral_reachability/codex_p1_full_cem_controller_20261101_exclusive.py`。
它是原生环境中实际 RGB 重编码、动作回执、逐节点 CEM 与账本的已有接线，
但只绑定旧 root0/root1，控制 20 步，在 0/5/10/15 重规划；
**不能直接用于 dev303 15 步/仅 CP5 反馈合同**。
执行需要对应 root 的全新 prepare 与 SHA、固定输入闭包、显式 host-CUDA/物理授权、
合作锁和环境/硬件审查；完整性及公平性仍未认证，详见
`docs/D2P_P1_FULL_CEM_HOST_V2_20261101_EXCLUSIVE.md`。

`agent_lewm_bounded_native_cem_20261029_v1.py` 是已知两根的 H2 模型规划，
不是 dev303 的 H3 全时域加 CP5 再规划。`p1_cp5_cem_baseline_20261101_leibniz.py`
是另一已知根的 CP5 H3 模型规划，也不是完整 dev303 闭环。

## 新接线规格

`dev303_full_cem_cp5_wiring_20261102_exclusive.py` 的 `run_model_callbacks`
调用 `TorchScorer`，通过注入 leaf 的 `action_encode` 和 `predict` 用实际
状态历史、动作历史、goal、normalizer 和目标矩阵计算模型分数：

1. 初始观测下规划 H3（15 raw actions），只派发第一包五步。
2. 验证回执的已执行动作逐项等于派发前缀；要求 CP5 观测附匹配动作历史与
   RGB SHA。缺失或错误即失败，不退回预想尾段。
3. 使用 host 从该真实反馈构造的 context 规划 H2，派发完整 10 步尾部；
   不额外在 CP10 重规划。允许 native 提前停止，但不得补齐或重试。

每节点 4 轮 × 16 候选、4 elite、原始动作 clip 到 [-1,1]，包含中心；
64 次评分中按分数及索引选 winner，再复评一次。H3 的 action encode 和
predict 各 195 次，H2 各 130 次；批大小为 1，因此预测样本相同。attempted
和 returned 分开记录；失败时通过事件 sink 记录部分计数。
事件 sink 由 host 注入：此模块本身不提供持久化或可信时间。完成节点时
检查精确调用数；host 仍需在派发前通过持久预算账本强制上限与错误保留。

这些是 **planning/replay 部分的计划量和接口计数**，不是覆盖 encoder、
环境、重置、模型加载、缓存、反向、wall 和其他共享工作的一揽子账。
也不等于此前 selector 的 25,608 根阶段及 3,200 CP5 预测样本；不能称
同总预算。真实闭环的配对任务、反馈时点与实际资源必须另行审计。

## 验证与下一步

测试：`tests/test_dev303_full_cem_cp5_wiring_20261102_exclusive.py`，
使用合成 CPU leaf 与回执；覆盖 H3/H2、单次反馈、完整尾段、
动作/反馈缺失、winner replay、计数和失败前不派发。与既有控制器测试合计
`18 passed`；测试不是 CUDA、物理、模型忠实性或性能证据。

下一步是新建经独立审查的 dev303 原生 host adapter，绑定已固定的初始源、
目标、checkpoint 与 CP5 实际图像/动作、完整预算和失败日志，并在任何新的
物理执行前完成新协议冻结与授权。不能重用旧 root 的 prepare，不能根据已看过
的 dev303 结果声称前瞻独立确认。作者报告值不进入此对照。
