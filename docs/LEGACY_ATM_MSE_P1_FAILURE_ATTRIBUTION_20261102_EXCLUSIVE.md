# 旧 ATM/MSE P1 failure attribution（2026-11-02）

范围是旧 matched diagnostic 的 checkpoint selection，**不是 D2P P1**。
不训练、不重新采集、不修改任何既有 gate 或阈值。

## 逐 stratum 复算

入口 `le-wm-main/experiments/empirical_spectral_audit/legacy_p1_failure_attribution_20261102_exclusive.py`
调用现有严格 benchmark 的加载、匹配及 selector 实现，固定使用：

1. 原 `cross_environment_final_strict_20261013` 的两个 quality-matched
   records 目录全部 100 条，作为历史 provisional calibration-risk panel；
2. 原 `cross_environment_transfer_diagnostic_external_20261015` 的 input
   清单全部 140 条，作为 external-evidence panel。

未依据结果筛选。逐候选原始记录哈希、外部证据路径/哈希、模型标识、
诊断、风险、原 selector、可识别条件、失败原因、逐组结果和与历史
输出的逐字段比较见
`le-wm-main/results/legacy_p1_failure_attribution_20261102_exclusive_RUN01/report.json`
（SHA-256 `64e5ba46abf2aec9c9c8d26194faaab557a5ff7bb33c3b3cf96aaed6a693e287`）。
同目录 `report.md` 是 21 个 strata 的完整可读表。
240 条记录均加载且运行前后哈希一致；所有已保存分层的候选数、
匹配集合、可识别状态、失败原因、selector 和 delta 与原结果一致。
这是对当前文件字节的重算，不是历史时间戳认证或新的物理实验。

### 20261013 质量匹配历史结果

10 个 strata 中 7 可识别，3 因匹配集合仅一种架构不可识别；
7 组 normalized ΔR 均值 `0.0533572713`，
原 primary bootstrap CI `[0.0114990057, 0.1151671074]`，
单侧 Monte Carlo sign-flip `p=0.0620968952`，gate false。
附件的 `0.11661` 是同一保存报告里另一 bootstrap 汇总的上界，
不是 primary 检验的区间上界；`p=0.0630` 也不是此文件的原值。

更重要的边界：原项目已查明当时 `action_inverse_risk` 是 calibration
transition error，而不是独立动作干预。100 条记录没有独立
`external_evidence` 路径；不能把此统计作为真正 ATM 条件下的证明。
对应后续十种子 panel 的变化只是另一组历史分析，不能同此组混合。

### 20261015 外部证据 panel

11 个 strata 中 3 可识别，5 个匹配候选不足 2，3 个匹配后仅一种
架构。可识别组依次为 PointMaze stage 5（ΔR=0）、PointMaze stage 10
（ΔR=0.0092271566）和 PushT stage 75（ΔR=0）；原 wrapper 的
平均 `0.0030757189`，primary bootstrap CI `[0, 0.0092271566]`，
单侧 Monte Carlo sign-flip `p=0.4935253237`，gate false。
三组只有一组改变了 MSE selector 选择。PushT stage 75 两个 selector
选择相同且 regret `0.0325321`，不是零 regret。

140 条外部证据文件的动作风险和规划风险数值均与记录相符；这里只验证
保存文件，不证明来源防篡改或独立重放。PointMaze 50 条来源写明
`observed-action horizon consequence`，每个 evidence 的干预和规划
使用同一批独立 external roots；因此所谓独立是相对 fit/outcome export，
不是动作干预和规划后果之间的相互独立，也不是本次新执行的 RGB 闭环。
PushT 来源记录固定的 12 episode RGB-only 规划，是否真正源独立依赖原协议。
两环境风险定义、原始单位与 train-only normalization 不同；不要将
单一 raw regret 跨环境直接比较。

## 独立规划 gate 与额外旧 P1

固定 `results/independent_pointmaze_planning_gain_fixed_20261013.json`
作为输入，使用原 `independent_planning_gain_gate` 默认参数在 `/tmp`
重算，与 `results/independent_pointmaze_planning_gain_gate_fixed_20261013.json`
逐字节一致（SHA-256 `6636a107bf66cd01c117d90d700bfc73c75651ae0efa5f4c4e99c4e9960aebd1`）。
32 对、无 invalid records、protocol gate true，但 mean `0.1006652`、
CI `[-0.0277088, 0.2478060]`、p `0.0876956`，statistical gate false。
此项为不同固定方法和 outcome，不是上面 cross-environment gate 的替代。
这些 fresh/frozen/permission 标志是输入协议声明，非本次新物理认证。

本地另有 20261021 的 `matched_decision_transfer_p1_20261021/diagnostic.json`：
FH-AIC 增量、50 条、10 个 PointMaze 分层，ΔR `-0.00824683`、
CI `[-0.0233452, 0.00402417]`、p `0.874953`，claim false。它是另一个
MSE+ATM 对 MSE+ATM+FH-AIC 的同环境 task-effect adaptation pilot，不是
上面 MSE-only 对 full geometry 的同一 estimand；其中 ATM 字段的真实
干预来源尚需独立核对，不能凭 `independent_external_evidence=true` 认证。

## 剩余缺口

- 分层跨 stage 重用训练 seed 和 outcome 的程度未完整验证；不能把 7、
  3 或 10 个 strata 宣称为相互独立的模型/episode 实验。
- 现有原 benchmark 使用稳定排序给精确并列分配不同序号，选择可能
  依赖输入顺序；本次忠实复现，没有更改排名规则或宣称结果对排列稳健。
- 四个几何指标方向来自原协议的 lower-is-better 声明；这里没有新的
  单调性验证，也没有把单指标事后选择当确认性检验。
- 对不可识别组，缺少足够符合原匹配条件且跨架构的候选；对 gate false
  的可识别组，证据不足以确认预设正增量，不等于证明真实效应为零。
- 需验证根级身份、独立动作干预、真实独立 planning episodes、
  pre-outcome 选择和计算成本，才能进一步判断归因及推广。

测试：`tests/test_legacy_p1_failure_attribution_20261102_exclusive.py`、
`tests/test_transfer_theorem_and_gates.py` 和
`tests/test_audit_matched_decision_transfer_p1.py` 共 15 passed。
测试含 fixture，不能算科学证据。
