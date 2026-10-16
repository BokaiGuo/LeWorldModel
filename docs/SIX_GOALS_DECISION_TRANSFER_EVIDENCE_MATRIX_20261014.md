# 六大科学目标：Decision-Transfer Sufficiency 证据矩阵（2026-10-14）

本文件是当前研究的 reader-facing evidence ledger。它把六个目标统一到有限时域的 decision-transfer sufficiency，但不把局部正结果升级为普适 world model、在线规划或通用 certificate。

## 统一研究对象

对 frozen representation Phi、动作序列 `a[t:t+h-1]` 和任务读出 `M_task`，验收对象是 `Phi_h(history)` 是否足以保持 `P(y[t+h] | history, a[t:t+h-1])` 的 action ranking、task transfer 和闭环 consequence。

形式化控制对象为 `C_H = (G_H^Phi, I_H, S_H, M_H, Ebar_H)`：`G_H^Phi` 是有限时域 action-to-task transfer，`I_H` 是动作可辨识性，`S_H` 是部署动作支持，`M_H` 是任务度量，`Ebar_H` 是独立校准的误差包络。

## 当前矩阵

| 目标 | 当前状态 | 已验证的最窄结论 | 当前关键证据 | 尚未通过的门 |
|---|---|---|---|---|
| 1. alignment metrics 普遍预测规划性能 | `partial / gate_failed` | 最新无泄漏 external records 在 PointMaze+PushT 形成 3 个可识别 matched strata；`Delta R = R_MSE - R_full = +0.00308`，bootstrap CI `[0, 0.00923]`，sign-flip `p=0.4935`，预注册 positive gate 未通过。 | `results/matched_learned_transfer_benchmark_external_crossenv_20261015/`; `results/cross_environment_transfer_diagnostic_external_20261015/` | 仍需更多独立 environment-level matched strata；当前结果不支持 alignment 的普遍预测性。 |
| 2. learned adapter 稳定解决姿态闭合 | `negative / partial` | repaired RGB MPC 改善 block-XY 和 endpoint pose error，但两种 action-block 均 strict success `0/8`；真实边界 signed-torque audit 的 angle improvement 为正但 CI 跨零。 | `results/pusht_repaired_predictor_mpc_visual_20261014/`; `results/pusht_repaired_predictor_mpc_visual_actionblock1_20261014/`; `results/pusht_signed_torque_audit_boundary_corrected_20261014/` | terminal-pose root、姿态/XY 联合 success、fresh seeds 和独立初始状态仍未通过。 |
| 3. LeWM latent 具有可靠物理因果动力学 | `partial` | 接触分层 paired `+a/-a/zero` 显示 effect 强烈依赖 mode；specialist geometry 只在 free/approach 有利，在 normal/tangential contact 不稳定。真实边界 chart 修复后 signed direction accuracy `0.609`，不能沿用旧 center-radius chart 的 `0.870`。 | `results/pusht_contact_stratified_paired_intervention_20261014/`; `results/pusht_signed_torque_audit_boundary_corrected_20261014/` | frozen RGB latent 的 cross-fitted physical effect、action ranking、uncertainty calibration 和多步 causal consequence。 |
| 4. frozen LeWM 支持视觉在线规划 | `partial_negative` | RGB-only repaired predictor MPC 运行完整，contact rate `1.0`；repair 相对 base 改善 held-out block-XY（`172.7 -> 159.5`，short-pulse variant `145.4`），但 success `0/8`。TwoRoom/Reacher 有 scoped live RGB success panels；修复 Reacher 的 history 接口错配后，64-root shared-bank ranking regret 为 `0.2544 [0.1190,0.4142]`，但仍随 target geometry 变化。 | `results/pusht_repaired_predictor_mpc_visual_20261014/`; `results/cross_environment_replication_20261015/`; `results/reacher_shared_candidate_ranking_aggregate_64root_history3_20261015.json` | 无 privileged successor verification 的独立闭环 success；跨目标 geometry 的 ranking/angle gate；成功/拒绝率与 latency 的完整 gate。 |
| 5. 建立普适 world model | `not_established` | 当前更接近跨环境 transfer-risk diagnostic；PointMaze/PushT 的 matched panel 仍是条件性证据。部署匹配 history=3 后 Reacher pooled 64-root inversion 为 `0.4700 [0.4476,0.4928]`，两个 target geometry 仍有明显差异，说明 action interface 依赖任务几何，不能推出 universal learned dynamics。 | `results/cross_environment_transfer_diagnostic_final_quality_20261014/`; `results/reacher_shared_candidate_ranking_aggregate_64root_history3_20261015.json` | 多任务联合 action interface、held-out domain、zero-shot/few-shot 区分、独立 architecture replication。 |
| 6. commutator 是通用 transfer certificate | `negative / theory-partial` | 可执行 finite-horizon 分解已通过 tightness、matched-diagnostic counterexample、non-normal transient、local nonlinear remainder 和 fail-closed control audit；因此 commutator 只能作为联合 certificate 的解释项，learned universal certificate 仍未验证。 | `results/finite_horizon_transfer_risk_decomposition_20261015/`; `results/covariant_transfer_risk_control_audit_20261015/`; `le-wm-main/experiments/theory_spectral_transfer/risk_aware_control.py` | learned nonlinear local applicability、独立 calibrated Ebar_H、跨环境 selective coverage 和真实 learned consequence。 |

## 已修复的实验协议问题

1. PushT 数据源的旧压缩文件大小正确但内容损坏，已从 `hf-mirror.com` 重新获取并成功解压为约 46.3GB HDF5；此前依赖该文件的失败不能作为模型负结果。
2. signed-torque audit 原先使用 center-radius root 和 center/vertex action frame；对非圆形 T 物体这不是真实 contact manifold。现已改为实际 polygon boundary root 和 boundary inward normal，并保留旧结果作为 recovery/invalid-chart 对照。
3. `terminal_pose` root builder 在当前 RGB support-gap=0.35 协议下没有可接受 root；这被记录为 root-construction boundary，而不是静默放宽到测试集可用的根。

## 当前不允许的论文表述

- “alignment metrics 普遍预测 learned-world-model planning”；
- “learned adapter 已稳定解决姿态闭合”；
- “LeWM latent 已构成可靠物理因果模型”；
- “frozen LeWM 已支持成功的视觉在线规划”；
- “已建立 universal world model”；
- “commutator 是通用 transfer certificate”。

## 当前允许的主结论

> Frozen visual world models exhibit finite-horizon decision-transfer only when the representation is action-identifiable, support-covered, mode-aware and independently calibrated. Correcting the temporal, action-chart and data-integrity interfaces can produce local transfer gains, but prediction repair alone does not imply terminal pose closure or universal visual planning.

## 证据状态

当前整体状态为 `argument_shaped`，不是 `oral_ready`：统一对象和失败边界已经明确，但目标 1 的预注册增量 gate、目标 2/4 的独立终端闭环 gate、目标 5 的普适性 gate 和目标 6 的 learned calibrated certificate 仍未闭合。

## 2026-10-15 continuation: offline-to-closed-loop readout check

TwoRoom 的新 32-episode root-disjoint RGB panel 在同一 seed 集合上比较了
raw frozen candidate-bank utility、已拟合 latent/action ranker 和
action-conditioned physical readout。raw 得到 `14/32`，ranker `10/32`，
physical readout `8/32`，随机控制为 `3/32`。因此 replay-regret 上的离线
ranker 改善没有传递为闭环收益；目标 4 的当前正式控制器保留 raw frozen
candidate-bank LeWM，readout 只作为诊断。

Reacher 上的 nested calibration 也在两个 target geometry 中均选择精确
raw fallback（`alpha=0`），而 effect-geometry 增强没有产生 held-out 正向
收益。这把当前负结果定位为 geometry-conditional action-identifiability 与
offline-to-online readout gap，而不是继续增加 readout 复杂度即可解决的
问题。详见 `docs/EXECUTION_REPORT_20261015.md` 与
`le-wm-main/results/live_tworoom_ranker_validation_20261015/`。

随后对 Reacher 的 action support 做了独立干预。固定 sequence bank 在
离线 replay 中显著降低 terminal error，但在闭环中消耗了 CEM 的连续探索
预算：同一 32-seed、`256/8/16` panel 上，raw continuous CEM 为
`21/32` 与 `7/32`，full bank 为 `14/32` 与 `1/32`，compact bank 为
`20/32` 与 `1/32`。因此当前方法论结论是：action support 是负结果的真实
组成部分，但在线修复应是 support-aware continuous exploration，而不是
固定 bank 替代 CEM。该结果支持 Reacher scoped RGB planning consequence，
不支持普适 learned dynamics 或稳定 pose closure。

## 2026-10-15 Cube observability continuation

`results/cube_rgb_observability_20261015/guided_panel80x24/observability.json`
是一个 mechanism-only、root-disjoint 的 RGB versus frozen-LeWM effect
diagnostic。它不是 RGB-only planning evidence：guided collection 使用初始
block-to-effector 方向只为获得稀疏接触样本，且该方向不进入特征或 runtime
policy。80 roots / 1,920 transitions 的 physical-motion rate 为 `10.31%`，
gripper-contact flag 仅 `2.29%`。在 current+action readout 上，physical-motion
held-out balanced accuracy 为 RGB `0.756`、latent `0.779`；加入 goal 后 latent
降为 `0.562`。该结果把 Cube 的边界进一步定位为 sparse contact support 与
goal-conditioned action-effect transfer，而非可由单纯增大 CEM budget 或增加
一个 goal readout 解决的问题。

同一审计的 RGB color-geometry control 位于
`results/cube_rgb_observability_20261015/guided_panel80x24_color/`：它的
held-out block-goal RMSE `0.0642` 与 frozen latent `0.0636` 接近，但 sparse
physical-motion balanced accuracy 为 `0.50`。因此静态 RGB 物体中心可见性
本身不能闭合接触 action-effect transfer，目标 3/4 仍不能升级。
