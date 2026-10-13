# 六大科学目标：Decision-Transfer Sufficiency 证据矩阵（2026-10-14）

本文件是当前研究的 reader-facing evidence ledger。它把六个目标统一到有限时域的 decision-transfer sufficiency，但不把局部正结果升级为普适 world model、在线规划或通用 certificate。

## 统一研究对象

对 frozen representation Phi、动作序列 `a[t:t+h-1]` 和任务读出 `M_task`，验收对象是 `Phi_h(history)` 是否足以保持 `P(y[t+h] | history, a[t:t+h-1])` 的 action ranking、task transfer 和闭环 consequence。

形式化控制对象为 `C_H = (G_H^Phi, I_H, S_H, M_H, Ebar_H)`：`G_H^Phi` 是有限时域 action-to-task transfer，`I_H` 是动作可辨识性，`S_H` 是部署动作支持，`M_H` 是任务度量，`Ebar_H` 是独立校准的误差包络。

## 当前矩阵

| 目标 | 当前状态 | 已验证的最窄结论 | 当前关键证据 | 尚未通过的门 |
|---|---|---|---|---|
| 1. alignment metrics 普遍预测规划性能 | `partial` | PointMaze+PushT 的 7 个 matched strata 中，Delta R = R_MSE - R_full = +0.0534，95% bootstrap CI `[0.0115, 0.1166]`；但 sign-flip `p=0.0630`，预注册 positive gate 未通过。 | `results/cross_environment_transfer_diagnostic_final_quality_20261014/` | 更多独立 environment-level strata；不能把 CI 不跨零替代预注册 sign-flip gate；不能称普遍预测。 |
| 2. learned adapter 稳定解决姿态闭合 | `negative / partial` | repaired RGB MPC 改善 block-XY 和 endpoint pose error，但两种 action-block 均 strict success `0/8`；真实边界 signed-torque audit 的 angle improvement 为正但 CI 跨零。 | `results/pusht_repaired_predictor_mpc_visual_20261014/`; `results/pusht_repaired_predictor_mpc_visual_actionblock1_20261014/`; `results/pusht_signed_torque_audit_boundary_corrected_20261014/` | terminal-pose root、姿态/XY 联合 success、fresh seeds 和独立初始状态仍未通过。 |
| 3. LeWM latent 具有可靠物理因果动力学 | `partial` | 接触分层 paired `+a/-a/zero` 显示 effect 强烈依赖 mode；specialist geometry 只在 free/approach 有利，在 normal/tangential contact 不稳定。真实边界 chart 修复后 signed direction accuracy `0.609`，不能沿用旧 center-radius chart 的 `0.870`。 | `results/pusht_contact_stratified_paired_intervention_20261014/`; `results/pusht_signed_torque_audit_boundary_corrected_20261014/` | frozen RGB latent 的 cross-fitted physical effect、action ranking、uncertainty calibration 和多步 causal consequence。 |
| 4. frozen LeWM 支持视觉在线规划 | `partial_negative` | RGB-only repaired predictor MPC 运行完整，contact rate `1.0`；repair 相对 base 改善 held-out block-XY（`172.7 -> 159.5`，short-pulse variant `145.4`）和 endpoint error（`262.3 -> 126.7`），但 success `0/8`。 | `results/pusht_repaired_predictor_mpc_visual_20261014/`; `results/pusht_repaired_predictor_mpc_visual_actionblock1_20261014/` | 无 privileged successor verification 的独立闭环 success；第二环境；成功/拒绝率与 latency 的完整 gate。 |
| 5. 建立普适 world model | `not_established` | 当前更接近跨环境 transfer-risk diagnostic；PointMaze/PushT 的 matched panel 仍是条件性证据，不能推出 universal learned dynamics。 | `results/cross_environment_transfer_diagnostic_final_quality_20261014/`; PointMaze history artifacts | 多任务联合 action interface、held-out domain、zero-shot/few-shot 区分、独立 architecture replication。 |
| 6. commutator 是通用 transfer certificate | `negative / theory-partial` | 普通 commutator 只能是联合 certificate 的解释项；当前 finite-horizon theorem、metric-aware tube、G_H、action/support/readout gates 已代码化，但 learned universal certificate 未被验证。 | `FINITE_HORIZON_TRANSFER_RISK_THEOREM_20261013.md`; `le-wm-main/experiments/theory_spectral_transfer/risk_aware_control.py` | learned nonlinear local applicability、独立 calibrated Ebar_H、tightness/counterexample 与跨环境 selective coverage。 |

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
