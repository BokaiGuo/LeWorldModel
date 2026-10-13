# 研究方案实现完整性审计（2026-10-13）

## 结论

研究方案已形成可运行的统一代码骨架，但尚未完成“六项目标的科学证据闭环”。
当前完成的是理论、协议、导出和 fail-closed 控制接口；尚未完成的是跨环境独立
learned 证据、稳定视觉规划和物理因果验证。

## 已落地

| 模块 | 当前实现 | 验证状态 |
|---|---|---|
| 有限时域理论 | `finite_horizon_transfer_risk_decomposition`、tightness、matched scalar/nonlinear adversarial counterexample、非正规 transient、局部 nonlinear applicability checker | theorem runner 已输出；本轮 fail-closed tests 通过 |
| 控制理论接口 | `covariant_transfer_risk_mpc`、`G_h=D A^(h-1)B`、metric-tube certificate、action/transition/tube 三重 gate | synthetic permit/fallback audit 已通过 |
| learned operator export | 每个 record 导出 `A/B/D/G_h` 和风险项可识别性 | 90 个真实 checkpoint records 已生成 |
| matched learned audit | 固定 MSE、spectrum、stage、latent dimension、action distribution 后做最大兼容 clique | 1 个 PushT matched stratum；跨环境 gate 仍关闭 |
| temporal memory | causal delay、event-aware、Fourier/phase controls、fit-only selector | 10 个 split；事件 lift 胜 base，但未稳定胜历史 Fourier |
| 测试和协议 | schema、泄漏检查、控制器、理论分解和 builder tests | 全量回归 `595 passed, 6 skipped, 26 warnings` |

本轮新增并验证：

- `docs/FINITE_HORIZON_TRANSFER_RISK_THEOREM.md`：正式 theorem/proof、estimability
  table、tightness、linear/nonlinear failure boundary；
- `experiments/theory_spectral_transfer/nonlinear_transfer_certificate.py`：对
  local Jacobian、Taylor remainder、action support、独立 successor split 和
  readout calibration 的 fail-closed checker；
- `experiments/empirical_spectral_audit/run_cross_environment_transfer_diagnostic.py`：
  per-environment matching、worst-environment、leave-one-environment-out 和跨环境 gate；
- `experiments/empirical_spectral_audit/independent_planning_gain_gate.py`：冻结指标/权重、
  paired fresh planning episodes、sign-flip/bootstrap 和 no-privileged gate；
- `experiments/empirical_spectral_audit/build_independent_pointmaze_planning_gain.py`：从显式
  指定 checkpoint 构造 paired baseline-vs-diagnostic planning artifact，不搜索 outcome；
- `results/finite_horizon_transfer_theorem_audit_20261013/`：tightness=true、两个
  adversarial counterexample 均显示不同 finite-horizon transfer；
- `results/cross_environment_transfer_diagnostic_pointmaze_20261013/`：当前真实
  PointMaze panel 为 `gated_cross_environment_claim`，没有被误报为正向证据。
- `results/independent_pointmaze_planning_gain_20261013.json` 与
  `results/independent_pointmaze_planning_gain_gate_20261013.json`：点估计
  `Delta=0.0259`，但 95% CI `[-0.0627, 0.1111]`、sign-flip `p=0.2813`，因此
  `gated_independent_planning_gain`；独立正向收益尚未成立。

全量回归：`595 passed, 6 skipped, 26 warnings`。

## 六项目标状态

1. **alignment 普遍预测 learned planning：未完成。** 受控线性 benchmark 成立；
   真实 panel 只有一个环境中的一个 identifiable stratum，不能给出跨环境相关性或
   universal selection 结论。
2. **learned adapter 稳定姿态闭合：未完成。** 接触保持已有局部改善，但 angle/action
   effect 在接触模式和区域间仍不可稳定识别；历史 pure learned strict success 仍低。
3. **可靠物理因果动力学：未完成。** 当前是 latent/action-associated transfer；独立
   action-balanced counterfactual、物理读出校准和不确定性覆盖尚未闭合。
4. **frozen LeWM 视觉在线规划：未完成。** 仍缺 RGB-only runtime successor verification
   关闭、独立环境和 fresh initial states 下的正向闭环结果。
5. **普适 world model：未完成。** 现有 cross-environment geometry 是 proxy，尚无
   zero-shot/few-shot task transfer、OOD rollout 和独立任务收益的闭环。
6. **commutator 通用 certificate：理论上不可成立。** 已正确降级为 metric- and
   horizon-specific diagnostic；新控制接口使用完整风险包和 gates，而不是 commutator
   单独放行。

## 关键最新 artifact

- 控制理论说明：`COVARIANT_TRANSFER_RISK_CONTROL_20261013.md`
- 控制器：`le-wm-main/experiments/theory_spectral_transfer/risk_aware_control.py`
- synthetic control audit：`le-wm-main/results/covariant_transfer_risk_control_audit_20261013/`
- learned operator panel：`le-wm-main/results/learned_transfer_operator_panel_20261013/`
- matched audit：`le-wm-main/results/learned_transfer_operator_panel_20261013/matched_audit/`
- 六目标基线矩阵：`le-wm-main/results/six_goal_evidence_20261012/six_goal_evidence_matrix.md`
- temporal memory summary：`le-wm-main/results/adaptive_time_frequency_memory_selector_20261013/summary.json`

## 尚未完成的决定性实验

必须新增至少第二个环境和独立 action-balanced/planner-reachable/task-readout split，
让每个 learned record 具备可审计的 closure、off-manifold、action、transient 和
readout envelope；随后在 leave-one-environment-out 上测试 `G_h`/risk bound 对
multi-step transfer 和 planning risk 的预测。缺少任一项时，控制器保持 fail-closed，
不能将代码完成升级成科学目标完成。
