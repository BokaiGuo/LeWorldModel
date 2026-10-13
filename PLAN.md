# LeWorldModel 统一研究计划

本轮对话整理出的完整研究上下文、文献边界、负结果和证据门槛见
[RESEARCH_CONTEXT_20261012.md](RESEARCH_CONTEXT_20261012.md)。该档案是研究决策的上下文入口；
执行优先级仍以本文件和 `RESEARCH_PLAN_EVOLUTION_20261012.md` 为准。
谱分析主线的具体解释、完整公式推导和统一证据边界见
[SPECTRAL_ANALYSIS_RESEARCH_GUIDE_20261012.md](SPECTRAL_ANALYSIS_RESEARCH_GUIDE_20261012.md)；
原始逐字记录见
[SPECTRAL_ANALYSIS_RESEARCH_RECORD_20261012.md](SPECTRAL_ANALYSIS_RESEARCH_RECORD_20261012.md)。
项目中可校验的逐字对话导出见
[CONVERSATION_ARCHIVE_20261012.md](CONVERSATION_ARCHIVE_20261012.md)。
本轮代码与证据完整性审计见
[IMPLEMENTATION_COMPLETENESS_AUDIT_20261013.md](IMPLEMENTATION_COMPLETENESS_AUDIT_20261013.md)。

## 新增 Workstream：GDD-WM

[Decision Preference → Predictive Geometry (GDD-WM)](docs/GDD_WM_DECISION_GEOMETRY_RESEARCH_PLAN_20261013.md)
是现有 LeWorldModel 研究的新增、独立追踪的 proposed workstream，不替换谱
transfer、horizon-conditioned memory、PushT 接触控制或既有负结果。其目标是检验：

本次扩展的完整执行计划见
[GDD-WM：将执行决策知识蒸馏为预测几何](docs/GDD_WM_EXECUTION_GROUNDED_GEOMETRY_PLAN_20261014.md)。
该文件补充了 Preference Stability、Geometric Distillation Gap、realized/predicted
双路径损失、RDMC 理论诊断、D0--D4 门禁、student-reachable candidate mining 和
Euclidean 不可表示时的 quasi-metric fallback。

```text
execution-grounded preference
  -> stable candidate preference
  -> candidate-set-independent predictive geometry
  -> preserved multi-step prediction
  -> lower decision regret
  -> independent held-out planning consequence
```

机器可读协议见
[`docs/GDD_WM_PROTOCOL_TEMPLATE_V0_1.yaml`](docs/GDD_WM_PROTOCOL_TEMPLATE_V0_1.yaml)。
当前状态固定为 `Proposed / Not Yet Validated`：仓库没有可直接声明为 D-JEPA
teacher 的 checkpoint/data package，D0 shared candidate dataset、D1 teacher
stability audit 和 D2 projection-only distillation 尚未运行。现有 PointMaze JEPA
permission-gate 结果只提供候选基础设施；DINO 的 `rejected_fail_closed` 结果不能
被重新解释为 teacher preference。

GDD-WM 的执行依赖已有 action-effect permission gate，优先顺序为：

1. D0：冻结 shared candidate-set schema、root-disjoint splits、action support 和
   teacher/outcome provenance；当前 schema/provenance validator 为
   `le-wm-main/experiments/gdd_wm/build_shared_candidate_dataset.py`，只接受外部
   candidate table，不生成 teacher 或 outcome；
2. D1：在独立 candidate-set bootstrap 上审计 preference stability，并与真实执行
   consequence 核对；不稳定或 execution-inconsistent 的 pair 不得蒸馏；
3. D2：先做低容量 projection-only geometry distillation，同时冻结 dynamics，
   检验 predicted/realized ranking 与 predictive preservation；
4. 只有 D0–D2 及独立 held-out planning gate 同时通过，才考虑联合 encoder/predictor
   微调。任何单独的 teacher agreement、realized geometry 改善或 latent MSE 改善都
   不足以升级 GDD-WM claim。

不能把附件中关于“已生成 ZIP/PATCH、已推送 GitHub 或已完成验证”的描述当作仓库事实；
当前实际变更仅包括上述研究计划与协议入口，未覆盖既有 artifact。

2026-10-14 已将下一阶段 P0/P1 首批实现落到
`le-wm-main/experiments/decision_benchmark/`：PointMaze readout 截距修复、独立
action-identifiability/Jacobian 审计、等时长 goal-conditioned candidate bank、共享候选
regret evaluator。实现与真实运行边界见
[`docs/DECISION_BENCHMARK_IMPLEMENTATION_20261014.md`](docs/DECISION_BENCHMARK_IMPLEMENTATION_20261014.md)。
首个 `paired_order_v2` bank 已通过 D0 schema/provenance 检查，但由于新候选没有 RGB
realized latent，D1 仍 fail-closed；该结果不构成 GDD-WM 或 frozen visual planning 的正向
证据。

最新进展与当前释放边界见
[`docs/RESEARCH_PROGRESS_20261014.md`](docs/RESEARCH_PROGRESS_20261014.md)。2026-10-14
新增的跨环境 matched audit、接触分层干预和 polygon-boundary signed-torque 修复均已
保留独立 artifact；跨环境 `Delta R` 点估计为正但预注册 sign-flip gate 未通过，PushT
严格姿态闭合仍为负，不能将这些结果升级为 universal alignment、universal world model
或视觉在线规划结论。

## 研究定位

本计划的最新演进、阶段门和论文级决策树见
[RESEARCH_PLAN_EVOLUTION_20261012.md](RESEARCH_PLAN_EVOLUTION_20261012.md)。该文件是当前执行优先级的规范版本。
新增的投影—记忆—混合系统—能量工具见
[MATHEMATICAL_PHYSICS_TOOLKIT_20261012.md](MATHEMATICAL_PHYSICS_TOOLKIT_20261012.md)。
新增的“升维必须对 action-to-task transfer 充分”的统一规范见
[LIFTED_TRANSFER_RESEARCH_SPEC_20261012.md](LIFTED_TRANSFER_RESEARCH_SPEC_20261012.md)。
有限时域 transfer-risk 定理、证明、tightness/counterexample 和非线性适用条件见
[FINITE_HORIZON_TRANSFER_RISK_THEOREM_20261013.md](FINITE_HORIZON_TRANSFER_RISK_THEOREM_20261013.md)；
其可执行合成审计入口为
`le-wm-main/experiments/theory_spectral_transfer/run_finite_horizon_transfer_risk_decomposition.py`。
更完整的 reader-facing theorem package（含 matched scalar 与 nonlinear adversarial
counterexample、可估计性表和局部 learned applicability）见
[`docs/FINITE_HORIZON_TRANSFER_RISK_THEOREM.md`](docs/FINITE_HORIZON_TRANSFER_RISK_THEOREM.md)。
非线性 fail-closed checker 为
`le-wm-main/experiments/theory_spectral_transfer/nonlinear_transfer_certificate.py`；
跨环境诊断为
`le-wm-main/experiments/empirical_spectral_audit/run_cross_environment_transfer_diagnostic.py`；
独立规划收益 gate 为
`le-wm-main/experiments/empirical_spectral_audit/independent_planning_gain_gate.py`。
这些接口已通过全量测试，但当前真实 PointMaze panel 仍然是
`gated_cross_environment_claim`，不能把工具可运行性写成正向 learned 证据。
诊断到控制的协变 transfer-risk 管束接口见
[COVARIANT_TRANSFER_RISK_CONTROL_20261013.md](COVARIANT_TRANSFER_RISK_CONTROL_20261013.md)；
其实现为
`le-wm-main/experiments/theory_spectral_transfer/risk_aware_control.py`，
合成审计为
`le-wm-main/experiments/theory_spectral_transfer/run_covariant_transfer_risk_control_audit.py`。
进一步将 horizon-conditioned memory、support/action/successor gates 和 fail-closed
release 统一为 HTSC 的正式控制对象见
[HORIZON_CONDITIONED_TRANSFER_SUFFICIENT_CONTROL_20261013.md](HORIZON_CONDITIONED_TRANSFER_SUFFICIENT_CONTROL_20261013.md)；
其 release gate 实现为
`le-wm-main/experiments/theory_spectral_transfer/horizon_sufficient_control.py`。

2026-10-14 PointMaze 负结果根因审计与协议修复见
[`docs/POINTMAZE_NEGATIVE_RESULT_ROOT_CAUSE_20261014.md`](docs/POINTMAZE_NEGATIVE_RESULT_ROOT_CAUSE_20261014.md)。
修复了 external action fan 的 action-support 与 horizon-clock 错配，并重新完成
50-checkpoint v2 panel；strict matched benchmark 仍因单环境、CI 跨零而 gated。
新增 intervention-supervised task-effect adapter
`le-wm-main/experiments/empirical_spectral_audit/train_pointmaze_intervention_effect_adapter.py`，
用于区分 action excitation、latent history 和 privileged observability，不能并入原
geometry selector estimand。

短期历史、Fourier/时频特征、事件记忆与接触模式的文献边界、当前负结果和冻结实验协议见
[MEMORY_EVENT_CONTACT_LITERATURE_20261013.md](MEMORY_EVENT_CONTACT_LITERATURE_20261013.md)。
逐篇核对的来源—机制—边界证据地图见
[LITERATURE_EVIDENCE_MAP_20261013.md](LITERATURE_EVIDENCE_MAP_20261013.md)；
其中将 frozen RGB latent、显式接触结构、action-conditioned training 和闭环规划的证据边界分开。

`theory_spectral_transfer`、`empirical_spectral_audit`、LeWM dynamics、adapter
和 PushT planning 现在属于同一个研究，而不是两条独立路线。统一科学问题已经从“alignment 是否能修好规划”收缩为：

> 在 prediction quality、action distribution 和 predictor spectrum 匹配后，covariance–predictor–metric 几何是否还能识别 planner-relevant transfer risk，以及 action-transition mismatch、non-normal transient 和 partial observability 何时使这种识别失效？

统一证据链为：

```text
理论可识别性
  -> learned latent instantiation
  -> action-conditioned dynamics
  -> geometry-aware intervention / adapter
  -> cross-environment transfer
  -> frozen visual planning
```

两个目录仍然保留，原因是兼容既有 CLI、测试和结果路径；它们是统一研究中的模块，不再分别追踪科学问题或分别形成结论。

## 统一对象与证据门槛

所有层级围绕同一对象 `(Phi, C_Phi, A_Phi, B_Phi, D_Phi, M_task)`；原来的
`(C, P, M, B_u)` 是 `Phi=identity` 的特例：

- `C`：latent covariance；
- `P`：predictor 或局部 Jacobian；
- `M`：预先声明的 task/action metric；
- `B_u`：action-conditioned effect operator。

对 frozen visual latent，默认 lifting 为
`eta_t = Phi(z_{t-L:t}, a_{t-L:t-1})`，并显式估计 `A_Phi`、`B_Phi`、`D_Phi`。
规划相关对象是 `G_h^Phi = D_Phi A_Phi^(h-1) B_Phi`，而不是单独的
`spectrum(P)`。历史 lifting 只有在同时改善 velocity/mode observability、action
inverse risk 和 held-out action-to-task transfer 时，才被视为有效干预。

有限时域风险分析现在固定使用五项分解：lifting/operator closure、planner-reachable
off-manifold residual、action-effect mismatch、non-normal transient excess 和
task readout mismatch。该分解是 conditional bound；off-manifold、action 和
readout 项必须由独立 transition/task data 提供上界，不能由 covariance、spectrum
或 commutator 单独推断。

每个 learned-model 结论必须同时记录：训练/验证/测试 split、latent dimension、完整 predictor spectrum、training stage、action distribution、architecture、seed、环境、horizon，以及 held-out transfer/planning outcome。没有 MSE+spectrum matched pool 时，结论状态必须是 `non-identifiable`，不能通过放宽 tolerance 或挑选 snapshot 补齐。

`commutator` 的地位固定为 metric- and horizon-specific diagnostic；不得称为通用 certificate。`alignment` 只有在 prediction quality 匹配后仍能预测独立 held-out outcome，才可升级为 conditional transfer-risk diagnostic。

## 从诊断到控制的新增闭环

当前将理论推进为一个可执行、条件式的控制协议，而不是把 no-go 结果改写成万能证书：

```text
external task metric
  -> A_Phi, B_Phi, D_Phi
  -> G_h^Phi = D_Phi A_Phi^(h-1) B_Phi
  -> finite-horizon risk envelope
  -> action-identifiability gate + real/predicted transition gate
  -> covariant risk-aware short-horizon control or fail-closed fallback
```

`covariant_transfer_risk_mpc.v1` 要求独立 action-balanced intervention、独立
planner-reachable transition 和独立 task/readout calibration；缺任一项都不能放行 learned
动作。`learned_transfer_operator.v1` 已接入 90 个真实 checkpoint records，导出
`A/B/D/G_h`，但当前 risk envelope 仍为
`blocked_missing_independent_components`。因此该控制层已经代码化并通过合成审计，
尚未证明能改善 learned planning 或姿态闭合。

## 四个研究层级

### 1. 理论可识别性

保留并扩展理论 toolkit：gauge-aware metric、similarity invariance、principal angles、projector overlap、Schur geometry、non-normal transient amplification、local-Jacobian remainder，以及 covariance-only whitening 的 no-go 结果。理论输出不是独立论文结论，而是规定 learned-model 实验可以声称什么。

### 2. Learned latent 实例化

将 `empirical_spectral_audit` 的 latent、dynamics、predicted dynamics、residual、physical、counterfactual 和 action-effect 字段映射到 `(C, P, M, B_u)`，统一报告 alignment、commutator、Schur、transient、observability 和 multi-horizon transfer。

### 3. 诊断到干预

统一纳入 action-tangent alignment、reachability alignment、nuisance suppression、residual-band regularization、contact/event auxiliary、local Jacobian stabilization、learned adapter 和 metric-aware model selection。失败的 adapter、ranker 和 composite 保留为预注册负消融，不能通过调权重或 snapshot 选择删除。

### 4. 物理与规划验证

按强度递进验证：latent multi-step transfer、decoded physical action effect、paired successor verification、learned adapter 姿态闭合、cross-environment transfer、frozen visual online planning。privileged simulator-state planning 只能作为 interface/control，不得替代视觉规划证据。

## 统一运行入口与兼容模块

现有入口继续可用：

- 理论：`experiments/theory_spectral_transfer/run_analysis.py`；
- latent audit：`experiments/empirical_spectral_audit/run_audit.py`；
- dynamics/adapter/planning：`experiments/run_*` 与 `experiments/online_planning_protocol.py`；
- 统一匹配审计：`experiments/empirical_spectral_audit/run_matched_learned_transfer_benchmark.py`。
- Oral 级统一诊断入口：`experiments/empirical_spectral_audit/run_oral_level_unified_diagnostic_audit.py`；它汇总 checkpoint 字段覆盖、严格 matched strata、已有 controlled/synthetic artifact、LOEO 和统计门禁，不把缺失字段或不可识别 strata 转成正结果。
- Oral 实验覆盖矩阵：`experiments/empirical_spectral_audit/oral_experiment_coverage_audit.py`；逐项审计 MSE、train-only latent variance、alignment、principal-angle/projector、commutator、Schur/resolvent/transient、ATM、real/predicted transfer、reachability、elite regret、planning risk，以及 architecture/seed/snapshot/horizon/environment、matched、LOEO 和统计门禁。字段可用不等于 learned matched estimand 可识别；当前 PointMaze+PushT 结果仍保持 `non-identifiable`。
- Best-Paper 工具审计：`experiments/empirical_spectral_audit/run_best_paper_readiness_audit.py`；对每个 learned checkpoint 输出 `G_h`、有限时域 transfer-gain proxy、checkpoint ranking、action-identifiability/non-normal/off-manifold warnings、校准状态和 fail-closed permission，并单独审计独立真实收益 gate。

新增或改造的结果必须带统一 `study_scope: covariance_predictor_unified`、`protocol_version` 和 `claim_boundary` 字段。旧结果不覆盖，旧目录不改名；汇总时按统一 evidence matrix 读取。

## 实验推进顺序

1. 冻结统一 manifest，并实现 raw/delay/action-history/interaction/mode lifting 的同一 runner。
2. 先分离 observability gate 与 action-effect gate；没有 velocity/mode/action recovery 的独立改善，不进入 planning 解释。
3. 先完成 protocol audit：joint PE 不作为唯一动作辨识门；使用 action-only/residualized-action PE，并确保每个 episode 有足够的 causal windows。没有足够窗口的 export 标为 `history_unsupported`，不作为 lifting 负结果。
4. 保留 historical first-harmonic Fourier 作为固定强基线；在同一 feature budget、同一 action interface 和同一 split 下比较 raw history、causal delay、event-aware、GRU/SSM、event-gated SSM 和 retrieval memory。
5. 为事件记忆增加 successor-mode、phase/time-to-transition 和 uncertainty 输出；先验收 event/mode observability、action ranking 和 real/predicted successor gate，再进入 planning。
6. 固定已确认的 horizon-aware policy：`h=1 -> fixed Fourier`、`h=4 -> Fourier+phase`；`h=2` 在 selector stability 不足时 fail-closed 回退。selector 只作为 selection-instability diagnostic，不使用 held-out planning 结果更新策略。
7. 新增候选 profile `carrier_plus_phase_for_multistep_v2`：在新的 split seeds 上检验 `h>=2 -> Fourier+phase`；该 profile 在跨架构/跨环境前保持为 confirmatory transfer-sufficiency evidence，不回写旧 protocol。
7. 在 lifting 固定后建立 matched learned strata；没有两个 architecture、两个 seed、两个环境的 identifiable pool，就不写 learned model-selection 结论。升维比较必须包含 causal delay、interaction、局部时频和 fit-only information/transfer audit。
8. 做 shared-candidate、cross-domain action-consistency 与 CEM-stage ranking，对照最近的 decision-alignment 机制。
9. 做固定 spectrum/alignment/MSE 的 Schur/transient 干预，验证非正规性是否提供独立 finite-horizon 风险信息。
10. 将五项 bound 的各项估计接入 matched learned intervention：训练集只估 closure，独立 action-balanced transition split 估 action 项，planner-reachable split 估 off-manifold 项，独立 task labels 估 readout 项；输出 `G_h^Phi`、bound 和 held-out risk 的逐环境记录。
11. 做 leave-one-environment-out 和预声明 external task/action metric；仅在前述 gates 通过后重启 PushT mode-conditioned adapter。
12. 最后做 frozen visual planning。任一 gate 失败时，按演进方案的结果决策树自动降级，不得包装成普适 world model。

## 不可越过的结论边界

- action-correlated latent signal 不等于物理因果动力学；
- matched probe 不等于 independently trained neural checkpoint；
- privileged planning 不等于 frozen visual online planning；
- 单环境或单架构结果不等于普适迁移；
- 单个 commutator 不等于 transfer certificate；
- 当前 90 条 learned panel 记录虽覆盖 3 架构、5 seed、6 snapshot，但只有 PushT，且 ATM/真实预测 transition/外部 action-effect 字段缺失；统一入口输出 `non_identifiable`，不能据此声称 learned-model geometry 选择有效。
- 测试通过、artifact 完整或 GPU 可用不等于科学假设成立。

## 六目标 Decision-Transfer 证据矩阵（2026-10-14）

当前六个科学目标的统一状态、最新 PushT 修复结果和禁止升级的 claim
边界见：[SIX_GOALS_DECISION_TRANSFER_EVIDENCE_MATRIX_20261014.md](docs/SIX_GOALS_DECISION_TRANSFER_EVIDENCE_MATRIX_20261014.md)。
PushT 数据完整性、真实 polygon contact chart、接触分层干预和 RGB-only
repaired MPC 的根因记录见：[PUSHT_NEGATIVE_RESULT_REPAIR_20261014.md](docs/PUSHT_NEGATIVE_RESULT_REPAIR_20261014.md)。
