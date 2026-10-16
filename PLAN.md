# LeWorldModel 统一研究计划

本轮对话整理出的完整研究上下文、文献边界、负结果和证据门槛见
[RESEARCH_CONTEXT_20261012.md](RESEARCH_CONTEXT_20261012.md)。该档案是研究决策的上下文入口；
项目负责人对整个研究的理解与统一问题定义见
[docs/PROJECT_UNDERSTANDING_FROM_USER_20261014.md](docs/PROJECT_UNDERSTANDING_FROM_USER_20261014.md)。
该记录用于保持研究定位、负结果解释和证据边界的一致性；实验事实仍以对应 artifact 为准。
执行优先级仍以本文件和 `RESEARCH_PLAN_EVOLUTION_20261012.md` 为准。
谱分析主线的具体解释、完整公式推导和统一证据边界见
[SPECTRAL_ANALYSIS_RESEARCH_GUIDE_20261012.md](SPECTRAL_ANALYSIS_RESEARCH_GUIDE_20261012.md)；
原始逐字记录见
[SPECTRAL_ANALYSIS_RESEARCH_RECORD_20261012.md](SPECTRAL_ANALYSIS_RESEARCH_RECORD_20261012.md)。
项目中可校验的逐字对话导出见
[CONVERSATION_ARCHIVE_20261012.md](CONVERSATION_ARCHIVE_20261012.md)。
本轮代码与证据完整性审计见
[IMPLEMENTATION_COMPLETENESS_AUDIT_20261013.md](IMPLEMENTATION_COMPLETENESS_AUDIT_20261013.md)。

科研工程治理与入口分类见
[docs/RESEARCH_ENGINEERING_GOVERNANCE_20261014.md](docs/RESEARCH_ENGINEERING_GOVERNANCE_20261014.md)
和 [docs/EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md)；统一执行架构决策见
[docs/decisions/ADR-001-unified-research-execution.md](docs/decisions/ADR-001-unified-research-execution.md)。
新增实验必须先遵守 input fingerprint、artifact schema、stage status 与显式
evidence manifest 契约，再接入新的研究模块。
本轮顶会严格审稿意见与全部研究方向的验收门槛见
[docs/REVIEW_CRITIQUE_AND_EXECUTION_CRITERIA_20261014.md](docs/REVIEW_CRITIQUE_AND_EXECUTION_CRITERIA_20261014.md)。

文献整合后的 predictive/action/decision sufficiency 三层协议、实验字段和硬否决规则见
[docs/DECISION_SUFFICIENCY_RESEARCH_PLAN_20261014.md](docs/DECISION_SUFFICIENCY_RESEARCH_PLAN_20261014.md)。

ATM → HTSC → GDD-WM 的统一论文主线、阶段门和当前 P0/P1 证据状态见
[docs/UNIFIED_DECISION_TRANSFER_PAPER_PLAN_20261014.md](docs/UNIFIED_DECISION_TRANSFER_PAPER_PLAN_20261014.md)。

H-JEPA 启发的“信息价值随 horizon 与 planning phase 改变”已经纳入现有主线，
正式方法与验收协议见
[Phase- and Horizon-Conditioned Transfer Sufficiency](docs/PHASE_HORIZON_CONDITIONED_TRANSFER_SUFFICIENCY_20261016.md)。
该扩展不把层级预测自动等同于决策充分性：高层表示仍须通过 action relevance、
deployment support、subgoal reachability 和独立闭环 consequence gate。

本轮实际执行状态、GPU 阻塞边界、代码修复和可恢复运行目录见
[docs/EXECUTION_REPORT_20261014.md](docs/EXECUTION_REPORT_20261014.md)。

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
teacher 的 checkpoint/data package。D0 schema audit 已运行，D1 preference
stability audit 已实现并在当前 bank 上 fail-closed，D2 projection-only distillation
入口已实现但在 D1 失败时 fail-closed。现有 PointMaze JEPA
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

2026-10-15 新增了
`le-wm-main/experiments/gdd_wm/build_repeated_candidate_contexts.py`，能够按精确
root/action-sequence hash 构造多个 candidate-set contexts；在现有 PointMaze bank 上
匹配到 40 个 root、410 个 root/action pair。D1 的 stability 因此已可估计，但新的 D0
preflight 同时检测出 `teacher_score = -real_task_cost` 的精确泄漏，并将
`d1_ready=false`。这把结构缺失与 teacher provenance 失败分开；在获得独立 frozen
teacher 和完整 runtime references 前，D2 仍禁止启动。

最新进展与当前释放边界见
[`docs/RESEARCH_PROGRESS_20261014.md`](docs/RESEARCH_PROGRESS_20261014.md)。2026-10-14
新增的跨环境 matched audit、接触分层干预和 polygon-boundary signed-torque 修复均已
保留独立 artifact；跨环境 `Delta R` 点估计为正但预注册 sign-flip gate 未通过，PushT
严格姿态闭合仍为负，不能将这些结果升级为 universal alignment、universal world model
或视觉在线规划结论。

### 文献更新后的双主线边界

value equivalence、state abstraction、bisimulation 和 DeepMDP 已经说明：决策表示不必
恢复全部环境动力学，只需在声明的任务/价值函数类上保持决策相关更新。因此本项目不把
“只保留决策相关信息”本身作为新颖性，而将新增问题收紧为：

> 在 frozen RGB latent、有限动作激励、部分可观测和固定有限 horizon 下，能否利用独立真实 successor 数据，对候选动作的 transfer uncertainty 进行校准，并决定何时允许规划？

HTSC 作为第一条科学主线，回答 **Can we trust the world model?**；它的输出应是候选
动作级风险上界和 `permit/abstain`，而非无条件的 commutator/alignment certificate。
GDD-WM 作为第二条科学主线，回答 **Can reliable decisions improve the world model?**；
它利用已执行验证的动作偏好学习预测几何，并必须在新候选分布和独立规划结果上证明收益。

两条线共享 candidate data、\(G_h^\Phi\)、action support、real/predicted successor
和闭环评估，但不共享成功判据。GDD-WM 的重训练收益不能反向证明 frozen latent 原本
充分；HTSC 的安全拒绝也不能被写成 GDD-WM 蒸馏成功。

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

## 文献约束下的决定性实验

下一阶段不以增加更多孤立指标为目标，而是建立同一测试起点、同一候选动作集和真实后果
记录上的统一 benchmark。每个候选动作应同时记录：

1. 真实 successor 与任务后果；
2. RGB successor 与 learned prediction；
3. action-identifiability / inverse risk；
4. geometry diagnostics 与 \(G_h^\Phi\)；
5. action ranking、regret 和证书输出；
6. 选择性规划的独立闭环结果。

主要估计顺序固定为：

```text
action identifiability
  -> real/predicted consistency
  -> incremental geometry
  -> decision certificate
  -> closed-loop consequence
```

控制 action risk 后的 geometry 增量是重要检验，但不能自动解释为 geometry 的独立因果
贡献；如果 action risk 是 geometry 影响 planning 的中介变量，还必须使用 matched intervention
和针对性的 Schur/transient 干预。若证书误差界过宽或候选动作超出支持集，协议必须
`abstain`，而不是为了得到正结果放宽门槛。

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

### 2026-10-16：context-safe capacity 修复后的非 PushT 证据

`audit_train_only_teacher_capacity.py` 已修复 context 合并统计 bug：不同
history context 不再共享一个 candidate ranking group。TwoRoom 与 Reacher
的 corrected CUDA artifact 分别位于
`le-wm-main/results/gdd_wm_cross_environment_20261016/tworoom_capacity_root_centered_context_safe/`
和
`le-wm-main/results/gdd_wm_cross_environment_20261016/reacher_capacity_root_centered_context_safe/`。
结果显示固定 residual readout 在 root-centered regret 上有跨环境诊断信号，
但尚未满足独立 architecture/seed、canonical goal-relative action frame 和
闭环 planning consequence 三重门槛；旧 mixed-context 结果不得继续作为证据。

Canonical follow-up is now complete: TwoRoom/Reacher panel exporters write
evaluator-only source/goal coordinates, and the goal-relative/root-centered
rerun removes the raw-action gain (all four environment/readout intervals cross
zero). The next repair target is therefore the learned action-effect interface
and action support, not MLP capacity or selector tuning. Any new positive result
must be obtained on this canonical protocol and then transferred to an
independent closed-loop consequence.

### 2026-10-16：Reacher proposal-relative decision repair

旧在线 `4.30%/4.48%` support 统计把未激活 planner rows 放入分母，不能再作为
“部署仅有约 4% 支持”的证据；旧 `98.44%` empirical gate 对有限 candidate
cloud 也过严。现已实现 cloud 内 action 标准化的 `proposal_relative` chart，并
用 calibration cell 的单侧 90% Wilson lower bound 的第 5 百分位冻结支持阈值
`0.927105`。该规则不读取在线 outcome。

CUDA 离线 outcome 上 action-only/source-action/full-predicted/shuffled regret
为 `0.47152/0.21495/0.25498/0.19389`。因此 source-action decision signal
成立，但 predicted dynamics 仍未通过 no-dynamics/shuffled control。两批互斥
fresh roots 在线结果分别为 `32/64 -> 38/64` 与 `36/64 -> 39/64`；按 batch
分层、root 为 bootstrap unit 的 128-root 汇总为 `68/128 -> 77/128`，paired
`+0.0703125`，95% CI `[+0.015625,+0.1328125]`。第二批单独 CI 仍跨零，
continuous qpos 指标也均跨零，必须同时报告。

当前新结论限定为：proposal-relative、phase-gated source-action readout 在
Reacher fresh RGB 闭环上提高二值成功率。它不是 learned predicted-dynamics
修复、因果物理 latent、跨环境证据或通用 world model。下一条关键门槛是让
predicted dynamics 在冻结协议下稳定超过 no-dynamics 与 shuffled-prediction，
并在未见环境复现独立 consequence。主 artifact：
`le-wm-main/results/gdd_wm_cross_environment_20261016/live_reacher_top1_proposal_relative_wilson_two_batch_128_stratified.json`。

该 predicted-dynamics 门槛随后进行了两次直接干预。固定 `0.5` model-cost
fusion 在 fresh 64 roots 上把 final qpos distance 显著降低，但 success 从
`39/64` 降至 `34/64`，且不优于 shuffled-cost placebo。calibration-only top-k
tie-break 又检验 `k=2..64`，所有 true-vs-shuffled CI 均跨零，合法 selector
回退 `k=1`；exploratory fresh `k=2` 的 true/shuffled 均为 `39/64`。因此继续
调 runtime 权重已经被否证，当前正式控制器保持 proposal-relative
no-dynamics。下一修复必须进入 action-contrastive predictor / GDD-WM
distillation，并继续要求 no-dynamics、shuffled 与独立闭环 consequence 三道
门同时通过。

## 2026-10-15 continuation: cross-environment evidence and D1/D2 closure

本轮继续推进不再扩充 PushT 单环境实验，而是完成两项可验证闭环：

- `experiments/summarize_cross_environment_planning.py` 生成
  `results/cross_environment_planning_evidence_20261015/`，按原生 estimand
  汇总 TwoRoom、Reacher、Cube 和 PointMaze。TwoRoom/Reacher 的 live RGB
  paired consequence 为正，Cube 是零收益 failure boundary，PointMaze
  independent gate 仍不显著；四者禁止合并成 universal success rate。
- `experiments/gdd_wm/audit_teacher_candidate_packages.py` 审计现有 24 个
  candidate packages，结果 `0/24` eligible。原因是
  `offline_environment_cost_v1`、`teacher_score=-real_task_cost` 泄漏、原始
  package 缺少 repeated contexts 和 outcome/test runtime refs。这一结果已
  固化到 `results/gdd_wm_teacher_candidate_audit_20261015/`，D2 不绕过 D1。

同时，D2 projection runner 已加入 feature-table provenance SHA-256、输入泄漏
声明和 train-sample-rank 上界；全套回归为 `744 passed, 3 skipped`。下一阶段
只有在获得独立 frozen teacher、重复 candidate contexts 和完整 realized refs
后才运行 D2；在此之前不得用已有 offline cost 标签制造正向蒸馏结果。

本轮进一步加入 `experiments/gdd_wm/build_train_only_readout_teacher_package.py`
作为 D1 infrastructure control：从两个 checkpoint-stage contexts 构造
640 个共同 root/action pairs，使用 train roots-only 的 frozen predicted-latent
readout，并保留完整 RGB successor refs。该 control 的 package audit 为
`1/1 eligible`，但 D1 flip rate 为 `0.3081`、held-out outcome consistency
仅 `0.5800`，因此明确证明“数据结构门禁可通过”不等于“teacher 有效”。该
artifact 不能被当作 D-JEPA teacher，也不能开启 D2。

在该 control 上又完成了预先冻结的五类 feature-family intervention，并用
root-bootstrap 汇总。`action_summary` 在 PointMaze outcome roots 上优于
`predicted_terminal`，但这是 action-distribution confound；source/delta/action
没有稳定增益。因此后续 D1/D2 协议必须把 action-only 作为强制外部 baseline，
不能把 action-only 的正向结果解释成 latent geometry 或 D-JEPA teacher 成功。

进一步加入 action-residualized teacher：先用 train-only action summary 拟合
task cost，再让 latent family 解释残差。结果相对 action-only 没有增量收益
（predicted-terminal consistency delta `-0.0247`，CI 跨零；regret delta
`+0.0250`）。因此后续真正的 GDD-WM teacher 必须通过
`Delta Risk | action summary` 门禁，不能只报告未条件化的 teacher agreement。

2026-10-16 又完成了固定容量 CUDA 对照
`experiments/gdd_wm/audit_train_only_teacher_capacity.py`：在相同
`source_delta_action`、相同 train/outcome split 和相同 action baseline 下，线性
residual 的 held-out regret delta 为 `-0.1568`
（95% CI `[-0.2602,-0.0678]`），固定 `[64,32]` SiLU MLP 的 delta 为
`-0.0631`（CI 跨零）。因此当前负结果不能简单归因于线性 readout 容量不足；
它更支持“低样本/动作支持敏感，可能存在有限线性信号，但尚未证明跨环境的独立
latent transfer 信息”的诊断。该结果仍是 PointMaze 诊断控制，不开启 D2、D-JEPA
teacher 或 planning claim。artifact 位于
`le-wm-main/results/gdd_wm_nonleaking_teacher_control_20261015/capacity_audit/`。

随后在两个独立 seed 的 canonical goal-relative candidate contexts 上复核：raw-action
线性 residual 的 regret delta 缩小为 `-0.0305`（CI 跨零）；改用
`--action-frame goal_relative` 后为 `0.0000`，再用
`--target-mode root_centered` 进行 within-root action-effect 拟合仍为 `0.0000`。
固定 MLP 的对应 delta 均跨零。由此将前一轮 raw absolute-cost 正点估计定位为
candidate-coordinate/root-cost confound；后续 GDD-WM 诊断必须显式报告 canonical
action frame 和 root-centered estimand，不能把该正点估计当作 latent 增量或 D2 证据。

Canonical follow-up is now complete: TwoRoom/Reacher panel exporters write
evaluator-only source/goal coordinates, and the goal-relative/root-centered
rerun removes the raw-action gain (all four environment/readout intervals cross
zero). A subsequent frozen RGB goal-latent conditioning arm also fails to give
stable held-out regret improvement. The next repair target is therefore the
learned action-effect interface and action support, not MLP capacity or
selector tuning. Any new positive result must use this canonical protocol and
then transfer to an independent closed-loop consequence.

### 2026-10-16：HTSC checkpoint-provenance 与 shared-world repair

HTSC 的 split-conformal 实现已按有限样本公式
`ceil((n+1)(1-alpha))` 修复，并新增 `n={8,10,20,40}` 的 exact rank
枚举测试。旧 8-root calibration 在 `alpha=0.1` 下没有有限上界，必须
fail-closed。进一步审计发现，旧 candidate-bank 的后置 split 与 learned
predictor 原始 episode partition 重叠；新 `checkpoint_protocol_v1` 强制继承
checkpoint 的 `54/14/32` train/calibration/outcome roots。

在 100-root、16-candidate、真实 RGB successor 的 CUDA panel 上，frozen
fixed-bank contrast bound 在独立 outcome 达到 `31/32=96.875%` coverage，
shared-world bank 达到 `32/32`，但两者 exact-winner release 均为 `0`。这验证
了 marginal fixed-bank coverage，不验证 adaptive CEM conditional coverage。

根因干预显示，旧 root-conditioned bank 的候选 ID 编码 goal direction：repair
regret `0.0832`，反而差于 Bank-only `0.0132`。换成预先冻结的 shared-world
action bank 后，train-root-only intervention effect adapter 在 32 个独立 outcome
roots 上达到 regret `0.0328`，优于 Bank-only `0.0851`（paired gain `+0.0523`,
95% CI `[+0.0139,+0.0923]`）、No-dynamics `0.2034` 和 shuffled `0.1602`。
未见 candidate-ID 对照相对 shuffled 的 gain 为 `+0.0310`，CI
`[+0.0110,+0.0535]`；matched temporal-order gain over Bank-only 为 `+0.0394`，
CI `[+0.0190,+0.0602]`。generic RGB successor residual adapter 则恶化
held-out successor RMSE，并保持零 release。

正式判定是 `Know + Repair + Abstain`：coverage 与 intervention-supervised
decision repair 成立，但 exact-winner 和预声明 epsilon-regret certificate 仍
零放行。因此该结果不是 frozen LeWM dynamics、在线 planning 或 universal
world-model 证据。汇总 artifact：
`le-wm-main/results/htsc_checkpoint_provenance_20261016/summary.json`。

### 2026-10-16：PointMaze 单次视觉预处理接口修复

进一步定位到离线--在线断裂的实现根因：temporal exporter 先在 dataset 中执行
视觉 transform，又把 tensor 交给拥有同一 preprocessing 的 `model.encode`，
导致离线 latent 为双重 transform，而 RGB runtime 为单重 transform。现在 dataset
保持 raw 0--255 observation，`model.encode` 成为唯一 preprocessing owner，并新增
防回归测试；修复后的 exporter 与 runtime 编码逐元素等价。

100-root、`54/14/32` split 的 CUDA 重跑表明，问题并未因此完全闭合。探索性
deployment-aligned rank 8 同时通过 Bank-only、No-dynamics、shuffled、未见
candidate 和 matched-order 增量门，但 bounded-regret calibration bound 为
`0.30093`，在预声明 `epsilon=0.25` 下 `0/32` 放行。rank 16 将 bound 降到
`0.08918` 并放行 `28/32`，但 Bank-only CI 跨零且未见-candidate gain 为负。
旧 PCA 虽可放行 `24/32` 且 outcome 零 violation，但 PCA 来自错误双重预处理，
只能保留为历史诊断。

因此当前不按已查看的 outcome 选择 rank，正式状态仍为 deployment-aligned
`Repair but Abstain`。下一次 confirmatory 运行必须先仅凭 train/calibration 冻结
chart，再使用新的 locked outcome；只有完整 controls、useful release 与独立 RGB
闭环 consequence 同时通过，才能升级 HTSC 或 learned-dynamics 主张。
