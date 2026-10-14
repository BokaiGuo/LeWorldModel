# LeWorldModel 统一研究项目理解

最后检查：2026-10-12。当前执行方案见
[RESEARCH_PLAN_EVOLUTION_20261012.md](RESEARCH_PLAN_EVOLUTION_20261012.md)。
项目负责人对整体研究的综合理解记录见
[docs/PROJECT_UNDERSTANDING_FROM_USER_20261014.md](docs/PROJECT_UNDERSTANDING_FROM_USER_20261014.md)。
该文件记录研究定位、统一架构、负结果的共同解释和当前证据缺口，不替代实验结果。
科研工程治理、入口注册表和保留历史入口的架构决策见
[docs/RESEARCH_ENGINEERING_GOVERNANCE_20261014.md](docs/RESEARCH_ENGINEERING_GOVERNANCE_20261014.md)、
[docs/EXPERIMENT_REGISTRY.md](docs/EXPERIMENT_REGISTRY.md) 和
[docs/decisions/ADR-001-unified-research-execution.md](docs/decisions/ADR-001-unified-research-execution.md)。
顶会严格审稿意见、Major Concerns、决定性实验矩阵和 Oral/Best Paper 证据门槛见
[docs/REVIEW_CRITIQUE_AND_EXECUTION_CRITERIA_20261014.md](docs/REVIEW_CRITIQUE_AND_EXECUTION_CRITERIA_20261014.md)。
数学物理工具包见
[MATHEMATICAL_PHYSICS_TOOLKIT_20261012.md](MATHEMATICAL_PHYSICS_TOOLKIT_20261012.md)。
本轮对话的完整结构化上下文、文献重叠、证据边界和负结果账本见
[RESEARCH_CONTEXT_20261012.md](RESEARCH_CONTEXT_20261012.md)。
新增的 GDD-WM（Decision Preference → Predictive Geometry）Workstream 研究计划见
[docs/GDD_WM_DECISION_GEOMETRY_RESEARCH_PLAN_20261013.md](docs/GDD_WM_DECISION_GEOMETRY_RESEARCH_PLAN_20261013.md)，
机器可读协议见
[docs/GDD_WM_PROTOCOL_TEMPLATE_V0_1.yaml](docs/GDD_WM_PROTOCOL_TEMPLATE_V0_1.yaml)。
本次纳入的完整执行扩展见
[docs/GDD_WM_EXECUTION_GROUNDED_GEOMETRY_PLAN_20261014.md](docs/GDD_WM_EXECUTION_GROUNDED_GEOMETRY_PLAN_20261014.md)，
其中冻结了 Preference Stability、realized/predicted geometry、RDMC、D0--D4 门禁和
Euclidean 不可表示时的 fallback。
短期历史、Fourier/时频、事件记忆和接触模式的文献边界与冻结实验协议见
[MEMORY_EVENT_CONTACT_LITERATURE_20261013.md](MEMORY_EVENT_CONTACT_LITERATURE_20261013.md)。
逐篇证据核验和文献边界见
[LITERATURE_EVIDENCE_MAP_20261013.md](LITERATURE_EVIDENCE_MAP_20261013.md)。
HTSC 控制理论及其 fail-closed gate 见
[HORIZON_CONDITIONED_TRANSFER_SUFFICIENT_CONTROL_20261013.md](HORIZON_CONDITIONED_TRANSFER_SUFFICIENT_CONTROL_20261013.md)。
谱分析主线的解释版与公式推导见
[SPECTRAL_ANALYSIS_RESEARCH_GUIDE_20261012.md](SPECTRAL_ANALYSIS_RESEARCH_GUIDE_20261012.md)；
不可变的原始逐字记录见
[SPECTRAL_ANALYSIS_RESEARCH_RECORD_20261012.md](SPECTRAL_ANALYSIS_RESEARCH_RECORD_20261012.md)。

## 一、研究不再拆成两条线

仓库包含上游 LeWorldModel 实现和围绕它建立的统一研究扩展。此前将
`theory_spectral_transfer` 与 `empirical_spectral_audit` 分别追踪，是代码兼容层面的安排，不是科学问题的划分。现在统一研究范围为：

```text
covariance–predictor spectral geometry
        ↓
learned LeWM latent dynamics
        ↓
action-conditioned effect / adapter
        ↓
PushT physical verification
        ↓
cross-environment transfer and frozen visual planning
```

统一核心问题已经从“alignment 是否能解决规划”收缩为：

> 当 prediction quality、action distribution 和 predictor spectrum 已匹配时，covariance–predictor–metric 几何能否识别 planner-relevant transfer risk，以及 action-transition mismatch、non-normal transient 和 partial observability 何时使这种识别失效？

`theory_spectral_transfer/` 和 `empirical_spectral_audit/` 仍保留原目录、CLI 和测试，保证历史 artifact 可复现；它们现在分别承担统一研究的理论层和 learned-model 实例化层，不再各自形成独立结论。

### 新增 GDD-WM Workstream（Proposed / Not Yet Validated）

GDD-WM 将当前研究从“诊断 transfer 风险”扩展到一个可检验的决策偏好蒸馏问题：

```text
D-JEPA-style execution preference
  -> teacher preference stability
  -> realized/predicted geometry distillation
  -> preserved action-conditioned prediction
  -> fixed-candidate regret
  -> independent closed-loop consequence
```

它与现有主线的关系是：现有 LeWorldModel 提供 frozen representation、
action-effect permission gate、transfer diagnostics 和 fail-closed planning protocol；
GDD-WM 检验这些基础设施能否支持把稳定的 execution-grounded preference 内化到
不依赖 candidate set 或 teacher runtime 的 predictive geometry。该方向不覆盖、也不改写
JEPA 的条件性正结果、DINO 的 `rejected_fail_closed`、PushT 姿态闭合失败或
commutator 的非通用性。

当前没有 D-JEPA teacher checkpoint/data package，D0 shared candidate dataset、D1
teacher preference stability audit 和 D2 projection-only geometry distillation 均为
`not-run`。因此现有 PointMaze JEPA 结果只能作为候选基础设施，不能写成 GDD-WM 已
蒸馏或已改善闭环规划的证据。只有在 preference stability、predicted/realized
ranking、multi-step predictive preservation、fixed-candidate regret 和独立规划收益
同时通过后，才允许升级 claim。

当前已落地的仅是 D0 的 schema/provenance validator：
`le-wm-main/experiments/gdd_wm/build_shared_candidate_dataset.py`。它可以拒绝缺失
字段、候选集合大小不符、root 泄漏、action support 缺失和 outcome 泄漏，并输出
`schema_audit_only_not_teacher_validation`；没有真实 teacher/data 输入时不会生成
伪造的 candidate dataset。

## 二、统一数学对象

所有实验围绕 action-history lifted 对象
`(Phi, C_Phi, A_Phi, B_Phi, D_Phi, M_task)`；原来的 `(C, P, M, B_u)` 是
`Phi=identity` 的特例：

- `C`：latent covariance；
- `P`：predictor 或局部 Jacobian；
- `M`：预先声明的 task/action metric；
- `B_u`：动作条件下的 effect operator。

其中 `eta_t = Phi(z_{t-L:t}, a_{t-L:t-1})`，`A_Phi` 表示 lifted history 的传播，
`B_Phi` 表示 action tangent，`D_Phi` 表示 task readout。真正的有限时域规划对象为
`G_h^Phi = D_Phi A_Phi^(h-1) B_Phi`。因此升维必须分别通过 observability、action
identifiability 和 external task transfer 三道 gate，不能用 latent separability
替代规划收益。

理论量与 learned-model 量的对应关系为：

```text
(C, P, M, B_u)
      ↕
(latent covariance, learned predictor/local Jacobian,
 task-space/action-effect metric, observed-minus-zero action effect)
```

因此，alignment、principal angle、projector overlap、commutator、Schur off-diagonal energy 和 transient amplification 不是一组孤立指标，而是同一 covariance–predictor pair 在不同几何和时间尺度下的诊断组件。

## 三、研究定位的进一步收紧：可辨识、可认证、可改造

结合新增的 value equivalence、state abstraction、bisimulation 与 action-identifiability
文献，项目的核心创新不应表述为“提出一种更好的 world model”，而应表述为：

> **Decision Transfer Identifiability and Trustworthy Planning in Visual World Models**：给定有限视觉历史、受限动作支持和具体控制目标，能否事先判断 learned world model 对候选动作的预测是否可靠到足以进行规划？

需要严格区分三个层次：

| 层次 | 问题 |
| --- | --- |
| Identifiability | 真实动作后果能否从当前视觉历史和动作数据中辨识？ |
| Transfer sufficiency | latent 是否保留足以比较候选动作的任务相关动力学信息？ |
| Decision reliability | 这些信息是否足以预测 action ranking、regret 和闭环结果？ |

这也收紧了 novelty 边界：value-equivalent abstraction 已经说明不必恢复全部动力学，
因此“只保留决策相关信息”本身不是新贡献。LeWorldModel 需要证明的是，在 frozen RGB、
有限动作激励、部分可观测、固定 horizon 和真实 successor 校准条件下，是否能够形成候选
动作级、可执行的 transfer-sufficiency 判断。

### HTSC：条件性决策证书

HTSC 的目标不是宣称 commutator 或 alignment 是通用证书，而是定义一个依赖校准数据的
条件性输出：

\[
\mathsf{Cert}_H\bigl(\Phi(\mathcal H_t),\mathcal U_H,g,\mathcal D_{\mathrm{cal}}\bigr)
\to
\left(\text{permit/abstain},\{\epsilon_H(U)\}_{U\in\mathcal U_H}\right).
\]

如果对两个候选动作有

\[
\widehat J(U_i)+\epsilon_H(U_i)
<
\widehat J(U_j)-\epsilon_H(U_j),
\]

则在误差上界确实由支持集内独立校准数据覆盖时，才可把该排序作为可靠的规划依据。
因此统一桥梁是：

```text
transfer uncertainty -> action-ranking confidence -> planning permission
```

当前研究必须继续验证 \(\epsilon_H\) 是否在有限 RGB 干预数据下足够窄、是否覆盖 planner
自适应选择造成的分布偏移；界过宽时应 `abstain`，不能把工具输出写成成功规划。

### HTSC 与 GDD-WM 的关系

两条主线共享数据、\(G_h^\Phi\)、action support、real/predicted successor 和独立规划
评估，但科学目标不同：

- **HTSC：Can we trust the world model?** 对冻结模型进行判断，允许或拒绝候选规划；
- **GDD-WM：Can reliable decisions improve the world model?** 利用执行验证过的动作偏好
  改造预测几何，并在新候选分布上获得独立收益。

GDD-WM 经过重新训练得到的收益不能证明原 frozen latent 已经充分；HTSC 的 `abstain` 也
不能被重新包装成 GDD-WM 的蒸馏成功。两者的协议和结果必须分开报告。

## 四、四个研究层级

### 1. 理论可识别性

`theory_spectral_transfer` 负责规定哪些量在什么条件下可识别：

- spectrum-only 信息不能确定 eigenspace alignment；
- gauge-aware metric 下定义协变 alignment 和 commutator；
- principal angles 与 projector overlap 描述子空间关系；
- Schur geometry 描述非正规算子中的耦合；
- transient amplification 不能由 spectrum 或 commutator 单独决定；
- 非线性模型只能通过 local Jacobian 和 remainder bound 做有限范围分析；
- covariance-only whitening 会令普通 alignment 退化，属于 no-go 结果。

这些结果是 learned-model 实验的识别条件和 claim gate，不是脱离实证的普适规划定理。

### 2. LeWM latent 实例化

`empirical_spectral_audit` 与 enriched export 将真实 latent 映射到统一对象，消费并审计：

- latent、dynamics、predicted dynamics、prediction residual；
- action、state、goal、episode/phase；
- physical、counterfactual 和 action-effect 字段；
- 多 horizon transfer、observability、controllability 和 task/action metric。

最新 horizon-aware 跑批进一步冻结了记忆策略：`h=1` 使用 fixed Fourier，`h=4`
使用 Fourier+phase，`h=2` 在 selector stability 不足时 fail-closed 回退。一个
h=1 split 稳定偏向 gated residual，但不改变跨 horizon 的正式 policy。selector
现在只作为 selection-instability diagnostic；完整结果和边界见
[MEMORY_EVENT_CONTACT_LITERATURE_20261013.md](MEMORY_EVENT_CONTACT_LITERATURE_20261013.md)。

当前 PushT/PointMaze enriched audit 的关键结果是：PushT effect-alignment rank correlation 约 `-0.182`、PointMaze 约 `0.037`；joint physical-transient correlation 仅约 `0.214` 和 `0.142`。这说明 geometry-to-transfer 关系具有环境依赖，不能升级成普适 model-selection criterion。

### 3. 诊断到干预

LeWM dynamics、directional training、nuisance suppression、residual-band regularization、local Jacobian stabilization、learned adapter 和 metric-aware selection 都属于统一研究的 intervention layer。

adapter 的每一项必须通过独立训练/验证/测试 split、fresh seeds、held-out contact states、full-fan ranking、runtime successor verification 和 paired baseline。失败版本保留在负消融中；不得通过调 composite 权重、挑 snapshot 或删掉难例来制造正结果。

### 4. 物理与规划验证

验证强度按以下顺序增加：

1. latent multi-step transfer；
2. decoded physical action effect；
3. paired observed/zero successor verification；
4. learned adapter 姿态闭合与终止接口；
5. cross-environment transfer；
6. frozen visual online planning。

privileged simulator-state planning 只作为 interface/control，不能替代视觉在线规划。action-correlated latent signal 也不能直接称为物理因果动力学。

## 五、统一实验流水线

```text
理论 synthetic identification
        ↓
LeWM latent diagnostic
        ↓
action-conditioned bridge
        ↓
geometry-aware intervention / adapter
        ↓
cross-environment matched audit
        ↓
visual planning validation
```

主要入口：

- `experiments/theory_spectral_transfer/run_analysis.py`：理论谱转移和非正规性诊断；
- `experiments/empirical_spectral_audit/run_audit.py`：latent 条件化审计和离线干预；
- `experiments/empirical_spectral_audit/run_matched_learned_transfer_benchmark.py`：固定 MSE、latent dimension、predictor spectrum、training stage 和 action distribution 的 matched learned-model audit；
- `experiments/empirical_spectral_audit/run_oral_level_unified_diagnostic_audit.py`：统一汇总 Oral 级诊断比较、字段覆盖、matched gate、跨环境/统计门禁和已有理论/合成 artifact；显式保留 `non_identifiable`、`partial` 与 `synthetic_only`。
- `experiments/empirical_spectral_audit/run_best_paper_readiness_audit.py`：逐 checkpoint 生成 `G_h` 与风险代理、fail-closed permission、显式理论反例和独立真实收益 gate；工具通过不等于 learned planning 或 Best-Paper 证据通过。
- `experiments/run_*adapter*.py`、`experiments/run_*planning*.py`：LeWM dynamics、adapter、PushT 与视觉规划验证。

所有新 artifact 应声明 `study_scope: covariance_predictor_unified`、`protocol_version`、architecture、seed、environment、horizon、split 和 `claim_boundary`。旧 artifact 不覆盖、不改名，统一汇总时作为历史证据或 negative audit record 读取。

## 六、当前证据边界

已经成立的是：

- 统一的 gauge-aware、covariant-metric、non-normality-aware diagnostic framework 已实现；
- 受控 fixed-spectrum benchmark 证明 alignment 在声明的 operator family 中能提供 spectrum 之外的 transfer 信息；
- learned latent 中存在 action-correlated、局部可观测的 dynamics signal；
- PushT 的 joint endpoint objective 是错误接口，receding block-pose privileged interface 能修复该层失败。

尚未成立的是：

- alignment metrics 普遍预测 learned-world-model 规划性能；
- learned adapter 稳定解决姿态闭合；
- LeWM latent 具有可靠物理因果动力学；
- frozen LeWM 支持视觉在线规划；
- 建立普适 world model；
- commutator 是通用 transfer certificate。

其中最后一项已明确降级为 metric- and horizon-specific diagnostic。没有至少两个 architecture、两个 seed、两个 environment 的 identifiable matched strata 时，任何“普适 model selection”结论都必须标记为 `non-identifiable`。

## 七、统一研究的最终判定

研究不是“理论论文加一个附属 PushT 实验”，也不是“失败的 PushT planner”。它研究的是：

> 为什么可识别的表示几何、局部动力学结构和 action-correlated latent signal，仍不足以自动保证物理因果性、跨环境 transfer 与视觉规划；以及在什么额外条件下，这个 gap 才能被 geometry-aware intervention 关闭。

只有当 matched learned-model transfer、action-effect ranking、姿态闭合和 frozen visual planning 的 gates 同时通过，才可将结论升级为 conditional transfer-risk diagnostic 或更强的方法性主张；否则负结果本身就是统一框架的边界证据。

## 八、2026-10-12 演进后的执行判定

近期文献已经覆盖“prediction error 不等于 planning success”“action-effect ranking
需要显式对齐”“contact mode 需要显式建模”等一般性发现。因此本项目不再把这些
机制本身作为 novelty，而是把它们作为竞争性 diagnostic controls。

当前论文主线固定为三层：

1. **理论层**：证明 spectrum-only、covariance-only whitening 和未经声明 metric 的
   alignment 在 gauge/identifiability 上的边界，并分离 alignment 与 non-normal transient；
2. **实证层**：在 matched learned strata 中检验 geometry 是否对 ATM/action-consistency
   之外仍有增量预测力；
3. **边界层**：用 PushT 说明 mode/precondition 能改善接触保持，但若 mode 内 action
   sign 仍不可辨识，不能把局部 adapter 解释成物理因果 dynamics。

新增的 lifting 层位于实证层之前：先比较 single-frame、delay、action-history、
interaction 和 mode-conditioned lifting 是否提高 velocity/mode/action recovery，再
检验 `G_h^Phi` 的 real-to-predicted transfer。只有第二道 gate 通过，才允许解释
planning risk；否则结果只能称为 representation-level separation。

执行优先级固定为：lifting observability audit → unified diagnostic runner → matched
learned strata → cross-domain
action consistency → Schur/transient intervention → cross-environment leave-one-out
→ 仅在前述 gates 通过后进行 mode-conditioned PushT adapter 和 frozen visual planning。

如果 geometry 只有在 ATM/decision metric 加入后才有效，最终贡献写成“spectral geometry
是 action-transition transfer 的条件性解释量”；如果 matched learned 仍失败，最终
贡献写成“谱、协方差 alignment 与局部 action signal 共同不足以识别 planner-relevant
transfer 的严格边界”。两种结果都不能升级为 universal certificate 或普适 world model。
