# LeWorldModel 研究上下文档案

版本：2026-10-12  
用途：记录本轮研究对话形成的统一问题定义、文献边界、数学工具、实验事实、负结果和后续证据门槛。  
状态：研究档案，不是论文正文；任何数字和结论进入论文前仍需由对应 artifact、冻结协议和独立测试重新核验。

## 0. 如何使用本档案

本档案将此前看似分散的讨论统一到一个研究对象中。它不逐句复制聊天记录，而是保留能够影响研究决策的内容：

- 哪个科学问题是当前主线；
- 哪些理论量已经证明、哪些只是标准工具的重新组织；
- 哪些实验证据为正、为负或尚不可识别；
- 哪些文献已经覆盖相邻问题；
- 哪些 claim 可以写，哪些 claim 必须禁止；
- 为达到 oral / best-paper 水平还缺什么可证伪证据。

本文件与以下文档的关系如下：

| 文档 | 职责 |
|---|---|
| `PLAN.md` | 当前执行计划、研究层级和实验顺序 |
| `PROJECT_UNDERSTANDING_zh.md` | 项目结构与稳定的研究理解 |
| `RESEARCH_PLAN_EVOLUTION_20261012.md` | 方案演进、决策树和主叙事 |
| `LIFTED_TRANSFER_RESEARCH_SPEC_20261012.md` | 升维表示的正式 estimand、gates 和实现规范 |
| `MATHEMATICAL_PHYSICS_TOOLKIT_20261012.md` | 数学、控制、物理、信号和信息论工具 |
| `MEMORY_EVENT_CONTACT_LITERATURE_20261013.md` | 短期历史、Fourier/时频、事件记忆、接触模式的文献边界和实验协议 |
| `SPECTRAL_ANALYSIS_RESEARCH_GUIDE_20261012.md` | 谱分析主线的具体解释与公式推导 |
| `SPECTRAL_ANALYSIS_RESEARCH_RECORD_20261012.md` | 相关原始逐字记录，不做内容改写 |
| 本文档 | 完整上下文、研究边界、文献地图和证据账本 |

新增 Workstream `GDD-WM` 的研究计划与协议分别见
[`docs/GDD_WM_DECISION_GEOMETRY_RESEARCH_PLAN_20261013.md`](docs/GDD_WM_DECISION_GEOMETRY_RESEARCH_PLAN_20261013.md)
和
[`docs/GDD_WM_PROTOCOL_TEMPLATE_V0_1.yaml`](docs/GDD_WM_PROTOCOL_TEMPLATE_V0_1.yaml)。
2026-10-14 提交的执行导向扩展、双路径几何损失、RDMC 条件诊断和 D0--D4 阶段门见
[`docs/GDD_WM_EXECUTION_GROUNDED_GEOMETRY_PLAN_20261014.md`](docs/GDD_WM_EXECUTION_GROUNDED_GEOMETRY_PLAN_20261014.md)。
该 Workstream 的状态固定为 `Proposed / Not Yet Validated`，不改变本文档已经记录的
旧结果和负结果。

## 1.1 新 Workstream：Decision Preference → Predictive Geometry

GDD-WM 是现有 LeWorldModel 研究之上的新增方向，而不是替换原来的谱分析、有限时域
transfer、horizon-conditioned memory 或 PushT 分支。它引入的科学问题是：

> D-JEPA 风格的 execution-grounded candidate preference，能否在不依赖 candidate
> set、Set Transformer 或 teacher runtime 的条件下，被蒸馏为 intrinsic predictive
> geometry，并保持 action-conditioned multi-step prediction？

与现有主线的关系为：

```text
frozen LeWM / JEPA
  -> candidate futures + transfer diagnostics
D-JEPA-style teacher
  -> execution-grounded preference
GDD-WM student
  -> realized/predicted geometry distillation
  -> candidate-set-independent MPC
  -> independent held-out planning audit
```

GDD-WM 的必要证据链是：

```text
teacher preference stability
 -> geometrizable preference
 -> realized geometry improvement
 -> predicted geometry improvement
 -> predictive consistency preserved
 -> decision regret reduced
 -> independent closed-loop planning gain
```

这条链上任何一个环节失败，都应保留为方法边界，而不是继续通过调权重、选择
snapshot、改变 candidate bank 或读取测试结果制造正结果。D0–D2 当前均为 `not-run`：

- D0：建立共享 candidate-set 数据协议，记录 root-disjoint split、候选生成器、CEM
  stage、action support、teacher provenance 和 realized/predicted 引用；
- D1：进行 candidate-set bootstrap / leave-one-candidate-out 的 teacher preference
  stability，并与独立真实执行 consequence 交叉核验；
- D2：先冻结 dynamics，做低容量 projection-only geometry distillation，检查
  realized/predicted ranking、multi-step prediction 和 fail-closed permission。

D0 的 schema/provenance validator 已落地于
`le-wm-main/experiments/gdd_wm/build_shared_candidate_dataset.py`，并有对应单元测试。
它只审计外部输入，不生成 candidate、teacher score、执行 outcome 或 latent 引用；
因此其通过状态最多是 `schema_audit_only_not_teacher_validation`，不能被写成 D0–D2
科学结果。

现有 PointMaze JEPA permission-gate 结果只能作为 D0–D2 的候选基础设施；DINO 已因
action-effect 不可辨识被 `rejected_fail_closed`，不能拿来生成 teacher preference。
当前仓库不存在可直接声明为 D-JEPA teacher 的 checkpoint 或 candidate dataset，附件
中“已生成 ZIP/PATCH、已完成完整仓库验证”的描述不属于本仓库事实。

已有逐字对话导出及其校验信息见
[CONVERSATION_ARCHIVE_20261012.md](CONVERSATION_ARCHIVE_20261012.md)。本文件本身是结构化研究档案，
不是 UI 聊天的逐字转录。

## 1. 研究主线的最终收敛

研究不再试图证明“某个 alignment 指标可以直接修好 world-model planning”，也不把 PushT 的一次成功当作普适 world model 证据。当前唯一主问题是：

> 在 prediction quality、action distribution、latent dimension、training stage 和 predictor spectrum 已经匹配后，协方差—预测器—外部任务度量的几何，能否为独立的 multi-step transfer risk 或 planning risk 提供额外、可重复、可跨环境检验的信息；以及 partial observability、memory、action-transition mismatch 和 non-normal transient 在何时使这种信息失效。

因此，谱分析不再是孤立的 eigenvalue 排序，而是 learned latent 的 action-to-task transfer 诊断的一部分。核心关系是：

\[
\eta_t=\Phi(z_{t-L:t},a_{t-L:t-1}),
\]

\[
\eta_{t+1}=A_\Phi\eta_t+B_\Phi a_t+
\sum_{\ell=1}^{L-1}K_{\Phi,\ell}\eta_{t-\ell}+\epsilon_t,
\qquad y_t=D_\Phi\eta_t,
\]

\[
G_h^\Phi=M_y^{1/2}D_\Phi A_\Phi^{h-1}B_\Phi M_a^{1/2}.
\]

`G_h^Phi` 而非单独的 `spectrum(P)` 才是有限时域规划相关对象。升维只有在同时改善历史可观测性、action identifiability 和 held-out action-to-task transfer 时才是有效干预；“把两个平面上不可分的点映射到高维后变得可分”只是动机，不是科学结论。

## 2. 研究范围的统一

此前项目被人为描述成两条路线：

1. `theory_spectral_transfer`：谱、协方差、alignment 和交换子理论；
2. `empirical_spectral_audit`、LeWM dynamics、adapter、PushT planning：实际模型和规划实验。

现在二者明确属于同一个研究的不同层级：

```text
gauge-aware covariance–predictor geometry
        ↓
learned LeWM latent / action-conditioned dynamics
        ↓
history lifting and task/action metric
        ↓
contact-aware adapter and successor verification
        ↓
cross-environment transfer
        ↓
frozen visual online planning
```

两个代码目录保留原路径、CLI、测试和结果目录，只是为了兼容既有实现与 artifact；不再把它们作为两个互不相干的科学项目分别追踪，也不把理论 benchmark 的正结果自动外推到真实 learned checkpoint。

统一对象为

\[
(\Phi,C_\Phi,A_\Phi,B_\Phi,D_\Phi,M_{task}),
\]

其中 `C_Phi` 是 lifted covariance，`A_Phi` 是历史状态传播，`B_Phi` 是 action tangent，`D_Phi` 是任务读出，`M_task` 是预先声明且不使用测试集选择的外部 metric。原来的 `(C,P,M,B_u)` 是 `Phi=identity` 的特例。

## 3. 从对话中形成的研究演进

### 3.1 早期问题：谱与 alignment 是否能解释规划

早期主张倾向于把 predictor eigenvalue spectrum、latent variance、alignment 或 commutator 当作 world-model quality 或 planning quality 的代理。严格审查后发现，这些量分别描述不同层面：

- 相同 eigenvalues 不代表相同 eigenvectors 或 eigenspaces；
- commutator 描述在声明 metric 下的不可交换性，但不是 transfer 的充分统计量；
- principal angles / projector overlap 描述子空间关系，但不等于 action reachability；
- spectrum 不包含 eigenvector geometry，也不能单独决定 non-normal transient amplification；
- latent distance 或 static decoder 不能保证 action ranking、contact mode 或多步闭环正确。

于是研究从“证明 alignment 普遍有效”改为“构造并审计 covariance–predictor–metric diagnostic，并明确它何时失效”。

### 3.2 理论修正：gauge、metric、非正规性和非线性

严格审稿意见指出：原理论更接近已知线性代数事实的重新组织，且主要覆盖对称/正规算子。后续理论边界必须明确：

1. 任意 `GL(d)` 重参数化下，Euclidean commutator 和普通 Euclidean alignment 不具有坐标不变性；
2. 只有声明协变 metric 后，alignment、commutator 和子空间量才有明确的几何意义；
3. covariance-only whitening 会把普通 alignment 退化为恒等映射，不能把 whitened alignment 当作非平凡诊断；
4. 非正规算子的 Schur off-diagonal、resolvent 和 transient amplification 不能由 spectrum 或 commutator 单独决定；
5. 非线性 learned model 只能在局部 Jacobian 与 remainder bound 的有限范围内分析。

因此 commutator 的最终地位降级为：

> metric-specified, finite-horizon diagnostic component，而不是通用 transfer certificate。

### 3.3 实验修正：从 PushT 成败转向可识别性和机制诊断

PushT 规划失败逐步拆解为多个层次：

- joint endpoint objective 同时要求接触、推动、姿态纠正和 agent 回位，接口耦合导致过早脱离接触；
- privileged receding block-pose objective 能修复接口层失败，但不等于视觉 LeWM planner 成功；
- learned adapter 的局部 action-effect ranking 在部分 case 有信号，却没有稳定解决 torque sign、接触模式转移和姿态终止；
- 视觉观测、接触状态和历史动力学之间存在 partial observability 与 memory mismatch；
- 单一 effect regressor 先回归三维 effect、再由 planner 排序，可能在 angle wrap、接触模式切换和多步误差下放大错误排序。

升维的正确问题因此变成：它是否恢复了 velocity / mode observability、action identifiability 和外部 task transfer，而不是它是否让 latent feature 更容易分离。

## 4. 已有实验事实与证据边界

### 4.1 受控 predictor benchmark：条件性正结果

已有受控 benchmark 使用 9 个 predictor、固定 eigenvalue spectrum、固定 eigenvalue pairing 和 rotation plane，在训练集计算 alignment、在独立测试状态评估 held-out transfer，并进行 128 次重复。报告结果为：

- alignment-aware selection win rate：`1.000`；
- spectrum-only random tie-break win rate：`0.078`；
- alignment selection regret：`0`；
- spectrum-only regret：约 `0.376`；
- alignment–held-out transfer rank correlation：约 `0.999`；
- matched pair 的一步 MSE 差异置信区间包含 0；
- 8-step transfer 存在稳定差异。

该结果只支持：

> 在固定 spectrum、固定 eigenspace family、固定 operator parameterization 的受控条件下，training-only covariance–predictor alignment 可以提供 spectrum 之外的 transfer 信息。

它不能支持普适 learned-model selection。真实 learned checkpoint 必须另行构造 MSE-matched、spectrum-matched、stage-matched strata，不能将 controlled intervention 的因果性外推到独立训练的神经网络。

### 4.2 真实 learned model：当前没有稳定复制

真实 learned checkpoint 中，alignment 并未稳定优于 prediction MSE、variance 或其他 baseline；已有跨环境审计显示 PushT、PointMaze 等的 geometry-to-transfer 关系环境依赖，不能升级为普适模型选择准则。正确的处理是：

- standalone alignment 失败结果必须保留；
- 先用 validation MSE 做质量过滤，再在 prediction-matched pool 中使用 geometry 作为 tie-breaker；
- 若 matched pool 不存在，结论写 `non-identifiable`，不通过放宽 tolerance、改变 composite 权重或挑选 snapshot 补齐；
- 需要控制 MSE、training stage、variance、action risk 后报告 geometry 的增量解释力。

### 4.3 PushT privileged planning：接口修复，不是视觉成功

此前 receding block-pose repair 给出过稳定的 privileged interface 改善：joint endpoint 为 `0/16`，receding block-pose repair 为 `16/16`，block XY 约从 `141.74` 降到 `11.25` px，angle error 约从 `1.657` 降到 `0.165` rad。它证明的是 joint endpoint objective 与姿态/接触任务不匹配，以及 receding task-space objective 的接口价值；它不证明 frozen visual latent 或 learned adapter 已经规划成功。

### 4.4 learned adapter 与姿态闭合：明确负结果

当前 learned adapter 的失败不是通过继续调权重即可合理视为解决。对话中保留的关键负结果包括：

- learned adapter 严格成功率 `26/500 = 5.2%`，角度误差约 `1.458 rad`，未形成稳定姿态闭合；
- 某次 mixed trajectory pilot 中 seed 9173 为 `1/4` strict success，block XY 均值约 `51.3 px`，angle 仍约 `1.31 rad`；
- observed local torque-to-angle calibration 的双 seed 仍仅约 `3/8`、`2/8`；
- 标准化 contact-frame KNN adapter 在一个 seed 升至 `4/8`，另一个仍为 `2/8`，跨 seed 不稳定；
- hierarchical translation + full-fan torque verification + contact recovery + safe return 的 32-case 汇总为 strict `3/32 = 9.375%`，corrected verification baseline 为 `6/32 = 18.75%`，paired block XY 基本无改善；
- 失败集中在接触模式下的 torque/action-effect ranking 与姿态终止接口，而非简单的视觉坐标读出。

因此当前可写为：latent 中可检测到局部 action-correlated dynamics，但 learned adapter 尚未稳定解决姿态闭合。

### 4.5 frozen visual planning 与跨环境：负结果

- frozen visual planning 当前约为 `1/200`，且相对于 zero baseline 的 block-XY 误差显著恶化，说明 vanilla planner 具有负收益；
- cross-environment 结果只有 PushT 对 action 敏感，TwoRoom 和 PointMaze 出现反向关系，不能支持普适迁移；
- `389 passed` 或其他测试通过只能说明实现和审计可靠，不能替代视觉规划、物理因果性或普适 world-model 证据。

### 4.6 lifting / signal / information audit：第一轮结论

第一轮真实 enriched export 中，JEPA–PushT 在一次切分上 causal-delay lifting 改善 held-out task-effect RMSE，但 interaction 和时频特征恶化；PE 未通过，memory closure 没有稳定优于 Markov；PointMaze 由于历史支持不足，causal signal gate unavailable。结论是：升维有局部表示收益，但尚未证明 transfer-sufficient lifting，更不能直接升级为因果动力学。

### 4.7 统一协议修正后的 split-level 结果

在同一 `jepa_pusht_windows4_metrics_enriched_export.json` 上完成了 10 个
seeded episode splits；fit-only PCA、rank-matched feature budget、episode-disjoint
outcome 和 paired maximum-history support 全部固定。结果保存在
`results/protocol_corrected_temporal_export_20261012/jepa_pusht_windows4_composite_multiseed_summary.json`：

- causal delay：`10/10` split 胜出，matched 平均 RMSE 降幅约 `15.5%`；
- state+action time-frequency：`10/10` split 胜出，matched 平均降幅约 `24.9%`；
- paired memory-4：`8/10` split 胜出；
- interaction：matched 仅 `3/10` 胜出，平均降幅约 `-7.4%`。

这表明原先的时频负结果主要包含“没有保留 action interface”和“未统一 feature
budget”的协议问题，但 interaction 负结果仍然存在。进一步把完整过去 action
history 加入 finite-memory model 后，paired memory-4 只剩 `1/10` 胜出，结果见
`results/protocol_corrected_temporal_export_20261012/jepa_pusht_windows4_action_history_multiseed_summary.json`。
因此当前更具体的机制解释是：在每个 split 仅有约 `52` 个 fit rows 时，完整
history 的参数维度超过有效样本支持，导致低样本/高维过拟合；“增加历史”本身不是
解决方案。该结果属于统一 covariance–predictor–memory–input–output 对象的负诊断，
不是独立的 temporal 支线。

同一统一审计随后发现并修复了 lifted pose adapter 的另一个接触接口错误：接触丢失
时，部署代码指向 block centroid，而训练/action bank 使用的是 support-boundary
inward normal。改为复用声明的 support-normal 接口后，在相同两个 seed 上 final XY
从 `108.84/90.83 px` 降到 `48.19/30.23 px`；strict closure 从 `2/32, 4/32`
变为 `2/32, 8/32`，但 final angle error 仍为 `1.64/1.35 rad`。因此这验证了
mode-recovery implementation cause 并改善了平移，却没有解决 torque ranking 和
terminal interface；不能升级为稳定 learned pose closure。对应 artifact 位于
`le-wm-main/results/negative_cause_diagnosis_20261012/lifted_pose_regressor_supportnormal_seed9173/`
和 `..._seed9181/`，修复前结果保留为 paired baseline。

### 4.8 短期历史与事件记忆：文献边界和当前结果

本项目对“latent 没有可利用记忆”的早期假设进行了 protocol-corrected 审计。当前最准确的结论不是 Fourier 或事件记忆单方面胜出，而是：

- causal delay、state+action time-frequency 和 paired memory-4 在 PushT split-level audit 中分别达到 `10/10`、`10/10` 和 `8/10` split 胜出，说明 frozen latent 轨迹中存在可利用的短期历史；
- interaction lifting 仅 `3/10` 胜出，完整过去 action history 降至 `1/10`，支持低样本/高维过拟合机制；
- fit-only adaptive selector 在 10 个 outer split 上改善 current-state/action baseline，但没有可靠超过 fixed first-harmonic Fourier control；
- 因此事件记忆目前是 `partial / conditional transfer evidence`，不是已解决的通用 memory，也不是姿态闭合或因果动力学证据。

文献调研显示，类似失败通常被拆成两个对象：一是 partial observability 下的 hidden-state/memory inference，二是接触丰富任务中的 hybrid contact-mode transition。PlaNet/RSSM 类方法用随机隐状态和 latent overshooting 处理前者；Visual Foresight 类方法用局部动力学、短 horizon 和 receding-horizon observation 降低长 rollout 误差；ContactNets、contact-aware MPC 和 D-LGP 类方法显式表示接触几何、离散 mode 和 mode transition。近期 episodic/retrieval memory 与 phase-aware predictor 进一步强调 cue-gated retrieval、delta reconstruction、episode/phase boundary 和 future/action conditioning。

这些方法共同支持的工程方向是 `event-gated memory -> successor-mode prediction -> mode-conditioned effect -> short pulse -> RGB re-observation -> recovery`，而不是继续堆叠固定谐波。频域特征仍保留为强基线：它可能对平稳/周期/冲击结构提供更强低维正则化；事件特征优于 current state 并不意味着它应稳定超过 Fourier。

完整文献—机制—代码映射、引用链接、指标和升级门槛见
[MEMORY_EVENT_CONTACT_LITERATURE_20261013.md](MEMORY_EVENT_CONTACT_LITERATURE_20261013.md)。后续只有在事件/mode 可辨识性、real/predicted successor consistency、multi-step transfer、contact retention 和 angle closure 同时通过独立 RGB-only 协议后，才允许把该方向升级为有效规划干预。
逐条来源核验、机制边界和方法论调整另见
[LITERATURE_EVIDENCE_MAP_20261013.md](LITERATURE_EVIDENCE_MAP_20261013.md)。
控制理论版本的定义、soundness proof、tightness/counterexample 和 fail-closed
接口见
[HORIZON_CONDITIONED_TRANSFER_SUFFICIENT_CONTROL_20261013.md](HORIZON_CONDITIONED_TRANSFER_SUFFICIENT_CONTROL_20261013.md)。

### 4.9 Horizon-aware frozen policy：selector 稳定性是诊断量

新的真实跑批在未见 split 上复核了预先冻结的 horizon-aware memory policy。已确认的定性结果为：`h=1` 正式使用 fixed Fourier，其中一个 split 的 selector 稳定偏向 gated residual；`h=2` 的 selector 在所有 split 上因稳定性不足而回退；`h=4` 的五个 split 均回退到预先冻结的 Fourier+phase。单个 h=1 split 的局部偏好不改变正式 policy，也不能用于事后改权重。

因此当前应把两种对象分开：

- **frozen policy**：由 calibration 阶段预先确定，跨 horizon 固定执行；
- **selector stability**：用于诊断 split sensitivity、memory-family heterogeneity 和 fail-closed 风险。

完整数值表、每个 split 的稳定性频率、fallback 频率和 paired held-out regret 仍需由 append-only 汇总 artifact 固化；在此之前，该批结果标记为 `partial`，不能写成已验证的 planning gain。当前最有价值的规律是：

> memory selection exhibits a horizon-dependent stability phase transition：短 horizon 可能存在局部事件收益，中间 horizon 的选择最不稳定，而较长 horizon 需要预先声明的 phase-aware policy。

## 5. 负结果的机制解释与应对方向

### 5.1 非 Markov 投影与历史缺失

单帧 frozen latent 可能把速度、接触持续时间、冲击和 mode transition 投影掉。应使用 causal delay / action-history lifting，拟合 finite-memory realization，并以 memory closure 和 held-out effect RMSE 为 gates。若 lifting 后 PE 或 closure 不通过，不能把它解释为动力学恢复。

文献调研后，历史接口进一步固定为 event-gated memory、successor-mode head、phase/time-to-transition auxiliary 和 retrieval/recap memory 的可比较组合；Fourier/first-harmonic 作为固定强基线，不得因事件特征的局部收益而删除。事件或时频特征若只改善表示误差、不改善 action ranking、真实/预测 successor 或任务空间指标，标记为 `representation-only`。

在当前冻结策略中，`h=1 -> fixed Fourier`、`h=4 -> Fourier+phase`；`h=2` 若 selector stability gate 不通过则回退到 fixed Fourier。该策略是方法协议，不是由测试集结果动态选择的 controller。

### 5.2 action-transition mismatch 与错误排序

latent 中有 action-correlated signal 不等于预测的 action effect 在真实环境中可执行。需要 shared-candidate temporal ranking、real/predicted transition consistency gate、inverse-dynamics probe、counterfactual action contrast 和 runtime successor verification。必须比较 predicted successor 与 observed successor，不能只在离线 effect regression 上验收。

### 5.3 接触模式和 hybrid dynamics

PushT 的接触是混合系统，不应让单一连续回归器隐式承担 free/contact/sticking/sliding/lost-contact 的 successor mode 识别。可行结构为：

\[
p(m_{t+1}\mid\phi_t,a_t),\qquad
\widehat{\Delta z}=\sum_m p(m_{t+1}=m\mid\phi_t,a_t)F_m(\phi_t,a_t),
\]

并使用短 pulse、receding observation、uncertainty gate、contact reacquisition 和终止条件。mode head 只有在 RGB-only held-out contact states、contact-loss rate、姿态闭合和 matched zero baseline 上稳定改善时才算成功。当前相关 specialist、structured ridge、online calibration 和 KNN 结果均未达到这一门槛。

### 5.4 非正规 transient 与有限时域风险

预测器 spectrum 可能稳定而 finite-horizon rollout 仍爆炸或发生 rank inversion。需要固定 spectrum 后干预 Schur off-diagonal、resolvent norm、pseudospectrum 和 transient gain；报告 horizon-dependent transfer、not just one-step MSE。commutator 只能作为其中一个诊断量，不能单独担任 certificate。

### 5.5 规划接口与终止接口

joint endpoint 将接触、物体位姿、agent 回位混成一个目标，导致 planner 为降低 agent endpoint cost 而脱离接触。应将 task-space block pose 作为主要目标，采用 receding-horizon、contact-safe return 和显式 terminal-success verifier；但必须区分 privileged simulator-state interface repair 与视觉 learned planner 的科学 claim。

## 6. 相关文献与重叠边界

以下工作已覆盖相邻机制，不能把这些内容单独包装为创新：

| 工作 | 已覆盖的问题 | 本研究剩余可检验差异 |
|---|---|---|
| ATM: *Why Latent World Models Can Fail to Plan* ([arXiv:2606.09028](https://arxiv.org/abs/2606.09028)) | action identifiability、Bayes inverse risk、真实/预测 transition mismatch | 在 action risk 已匹配后，协方差—预测器—metric geometry 是否还提供 transfer 信息 |
| AD-WM ([arXiv:2609.30264](https://arxiv.org/abs/2609.30264)) | action recovery、residual dynamics、counterfactual MPC | 将 action-transition consistency 与 gauge-aware/non-normal transfer geometry 统一审计 |
| D-JEPA ([arXiv:2609.24749](https://arxiv.org/abs/2609.24749)) | decision-aligned candidate futures | 测量 alignment/Schur/transient 是否在 decision alignment 之后仍有增量信息 |
| Temporal-Distance-JEPA ([arXiv:2607.25337](https://arxiv.org/abs/2607.25337)) | temporal progress、跨轨迹 negatives、rollout consistency | 对 history lifting、task readout 和 predictor spectrum 做 matched transfer audit |
| *On the Identifiability of Controlled World Models* ([arXiv:2607.22430](https://arxiv.org/abs/2607.22430)) | representation/transition identifiability、spectral separation、action excitation | 将可识别性条件落实到 covariance–predictor–metric diagnostic 与规划 risk |
| ContactNets ([arXiv:2009.11193](https://arxiv.org/abs/2009.11193)) | signed distance、contact Jacobian、互补约束和接触模式 | 检验 RGB frozen latent 是否能在不使用 privileged contact state 的条件下恢复同类 mode observability |
| Visual Foresight / local dynamics ([arXiv:2206.14802](https://arxiv.org/abs/2206.14802)) | 局部动力学、短 horizon lookahead、receding control | 将短时 action-to-task transfer 与 latent spectral geometry 联系并做独立 held-out 审计 |
| seq-JEPA | transformation-aware invariant/equivariant representation | 明确区分：seq-JEPA 研究架构如何诱导 transformation-aware representation；本研究研究 covariance-relative linear transfer quantity 能否从 predictor eigenvalues 中识别 |

主文应使用 `group-unspecified, rather than equivariance-aware`，而不是把 `symmetry-agnostic` 写成似乎对任意 symmetry 都适用。当前研究不预先指定 transformation group，也不声称建立等变 world model。

## 7. 六个目标的当前状态和可接受升级方式

| 目标 | 当前状态 | 只有满足什么才可升级 |
|---|---|---|
| alignment 普遍预测规划性能 | 仅受控 benchmark 成立；真实 learned model 未稳定成立 | 多架构、多 seed、多环境、MSE/spectrum/stage/action matched，geometry 对独立 planning risk 有增量预测 |
| learned adapter 稳定姿态闭合 | 明确失败且跨 seed 不稳定 | RGB-only successor-mode、torque ranking、终止接口和 fresh held-out episodes 同时通过预注册门槛 |
| latent 可靠物理因果动力学 | 只有局部 action-correlated observability | intervention、counterfactual、PE、mode-conditioned dynamics、真实/预测 successor 一致性和跨状态泛化 |
| frozen LeWM 视觉在线规划 | 当前负结果，约 `1/200` | 完全 RGB-only、独立 task/episode、paired zero baseline、multi-seed、多环境稳定改善 |
| 普适 world model | 尚未成立 | 只能先收缩为 scoped generalization；需要 leave-one-environment-out 和独立任务收益 |
| commutator 通用 certificate | 已被理论边界否定 | 不应再追求此 claim；改为 metric- and horizon-specific diagnostic，并与 transfer function、Schur/transient 组合 |

## 8. Oral / Best Paper 证据门槛

### Oral 级最低要求

1. 完成 similarity / gauge-aware transfer theorem；
2. 给出 minimal realization identifiability、finite-memory companion equivalence 和 non-normal finite-horizon error bound；
3. 完成 fixed-spectrum / fixed-MSE synthetic intervention；
4. 至少 3 个 architecture、5 个 seed、4 个 environment 的 learned benchmark；
5. 使用 root-disjoint / episode-disjoint test，并完成 leave-one-environment-out；
6. 证明 geometry 在 MSE 和 action risk 之外仍有增量解释力；
7. transfer-aware selection 在独立 held-out planning 上降低 regret；
8. 完整解释 PushT 的 observability、mode transition、ranking、contact loss 和 terminal interface 失败链条。

### Best Paper 级额外要求

- 一个跨领域可复述的新规律，而不是若干指标的拼接；
- 一个其他研究者可直接运行的 transfer-risk / model-selection 工具；
- 跨环境的独立规划收益，且不依赖 privileged state；
- 至少一个强负结果由理论提前预测；
- 与 ATM、AD-WM、D-JEPA 等最近工作有清晰、可验证的增量；
- 不用测试集选择 metric、权重、snapshot 或困难样例。

## 9. 冻结后的实验协议

每个 learned-model 实验必须记录：architecture、seed、environment、latent dimension、training stage、action distribution、training/validation/test split、horizon、完整 predictor spectrum、one-step 与 multi-step MSE、variance、alignment、commutator、Schur geometry、transient gain、action risk、planning score 和 oracle regret。

### 9.1 Matched-quality model-selection protocol

1. 使用 validation MSE 做预先声明的质量过滤，例如最优值容差内的 checkpoint；
2. 只在 MSE、spectrum、stage、latent dimension 和 action distribution 匹配的 pool 内计算 geometry；
3. alignment、principal angle、projector overlap、Schur、transient 分开报告，不随意调 composite 权重；
4. planning 使用独立 episode/task，报告每个 seed 的 regret、置信区间和 rank correlation；
5. 如果没有可识别 matched strata，报告 `non-identifiable`，停止升级 claim。

### 9.2 Lifting / causal transfer protocol

1. 所有 lifting basis、POD、metric 和 decoder 只从 fit episodes 估计；
2. 先做 PE、memory closure、mode observability，再解释 `B_Phi` 或因果 action effect；
3. 对同一 source state 使用 action contrast，比较真实/预测 successor，而不是只比较 latent reconstruction；
4. 以 held-out multi-horizon effect RMSE、ranking regret、inverse action risk 和 task-space error 为主指标；
5. interaction、causal delay、频域和信息量特征若只改善表示而不改善 transfer，标记为 representation-only。

### 9.3 Contact-aware planning protocol

1. 训练 contact-mode successor head 和 mode-conditioned effect heads；
2. 使用 shared candidate set，避免每个模型看到不同候选导致排名不可比；
3. 使用短 pulse 和 receding RGB observation；
4. uncertainty / lost-contact 时禁止继续 torque，进入 contact reacquisition；
5. 独立验证 contact loss、angle closure、object progress、terminal success，并和 zero / privileged interface baseline 配对；
6. 任一组件只改善 XY 但不改善角度闭合时，不能称为姿态解决。

## 10. 统一的结果标签和论文措辞

建议所有 artifact 使用以下状态：

- `validated`：预注册主指标、独立测试和不确定性均通过；
- `partial`：只通过中间 gate 或单环境/单 seed；
- `negative`：预注册结果方向明确失败；
- `non-identifiable`：匹配池、PE、history support 或 mode labels 不足；
- `proxy/smoke`：只用于代码和接口检查；
- `environment-blocked`：运行环境导致不能获得科学证据；
- `not-run`：尚未执行。

允许的当前主结论：

> We establish a gauge-aware, covariant-metric, non-normality-aware diagnostic framework and use it to show that spectral structure, latent action-correlated signals, and local predictive observability do not automatically yield reliable visual planning or universal world-model transfer.

中文对应表述：

> 本研究建立了具有明确坐标规范、协变任务度量和非正规性扩展的诊断框架，并以实验证明：谱结构、潜空间中的动作相关信号和局部可预测性，并不会自动转化为可靠的视觉规划或普适 world-model 迁移。

禁止的表述：

- “alignment 普遍优于 MSE”或“alignment 是通用模型选择准则”；
- “commutator 是通用 transfer certificate”；
- “action-correlated signal 证明了物理因果动力学”；
- “privileged simulator-state planning 证明了 frozen visual planning”；
- “单环境、单架构或单个成功率证明了普适 world model”；
- “通过测试、artifact 完整或 GPU 可用等于科学假设成立”。

## 11. 关键实现与结果入口

统一入口和当前规范文档：

- 理论：`experiments/theory_spectral_transfer/run_analysis.py`；
- latent audit：`experiments/empirical_spectral_audit/run_audit.py`；
- dynamics / adapter / planning：`experiments/run_*` 与 `experiments/online_planning_protocol.py`；
- matched learned transfer：`experiments/empirical_spectral_audit/run_matched_learned_transfer_benchmark.py`；
- lifting / hybrid / signal audit：见 `LIFTED_TRANSFER_RESEARCH_SPEC_20261012.md` 中的实现清单；
- PushT privileged repair：`experiments/run_pusht_receding_block_pose_planner.py`；
- 统一计划：`PLAN.md`、`RESEARCH_PLAN_EVOLUTION_20261012.md`。

重要 artifact（均不得覆盖或删除）：

- `results/covariance_predictor_diagnostic_benchmark_20261008/diagnostic.json`；
- `results/covariance_predictor_diagnostic_benchmark_20261008/diagnostic.md`；
- `results/pusht/planning_repair_paired_statistics_deep_rescue_20261006/planning_repair_paired_statistics.json`；
- `results/pusht/lewm/hierarchical_fullfan_return_ablation_20261006/hierarchical_fullfan_return_ablation.json`；
- `results/projected_hybrid_physics_audit_20261012/`；
- `results/projected_hybrid_physics_audit_20261012_action_history/`；
- `results/protocol_corrected_temporal_export_20261012/jepa_pusht_windows4_composite_multiseed_summary.json`；
- `results/protocol_corrected_temporal_export_20261012/jepa_pusht_windows4_action_history_multiseed_summary.json`；
- `results/six_goal_evidence_20261012/`；
- 各 `negative_cause_diagnosis_*`、adapter smoke 和 frozen visual planning 结果目录。

## 12. 下一阶段执行顺序

1. 冻结统一 manifest、版本和 metric；
2. 修复 action-support / zero-selection invariant；
3. 做 encoder-level inverse-dynamics action-sensitivity probe；
4. 将一步 effect ranking 改成 shared-candidate temporal ranking；
5. 增加 real/predicted transition transfer gate；
6. 建立 contact-mode conditioned adapter、uncertainty gate 和 online residual；
7. 在 matched MSE/spectrum/stage/action distribution strata 中做跨架构、跨 seed、跨环境验证；
8. 做 fixed-spectrum Schur/transient intervention 和 leave-one-environment-out；
9. 只有 geometry 的条件性增量证据成立后，才重启 RGB-only PushT mode-aware planner；
10. 最后再评估 frozen visual planning 与 scoped generalization。任一 gate 失败时，按证据状态降级，不用调权重或挑 snapshot 覆盖负结果。

本档案的核心判断是：研究的价值不再取决于是否把所有负结果“调成正结果”，而取决于能否把负结果变成由识别条件、transfer geometry、hybrid mode 和 finite-horizon dynamics 共同解释的可复现边界；若跨环境 matched-quality 实验成功，则主张 conditional transfer-risk diagnostic，若失败，则诚实地保留为 geometry-insufficiency 的严格边界结果。

2026-10-13 新增的 horizon-conditioned transfer-sufficiency runner 将 selector 永久降级为诊断项，正式 deployment policy 使用预先声明的 horizon policy。fit-only POD rank 32 的新 split-seed confirmation 中，候选 `carrier_plus_phase_for_multistep_v2` 相对 fixed Fourier 在 h=2 的 5 个 episode-holdout split 中胜 4/5，在 h=4 胜 5/5；这是同一 JEPA-PushT export 上的正向 transfer-sufficiency proxy，不是跨架构、跨环境或 online planning 证据。artifact 见 `le-wm-main/results/horizon_conditioned_transfer_sufficiency_v2_confirm_20261013/summary.json`。

2026-10-13 新增 Oral 级统一诊断入口 `experiments/empirical_spectral_audit/run_oral_level_unified_diagnostic_audit.py`。它对 `learned_transfer_operator_panel_20261013` 的 90 条 checkpoint 记录执行了字段覆盖、面板设计门禁和严格 MSE+spectrum matched audit，并汇总既有 fixed-quality synthetic、horizon-operator、gauge/non-normality 与跨环境 artifact。当前实证状态为：3 架构、5 seed、30 snapshot/架构均存在，但只有 PushT；严格 matched 仅有 1 个局部 stratum，未达到两环境独立复制，因此 learned intervention、LOEO、paired bootstrap/permutation/Holm 均保持 gated/non-identifiable。由 A/B 推导的 resolvent 与有限时域 reachability 只标为 `derived_operator_proxy`，不能替代 ATM 或独立 real/predicted transition 数据。审计产物见 `le-wm-main/results/oral_level_unified_diagnostic_audit_20261013/`。

2026-10-13 新增 Best-Paper readiness audit `experiments/empirical_spectral_audit/run_best_paper_readiness_audit.py`。它逐 checkpoint 计算 h∈{1,2,4,8,16} 的 `G_h=D A^{h-1}B` 和有限时域 transfer-gain proxy，输出诊断 ranking、action-identifiability/off-manifold/readout/real-predicted successor warnings，并在缺独立风险包络时强制 `rejected_fail_closed`。90 条当前 learned records 全部生成了 `G_h`，但 90/90 被 fail-closed；显式同谱、同 covariance、同声明 MSE/action-risk、同 commutator 而不同 `G_h` 的有限维 counterexample 通过。结果树中没有符合“action ranking inversion + multi-step task error + contact retention + angle closure + planning regret、无 privileged state、无 test snapshot selection”的独立收益 schema，因此 Best-Paper 状态仍为 `not_ready_without_independent_real_gain_and_cross_environment_calibration`。产物见 `le-wm-main/results/best_paper_readiness_audit_20261013/`。

2026-10-13 完成 Oral 实验覆盖与匹配原因审计。修复 `run_oral_level_unified_diagnostic_audit.py` 对 `action_inverse_risk` 别名的误报，并从 train-only fit export 计算 latent variance、从独立 held-out outcome 记录计算 candidate elite regret。新增 `experiments/empirical_spectral_audit/oral_experiment_coverage_audit.py` 及 artifact `le-wm-main/results/oral_experiment_coverage_audit_pointmaze_pusht_final_20261013/`：PointMaze 100 条记录的 MSE、latent variance、alignment、commutator、Schur、transient、ATM、elite regret、planning risk 均有来源；principal-angle/projector、real/predicted transfer、reachability、resolvent 在该 panel 仍缺失。PointMaze+PushT 仅有 2 个环境，horizon 支持只有 1/2/4，且严格 matched strata 为 0。训练集共同窗口搜索 `le-wm-main/results/pointmaze_common_fit_window_search_20261013/` 的最佳窗口只有单一 `pointmaze_jepa_mlp` architecture 的 3 个 seed；0.25/0.50 tolerance exploratory runs 仍没有跨 architecture clique。因此当前根因是 learned representation/action-support 的质量与谱重叠不足，而不是 selector 统计实现错误；formal claim 继续保持 `non-identifiable`。

2026-10-13 继续完成 PointMaze learned-adapter 负结果定位与修复实验。`train_pointmaze_learned_adapter_panel.py` 新增了四类不改变 outcome 协议的 train-only intervention：`standardized_mlp`（absolute target 标准化）、`task_aligned_mlp`（加入 train-only task-readout loss）、`task_effect_mlp`（仅预测 train-fitted task effect）和 `task_effect_lowrank_mlp`（train-only action PCA-4）；同时新增 `--outcome-count` 以便显式运行未覆盖旧 artifact 的 exploratory split。结果树：`le-wm-main/results/pointmaze_standardized_mlp_panel_rawaction_fixed_20261013/`、`pointmaze_task_aligned_mlp_panel_rawaction_fixed_20261013/`、`pointmaze_task_effect_mlp_panel_rawaction_fixed_20261013/`、`pointmaze_task_effect_lowrank_mlp_panel_rawaction_fixed_20261013/`。在原始 32-row matched support 上，DINO 的 raw task-effect gain 约 `-0.0417`；standardization 后约 `-0.2388`，task-aligned 后约 `-0.1154`，说明尺度和 readout mismatch 是部分原因，但不能单独修复 DINO。JEPA 在这些干预下均保持正向，task-effect MLP 的 seed-cluster gain 约 `+0.1741`，sign-flip `p≈0.0299`。

同一 runner 现在加入 calibration-only `calibration_task_rmse_penalty_beta1` fail-closed gate，并将 gated 结果写入 `heldout.planning_details_conservative`；汇总器同步报告 raw 与 conservative 两列。Artifact `le-wm-main/results/pointmaze_task_effect_mlp_gated_panel_rawaction_fixed_20261013/` 显示：JEPA gated gain 约 `+0.0980`，5 seed 正向；DINO gated gain 约 `+0.00449`，bootstrap CI `[0.00014,0.00884]`，但 sign-flip `p≈0.122`，因此只能结论为“消除负点估计并安全回退到 h=1”，不能称稳定正收益。该结果将 DINO 的剩余问题定位为 representation-conditioned action/task transfer 不可辨识与 calibration-to-outcome 分布漂移，而不是单纯 selector bug；DINO 当前仍不能支持普适 learned-world-model claim。

随后完成两项排除性干预：将 train-only PCA rank 从 8 提升到 16，以及将 adapter 输入从 formal shared raw action chart 切换为 DINO/JEPA export-native normalized chart。rank-16 DINO raw task-effect gain 约 `-0.144`；export-native chart 与 raw chart 的 DINO 结果相同（raw gain 约 `-0.0417`，conservative gain 约 `+0.00449`）。因此当前负结果不是 PCA 截断或 action 坐标转换造成的。新增 artifact 为 `le-wm-main/results/pointmaze_task_effect_mlp_rank16_rawaction_fixed_20261013/` 和 `le-wm-main/results/pointmaze_task_effect_mlp_exportchart_exploratory_20261013/`。正式方法决策固定为：先估计 calibration-only finite-horizon transfer uncertainty；若候选 horizon 不通过 gate，则拒绝 learned candidate 并回退 h=1。不能用 DINO 的非显著安全回退改写成 positive learned transfer。

2026-10-13 继续完成了两项代码级修复与验证。第一，将 calibration-only gate 升级为 horizon-conditioned `planning_permission`：每个 horizon 使用自己的 train-only task-effect scale；h=1 需要 normalized task RMSE `<=0.75` 且 calibration action-inverse risk `<=0.05`，其他 candidate horizon 的 normalized task RMSE 上限为 `1.25`。权限失败时不再把保守候选当作 learned success，而是在 `heldout.planning_details_conservative` 明确写入 `rejected_fail_closed`、拒绝原因、允许 horizon 和阈值。新 artifact `le-wm-main/results/pointmaze_task_effect_mlp_permission_horizonscaled_rawaction_fixed_20261013/` 显示：JEPA 的 25/25 checkpoint 通过，conservative `Delta R=+0.0980`，5 seed sign-flip `p=0.0306`，bootstrap 95% CI `[0.0649,0.1328]`；DINO 的 25/25 checkpoint 全部拒绝，原因均为 `h1_task_effect_not_identifiable` 与 `action_effect_not_identifiable`，其正式结果为固定 h=1 fallback、`Delta R=0`。这解决的是负 transfer 的安全传播问题，不是把 DINO 表示改造成可规划表示。

第二，新增 `audit_dino_vector_artifact_join.py` 检查旧 DINO 200-episode vector-only artifact 是否可以无歧义扩大正式 support。审计显示 episode 前缀与 32-row enriched export 一致，但 vector artifact 缺少 sample/episode IDs、action blocks、source states 和 target states；h=1/2/4 的 predicted vector 可以作相似性诊断，但 target vector 与 enriched target 的 RMSE 约为 `0.593/0.597/0.593`，不能作为同一正式 transition 表。artifact `le-wm-main/results/pointmaze_dino_vector_artifact_join_audit_20261013.json` 因而标记为 `formal_action_conditioned_join=rejected_fail_closed`、`vector_only_joined_diagnostic=available`。因此不能通过拼接旧向量、复制 action 或补造 state 来人为扩大 DINO 样本；若要解决 DINO 的表示条件，仍需要重新导出包含完整 action/state/source/target schema 的 DINO checkpoint 数据。

2026-10-13 GDD-WM 接入后的首次执行审计已完成。D0 validator 将历史
`horizon_matched_ranking.json` 判为 `rejected_fail_closed`：该 artifact 只有旧的
ranking rows，没有 GDD-WM 所需的 records/provenance/teacher 字段，不能直接转换为
D-JEPA teacher 数据。重新运行 Oral unified audit 后，当前 panel 仍为单环境、仅 1 个
可识别 stratum，claim 继续为 `non-identifiable`；Best-Paper audit 的 synthetic
counterexample 和工具 schema 通过，但独立真实收益 gate 仍为
`blocked_missing_independent_gain_schema`。对既有 JEPA PointMaze record 的独立 planning
gain 重跑得到 baseline loss `0.35498`、diagnostic loss `0.25918`、mean gain
`0.09581`（10 个 paired outcomes）；这是既有 PointMaze 条件性结果的复核，不是
GDD-WM 蒸馏结果。

使用项目 `.venv` 的相关回归测试为 `14 passed`。全量 pytest 已进入正确依赖环境，
在 46 passed、2 skipped 后卡在外部 DINOv2 TorchHub/socket 下载测试，超时停止；这属于
外部模型下载阻断，不是本轮 GDD-WM 或 D0 代码失败。

2026-10-14 完成 PointMaze learned-transfer 负结果的协议级根因审计。旧 external
`+a/-a/zero` evaluator 只填充四个 action block 的第一个、只模拟 5 步，和 adapter
的四-block 输入及 h={1,2,4} 输出不一致；已新增 v2 evaluator，改为四个 block 全部
填充、对角 action fan、在 5/10/20 步记录标签，并保留 v1 artifact。v2 panel 为
50/50 reports、0 failures；strict benchmark 有 2/5 可识别 strata，但
`Delta R=0.0046`、95% CI `[0,0.0092]`、单环境，positive gate 仍为 false。

进一步新增独立 intervention-supervised task-effect adapter
`le-wm-main/experiments/empirical_spectral_audit/train_pointmaze_intervention_effect_adapter.py`。
在 root-disjoint fresh windows 上，frozen latent 的 action-inverse risk 均值为
`1.496`；加入 `z_t-z_(t-5)` 后降至 `1.282`；同样网络使用 privileged simulator
state upper bound 时为 `0.257`。这将主根因收敛为 partial observability / latent
history insufficiency，而不是继续调整 selector 或匹配阈值。完整数字、artifact 和
claim boundary 见
[`docs/POINTMAZE_NEGATIVE_RESULT_ROOT_CAUSE_20261014.md`](docs/POINTMAZE_NEGATIVE_RESULT_ROOT_CAUSE_20261014.md)。
当前结论仍不支持 universal world model、通用 commutator certificate 或跨环境
geometry selector；全量回归为 `617 passed, 3 skipped`。
