# LeWorldModel 研究方案演进：从 latent alignment 到可识别的 transfer-risk diagnostic

版本：2026-10-12  
依据：完整文献重审、现有理论结果、matched learned audit、LeWM dynamics/adapter/PushT 结果。

## 0. 方案重置

### 放弃的主叙事

不再把研究定义为：

> latent alignment 能解释并解决 world-model planning 失败。

近期 ATM、ACID、AD-WM、D-JEPA、TRM、criterion-aligned auxiliary loss、contact-mode
planning 和 physically grounded JEPA 已分别覆盖 prediction-to-planning mismatch、
action identifiability、candidate ranking、precondition 和 contact mode 的主要机制。

也不再追求没有证据基础的以下结论：普适 world model、通用 commutator certificate、
frozen RGB latent 的可靠物理因果性，或仅凭一个 local adapter 稳定完成 PushT 姿态闭合。

新的升维视角也不把“维度增加”本身当作贡献。只有当 lifting 同时改善历史可观测性、
action identifiability 和外部 task-space transfer，才算有效 intervention。

### 新的主叙事

本文研究：

> 当 predictor 的 spectrum、prediction MSE 和 latent covariance 已知时，哪些
> covariance–predictor–metric 几何量仍不可识别；这些量在加入 action-transition
> transfer、non-normal transient 和 planner-reachable risk 后，何时能提供超出
> prediction quality 的条件性信息。

核心对象为：

\[
(\Sigma, P, M, B_u, D_{\mathrm{real}\leftrightarrow\mathrm{pred}}),
\]

其中 \(\Sigma\) 是 latent covariance，\(P\) 是 predictor 或局部 Jacobian，\(M\)
是预先声明的外部 task/action metric，\(B_u\) 是 action-effect operator，
\(D_{\mathrm{real}\leftrightarrow\mathrm{pred}}\) 是真实/预测 transition 的跨域一致性。

最终研究对象不是单个 alignment score，而是一个可证伪的 diagnostic tuple：

\[
\mathcal{D}=
(A_M,\Theta_M,\Pi_{\mathrm{Schur}},G_H,
R_{\mathrm{inv}},R_{\mathrm{reach}},R_{\mathrm{elite}}).
\]

它分别对应 metric-relative alignment、principal-angle/projector overlap、Schur
non-normality、finite-horizon transient gain、inverse action risk、reachability
risk 和 CEM elite regret。

对接触丰富和部分可观测任务，实际分析对象进一步升级为 action-history lifted tuple：

\[
\eta_t=\Phi(z_{t-L:t},a_{t-L:t-1}),\qquad
\eta_{t+1}=A_\Phi\eta_t+B_\Phi a_t+\epsilon_t,\qquad
y_t=D_\Phi\eta_t.
\]

相应的规划相关有限时域 transfer operator 是：

\[
G_h^\Phi=D_\Phi A_\Phi^{h-1}B_\Phi.
\]

因此，单独分析 \(\operatorname{spec}(P_z)\) 不再是主对象；主对象是 action-tangent
spectral transfer，即哪些 lifted spectral modes 能被 action 激活并通过 task readout
形成可重复的外部结果。

## 1. 研究问题与可证伪假设

### RQ1：谱外几何是否在 matched quality 下提供 transfer 信息？

在 latent dimension、training stage、action distribution、one-step MSE 和 predictor
spectrum 匹配后，测试：

\[
H_1:
\mathcal{D}_{\mathrm{geometry}}
\text{ 能预测 held-out multi-step transfer，且优于 spectrum-only tie-break。}
\]

必须使用相同 candidate bank、相同 starts、root-disjoint test 和独立 planning outcome。
若没有可验证的 matched strata，结论标记为 `non-identifiable`，不进行因果解释。

### RQ2：action-transition transfer 是否解释 geometry 的剩余价值？

比较 covariance-relative geometry 与 ATM/ACID/AD-WM 风格 action diagnostics：

\[
H_2:
\text{在控制 MSE 后，}\mathcal{D}\text{ 的增量价值来自对真实/预测 action structure
的解释，而非 latent distance 的重新命名。}
\]

直接报告 nested regression、cross-fitted rank correlation、pairwise regret 和
leave-one-environment-out，而不是只报告 pooled correlation。

### RQ3：非正规性是否是独立的 finite-horizon 风险源？

在固定 spectrum、alignment 和 one-step MSE 后，只改变 Schur off-diagonal、resolvent
norm 或 transient gain，检验其对 rollout error、rank inversion 和 elite regret 的影响。

\[
H_3:
\text{non-normal transient 能解释 spectrum/alignment 无法解释的 horizon-dependent risk。}
\]

若只改变 commutator 而未隔离 transient amplification，不能声称非正规机制被验证。

### RQ4：接触丰富任务中的负结果是否是可诊断边界，而非 adapter 调参问题？

PushT 只承担机制边界验证：

\[
\text{pose/motion observability}
\rightarrow
\text{mode identifiability}
\rightarrow
\text{action consistency}
\rightarrow
\text{pose closure}.
\]

若 action consistency 和 mode-conditioned effect 在 root-disjoint contact states 上不
成立，则停止扩大 adapter；把“mode 内 torque sign 不可由当前 frozen RGB 接口识别”
作为正式 boundary result。

### RQ5：历史—动作 lifting 是否真正解除 latent aliasing？

比较 \(\Phi_0(z_t)=z_t\)、delay lifting、history-plus-action lifting、interaction
lifting 和 mode-conditioned lifting。该问题拆成两个不等价的 gate：

\[
\text{state/mode observability}
\neq
\text{action-effect identification}
\neq
\text{planning success}.
\]

只有当 lifting 先提高 held-out velocity/mode/action recovery，再提高
\(G_h^\Phi\) 的 real-to-predicted transfer，才允许继续检验 planning；仅 latent
separability 变好而 action-to-task transfer 不变，判为 representation-only gain。

## 2. 论文级证据架构

### 主贡献 A：gauge-aware spectral identifiability

保留并强化：

- full-\(GL(d)\) 下 Euclidean commutator 不协变；
- 声明 metric 后定义协变 alignment/commutator；
- covariance-only whitening 的 identity/no-go；
- principal angles/projector overlap 处理重复特征值与子空间；
- Schur geometry、resolvent/pseudospectrum 和 transient amplification 的分离；
- nonlinear predictor 只通过 local Jacobian + remainder bound 分析。

该贡献回答“什么量在什么 gauge/metric 下可识别”，不直接声称 planning gain。

### 主贡献 B：matched learned diagnostic benchmark

统一比较：

- validation/held-out MSE；
- variance、spectrum-only；
- covariance alignment、principal angle、projector overlap；
- commutator；
- Schur off-diagonal、resolvent、finite-horizon transient gain；
- inverse action risk、ATM cross-domain transfer；
- decision-metric alignment、reachability score、elite regret。

主 estimand 是在固定 quality 条件下对独立 held-out transfer/planning risk 的增量解释力。

### 主贡献 C：边界机制，而不是伪装成成功 planner

learned LeWM/PushT 结果用于证明以下链条的边界：

\[
\text{action-correlated signal}
\not\Rightarrow
\text{cross-domain intervention fidelity}
\not\Rightarrow
\text{stable contact pose closure}.
\]

这使正结果、负结果和旧 adapter 消融都服务于同一科学问题。

### 主贡献 D：action-history lifting 的适用条件

把 delay/Hankel、Fourier-like temporal features、interaction features 和 learned
history encoder 作为可比较的 observable lifting family，明确检验 stable embedding、
action excitation、task metric specification 和 finite-horizon stability 四个条件。
该贡献不是“升维总能解决规划”，而是识别何时升维只能暴露状态、何时还能形成可用的
action-to-task transfer。

## 3. 实验阶段与停止门

### Gate 0：协议冻结与 provenance

冻结 checkpoint、seed、architecture、stage、latent dimension、action distribution、
horizon、metric、candidate bank、train/calibration/test split 和 planner budget。所有
结果写入统一 manifest，禁止测试集选 metric、权重或 snapshot。

### Gate 1：可观测性与动作辨识

对每个环境报告：pose、velocity、relative motion、mode probe；real transition 与
predicted transition 的 inverse action risk；ATM 四格；action-shuffled、zero-effect、
same-state shared-candidate controls。Gate 1 未通过时，任何 covariance geometry 对
planning 的解释都只能是相关性，不能是 causal/action-effect 结论。

在本 gate 内先比较 lifting family：raw single-frame、latent delay、action-history、
interaction cross-feature 和 mode-conditioned history。必须分别报告 pose、velocity、
mode、action recovery；不允许以 pose probe 的改善代替 velocity 或 action-effect 的改善。

### Gate 2：matched learned strata

最低要求：两个 architecture、两个 seed、两个环境；优先四环境（PushT、TwoRoom、
PointMaze、OGBench-Cube/Reacher）。每个 matched pool 需同时满足 MSE/spectrum tolerance、
latent dimension 和 training stage 固定、action distribution 和 candidate bank 固定，
且 geometry 变化不是通过 test outcome 反向构造。

如果无法得到足够的真实 learned strata，就把结论限定为 synthetic/semisynthetic，
不得写成 learned model selection criterion。

### Gate 3：跨环境与外部 metric

使用 action-effect/task-space metric，不能使用 covariance-only whitening 作为唯一 metric。
进行 leave-one-environment-out：训练环境估计 metric/模型，未见环境只做一次固定评估。
主指标为 cross-fitted planning-risk rank correlation、pairwise regret、calibration 和
worst-environment performance。

### Gate 4：PushT contact boundary

只在 Gate 1--3 的 action diagnostics 有证据时，运行 mode-conditioned adapter：history
encoder、successor-mode head、每个 mode 独立 effect/inverse head、calibrated support/risk
gate、short pulse + re-observation + recovery。若 contact retention 改善但 angle 没有跨
seed paired improvement，立即停止调权重，保留为“mode gate 与 pose closure 分离”的负结果。

### Gate 5：冻结视觉规划

必须使用 RGB-only runtime、独立 test episodes、zero/legacy/receding paired control，
并报告 action consistency、block XY、wrapped angle、strict success、recovery rate。
privileged simulator-state planning 只作为上界/接口诊断，不得合并到 frozen visual 结果。

## 4. 具体执行包

### Package A：统一 diagnostic runner

把现有 `run_matched_learned_transfer_benchmark.py` 扩展为单一 manifest 驱动的 runner，
每个 checkpoint/candidate 输出 quality（train/val/heldout MSE、variance、spectrum）、
geometry（alignment、principal angle、projector overlap、commutator）、non-normal
（Schur energy、resolvent、transient gain[h]）、decision（inverse risk、ATM、Plan-Real
rank、reachability、elite regret）和 outcome（multi-step transfer、planning risk、strict
success）。禁止使用一个 composite score 直接替代这些字段。

同时输出 lifting 元数据：`lift_family`、`history_length`、`action_history_length`、
`feature_dim`、`embedding_distortion`、`action_excitation_margin`、`A_phi/B_phi/D_phi`
估计方式和 `G_h_phi` 的 multi-horizon 误差。所有 lift 在相同 episode split 和 candidate
bank 下比较。

### Package B：竞争方法作为 diagnostic controls

实现最小、可复现的 inverse action recovery/cycle consistency、shared-candidate temporal
ranking、CEM-stage rank agreement、reachability temporal head、action-discriminative
residual probe。它们是竞争性 controls，不是把论文改写成 AD-WM 或 D-JEPA。

### Package C：matched intervention

优先采用可审计的线性/低秩 operator intervention，固定 eigenvalues 和 MSE，改变
covariance-relative orientation、Schur coupling 或 external metric。若真实 neural
checkpoint 无法严格保持这些条件，报告为 controlled identification，不外推到 learned
universal selection。

### Package D：PushT boundary adapter

只实现一个主版本：history + mode-conditioned effect + inverse action consistency +
support gate。旧 global/Ridge/KNN/angle-only/task-rank/hybrid 版本全部作为冻结负消融，
不再并行调参制造“最佳版本”。

其输入固定为 history/action lifting，输出同时包含 successor mode、mode-conditioned
effect、inverse action consistency 和 support risk；禁止把单帧 pose decoder 直接当作
该 adapter 的替代品。

## 5.1 Action-history lifting 实验矩阵

| lifting | 目的 | 主要否证 |
|---|---|---|
| `Phi0`: current latent | 单帧基线 | velocity/mode/action recovery 不高于 chance |
| `PhiDelay`: `[z_t,...,z_{t-L}]` | 恢复历史可观测性 | mode 提升但 action transfer 不提升 |
| `PhiAct`: delay + action history | 对齐施加过的 action | inverse risk/ATM 不改善 |
| `PhiInt`: delay + `z\otimes a`/finite differences | 分离 action tangent | one-step 好但 shared-candidate rank 不改善 |
| `PhiMode`: history + successor-mode gating | 分离 hybrid regimes | retention 提升但 angle closure 无 paired gain |
| `PhiFourier`: temporal frequency features | 检验信号处理类 lifting | 只提高 latent MSE/谱可读性而不提高 task transfer |

所有 lifting 使用同一 frozen LeWM encoder、同一 train/calibration/test roots、同一
candidate bank 和同一 external task metric。`PhiFourier` 是诊断对照，不预设其适合
非平稳接触事件。

核心报告对象：

\[
\mathcal I_{\mathrm{act}}
=I(A_t;\Delta\eta_t\mid\eta_t),\qquad
G_h^\Phi=D_\Phi A_\Phi^{h-1}B_\Phi,
\]

并同时报告 action inverse risk、ATM、projector/alignment、Schur/transient、decoded
XY/angle/contact 和 strict success。

## 5. 统计与报告规则

- 每个 seed 单独报告，再给 paired aggregate；
- 主要相关性用 cross-fitting，禁止 train/test metric 混算；
- 多指标比较控制 multiplicity，优先报告 bootstrap/paired CI；
- planning risk 使用共享 candidate/start，避免 action availability confound；
- success、XY、angle、contact retention 必须同时满足主成功判据；
- 只在 matched-quality pool 中解释 geometry 的增量价值；
- 负结果保留原始 protocol、代码、artifact 和 audit record；
- 代码测试通过只说明实现可靠，不升级科学假设。

## 6. 结果决策树

### 情形 A：matched learned + cross-environment 成功

主结论可升级为：

> covariance–predictor geometry is a conditional, metric- and horizon-specific
> transfer-risk diagnostic after predictive quality and action consistency are matched.

仍不能称为 universal certificate。

### 情形 B：geometry 只有在加入 ATM/decision metric 后有效

主结论改为：

> spectral geometry is an explanatory refinement of action-transition transfer,
> not a standalone planning signal.

这是目前最可能、也最有科学价值的成功形态。

### 情形 C：matched learned 仍失败

论文转为严格 boundary paper：

> spectrum, covariance alignment and local action-correlated structure are jointly
> insufficient to identify planner-relevant transfer under off-manifold and partially
> observed dynamics.

此时不再投入更多 PushT adapter；把失败作为对“谱摘要可作模型选择”的否证。

## 7. 最终论文定位

建议标题方向：

> **When Spectra Are Not Enough: A Gauge-Aware Covariance–Predictor Diagnostic for Latent Transfer**

主摘要/引言应该承诺：一个有 gauge 与 metric 边界的诊断框架、一个固定 spectrum 的
可控识别实验、一个跨 learned model 的 matched-quality 审计，以及一个说明
action transfer、non-normality 和 contact partial observability 如何限制外推的边界实验。

不要承诺普适 world model、通用 commutator certificate、frozen LeWM 已解决视觉规划，
或 learned adapter 已稳定闭合姿态。

