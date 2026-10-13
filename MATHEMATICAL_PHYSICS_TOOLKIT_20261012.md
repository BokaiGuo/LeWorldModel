# Action-lifted 数学与数学物理工具包

本文件把投影/记忆、历史 lifting、控制理论、Koopman/EDMD、混合系统、Lie 群、能量
方法、谱几何和鲁棒可达性正式并入 LeWorldModel 统一研究。它们服务于同一个问题：
哪些 latent 方向既能被观测、又能被动作激活，并且能够通过外部 task metric 产生
可重复、有限时域且物理一致的 transfer。

## 1. 统一对象

对 frozen visual latent 定义：

\[
\eta_t=\Phi(z_{t-L:t},a_{t-L:t-1}),
\qquad
\eta_{t+1}=A_\Phi\eta_t+B_\Phi a_t+\varepsilon_t,
\qquad
y_t=D_\Phi\eta_t.
\]

这里 \(\Phi\) 可以是 single-frame、delay/Hankel、action-history、interaction、
Fourier-like temporal 或 mode-conditioned lifting；\(A_\Phi\) 是 lifted dynamics，
\(B_\Phi\) 是 action-effect operator，\(D_\Phi\) 是 task-space readout。

真正的有限时域 action-to-task transfer 是：

\[
G_h^\Phi=D_\Phi A_\Phi^{h-1}B_\Phi,
\]

而不是单独的 \(\operatorname{spec}(A_\Phi)\)。

对 frozen latent 作为完整物理状态的投影，进一步定义统一数学物理对象：

\[
\mathcal W_\Phi=(\Pi,K_\Phi,A_\Phi,B_\Phi,D_\Phi,M_{task},\Gamma_H).
\]

其中 \(\Pi\) 是物理状态到 latent 的投影，\(K_\Phi\) 是 unresolved state 产生的
memory kernel，\(\Gamma_H\) 是有限时域 transient amplification。目标关系是：

\[
\operatorname{spec}(A_\Phi)\not\Rightarrow
\mathcal H_\Phi(z)
\not\Rightarrow
\text{planning success}.
\]

## 2. 工具与它们解决的具体问题

### 2.1 Delay/Hankel/Koopman lifting：hidden state observability

\[
\eta_t=[z_t,z_{t-1},\ldots,z_{t-L}].
\]

用于检验速度、接触持续时间和 mode transition 是否只是单帧 aliasing。必须同时
报告稳定嵌入 distortion、mode/velocity probe 和 held-out action recovery。升维不会
凭空创造摩擦或接触力信息。

### 2.2 Controlled Koopman、EDMD with control 与 bilinear lifting

接触模式需要至少允许：

\[
\eta_{t+1}=A_\Phi\eta_t+B_\Phi a_t+
\sum_j a_{t,j}B_{\Phi,j}\eta_t+\epsilon_t.
\]

这一区分 direct action effect 与 state-action interaction，直接针对当前 mode 内
torque sign 的不稳定。`run_action_lifted_system_audit.py` 提供局部线性/双线性拟合的
CPU 基础接口；它不是全局 causal model。

### 2.3 Mori--Zwanzig memory closure

若 \(z_t=\Pi(x_t)\) 是完整状态的投影，reduced process 通常不是 Markov：

\[
z_{t+1}=Az_t+Ba_t+\sum_{\tau=1}^{L}K_\tau z_{t-\tau+1}+\xi_t.
\]

因此必须比较 `K=0` 的 Markov adapter、finite-memory kernel、low-rank kernel 和
mode-conditioned kernel。即使 \(A\) 的 spectrum 固定，\(K_\tau\) 的变化也能改变
finite-horizon transfer。新增实现提供 memory-augmented companion operator 和对应
transfer/resolvent 计算。

### 2.4 Observability/controllability Gramians 与 PBH

\[
W_o(H)=\sum_{h=0}^{H-1}(A_\Phi^h)^\top D_\Phi^\top M_yD_\Phi A_\Phi^h,
\]

\[
W_c(H)=\sum_{h=0}^{H-1}A_\Phi^hB_\Phi M_aB_\Phi^\top(A_\Phi^h)^\top.
\]

真正可规划的方向必须属于 observable 与 controllable 的交集。PBH rank test：

\[
\operatorname{rank}[\lambda I-A_\Phi\;B_\Phi]=d,
\qquad
\operatorname{rank}\begin{bmatrix}\lambda I-A_\Phi\\D_\Phi\end{bmatrix}=d.
\]

它把姿态失败分解为不可控、不可观测、二者都满足但被 transient/metric/planner 破坏。

### 2.5 Persistent excitation 与 inverse risk

令 \(\psi_t=[\eta_t^\top,a_t^\top]^\top\)，检查：

\[
\lambda_{\min}\left(N^{-1}\sum_t\psi_t\psi_t^\top\right)\geq\alpha.
\]

不满足时，不允许把 Ridge/KNN 的 torque sign 当成已辨识的 \(B_\Phi\)。同时报告
real/predicted 两侧 inverse risk 及 ATM cross-domain transfer。

### 2.6 \(SE(2)\) Lie-group pose geometry

PushT object pose 属于 \(SE(2)\)，不应直接把 wrapped angle 当作普通欧氏坐标。对
相邻 pose 使用：

\[
\xi_t=\log(g_t^{-1}g_{t+1})\in\mathfrak{se}(2),
\]

得到 body-frame twist，再用 \(\operatorname{Ad}_{g_t}\) 转到 world frame。这样能分离
坐标表示错误与真实 mode-conditioned action effect。若 twist ranking 仍失败，问题
更可能来自 contact mode、hidden state 或 action excitation，而不是 angle wrap。

### 2.7 Task pullback metric 与 generalized spectrum

\[
G_\Phi(\eta)=J_{D_\Phi}(\eta)^\top M_{task}J_{D_\Phi}(\eta),
\qquad
\langle u,v\rangle_{G_\Phi}=u^\top G_\Phi v.
\]

它把 latent 几何连接到 block XY、angle、contact 等外部任务变量，避免 covariance-only
whitening 的 trivial alignment。所有 alignment/commutator 都必须记录 metric 来源。

### 2.8 Hybrid mode 与互补约束

接触任务用离散模式和连续状态联合表示：

\[
m_{t+1}=G(m_t,\eta_t,a_t),\qquad
\eta_{t+1}=A_{m_t}\eta_t+B_{m_t}a_t+\epsilon_t.
\]

接触的 privileged audit 可检查 \(0\leq\lambda_n\perp\phi(q)\geq0\) 和摩擦锥
约束；RGB-only 运行时只能学习 successor-mode probability，不能假装观测到法向力。
因此谱、alignment 和 transient 都应按 mode 条件化。

### 2.9 Schur、resolvent、Kreiss 与有限时域放大

\[
A_\Phi=Q(\Lambda+N)Q^*,
\qquad
\Gamma_H=\max_{1\leq h\leq H}\|A_\Phi^h\|_{G_\Phi}.
\]

同时报告 Schur off-diagonal、resolvent norm 和有限时域 gain；谱和 commutator 都不能
单独决定 transient amplification。

### 2.10 Closure/eigenpair residual

\[
r_{closure}=\|\eta_{t+1}-\Pi_{\mathcal H_\Phi}(A_\Phi\eta_t+B_\Phi a_t)\|,
\qquad
r_{eig}=\frac{\|A_\Phi v-\lambda v\|}{\|v\|}.
\]

它们防止把 delay 冗余、高方差噪声和 spurious eigenmode 当成动力学结构。

### 2.11 Counterfactual transfer 与鲁棒 reachability

同一 source state 的 matched actions \(a,a'\) 要求：

\[
\mathcal E_{cf}=\left\|
(\hat y^a-\hat y^{a'})-(y^a-y^{a'})
\right\|_{M_{task}}^2.
\]

若 \(\|\epsilon_t\|\leq\delta_t\)，可用 transient gain 传播：

\[
\|e_H\|\leq\sum_{j=0}^{H-1}\Gamma_{H-1-j}\delta_j.
\]

这为 candidate rejection、contact-safe tube 和 conformal fallback 提供接口，但
ensemble dispersion 不能自动称为 calibrated uncertainty。

### 2.12 能量、Lyapunov 与耗散审计

在 privileged simulator state 上定义能量或 task-relevant Lyapunov scalar \(V\)，检验：

\[
V_{t+1}\leq\rho V_t+\gamma\|a_t\|^2,\qquad \rho<1,
\]

或者离散耗散残差是否非负。该工具用于发现 rollout 爆炸、非物理 pulse 和 recovery
失稳，不把没有力/速度观测的 RGB latent 直接称为 port-Hamiltonian system。

### 2.13 多尺度 transfer consistency

对不同采样间隔 \(\Delta\) 分别估计 \(A_\Delta,B_\Delta,D_\Delta\)，比较：

\[
\mathcal T_h^{(\Delta)}=D_\Delta A_\Delta^{h-1}B_\Delta.
\]

只在单一采样率出现的 mode 可能是 aliasing 或 spurious eigenmode；跨尺度保持的
action-to-task transfer 才具有更强的物理稳定性证据。

## 3. 研究中的门控逻辑

1. `observability gate`：lifting 是否改善 velocity/mode/readout；
2. `memory gate`：finite-memory closure 是否显著优于 Markov closure；
3. `excitation gate`：当前数据是否足以识别 \(B_\Phi\)；
4. `mode/geometry gate`：mode-conditioned alignment、projector、Schur/transient；
5. `action-transfer gate`：real/predicted inverse risk 与 counterfactual transfer；
6. `energy/reachability gate`：耗散/稳定性与 robust task tube；
7. `planning gate`：独立 held-out multi-step outcome 和 paired closed-loop result。

任何前一门失败，都禁止用后一门的正向代理指标升级结论。

## 4. 当前实现与边界

新增 CPU primitives：

- `le-wm-main/experiments/empirical_spectral_audit/run_action_lifted_system_audit.py`；
- `le-wm-main/tests/test_action_lifted_system_audit.py`。

新增的投影—记忆—Lie—混合系统 primitive：

- `le-wm-main/experiments/empirical_spectral_audit/run_projected_hybrid_physics_audit.py`；
- `le-wm-main/tests/test_projected_hybrid_physics_audit.py`。

前者支持 local `A/B/D` fit、可选 bilinear interaction、PE、Gramians、PBH、Schur
transient、transfer operator 和 matched counterfactual contrast；后者支持 finite-memory
kernel、memory-augmented transfer/resolvent、\(SE(2)\) body twist、mode-conditioned
operator、passivity/Lyapunov residual 和 multiscale transfer consistency。它们只支持
诊断，不代表已完成真实 LeWM lifting、物理因果辨识或在线规划。

## 5. 下一步实验优先级

1. 在 synthetic controlled system 验证已知 controllability/observability 和 memory 边界；
2. 对 frozen LeWM 生成 root-disjoint `Phi0/PhiDelay/PhiAct/PhiInt/PhiMode` export；
3. 把 `B_Phi` 的 PE/PBH/Gramian、memory 和 `SE(2)` twist 诊断接入 PushT contact states；
4. 用 privileged state 做 mode/complementarity/energy audit，严格区分 RGB-only 与上界；
5. 只对 excitation 和 action-transfer gate 通过的 strata 运行 mode-conditioned adapter；
6. 最后再检验 geometry 是否对 \(G_h^\Phi\)、ATM 和 held-out planning risk 有增量解释力。

所有旧负结果保留；该工具不能把 action-correlated signal 改写为因果动力学，也不能
把 privileged simulator-state planning 改写成 frozen visual planning。

## 6. 新增：信号学与信息论的可证伪接口

### 6.1 Causal delay 与局部时频 lifting

对离散轨迹只使用当前及过去窗口：

\[
\Phi_{\mathrm{delay}}(t)=[z_t,z_{t-1},\ldots,z_{t-L+1}],
\qquad
\Phi_{\mathrm{tf}}(t)=\operatorname{rFFT}(z_{t-L+1:t}).
\]

局部频域特征只用于区分慢变、接触持续和冲击成分，不能把高频能量直接解释为接触力。窗口必须 episode 内且不访问未来帧；必须与参数量匹配的 random/PCA controls 比较。

### 6.2 Action-to-task sufficiency

升维的可验证版本是：

\[
\Delta y\perp x\mid(\Phi(x),a).
\]

有限数据中先报告 Gaussian plug-in 条件互信息

\[
I(\Delta y;\Phi_{\mathrm{extra}}\mid\Phi_{\mathrm{base}},a),
\]

再报告 cross-fitted held-out action-effect RMSE、candidate ranking 和 multi-step transfer。互信息只是 dependence diagnostic；若没有独立 transfer 改善，只能称为 information gain，不能称为 sufficient representation 或 causal dynamics。

### 6.3 Fit-only POD 与维数可比性

不同 lifting 的维数不同，不能直接比较 eigenvalues。真实高维 export 的计算审计允许使用 fit-only POD：basis 仅由 fit episodes 估计，outcome 只投影不重估，并同时记录 retained rank、explained variance 和 reconstruction error。rank-capped operator 的结果必须带有 rank 条件，不能外推为完整 latent 空间定理。

对应实现为：

- `run_lifted_signal_information_audit.py`：causal delay、局部 rFFT、conditional-MI 和 held-out transfer sufficiency；
- `run_real_enriched_hybrid_audit.py`：canonical export、episode-disjoint split、fit-only POD 及各 gate；
- `results/projected_hybrid_physics_audit_20261012/`：当前真实 export artifact。

