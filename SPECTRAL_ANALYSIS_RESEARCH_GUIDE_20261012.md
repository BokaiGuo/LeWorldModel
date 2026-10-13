# 谱分析主线：协方差—预测器传输诊断的具体解释与推导

> 本文是对 [SPECTRAL_ANALYSIS_RESEARCH_RECORD_20261012.md](SPECTRAL_ANALYSIS_RESEARCH_RECORD_20261012.md) 的解释版整理。原始记录保持不变；本文对其核心论证进行结构化重述、补充推导，并明确每个结论的适用范围。

## 1. 研究对象与一句话结论

本研究真正研究的对象不是一个孤立的特征值、交换子或 alignment 指标，而是一对带有外部度量的对象：

\[
(\Sigma,P;M).
\]

其中：

- \(\Sigma\) 描述 latent 在数据分布下的协方差或二阶几何；
- \(P\) 描述 predictor、局部线性化 dynamics，或某一时间步的有效传播算子；
- \(M\) 是预先声明的外部度量，例如 decoder pullback metric、action-effect metric 或 task-space metric。

核心可证伪命题是：

\[
\operatorname{spec}(P)
\;\not\Rightarrow\;
\mathcal T_h(\Sigma,P,M),
\]

即 predictor 的特征值谱一般不能唯一确定 covariance-relative、metric-relative 的有限时域传输行为。要判断一个 latent dynamics 是否适合迁移或规划，至少要同时考虑：

\[
\text{eigenvalues}
+\text{eigenspaces}
+\text{covariance/task metric}
+\text{non-normal transient}
+\text{action-to-task transfer}.
\]

这是一条诊断命题，不是“任何 alignment 都能预测规划”的普遍定理。

## 2. 统一的输入—状态—输出对象

为了把谱分析、有限记忆、action effect 和规划放进同一个数学对象，定义 lifted state

\[
\eta_t=\Phi(z_{t-L+1:t},a_{t-L+1:t-1}),
\]

其中 \(z_t\) 是 frozen latent，\(a_t\) 是动作，\(L\) 是声明的历史长度，\(\Phi\) 可以是：

- identity/current-state representation；
- delay 或 Hankel lifting；
- action-history lifting；
- Fourier/time-frequency lifting；
- mode-conditioned lifting；
- learned history encoder。

局部线性输入—输出模型写为

\[
\eta_{t+1}=A_\Phi\eta_t+B_\Phi a_t+\varepsilon_t,
\qquad
y_t=D_\Phi\eta_t+\nu_t.
\]

这里：

- \(A_\Phi\) 是 lifted latent 的传播算子；
- \(B_\Phi\) 是动作进入 lifted state 的方向；
- \(D_\Phi\) 是从 lifted state 到 task-space readout 的映射；
- \(\varepsilon_t\) 是 model residual；
- \(\nu_t\) 是 readout residual。

一次动作扰动 \(\delta a_t\) 在 \(h\) 步后的 task-space 一阶影响为

\[
\delta y_{t+h}
=
D_\Phi A_\Phi^{h-1}B_\Phi\,\delta a_t.
\]

因此，真正与规划相关的 finite-horizon transfer operator 是

\[
\boxed{
G_h^\Phi=D_\Phi A_\Phi^{h-1}B_\Phi
}
\tag{1}
\]

而不是单独的 \(\operatorname{spec}(A_\Phi)\)。

### 2.1 公式（1）的逐步推导

由状态方程：

\[
\delta\eta_{t+1}=B_\Phi\delta a_t.
\]

再传播一步：

\[
\delta\eta_{t+2}
=A_\Phi\delta\eta_{t+1}
=A_\Phi B_\Phi\delta a_t.
\]

继续递推：

\[
\delta\eta_{t+h}
=A_\Phi^{h-1}B_\Phi\delta a_t.
\]

由输出方程：

\[
\delta y_{t+h}
=D_\Phi\delta\eta_{t+h}
=D_\Phi A_\Phi^{h-1}B_\Phi\delta a_t.
\]

所以式（1）是由局部状态传播、动作注入和任务读出共同决定的。它直接说明三个事实：

1. 相同的 \(A_\Phi\) 谱不保证相同的 \(B_\Phi\)；
2. 相同的 \(A_\Phi,B_\Phi\) 不保证相同的 \(D_\Phi\)；
3. 相同的一步作用不保证相同的多步作用，因为 \(A_\Phi^{h-1}\) 会累积 eigenspace alignment 与 non-normal transient。

## 3. 同谱不同几何：二维精确推导

### 3.1 构造同谱 predictor family

令

\[
\Lambda=
\begin{bmatrix}
\lambda_1&0\\
0&\lambda_2
\end{bmatrix},
\qquad
R_\theta=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix},
\]

并定义

\[
P_\theta=R_\theta\Lambda R_\theta^\top.
\]

因为 \(R_\theta\) 是正交矩阵，

\[
P_\theta^k
=
R_\theta\Lambda^kR_\theta^\top,
\]

且对所有 \(\theta\) 都有

\[
\operatorname{spec}(P_\theta)=\{\lambda_1,\lambda_2\}.
\]

因此，改变 \(\theta\) 不改变 eigenvalue spectrum，只改变 eigenspace orientation。

### 3.2 与 latent covariance 的相对几何

令

\[
\Sigma=
\begin{bmatrix}
\sigma_1^2&0\\
0&\sigma_2^2
\end{bmatrix}.
\]

将 \(P_\theta\) 展开：

\[
P_\theta=
\begin{bmatrix}
\lambda_1c^2+\lambda_2s^2
&
(\lambda_1-\lambda_2)cs
\\
(\lambda_1-\lambda_2)cs
&
\lambda_1s^2+\lambda_2c^2
\end{bmatrix},
\]

其中 \(c=\cos\theta\)，\(s=\sin\theta\)。

交换子为

\[
[\Sigma,P_\theta]
=
\Sigma P_\theta-P_\theta\Sigma.
\]

由于对角项相互抵消，得到

\[
[\Sigma,P_\theta]
=
\begin{bmatrix}
0&
(\sigma_1^2-\sigma_2^2)(\lambda_1-\lambda_2)cs
\\
-(\sigma_1^2-\sigma_2^2)(\lambda_1-\lambda_2)cs&
0
\end{bmatrix}.
\]

因此

\[
\|[\Sigma,P_\theta]\|_F^2
=
2(\sigma_1^2-\sigma_2^2)^2(\lambda_1-\lambda_2)^2c^2s^2.
\]

使用

\[
2cs=\sin(2\theta),
\]

可得精确关系

\[
\boxed{
\|[\Sigma,P_\theta]\|_F
=
\frac{
|\sigma_1^2-\sigma_2^2|\,
|\lambda_1-\lambda_2|
}{\sqrt{2}}
|\sin(2\theta)|
}
\tag{2}
\]

式（2）揭示：

- 若 covariance isotropic，即 \(\sigma_1^2=\sigma_2^2\)，交换子恒为零；
- 若 predictor 有重复特征值，即 \(\lambda_1=\lambda_2\)，交换子恒为零；
- 只有当 covariance 有方向性、predictor 有谱间隔，orientation mismatch 才可被这个交换子检测；
- \(\sin(2\theta)\) 是二倍角关系，说明 \(\theta\) 与 \(\theta+\pi/2\) 在某些未定向子空间度量下可能不可区分。

这不是新的矩阵分析公理；本研究的贡献在于把它作为 covariance–predictor pair 的可检验诊断量，并测试它是否对 held-out transfer 有增量解释力。

### 3.3 covariance-relative finite-horizon energy

定义由 latent covariance 加权的传播能量：

\[
\mathcal E_h(\theta)
=
\left\|P_\theta^h\Sigma^{1/2}\right\|_F^2
=
\operatorname{tr}(P_\theta^{2h}\Sigma).
\]

因为

\[
P_\theta^{2h}
=
R_\theta
\begin{bmatrix}
\lambda_1^{2h}&0\\
0&\lambda_2^{2h}
\end{bmatrix}
R_\theta^\top,
\]

所以

\[
\mathcal E_h(\theta)
=
\lambda_1^{2h}(\sigma_1^2c^2+\sigma_2^2s^2)
+
\lambda_2^{2h}(\sigma_1^2s^2+\sigma_2^2c^2).
\]

整理得

\[
\boxed{
\mathcal E_h(\theta)
=
\frac{
(\lambda_1^{2h}+\lambda_2^{2h})
(\sigma_1^2+\sigma_2^2)
}{2}
+
\frac{
(\lambda_1^{2h}-\lambda_2^{2h})
(\sigma_1^2-\sigma_2^2)
}{2}
\cos(2\theta)
}
\tag{3}
\]

因此：

- 交换子范数对 orientation mismatch 呈 \(|\sin(2\theta)|\)；
- covariance-weighted transfer energy 对 orientation alignment 呈 \(\cos(2\theta)\)；
- 二者互补，不能把它们视为同一个量；
- 即使所有 \(P_\theta\) 的 spectrum 完全相同，\(\mathcal E_h(\theta)\) 仍可随 \(\theta\) 改变。

式（3）就是“同谱不等于同传输”的最小可复现实例。

## 4. 为什么普通 Euclidean commutator 不是 gauge invariant

### 4.1 一般可逆重参数化

设 latent 坐标发生可逆变换

\[
z'=Sz,\qquad S\in GL(d).
\]

如果 \(P\) 是线性算子，则

\[
P'=SPS^{-1}.
\]

如果 \(\Sigma\) 表示二阶协方差，则它按双线性形式变换为

\[
\Sigma'=S\Sigma S^\top.
\]

现在直接计算普通矩阵交换子：

\[
[\Sigma',P']
=
S\Sigma S^\top SPS^{-1}
-
SPS^{-1}S\Sigma S^\top.
\]

中间出现了 \(S^\top S\)，因此一般不能写成

\[
S[\Sigma,P]S^{-1}.
\]

只有在 \(S^\top S=I\) 的正交变换下，普通 Euclidean commutator 才保持这种形式。由此得到：

\[
\boxed{
[\Sigma,P]\text{ 不是一般 }GL(d)\text{ 重参数化下的坐标不变量。}
}
\]

这不是说交换子没有用，而是说必须同时声明 latent metric 与变换规则。

### 4.2 metric-aware adjoint 与协变交换子

令 \(G\succ0\) 是一个声明的 covariant metric。对任意算子 \(P\)，定义 \(G\)-adjoint：

\[
P^{\dagger_G}=G^{-1}P^\top G.
\]

定义 metric-aware non-normality operator：

\[
\mathcal N_G(P)
=
[P,P^{\dagger_G}]
=
PP^{\dagger_G}-P^{\dagger_G}P.
\]

若坐标变换为 \(z'=Sz\)，令

\[
P'=SPS^{-1},
\qquad
G'=S^{-\top}GS^{-1}.
\]

则

\[
\begin{aligned}
(P')^{\dagger_{G'}}
&=(G')^{-1}(P')^\top G'\\
&=(SG^{-1}S^\top)(S^{-\top}P^\top S^\top)
(S^{-\top}GS^{-1})\\
&=SP^{\dagger_G}S^{-1}.
\end{aligned}
\]

于是

\[
\begin{aligned}
\mathcal N_{G'}(P')
&=[SP S^{-1},SP^{\dagger_G}S^{-1}]\\
&=S[P,P^{\dagger_G}]S^{-1}\\
&=S\mathcal N_G(P)S^{-1}.
\end{aligned}
\]

所以 \(\mathcal N_G(P)\) 是协变对象；它本身的坐标矩阵会变化，但在同一 metric 结构下定义的谱、秩与相应不变量才有可解释性。

### 4.3 covariance-only whitening 的 no-go

若只使用 covariance 做 whitening，令

\[
W=\Sigma^{-1/2},
\qquad
\widetilde z=Wz.
\]

则

\[
\widetilde\Sigma
=
W\Sigma W^\top
=I.
\]

因此 covariance 的 eigenspace 不再提供任何方向偏好。若把“alignment”定义为 whitened covariance 与任何同维对象之间的普通 Euclidean alignment，那么 covariance 端已经退化为 identity：

\[
\operatorname{span}(\widetilde\Sigma)=\mathbb R^d.
\]

它只能给出平凡的 full-space overlap，而不能产生非平凡的方向诊断。这个 no-go 结果说明：

\[
\boxed{
\text{covariance-only whitening 不是一个非退化的 alignment metric。}
}
\]

要获得非平凡诊断，必须声明额外结构，例如：

- decoder pullback metric \(G_{\mathrm{dec}}=J_g^\top J_g\)；
- action-effect metric \(G_{\mathrm{act}}=B^\top W_yB\)；
- task-space metric \(G_{\mathrm{task}}=D^\top W_yD\)；
- 物理量或控制代价诱导的 metric。

## 5. 非正规算子：为什么 spectrum 和 commutator 都不够

### 5.1 Schur 分解

任意复方阵 \(P\) 都可写为

\[
P=QTQ^\ast,
\qquad
T=\Lambda+N,
\]

其中：

- \(Q\) 是酉矩阵；
- \(\Lambda\) 是对角矩阵，包含 eigenvalues；
- \(N\) 是严格上三角矩阵，表示 Schur off-diagonal coupling。

特征值只由 \(\Lambda\) 决定；有限时间行为还受到 \(N\) 的影响。

### 5.2 Jordan 例子

考虑

\[
P=
\begin{bmatrix}
\lambda&\gamma\\
0&\lambda
\end{bmatrix}
=
\lambda I+N,
\qquad
N=
\begin{bmatrix}
0&\gamma\\
0&0
\end{bmatrix},
\qquad N^2=0.
\]

由二项式展开：

\[
P^k=(\lambda I+N)^k
=
\lambda^kI+k\lambda^{k-1}N,
\]

因为含 \(N^2\) 的项全部为零。因此

\[
\boxed{
P^k=
\begin{bmatrix}
\lambda^k&k\gamma\lambda^{k-1}\\
0&\lambda^k
\end{bmatrix}.
}
\tag{4}
\]

即使 \(|\lambda|<1\)，中间时间的 off-diagonal 响应也可能很大。固定 \(\lambda\) 而改变 \(\gamma\)，不会改变 spectrum，却会改变 finite-horizon amplification。

若 covariance metric 取 \(G=I\)，则 covariance commutator

\[
[I,P]=0
\]

对所有 \(P\) 都成立；但式（4）的 transient term \(k\gamma\lambda^{k-1}\) 仍然存在。因此 covariance commutator 为零不能推出 transient amplification 为零。

### 5.3 Resolvent 与 pseudospectrum

对 \(z\neq\lambda\)，

\[
zI-P=
\begin{bmatrix}
z-\lambda&-\gamma\\
0&z-\lambda
\end{bmatrix},
\]

其逆为

\[
(zI-P)^{-1}
=
\begin{bmatrix}
(z-\lambda)^{-1}&
\gamma(z-\lambda)^{-2}\\
0&(z-\lambda)^{-1}
\end{bmatrix}.
\]

因此 resolvent norm 至少受到

\[
\frac{|\gamma|}{|z-\lambda|^2}
\]

控制。相同 eigenvalue 可以产生不同 pseudospectrum。对有限时域传输而言，需要至少共同报告：

\[
\|P^h\|,
\qquad
\|(zI-P)^{-1}\|,
\qquad
\|N\|_F,
\qquad
\text{Schur off-diagonal energy}.
\]

因此本文将 commutator 限定为：

\[
\boxed{
\text{metric-specified, finite-horizon diagnostic component}
}
\]

而不是通用 transfer certificate。

## 6. 非线性模型：local Jacobian 只能给局部结论

设真实 lifted dynamics 为

\[
\eta_{t+1}=F(\eta_t,a_t),
\]

在参考点 \((\eta_\star,a_\star)\) 附近令

\[
\delta\eta_t=\eta_t-\eta_\star,
\qquad
\delta a_t=a_t-a_\star.
\]

对 \(F\) 做 Taylor 展开：

\[
F(\eta_\star+\delta\eta_t,a_\star+\delta a_t)
=
F(\eta_\star,a_\star)
+
A_\star\delta\eta_t
+
B_\star\delta a_t
+
r_t,
\]

其中

\[
A_\star=\frac{\partial F}{\partial\eta}(\eta_\star,a_\star),
\qquad
B_\star=\frac{\partial F}{\partial a}(\eta_\star,a_\star).
\]

若 \(F\) 的 Jacobian 是 Lipschitz 的，且二阶导数上界为 \(L_F\)，则可得到

\[
\|r_t\|
\le
\frac{L_F}{2}
\left(
\|\delta\eta_t\|+\|\delta a_t\|
\right)^2.
\]

因此 local spectral analysis 只能支持：

- 小邻域内的局部传播；
- 给定 horizon 的 remainder-controlled transfer；
- 对局部 action effect 的近似。

它不能直接支持：

- 全局物理因果性；
- 任意远离参考点的 counterfactual；
- 无约束长时域 rollout；
- universal world model。

## 7. 从 latent signal 到物理 action effect：必须经过三道 gate

latent 中出现 action-correlated signal 只说明

\[
I(Z_{t+1};A_t\mid Z_t)>0
\]

或某个回归 probe 能降低误差。它不等于可执行的 physical effect。完整证据链应为：

### Gate 1：action excitation

动作必须在训练数据中有足够独立变化。对 residualized action

\[
\widetilde A_t
=
A_t-\mathbb E[A_t\mid Z_t]
\]

检查条件 Gramian

\[
\Gamma_A
=
\mathbb E[\widetilde A_t\widetilde A_t^\top].
\]

若 \(\lambda_{\min}(\Gamma_A)\) 很小，则不同动作方向不可辨识，回归器可以拟合但不能支持 counterfactual。

### Gate 2：real/predicted transition consistency

对同一 action candidate \(a\)，分别测量：

\[
\Delta y^{\mathrm{real}}(a)
=
y(o_{t+1}^{\mathrm{real}}(a))-y(o_t),
\]

\[
\Delta y^{\mathrm{pred}}(a)
=
\widehat y(\widehat o_{t+1}(a))-y(o_t).
\]

需要检验的不只是两者各自可预测，而是

\[
\operatorname{rank}
\left(
\Delta y^{\mathrm{real}}(a)
\right)
\approx
\operatorname{rank}
\left(
\Delta y^{\mathrm{pred}}(a)
\right)
\]

以及 pairwise ordering agreement。若 predicted domain 中 action 可解码、但 real/predicted ranking 不一致，planner 仍会失败。

### Gate 3：runtime successor verification

对 planner 选出的动作执行后，用下一帧真实观测检验：

\[
\|\widehat y_{t+1}-y_{t+1}^{\mathrm{real}}\|
\le \epsilon_y,
\]

\[
\|\widehat m_{t+1}-m_{t+1}^{\mathrm{real}}\|
\le \epsilon_m,
\]

其中 \(m_t\) 是接触或混合动力学 mode。若 successor verification 不通过，必须停止继续 rollout，转入短脉冲重规划或 contact recovery。

三道 gate 的逻辑是：

\[
\boxed{
\text{excitation}
\rightarrow
\text{transfer consistency}
\rightarrow
\text{runtime verification}
}
\]

它们分别回答“能不能辨识”“预测是否对应真实执行”“在线时是否仍然可信”。

## 8. 为什么 PushT 姿态闭合失败

PushT 不是单一连续 dynamics，而更接近混合系统：

\[
m_t\in
\{
\text{free},
\text{approach},
\text{sticking},
\text{sliding},
\text{lost-contact}
\}.
\]

模式转移写为

\[
m_{t+1}
\sim
p(m_{t+1}\mid m_t,\eta_t,a_t,o_{t+1}),
\]

模式条件 dynamics 写为

\[
\eta_{t+1}
=
F_{m_t}(\eta_t,a_t)+\varepsilon_t.
\]

现有单一 effect regressor 实际上假设

\[
\eta_{t+1}=F(\eta_t,a_t)
\]

能够同时解释 free、contact、sticking、sliding 和 lost-contact。这会产生两类错误：

1. 相同 action 在不同 mode 下效果符号不同；
2. 回归平均值在任何一个 mode 中都不是正确的 effect。

姿态闭合的合理接口应为

\[
\widehat p(m_{t+1}\mid \eta_t,a_t,o_{t+1}),
\]

\[
\widehat{\Delta y}_{t+1}
=
\sum_m
\widehat p(m_{t+1}=m\mid \eta_t,a_t,o_{t+1})
F_m(\eta_t,a_t),
\]

再由风险约束控制：

\[
a_t=
\arg\min_a
\mathbb E[\ell(y_{t+1},y^\star)]
+
\beta\,\Pr(m_{t+1}=\text{lost-contact})
+
\gamma\,\operatorname{Uncertainty}(a).
\]

这说明升维的作用是提高局部分离能力，但升维本身不会自动生成 successor-mode model。

## 9. 统一的实验 estimand

### 9.1 alignment 的条件性增量价值

不能比较“alignment 是否替代 MSE”。正确 estimand 是：

\[
\Delta_{\mathrm{geo}}
=
\mathbb E[
R_{\mathrm{plan}}
\mid
\mathrm{MSE},\mathrm{spec},d,\mathrm{stage},\mathrm{action\ distribution}
]
-
\mathbb E[
R_{\mathrm{plan}}
\mid
\mathrm{MSE},\mathrm{spec},d,\mathrm{stage},\mathrm{action\ distribution},\mathrm{geometry}
].
\]

操作上：

1. 先按 validation MSE 过滤质量相近的 checkpoints；
2. 再在 matched stratum 内比较 alignment、principal angles、Schur energy 与 held-out transfer；
3. 规划 episode 和 geometry fit episode 完全分离；
4. 以 seed/episode cluster 为独立单位；
5. 报告 regret、rank correlation 与 confidence interval。

若 matched stratum 不存在，结论是 non-identifiable，而不是 alignment 失败。

### 9.2 记忆提升的正确比较

有限记忆模型必须在相同 current transitions 上比较：

\[
\mathcal I_L
=
\{t:
t-L+1,\ldots,t
\text{ 都在当前 split 且来自同一 episode}\}.
\]

Markov 与 memory-\(L\) 都在 \(\mathcal I_L\) 上评估：

\[
\Delta_L
=
\operatorname{RMSE}_{\mathrm{Markov}}(\mathcal I_L)
-
\operatorname{RMSE}_{L}(\mathcal I_L).
\]

并且 memory-\(L\) 的输入应包括：

\[
(z_t,\ldots,z_{t-L+1},
a_t,\ldots,a_{t-L+1}),
\]

而不是只包括过去 latent 加当前 action。否则输入驱动系统中的历史 action 被错误地当成可由 latent 完全恢复。

### 9.3 planning risk 的独立评估

对于 candidate set \(\mathcal A_t\)，定义真实和预测 task effect：

\[
e_{\mathrm{real}}(a)
=
y_{t+1}^{\mathrm{real}}(a)-y_t,
\qquad
e_{\mathrm{pred}}(a)
=
\widehat y_{t+1}(a)-y_t.
\]

定义 pairwise rank agreement：

\[
\mathrm{RA}
=
\frac{1}{|\mathcal P|}
\sum_{(a_i,a_j)\in\mathcal P}
\mathbf 1
\left[
\operatorname{sign}
\bigl(q(e_{\mathrm{real}}(a_i))-q(e_{\mathrm{real}}(a_j))\bigr)
=
\operatorname{sign}
\bigl(q(e_{\mathrm{pred}}(a_i))-q(e_{\mathrm{pred}}(a_j))\bigr)
\right].
\]

其中 \(q\) 是预先声明的 task progress score。这个指标直接测量 planner 会不会选对动作，比只报告 latent one-step RMSE 更贴近控制。

## 10. 现有证据应如何解释

原始记录中最强的受控 benchmark 具有：

- 9 个 predictor；
- 完全相同的 predictor eigenvalue spectrum；
- 固定 eigenspace family 与 eigenvalue pairing；
- 训练集计算 alignment；
- 独立测试集评估 held-out transfer；
- 128 次重复。

该实验支持：

\[
\text{在固定 operator family 中，alignment 提供 spectrum 之外的 transfer 信息。}
\]

它不支持：

\[
\text{alignment 普遍选择真实 learned checkpoint，或直接预测 planning success。}
\]

真实 learned checkpoint 的 matched audit 若没有形成足够的 MSE+spectrum matched strata，则正确结论是：

\[
\text{当前协议无法识别 geometry 的独立增量价值。}
\]

这与“geometry 已被证明无用”不同，也与“geometry 已被证明普遍有效”不同。

对 PushT，同样要区分：

- privileged simulator-state interface repair；
- offline learned action-effect ranking；
- runtime successor verification；
- pure learned adapter closure；
- frozen RGB online planning。

它们是不同 estimand，不能用上一级结果填补下一级证据。

## 11. 论文主线的建议表述

建议将论文主张写成：

> We establish a gauge-aware, metric-relative, non-normality-aware covariance–predictor diagnostic framework. In a controlled fixed-spectrum family, the framework identifies transfer differences that eigenvalues alone cannot determine. On learned visual world-model exports, it further exposes the boundary between action-correlated latent structure and executable action-conditioned dynamics: local probes can be predictive while real/predicted transition ranking and contact-mode observability remain insufficient for stable planning.

中文表述为：

> 我们建立了一个具有规范感知、外部度量相对性和非正规性分析的协方差—预测器诊断框架。在固定谱的受控算子族中，该框架识别出仅凭特征值无法确定的传输差异；在 learned visual world-model 导出上，它进一步揭示了“latent 中存在动作相关结构”与“动作条件动力学可执行”之间的边界：局部 probe 可以具有预测性，但真实/预测 transition 排序和接触模式可观测性仍可能不足以支撑稳定规划。

不应使用的强表述包括：

- alignment 普遍预测 learned-world-model 规划性能；
- commutator 是通用 transfer certificate；
- frozen LeWM 已经支持视觉在线规划；
- latent 已经具有可靠物理因果动力学；
- learned adapter 已稳定解决 PushT 姿态闭合；
- 该方法已经建立普适 world model。

## 12. 读者应记住的最终逻辑

整项研究可以压缩为一条链：

\[
\boxed{
\text{gauge specification}
\rightarrow
\text{non-degenerate external metric}
\rightarrow
\text{eigenspace/Schur geometry}
\rightarrow
\text{finite-horizon }G_h^\Phi
\rightarrow
\text{action identifiability}
\rightarrow
\text{real/predicted consistency}
\rightarrow
\text{mode-aware closed-loop validation}
}
\]

每一步都对应一个不能跳过的科学问题：

1. 没有 gauge specification，alignment 与 commutator 可能只是坐标产物；
2. 没有 external metric，covariance-only whitening 会使 alignment 退化；
3. 没有 Schur/transient analysis，spectrum 不能描述 finite-horizon amplification；
4. 没有 \(G_h^\Phi\)，latent geometry 不能连接到 action-to-task transfer；
5. 没有 action excitation，counterfactual effect 不可辨识；
6. 没有 real/predicted transition consistency，offline regression 不能支撑 planning；
7. 没有 mode-aware closed-loop validation，局部 effect ranking 不能支撑接触姿态闭合。

因此，本研究的价值不是把所有负结果“调成正结果”，而是建立一套能够明确回答“谱信息、latent geometry 和局部可预测性在哪一步不再足够”的统一工具。

