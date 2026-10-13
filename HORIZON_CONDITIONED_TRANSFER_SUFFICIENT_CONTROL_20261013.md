# Horizon-Conditioned Transfer-Sufficient Control (HTSC)

## 1. 研究对象

HTSC 不是把 covariance、spectrum 或 commutator 单独升级成 certificate，而是把 learned latent 的可用性定义为一个有限时域、任务度量和动作支持集上的控制条件。

给定观测历史和动作历史，定义 horizon-specific lifting

\[
\eta_t^{(h)}=\Phi_h(z_{t-L:t},a_{t-L:t-1}),
\]

以及声明的 task metric \(M_y\succ0\)、候选动作集合 \(\mathcal A_h\) 和局部输入—输出模型

\[
\eta_{t+1}=A_h\eta_t+B_hu_t+r_t,
\qquad y_t=D_h\eta_t.
\]

真正的 decision-relevant object 是

\[
G_h^\Phi=D_hA_h^{h-1}B_h,
\]

而不是 predictor spectrum 或普通 Euclidean commutator。

## 2. Transfer-sufficient memory 的定义

在支持集 \(\mathcal S_h\) 上，\(\Phi_h\) 称为 \((\tau_h,\gamma_h,\kappa_h,\xi_h)\)-transfer-sufficient，当且仅当下面四个 gate 都通过：

1. **support gate**：部署候选动作到 calibration action support 的距离不超过 \(\kappa_h\)；
2. **action-identifiability gate**：候选 action effect 的最小奇异值不低于 \(\gamma_h\)，避免 mode 内 torque/action inversion；
3. **successor-consistency gate**：独立真实 transition 与预测 successor 的 task-metric 误差不超过 \(\xi_h\)；
4. **finite-horizon risk gate**：由独立 closure、off-manifold、action-effect、transient 和 readout envelope 形成的上界 \(\bar E_h\leq\tau_h\)。

四个 gate 的数据来源必须分离：训练/校准数据只能估计 representation 和局部模型；action intervention 数据估计 action gate；planner-reachable transition 数据估计 off-manifold/real-predicted gate；独立 task readout 数据估计 readout gate。

## 3. HTSC 控制律

固定一个不依赖 held-out outcome 的 memory policy \(\pi_h^{\mathrm{mem}}\)。当前 LeWM 实例为

\[
\Phi_h=F_h+g_hE_h,
\]

其中 \(F_h\) 是 fixed Fourier carrier，\(E_h\) 是 phase/event residual，\(g_h\in[0,1]\) 是因果事件门控。selector 不进入部署控制律，只输出 selection-instability diagnostic。

每次只在候选集合 \(\mathcal A_h\) 内解

\[
u_{t:t+h-1}^*=\arg\min_{u\in\mathcal A_h}
\left[
\|\hat y_{t+h}(u)-y_g\|_{M_y}^2+\lambda\bar E_h(u)
\right],
\]

然后执行一个短 pulse，重新观测并重新计算四个 gate。任何 gate 失败都返回预声明 fallback，而不是继续执行 learned action。

## 4. Soundness theorem

**定理（HTSC finite-horizon soundness）。** 假设：

1. 在声明的局部 tube \(\mathcal U_h\) 内，真实 lifted dynamics 与 learned dynamics 满足
   \(\eta_{k+1}=A_h\eta_k+B_hu_k+r_k^{\mathrm{cl}}+r_k^{\mathrm{off}}\)，且两条轨迹都留在 \(\mathcal U_h\)；
2. 对每一个被允许的候选序列，独立数据给出的 \(c_k,o_k,a_k,d,X\) 是真实 closure、off-manifold、action-effect 和 readout 误差的上界；
3. \(\bar E_h\) 使用

\[
\begin{aligned}
\bar E_h={}&\|D_hA_h^h\|\|e_0\|
 +\sum_{k=0}^{h-1}s_{h-1-k}(c_k+o_k+a_k)\\
 &+\sum_{k=0}^{h-1}(\|D_hA_h^{h-1-k}\|-s_{h-1-k})_+(c_k+o_k+a_k)+dX,
\end{aligned}
\]

其中 \(s_j=\|D_h\|\rho^j\)，\(\rho\) 是 \(A_h\) 的 spectral-radius upper bound；
4. support、action-identifiability、successor-consistency 和 risk gate 均通过。

则 HTSC 释放的 candidate 满足

\[
\|y_{t+h}-\hat y_{t+h}\|_2\leq\bar E_h\leq\tau_h.
\]

若任一 gate 不通过，HTSC 不释放 learned candidate，因此其控制合约不会把未识别的 action-to-task transfer 当成已认证行为。

### 证明

令 \(e_k=\eta_{t+k}-\hat\eta_{t+k}\)。相减得到

\[
e_{k+1}=A_he_k+v_k,
\qquad \|v_k\|\leq c_k+o_k+a_k.
\]

展开递推：

\[
e_h=A_h^he_0+\sum_{k=0}^{h-1}A_h^{h-1-k}v_k.
\]

读出误差满足

\[
y_{t+h}-\hat y_{t+h}=D_he_h+(D_h-\hat D_h)\hat\eta_{t+h}.
\]

三角不等式给出 \(\|D_hA_h^j\|\) 加权的 residual sum；将每个 gain 写成 \(s_j+(\|D_hA_h^j\|-s_j)_+\)，即可得到上述六项 bound。四个 gate 的作用是保证这个 bound 只在已识别的 action/support/local tube 内被使用；gate 失败时返回 fallback，故不会发生无证据的 learned release。\(\square\)

## 5. 该理论真正新增的内容

新增的不是“commutator 是证书”，而是以下控制对象和安全性质：

\[
\boxed{
\text{horizon-sufficient memory}
 +\text{ support gate}
 +\text{ action-identifiability gate}
 +\text{ finite-horizon risk envelope}
 +\text{ fail-closed receding control}
}
\]

它将记忆表示、learned dynamics、task metric、action coverage 和控制释放绑定到同一个可审计接口。普通 spectrum/alignment/commutator 只能作为 envelope 的解释项，不能替代 gate。

## 6. Tightness 与否证边界

- 标量系统 \(A=[a]\)、\(D=[1]\) 且所有 residual 同向时，triangle-inequality bound 可达到，说明 finite-horizon envelope 不是任意松弛；
- 两个系统可有相同 spectrum、covariance 和零 commutator，却因不同 \(B,D\) 产生不同 \(G_h\)，说明 geometry alone 不可识别 transfer；
- 若 action support 缺少某个 torque sign，action-identifiability gate 必须失败，即使 latent RMSE 很低；
- 若 selector 在 resampling 下发生 variant inversion，HTSC 使用预冻结 policy，不把 selector 的局部胜出升级为部署策略。

## 7. LeWM 的当前实例与证据边界

fit-only POD rank 32、episode-disjoint 的新 confirmation artifact：

`le-wm-main/results/horizon_conditioned_transfer_sufficiency_v2_confirm_20261013/summary.json`

给出同一 JEPA-PushT export 上的 transfer-sufficiency proxy：

- h=2：`Fourier+phase` 相对 fixed Fourier 在 5 个 split 中胜 4/5；
- h=4：`Fourier+phase` 在 5 个 split 中胜 5/5；
- h=1：正式策略保持 fixed Fourier；
- selector 的部署状态始终为 false，stability 只作风险诊断。

这些结果验证的是 finite-horizon action-to-task proxy，不是 action ranking、接触 mode 预测、姿态闭合、frozen online planning 或跨环境普适性。HTSC 理论只有在这些 gate 用独立数据真正通过后，才能实例化为 learned-model 控制证据。
