# Lifted Transfer Research Specification

版本：2026-10-12  
状态：代码已开始落地；真实 learned export 的结论仍受 PE、history support 和 mode labels gate 约束。

## 1. 唯一主命题

本研究不把“升维后更容易分离”作为结论，而检验一个更严格的命题：

> 一个 lifted representation 只有在能够作为 action-to-task transfer 的充分统计量时才有规划意义；它不需要恢复完整物理状态，但必须在独立数据上降低 action-conditioned task-effect risk。

对观测历史和动作历史定义

\[
\eta_t=\mathcal L_\Phi(o_{t-L:t},a_{t-L:t-1}),\qquad
\hat y_t=\mathcal P_\Phi(\eta_t),
\]

并拟合有限记忆输入输出实现

\[
\eta_{t+1}=A_\Phi\eta_t+B_\Phi a_t+
\sum_{\ell=1}^{L-1}K_{\Phi,\ell}\eta_{t-\ell}+\varepsilon_t,
\qquad
y_t=D_\Phi\eta_t.
\]

主 estimand 是

\[
\mathcal T_h^\Phi=M_y^{1/2}D_\Phi A_\Phi^{h-1}B_\Phi M_a^{1/2},
\]

以及同一 source state 上的 action contrast：

\[
E_{\rm cf}^\Phi=
\|[(\hat y^a-\hat y^{a'})-(y^a-y^{a'})]\|_{M_y}.
\]

latent reconstruction、mutual information 或 eigenvalue separation 只能作为中间诊断，不能替代这两个量。

## 2. 数学—物理—信号—信息论的统一对象

统一对象为

\[
\mathcal W_\Phi=(\Pi,K_\Phi,A_\Phi,B_\Phi,D_\Phi,M_{task},\Gamma_H),
\]

其中 \(\Pi\) 是从物理状态到观测 latent 的投影，\(K_\Phi\) 是 unresolved memory，\(A,B,D\) 是 lifted input-output realization，\(M_{task}\) 是预先声明的外部 metric，\(\Gamma_H\) 是有限时域 transient amplification。

各工具的职责固定如下：

| 层 | 可计算对象 | 要回答的问题 |
|---|---|---|
| gauge / realization | \(DA^{h-1}B\)、\(D(\lambda I-A)^{-1}B\) | 哪些量不依赖 latent 坐标？ |
| memory / Koopman | \(K_\ell\)、companion operator、closure residual | 单帧 latent 是否是非 Markov 投影？ |
| control | PE、PBH、\(W_c,W_o\)、Hankel singular values | 哪些方向既可激活又可读出？ |
| physics | \(SE(2)\) twist、mode path transfer、passivity | 任务变量和混合模式是否物理一致？ |
| non-normality | Schur、resolvent、transient gain | 一步误差如何变成 horizon risk？ |
| signal | causal delay、rFFT/STFT-like local spectrum | 速度、接触持续时间和冲击是否被历史恢复？ |
| information | Gaussian plug-in \(I(\Delta y;\Phi_{extra}\mid\Phi_{base},a)\) | lifting 是否增加 action-effect 信息？ |
| decision | held-out effect RMSE、ranking、regret | 信息是否真正转成 transfer？ |

信息量增加不等于因果性增加。只有 information gate 和 held-out transfer gate 同时通过，才允许称为 transfer-sufficient lifting。

## 3. 代码实现

- [run_projected_hybrid_physics_audit.py](/home/gbk/code/LeWorldModel/le-wm-main/experiments/empirical_spectral_audit/run_projected_hybrid_physics_audit.py)
  - finite-memory kernel 与正确的 companion operator；
  - finite-horizon Hankel geometry；
  - metric-weighted frequency response；
  - hybrid pathwise transfer。
- [run_lifted_signal_information_audit.py](/home/gbk/code/LeWorldModel/le-wm-main/experiments/empirical_spectral_audit/run_lifted_signal_information_audit.py)
  - episode 内 causal delay；
  - causal short-time rFFT features；
  - action interaction lifting；
  - conditional-MI plug-in 与 cross-fitted transfer sufficiency。
- [run_real_enriched_hybrid_audit.py](/home/gbk/code/LeWorldModel/le-wm-main/experiments/empirical_spectral_audit/run_real_enriched_hybrid_audit.py)
  - canonical enriched export loader；
  - episode-disjoint fit/outcome；
  - fit-only POD projection；
  - PE、memory、Hankel、frequency、signal/information gates。

所有真实 export 的 POD basis 只从 fit episodes 估计；原始 384 维 export 不被覆盖。rank-capped 结果必须带有 rank、解释方差和重构误差元数据。

## 4. 证伪协议

1. `action_excitation` 不通过：停止解释 \(B_\Phi\)，不得声称 action effect 已辨识。
2. `memory_closure` 不通过：停止把 delay lifting 解释为有效动力学恢复。
3. `mode_observability` 不可用：PushT 只能作为 contact-mode boundary，不能作为 hybrid planner 成功证据。
4. causal lifting 的 held-out effect RMSE 不优于 base：升维只算 representation gain，不算 transfer gain。
5. geometry 在控制 MSE、action risk 后无增量：降级为 descriptive diagnostic。
6. 只有在上述门通过后，才允许进行 matched model selection 或闭环 planner 实验。

## 5. 当前真实数据的第一轮状态

2026-10-12 的 canonical JEPA/DINO export 已完成 rank-32、episode-disjoint audit。当前 artifact 位于
`le-wm-main/results/projected_hybrid_physics_audit_20261012/`。

协议审计发现两类原始负结果不能直接解释为表示失败：联合 \([\eta,a]\) PE 在高维 latent/短样本下必然秩亏，因此已改为 action-only 与 residualized-action PE；旧 PointMaze/DINO export 每 episode 只有一个 window，因此不能用于否定 causal history。修正后的 PointMaze pilot 位于
`results/projected_hybrid_physics_audit_20261012/jepa_pointmaze_windows4.json`：20 个 episode、每个 4 个窗口；action-only/residualized PE 通过，finite-memory outcome RMSE 由 `0.1834` 降到 `0.1443`，delay/interaction/time-frequency held-out task-effect RMSE 相对 base 分别降低约 `35.97%/24.52%/37.11%`。该结果只有 8 个 outcome rows，仍是 protocol-corrected pilot。

协议修正后仍存在的负结果不能全部归因于协议：JEPA–PushT 的 action PE 通过但 residualized action 和 finite-memory closure 不能稳定通过，interaction 与时频 lifting 也没有改善 held-out effect；这表明接触任务还有 representation/mode/action-support 问题。

因此当前正确结论是：

> lifting 的价值取决于历史支持、动作激励、memory closure 和外部 task transfer；“维度更高”本身不是方法贡献。

下一阶段必须补足多 episode、多 seed、successor-mode probe 和 shared-source candidate ranking，再讨论 conditional model selection。旧的 PushT 负结果和 matched benchmark 结果全部保留。
