# LeWorldModel 顶会严格审稿意见与执行准则

## 来源与定位

本文件固化 2026-10-14 项目审稿意见，作为后续代码、协议和证据审计的执行准则。
它不是新的实验结果，也不改变既有负结果。核心判断是：项目已经形成了较完整的
诊断体系，但尚未完成“诊断失败 → 预测失败 → 认证决策 → 改善控制”的闭环。

当前模拟审稿结论：`Reject`。成熟度评分为：问题重要性 9/10、原创性 6/10、理论
深度 6/10、方法完整性 6/10、实验证据 3/10、可复现与证据规范 8/10、Oral 3/10、
Best Paper 2/10。评分用于确定工作优先级，不是会议录用预测。

## 已确认的实证边界

| 方向 | 当前结果 | 可支持表述 |
| --- | --- | --- |
| Cross-environment matched diagnostic | `Delta R=+0.05336`，sign-flip `p=0.0630` | 有正向信号，但预注册检验未通过 |
| PointMaze RGB successor | RMSE `0.2011 -> 0.1631` | 后继预测改善，不等于规划改善 |
| PointMaze causal history | regret `0.1897 -> 0.0798`，success `0.125` | 离线候选选择改善，闭环成功率未建立 |
| PushT contact/torque | signed direction accuracy `0.609`，strict pose `0/32` | 局部动作效应迹象，稳定姿态闭合失败 |
| GDD-WM | Proposed / Not Yet Validated | 没有已验证 teacher 或 student 蒸馏结果 |

这些结果不能升级为 universal alignment predictor、稳定 learned pose closure、因果
latent dynamics、frozen RGB online planning、universal world model 或 universal
commutator certificate。

## 七项 Major Concerns

### 1. HTSC theorem 还不是可用的 decision certificate

有限时域误差传播定理在假设成立时合理，但 closure、off-manifold、action-effect、
readout 上界和局部 tube 条件本身尚未由有限独立干预数据验证。当前任务状态误差界
也没有自动给出 planning cost、路径碰撞、接触失败或累计代价的界。

必须增加 cost-level certificate。对平方任务代价至少应明确推导：

\[
|J(y)-J(\hat y)|
\leq 2\|\hat y-g\|_M\epsilon+\epsilon^2.
\]

最终目标是独立 action intervention 与 successor calibration 支持的 candidate-wise
coverage，而不是把外部 `passed=True` 传入控制器。

### 2. 风险控制接口存在，但风险认证算法尚未闭合

`risk_aware_control.py` 能消费 action-identifiability、transition consistency 和
risk errors，但尚不能证明这些 evidence 来自有效覆盖、真实干预和独立 calibration。
必须把 evidence 生成、校准、permit/abstain 和 fallback 统一起来，并报告拒绝成本。

### 3. Cross-environment 统计独立性不足

matched strata 不等于独立环境、独立 world model 或独立训练重复。需要 model-level
外层留出、独立 backbone seeds、leave-one-environment-out、冻结指标权重，并分别报告
总体预测效应、条件增量效应和几何干预效应。

### 4. PointMaze 仍主要是 prediction/selection proxy

固定 candidate bank 的 regret 不等于 CEM/在线 MPC 的 regret。必须运行真正的
RGB-only receding-horizon MPC，比较 random、CEM initial、CEM elite、student-generated
candidates，并记录 optimizer-induced distribution shift、success、regret、latency、
fallback 和计算预算。

### 5. PushT 暴露 within-mode action identifiability 瓶颈

contact-mode observability 与 within-mode action-effect identifiability 必须分开。
增加 GRU、Fourier 或 MLP 不能自动恢复不可观测的接触点、摩擦、相对速度和切向载荷。
先做同一 mode、同一 root 的正向/反向/零动作干预，再判断历史或主动探测是否真正
降低不可辨识性；strict pose closure 不能由 signed effect 结果替代。

### 6. GDD-WM 可能只是 metric learning

冻结 predictor 上的 projection-only 目标满足
\(P^\top P=M\succeq0\)，本质上是低秩 Mahalanobis metric。必须比较 Euclidean、
Mahalanobis、pointwise MLP、ordinary ranking distillation、D-JEPA/DA-LeWM、
realized-only、realized+predicted 和 joint adaptation，并在 student-generated
candidate distribution 上验证，而不能只在 teacher candidates 上报告 ranking。

同时必须严格区分“动作条件预测器表达有向动力学”与“固定表示/终点代价/低秩投影
的不可表示性”，不能从欧氏距离对称性直接推出一般性 impossibility。

### 7. 尚未压过最接近工作

需要明确相对 ATM、AD-WM、DA-LeWM、D-JEPA、controlled-world-model identifiability
和 Value Equivalence 的增量：不是再增加指标，而是证明在 MSE、action inverse risk
和简单 decision alignment 匹配后，HTSC 仍能识别有限时域失败，且能在独立 RGB-only
闭环中降低错误放行和 regret。

## 必须完成的理论升级

### A. Evidence-Calibrated Decision Transfer Certificate

对视觉历史、目标和候选集合定义真实干预代价 \(J_H^*(U)\) 与模型代价
\(\widehat J_H(U)\)，从独立 action intervention/successor calibration 构造
\(\epsilon_H(U,\delta)\)，目标是：

\[
\Pr\left(\forall U\in\mathcal C_H:
|J_H^*(U)-\widehat J_H(U)|\leq\epsilon_H(U,\delta)\right)\geq1-\delta.
\]

在认证候选集合中使用 upper-confidence selection，regret 结论只能相对于允许集合
内的最优动作成立；必须同时报告 coverage、false release、fallback 和 rejection cost。

### B. Preference Geometrizability under Predictive Dynamics Constraints

对 \(\Delta_i=\widehat z_H(U_i)-z_g\)，检验是否存在
\(M\succeq0,\operatorname{Tr}(M)=1\) 满足偏好约束：

\[
\operatorname{Tr}\left(M(\Delta_j\Delta_j^\top-
\Delta_i\Delta_i^\top)\right)\geq m_{ij}.
\]

该 SDP 只能证明固定表示上的 PSD metric 可表示性。主贡献必须进一步连接 preference
violation、representation aliasing、预测保持和真实规划，而不是把 SDP 或 metric
learning 本身包装成新理论。

## 决定性实验协议

### Experiment 1：预测正确但选错动作

同一 RGB 起点、goal、action length、candidate bank 和预算，真实执行所有候选，
比较 MSE、uncertainty、Plan-Real/CEM-stage alignment、ATM、spectrum/covariance/
commutator、\(G_H^\Phi\)+action support 和 full HTSC。主估计量是控制 MSE、action
inverse risk、spectrum 和 stage 后的增量 real action regret。

### Experiment 2：RGB-only online planning

严格执行：

```text
RGB -> frozen LeWM -> candidate rollout -> HTSC -> execute -> new RGB
```

禁止 simulator-state successor 参与动作选择。比较原始 LeWM、fixed-horizon、
uncertainty-aware、ATM screening、HTSC 和 privileged oracle；报告 success、regret、
动作数、latency、fallback、coverage、false release 和计算预算。

### Experiment 3：PushT within-mode physics 与闭环分离

报告正/反/零动作的真实与预测 \(\operatorname{sign}(\Delta\theta(a)-
\Delta\theta(-a))\)、误差、CI、mode-conditioned calibration；之后再做 learned
RGB-only pose MPC。`0/32` 必须保留为负结果，不能用任何局部 signed-effect 提升替代。

### Experiment 4：HTSC Risk-Coverage-Utility

绘制 risk–coverage 曲线，并在相同 coverage 下比较 conditional regret、包含 fallback
的总体成功率、实际 coverage 和 false release。低 coverage 的“安全”不能被报告为控制
收益。

### Experiment 5：GDD-WM predictive geometry

严格执行 D0-D4：teacher/candidate provenance、preference stability、geometrizability
SDP、projection-only、Mahalanobis/MLP/ranking baselines、predicted/realized 双路径、
predictive preservation，最后在 student-generated candidates 和独立 planning split
上验证。D0 不能提升为 D1，D2 不能提升为 planning success。

## 最终实验矩阵

| 维度 | 最低覆盖 |
| --- | --- |
| 环境 | PointMaze、Reacher、PushT、3D manipulation |
| 模型 | LeWM/JEPA、DINO-based WM、独立 video world model |
| 重复 | 多个独立训练 seed；candidate、root、episode、model、environment 分开统计 |
| 数据 | train/calibration/intervention/final outcome 完全隔离 |
| 候选 | random、CEM initial、CEM elite、student-generated |
| 指标 | success、regret、risk coverage、false release、latency、fallback |
| OOD | 物理参数、接触模式、目标和初始状态变化 |

## 论文级停止条件

### 主会接收级

独立 RGB-only planning 中，HTSC 相对充分调优的固定 horizon 和 uncertainty baseline
取得稳定、公平、显著的闭环收益，并有完整消融和失败分析。

### Oral 级

新的可辨识性/证书理论能够解释既有方法解释不了的真实规划失败，并在跨模型、跨环境
上预测风险，改善 risk–coverage–utility 权衡。

### Best Paper 级

发现具有广泛意义的 world-model 决策失效定律，给出新的可计算保证或能力边界，且
通过独立物理干预和广泛闭环控制实验证明实际价值。

当前最值得验证的主命题是：

> 即使两个 world model 具有相近 latent prediction quality 和 action identifiability，
> 它们仍可能具有不同的 planning reliability；差异可由 horizon-conditioned
> transfer-sufficiency 条件解释，并由有限独立干预校准。

## 执行规则

- 所有新运行必须使用 GPU 优先；只有宿主 CUDA 检查失败才允许 CPU；
- 每个方向必须写入 manifest、input lock、stage status、metrics 和 evidence boundary；
- 代码完成、artifact 存在和 execution completed 不得自动变成 scientific support；
- 历史负结果必须保留；不能通过改阈值、挑 checkpoint 或重分类证据制造正结果；
- 缺少真实数据、checkpoint、独立 outcome 或 RGB runtime bridge 的方向必须输出
  `blocked`，而不是生成 proxy claim。
