# GDD-WM：从决策偏好到预测几何

状态：`Proposed / Not Yet Validated`

本 Workstream 将附件中的 DA-LeWM / D-JEPA 分析与当前 LeWorldModel 的
transfer-sufficient diagnostics 连接起来。它是现有研究的新方向，不替换
谱 transfer、horizon-conditioned memory、PushT 接触控制或现有负结果。

## 1. 统一定位

现有主线回答：

> 给定 frozen visual latent、action distribution、预测器和 task readout，哪些
> 有限时域 action-to-task transfer 风险可以被识别和门控？

GDD-WM 增加一个新的问题：

> D-JEPA 从真实执行结果中学到的候选偏好，有多少能够被蒸馏成一个不依赖
> candidate set 的 intrinsic predictive geometry，并在不牺牲预测一致性的条件下
> 改善 MPC 的局部决策？

统一关系为：

```text
frozen LeWM / JEPA
    -> candidate futures + transfer diagnostics
D-JEPA-style teacher
    -> execution-grounded preference
GDD-WM student
    -> realized/predicted geometry distillation
    -> native Euclidean MPC without teacher reranker
    -> independent held-out planning consequence
```

GDD-WM 的成功不能由 teacher ranking agreement 单独定义，必须同时通过：

1. preference stability；
2. realized-geometry ranking；
3. predicted-geometry ranking；
4. multi-step predictive consistency；
5. fixed-candidate decision regret；
6. independent closed-loop planning consequence。

## 2. 与现有结果的衔接

当前 PointMaze 结果提供了一个重要的条件边界：JEPA task-effect adapter 在
calibration-only permission gate 后保留正向 held-out consequence；DINO 因
action-effect 不可辨识而被 `rejected_fail_closed`。因此：

- 第一版 GDD-WM teacher 优先使用已经通过 action/task transfer gate 的 JEPA
  条件；
- DINO 只能作为 representation-conditioned failure boundary，不能在 gate
  失败时被用来制造偏好标签；
- PushT 的接触和姿态闭合负结果必须保留为 GDD-WM 的 failure boundary；
- 任何 GDD-WM 正结果都不能回写成 frozen LeWM 已支持 universal planning。

GDD-WM 的关系不是“alignment 修复了规划”，而是新增一条可检验的因果链：

```text
execution feedback
 -> stable local preference
 -> geometrizable preference
 -> preserved predictive dynamics
 -> lower decision regret
 -> closed-loop improvement
```

任意箭头断裂，都必须作为机制边界报告。

## 3. 核心对象和可表示性问题

对同一个 start-goal context (x) 和候选集合 (mathcal C)，teacher 输出
候选分数 (s_T(a_imid x,mathcal C))。student 必须在部署时使用集合无关的
成本：

\[
c_S(x,a_i)=\frac1d\left\|F_\phi(E_\theta(x),a_i)-E_\theta(g)\right\|_2^2.
\]

student 不得接收其他候选、teacher rank、candidate ID 或 Set Transformer
hidden state。

### 3.1 Preference Stability

对固定候选对 (i,j)，在不同补充候选集合上重复比较：

\[
P_{ij}=\Pr_{\mathcal C}[s_T(a_i\mid x,\mathcal C)<s_T(a_j\mid x,\mathcal C)],
\qquad
q_{ij}=2\left|P_{ij}-\frac12\right|.
\]

高 (q_{ij}) 只说明 teacher 偏好稳定，不说明它真实正确；必须与独立执行
结果核对。低稳定性 pair 不得被强制蒸馏到固定 Euclidean geometry。

### 3.2 Geometric Distillation Gap

在固定 student geometry class、参数预算和训练 split 下，记录验证集上无法
消除的 preference violation。它区分：

1. teacher 自身偏好不稳定；
2. 偏好稳定但固定 Euclidean geometry 不可表示；
3. 几何可表示但 student 优化失败；
4. geometry 成功但 predictor rollout 失败。

### 3.3 Explicit impossibility boundaries

固定 geometry 不能普遍表达：

- 同一候选对随 candidate set 翻转的 context-dependent preference；
- 有向、非对称的可达性；
- 相同终点但不同碰撞/接触路径；
- 单帧观测无法区分的速度、接触模式或隐藏动力学；
- 依赖历史和模式转换的长时域偏好。

这些不是实现 bug，而是 GDD-WM 必须测量的不可约误差。

## 4. GDD-WM Student 的训练对象

第一版只保留最小、可审计的两条 geometry path：

### Realized geometry

\[
z^{real}_{i,H}=E_\theta(h^{real}_{i,H}),
\qquad
c^{obs}_i=\|z^{real}_{i,H}-z_g\|_2^2/d.
\]

对 outcome-grounded 的成功—失败 pair 使用 weighted pairwise loss：

\[
\mathcal L_{geo}=\sum_{i\succ j}w_{ij}
\operatorname{softplus}\left((m+c_i^{obs}-c_j^{obs})/\tau\right).
\]

### Predicted geometry

\[
\hat z_{i,H}=F_\phi(E_\theta(h_t),a_i),
\qquad
c^{pred}_i=\|\hat z_{i,H}-z_g\|_2^2/d.
\]

\[
\mathcal L_{rollrank}=\sum_{i\succ j}w_{ij}
\operatorname{softplus}\left((m+c_i^{pred}-c_j^{pred})/\tau\right).
\]

### Predictive preservation

\[
\mathcal L_{pred}=\sum_h\omega_h\left\|
\hat z_{t+h}-\operatorname{sg}[E_{\bar\theta}(h_{t+h})]
\right\|^2.
\]

Success-mass distillation、DA-LeWM inverse/goal-action losses和 preference
mining 属于后续阶段，不能在 MVP 中同时改变所有变量。

## 5. D0–D2 实施路线

### D0：Shared Candidate Dataset

建立 candidate-set 级别的机器可读数据，而不是把单候选记录拼接成伪集合。
每条 candidate 至少包含：

```text
instance_id, split, start_episode_id, goal_id, candidate_id,
action_sequence, source_observation_ref, goal_observation_ref,
teacher_scores/ranks, teacher_model_id,
real_success, real_task_cost, realized_observation_ref,
predicted_latent_ref, realized_latent_ref,
action_support, candidate_generator, privileged_state_role
```

要求：

- train/calibration/outcome 以 instance 或 root disjoint；
- 所有方法共享 candidate bank；
- teacher 在 distillation 前冻结；
- outcome 不参与 teacher、metric、权重或 snapshot 选择；
- real state 只作为 offline consequence/label，不进入 RGB-only runtime；
- 明确 candidate generator、CEM stage 和 action support。

### D1：Teacher Preference Stability Audit

对同一起点和候选对做 candidate-set bootstrap / leave-one-candidate-out，输出：

- (q_{ij}) 和 preference flip rate；
- teacher 与真实执行结果的 pairwise accuracy；
- all/random/mid/elite 分层结果；
- success/failure inversion rate；
- decision margin 与 teacher disagreement；
- candidate-set size 和 generator 改变后的 stability。

D1 的 gate 不是“稳定就通过”，而是决定 pair 的去向：

```text
stable + execution-consistent -> eligible for distillation
stable + execution-inconsistent -> teacher failure, execution wins
unstable -> diagnostic only, do not distill to fixed geometry
insufficient support -> rejected_fail_closed
```

### D2：Projection-only Geometry Distillation

先冻结 dynamics 和大部分 encoder，只学习一个低容量、train-only projection
(P_\psi)，验证偏好是否至少能被投影几何表达。比较：

1. frozen baseline；
2. linear projection-only；
3. last-layer geometry fine-tuning；
4. realized-only；
5. realized + predicted geometry；
6. full GDD-WM（后续，不属于 D2）。

D2 的必要输出：

- fixed-candidate pairwise accuracy；
- top-1 decision regret；
- teacher gain retention；
- observed-vs-predicted ranking gap；
- latent prediction MSE / multi-step transfer；
- covariance/spectrum/variance and collapse diagnostics；
- held-out candidate-set stability；
- fail-closed permission。

D2 只有在 prediction preservation 和独立 decision consequence 同时不恶化
时，才允许进入联合 Encoder/Predictor 微调。

## 6. 实验矩阵

### 主比较

```text
LeWM baseline
DA-LeWM
D-JEPA teacher (inference-time relation model)
pointwise score distillation
projection-only GDD-WM
GDD-WM with predicted-geometry path
```

### 必要 ablation

```text
no teacher
teacher-only without execution validation
no preference stability weighting
no elite mining
no realized-geometry loss
no predicted-geometry loss
no predictive preservation
frozen encoder
frozen predictor
```

所有使用执行反馈的方法必须使用相同 feedback budget；所有部署方法必须报告
相同 CEM rollout budget 和 latency。不能把额外数据或额外 inference module 的
收益归因给 intrinsic geometry。

## 7. 统计和释放门

GDD-WM 结果必须沿用现有项目的 evidence gates：

- 多个独立 training seeds，而不是多个 evaluation episodes 代替训练重复；
- candidate-set、root、environment 分离；
- paired bootstrap 和 sign/permutation test；
- train/calibration/test 三分离；
- leave-one-environment-out；
- no privileged runtime successor verification；
- action support 和 mode/observability gate；
- 所有 matched stratum 样本数和失败原因公开。

最低成功条件是一个联合条件，而非单项指标：

\[
\text{decision regret}\downarrow
\land
\text{predicted-realized ranking gap}\downarrow
\land
\text{multi-step prediction not worse}
\land
\text{independent held-out planning consequence}>0.
\]

如果只有 teacher agreement 或 realized geometry 改善，而 predicted geometry、
物理 transfer 或闭环规划不改善，结论只能是“preference distillation
diagnostic”，不能称为成功的 intrinsic world-model geometry。

## 8. 失败边界和备用路线

若 D1 显示 teacher preference 不稳定，停止强制蒸馏，转为 preference stability
paper/diagnostic。若 D2 的 Geometric Distillation Gap 很大，则承认 fixed
Euclidean metric 不足，尝试 goal-conditioned quasi-metric 或保留轻量 relation
model，但不再声称 reranker-free Euclidean planning。

若 rank 变好但 prediction MSE、real/predicted consistency 或 contact retention
变差，标记为 shortcut / geometry cheating，并回退到 frozen representation。

GDD-WM 不能覆盖现有 DINO rejection、PushT 姿态闭合失败或 commutator
non-universality；它只能为这些边界提供新的偏好层诊断。

## 9. 当前状态

截至 2026-10-13：

- 本文档和协议模板已加入仓库；
- D0 的 schema/provenance validator 已实现于
  `le-wm-main/experiments/gdd_wm/build_shared_candidate_dataset.py`，它只规范化
  外部提供的 candidate table，不生成 teacher、action outcome 或 latent 引用；
- D0 validator 的 schema smoke tests 已加入
  `le-wm-main/tests/test_gdd_wm_d0_candidate_dataset.py`；
- D0 schema 已运行；D1 stability audit 已实现并在当前 bank 上 fail-closed；D2
  projection distillation 尚未运行；
- 当前仓库没有可直接声明为 D-JEPA teacher 的已验证 checkpoint/data package；
- 现有 PointMaze JEPA positive gate 只是候选基础设施，不是 GDD-WM 结果；
- 任何论文表述必须使用 `Proposed / Not Yet Validated`。

参考：

- [Decision-Metric Alignment in Latent World Models](https://arxiv.org/abs/2608.18746)
- [D-JEPA: A Decision-Aligned Latent World Model](https://arxiv.org/abs/2609.24749)

上述链接和附件中的论文解读用于提出假设；论文中的作者报告结果不等于本仓库已复现实证。
