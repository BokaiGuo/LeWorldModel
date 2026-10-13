# GDD-WM：将执行决策知识蒸馏为预测几何

状态：`Proposed / Not Yet Validated`

本文件把 2026-10-14 提交的研究方案正式纳入 LeWorldModel。它是
[GDD-WM 基础计划](GDD_WM_DECISION_GEOMETRY_RESEARCH_PLAN_20261013.md)的执行扩展，
不是既有谱 transfer、horizon-conditioned memory 或 PushT 负结果的替代解释。

## 1. 核心问题与研究边界

研究问题是：

> 能否把冻结的 D-JEPA 风格、execution-grounded candidate preference，蒸馏成
> 不依赖 candidate set、Set Transformer 或 teacher runtime 的 intrinsic predictive
> geometry，同时保持 action-conditioned multi-step prediction？

目标不是简单增加一个 ranking loss，而是同时满足：

```text
decision alignment
  + predictive consistency
  + candidate-set-independent inference
  -> lower fixed-candidate regret
  -> independent closed-loop planning consequence
```

“稳定”不等于“正确”：teacher preference 必须先与独立执行结果核对；
teacher 不一致、集合依赖或 action support 不足的 pair 只能进入诊断，不能进入蒸馏。

## 2. 与现有 LeWorldModel 的关系

```text
frozen LeWM / JEPA
  -> candidate futures + transfer-risk/action-effect gates
D-JEPA-style frozen teacher
  -> execution-grounded preference
GDD-WM student
  -> realized + predicted geometry
  -> native Euclidean cost for MPC
  -> independent held-out planning audit
```

现有主线提供 representation、dynamics、action-support、finite-horizon transfer 和
fail-closed infrastructure；GDD-WM 检验这些对象能否进一步被决策偏好塑形。它不改写：

- PointMaze JEPA 的条件性 permission-gate 正结果；
- DINO 的 `rejected_fail_closed`；
- PushT 中 mode observability 与 within-mode action identifiability 的分离；
- learned adapter 姿态闭合失败；
- alignment 的跨环境 positive gate 尚未通过；
- commutator 不是通用 transfer certificate。

当前仓库没有已验证的 D-JEPA teacher checkpoint/data package，因此下面全部内容是
协议与实现目标，不是已完成结果。

## 3. 可表示性：Preference Stability 与 Geometrizability

对固定 start-goal context、候选对 `(i,j)`，从不同补充候选集合中重复读取 teacher
偏好：

\[
P_{ij}=\Pr_{\mathcal C}[s_T(a_i\mid x,\mathcal C)<s_T(a_j\mid x,\mathcal C)],
\qquad q_{ij}=2\left|P_{ij}-\frac12\right|.
\]

`q` 衡量候选集合稳定性，不是正确性。蒸馏资格固定为：

| 条件 | 处理 |
|---|---|
| stable + execution-consistent | 可蒸馏 |
| stable + execution-inconsistent | 以执行结果为准，teacher failure |
| unstable | 诊断，不蒸馏到固定几何 |
| support 不足 | `rejected_fail_closed` |

Geometric Distillation Gap 定义为固定 student geometry class、容量预算和训练 split
下的验证偏好违背率。它区分 teacher 不稳定、Euclidean geometry 不可表示、优化失败和
predictor rollout 失败。

必须保留的结构性反例包括：候选集合导致偏好翻转、有向不可逆可达性、相同终点不同
碰撞路径、单帧相同而速度/接触模式不同，以及长时域路径依赖。增大模型容量不能把
这些数学不可表示性改写成优化问题。

## 4. Student 的部署接口

student 只允许使用：

\[
c_S(x,a_i)=\frac1d\left\|F_\phi(E_\theta(x),a_i)-E_\theta(g)\right\|_2^2.
\]

部署时不得输入其他 candidate、candidate ID、teacher rank、Set Transformer hidden
state 或后置 reranker。teacher 在蒸馏前冻结，部署时丢弃。

## 5. 双路径几何监督

### 5.1 Realized geometry

真实执行后的观测编码为

\[
z^{real}_{i,H}=E_\theta(h^{real}_{i,H}),\qquad
c^{obs}_i=\|z^{real}_{i,H}-z_g\|_2^2/d.
\]

对可信的 outcome-grounded pair 使用：

\[
\mathcal L_{geo}=\frac1{\sum w_{ij}}
\sum_{i\succ j}w_{ij}\operatorname{softplus}
\left((m+c^{obs}_i-c^{obs}_j)/\tau\right).
\]

### 5.2 Predicted geometry

student imagined rollout 为

\[
\hat z_{i,H}=F_\phi(E_\theta(h_t),a_i),\qquad
c^{pred}_i=\|\hat z_{i,H}-z_g\|_2^2/d,
\]

对应的 rollout-ranking loss 为：

\[
\mathcal L_{rollrank}=\frac1{\sum w_{ij}}
\sum_{i\succ j}w_{ij}\operatorname{softplus}
\left((m+c^{pred}_i-c^{pred}_j)/\tau\right).
\]

### 5.3 Success mass 与 predictive preservation

有至少一个真实成功候选时，才允许使用：

\[
p_i^S=\frac{e^{-c^{pred}_i/T}}{\sum_k e^{-c^{pred}_k/T}},\qquad
\mathcal L_{succ}=-\log\sum_{i:y_i=1}p_i^S.
\]

全失败集合不伪造成功标签，应使用连续 task cost 或跳过该 loss。预测保持项为：

\[
\mathcal L_{pred}=\sum_h\omega_h\left\|
\hat z_{t+h}-\operatorname{sg}[E_{\bar\theta}(h_{t+h})]
\right\|_2^2,
\]

并保留 SIGReg、防坍缩、DA-LeWM inverse/goal-action 和必要的 feature preservation。
完整目标按阶段启用，而不是同时调所有权重：

\[
\mathcal L=\mathcal L_{pred}+\lambda_{geo}\mathcal L_{geo}
+\lambda_{rank}\mathcal L_{rollrank}+\lambda_{succ}\mathcal L_{succ}
+\lambda_{sig}\mathcal L_{SIGReg}+\lambda_{inv}\mathcal L_{inv}
+\lambda_{goal}\mathcal L_{goal}+\lambda_{pres}\mathcal L_{pres}.
\]

## 6. Preference mining 与数据闭环

pair 分为：

1. `outcome-grounded`：同根一成一败，执行反馈优先；
2. `teacher-consistent`：跨 candidate-set 稳定且未被执行结果否证；
3. `elite-hard`：CEM 后期接近但真实结果不同的 pair。

建议权重为稳定性、执行可靠度和 elite 标志的乘积；但所有阈值必须在 calibration
split 冻结。数据闭环为：

```text
student CEM candidates
 -> root-disjoint execution
 -> frozen teacher preference + stability audit
 -> eligible pair mining
 -> student update
 -> new student-reachable candidate distribution
```

不能只在 teacher 的旧候选分布上训练后声称闭环泛化。

## 7. 可检验的理论结果

对真实执行半径 `r_i`、预测半径 `rhat_i`，若可信 pair 满足

\[
r_j-r_i\ge m,
\qquad
\|\hat z_{i,H}-E(h_{i,H})\|_2,
\|\hat z_{j,H}-E(h_{j,H})\|_2\le\epsilon_H,
\]

则

\[
\hat r_j-\hat r_i\ge m-2\epsilon_H.
\]

因此 `m > 2 epsilon_H` 时预测欧氏距离保持 pair 顺序。实现时输出：

\[
M^{robust}_{ij}=(r_j-r_i)-(e_i+e_j),
\]

以及真实标签 pair 中 `M > 0` 的比例 `RDMC`（Robust Decision-Margin Coverage）。
该结论是固定 pair、固定目标、有界 rollout error 下的条件保证，不是训练算法或成功率
的无条件定理。

## 8. 阶段化实验与门禁

### D0：Shared Candidate Dataset

使用现有 validator
`le-wm-main/experiments/gdd_wm/build_shared_candidate_dataset.py`，每条记录至少有
instance/root/split/goal/candidate、action sequence、teacher score/rank、真实成功与
task cost、realized/predicted 引用、action support、candidate generator 和 CEM stage。
要求 train/calibration/outcome/environment-test root-disjoint、共享 candidate bank、
teacher frozen、runtime teacher forbidden。

### D1：Teacher Preference Stability Audit

输出 pair stability/flip rate、teacher-vs-execution pair accuracy、all/random/mid/elite
分层、success/failure inversion、candidate-set size/generator sensitivity 和 support
审计。D1 通过不等于 teacher 正确；它只决定哪些 pair 可进入 D2。

### D2：Projection-only Geometry Distillation

冻结 dynamics，先比较 frozen baseline、linear projection、last-layer tuning、
realized-only 和 realized+predicted geometry。输出 observed/predicted ranking、
decision regret、prediction MSE、effective rank、collapse、held-out stability 和
fail-closed permission。D2 未同时通过 prediction preservation 与独立 decision
consequence，不得联合微调 encoder/predictor。

### D3：Decision-aware co-training

仅在 D2 通过后加入 rollout ranking、success mass、EMA target 和学生自身 candidate
mining。固定真实反馈预算、CEM budget、latency 和 candidate bank，并比较：

```text
LeWM / DA-LeWM
D-JEPA teacher (runtime upper reference)
pointwise score distillation
projection-only GDD-WM
realized-only / predicted-only
full GDD-WM
```

### D4：独立闭环验证

必须同时运行 fixed candidate pool 和 independent CEM-MPC。主结果报告 success/failure
inversion、decision regret、teacher-gain retention、RDMC、observed-predicted ranking
gap、multi-step transfer、contact/angle/XY、planning latency 和 FLOPs。

## 9. 必要消融与失败判据

必须有 `no teacher`、`teacher-only`、`no stability weight`、`no elite mining`、
`no realized geometry`、`no rollout ranking`、`no success mass`、`no predictive
preservation`、`frozen encoder`、`frozen predictor`，并使用多个独立训练 seed。

预注册失败判据：

| 观测 | 解释与处理 |
|---|---|
| teacher pair 经常翻转 | 只蒸馏跨集合共识；其余保留为 context-dependent boundary |
| stable teacher 与执行结果冲突 | execution wins，记录 teacher failure |
| geometry gap 随容量仍不降 | Euclidean 不可表示，转向 goal-conditioned quasi-metric |
| ranking 上升但 prediction 变差 | 预测保持失败，拒绝联合模型 |
| observed ranking 改善、predicted 不改善 | predictor 是瓶颈 |
| 两者都改善但闭环不改善 | 检查 candidate support、长时域和终止接口 |
| success 上升但 probe/OOD 退化 | 表示 shortcut/collapse，拒绝升级 claim |
| fixed pool 改善而 CEM 不改善 | planner distribution shift，启动 student-reachable mining |

在 TwoRoom、不可逆或强路径依赖任务中，若纯 Euclidean geometry 出现结构性不可表示
误差，备选只能是明确命名的 goal-conditioned quasi-metric，不能继续称为严格 Euclidean
GDD-WM。

## 10. 论文贡献候选与 claim gate

若全部门禁通过，可检验四项贡献：

1. `Decision Preference Geometrizability`：稳定、可表示与不可表示偏好的诊断；
2. `Execution-Grounded Geometric Distillation`：把可信执行偏好内化到 encoder geometry；
3. `Prediction-Preserving Decision Alignment`：realized/predicted geometry 与多步保持；
4. `Reranker-Free Closed-loop Planning`：student 脱离 teacher 后的独立闭环收益。

在门禁通过前，论文只能写成 proposed workstream / protocol；不能声称 GDD-WM 已经
建立 universal world model、已解决 PushT 姿态闭合、或已证明 D-JEPA 偏好可普遍几何化。

## 11. 项目实现映射

| 研究对象 | 当前入口 | 状态 |
|---|---|---|
| D0 schema/provenance | `experiments/gdd_wm/build_shared_candidate_dataset.py` | 已实现，只有 schema audit |
| shared candidate bank | `experiments/decision_benchmark/` | 部分基础设施，尚缺 D-JEPA teacher/realized RGB refs |
| D1 stability | 待新增 `gdd_wm/audit_preference_stability.py` | not-run |
| D2 projection | 待新增 `gdd_wm/train_projection_distillation.py` | not-run |
| D3 co-training | 待新增 `gdd_wm/train_gdd_student.py` | blocked on D0-D2 |
| D4 fixed/CEM evaluation | 复用 decision benchmark 与 planning gates | blocked on valid student |

本文件的实现顺序是 D0 -> D1 -> D2 -> D3 -> D4；任何阶段失败都要保存 artifact 和
失败标签，不能用后续阶段的 test consequence 反向选择 teacher、pair、权重或 checkpoint。
