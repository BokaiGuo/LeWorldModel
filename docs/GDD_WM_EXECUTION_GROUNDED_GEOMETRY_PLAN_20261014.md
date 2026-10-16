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
| shared candidate bank | `experiments/decision_benchmark/` | 部分基础设施；v5 已有 realized RGB refs，但尚缺独立 D-JEPA teacher |
| D1 stability | `experiments/gdd_wm/audit_preference_stability.py` | implemented; current bank rejected fail-closed |
| D2 projection | `experiments/gdd_wm/train_projection_distillation.py` | implemented; current D1 gate rejected and training correctly short-circuited |
| D3 co-training | 待新增 `gdd_wm/train_gdd_student.py` | blocked on D0-D2 |
| D4 fixed/CEM evaluation | 复用 decision benchmark 与 planning gates | blocked on valid student |

本文件的实现顺序是 D0 -> D1 -> D2 -> D3 -> D4；任何阶段失败都要保存 artifact 和
失败标签，不能用后续阶段的 test consequence 反向选择 teacher、pair、权重或 checkpoint。

### D0/D1 当前运行结果（2026-10-14）

历史审计 artifact 位于
`le-wm-main/results/gdd_wm_d0_d1_audit_20261014_v4/`；当前 continuation 位于
`le-wm-main/results/gdd_wm_d0_d1_audit_20261014_v5/`。v5 通过
`experiments/gdd_wm/join_rgb_realized_refs.py` 补齐 640/640 条 realized RGB
references，D0 schema/provenance 检查通过（640 rows、40 candidate sets、
train/calibration/outcome root split），因此 D1 的 runtime-reference support
gate 已通过。

D1 进一步发现当前 `teacher_score` 与 `real_task_cost` 精确满足
`teacher_score = -real_task_cost`。表面上的 execution consistency 为 1.0 是标签泄漏，
不是 teacher 的独立验证；审计已将
`teacher_execution_leakage.exact_negative_cost_match=true`，并拒绝所有 D1 eligible
pairs。此外每个 root 只有一个 `teacher_set_id`，因此 candidate-set preference
stability 的 flip rate 不可估计。当前 teacher 还是
`offline_environment_cost_v1`，不是已验证的 D-JEPA teacher。

v5 的 D1 仍然 fail-closed：每个 root 只有一个 candidate-set context，无法估计
preference stability；同时 teacher 是 `offline_environment_cost_v1`，且
`teacher_score = -real_task_cost`，execution consistency 属于标签泄漏而非独立
验证。因而 v5 不是 D-JEPA teacher 证据，D2 projection distillation 仍然 blocked。
下一份合格 D1 数据必须同时提供：同一
root/candidate pair 的至少两个独立 candidate-set contexts、独立 frozen teacher score、
execution-grounded realized observation refs，以及 teacher/execution 互不泄漏的 provenance。

### Train-only candidate-effect control（2026-10-15）

为推进 D0/D1 被阻塞后的可验证部分，使用带有完整 history 字段的
`jepa_history_near_goal_v5_source` 运行了 GPU train-root-only candidate-effect adapter。
它不是 D-JEPA teacher，也不使用 outcome labels 选择模型；只作为 decision-transfer
接口控制实验。history-latent+action 在 8 个 root-disjoint outcome roots 上的平均
effect RMSE 为 `0.3358`，平均 paired regret gain 为 `+0.1243`，paired success gain
为 `+0.125`；同一数据和 split 的 frozen-latent+action 对照分别为 `0.3518`、
`+0.0998` 和 `+0.025`。这说明短期 history 在一个 PointMaze 离线候选任务上有
增量 action-effect 信息，但仍不构成 D-JEPA preference stability、RGB successor
verification 或闭环规划证据。

artifact：
`le-wm-main/results/gdd_wm_candidate_effect_cuda_20261015/history_latent_action_v5.json`
和
`le-wm-main/results/gdd_wm_candidate_effect_cuda_20261015/frozen_latent_action_v5.json`。
由于该 source 的 manifest 明确标记 realized visual latent unavailable，D2 projection
distillation 仍保持 blocked；下一步必须获得独立 frozen teacher 和重复 candidate-set
contexts，不能用这个 positive control 绕过 D1 门禁。

### Realized-RGB successor subgate（2026-10-15 continuation）

现有 dataset-matched RGB successor artifact 已按 architecture × seed 聚合，未按 outcome
选择 case。`results/gdd_wm_rgb_successor_panel_20261015/aggregate.json` 包含 `mlp` 与
`standardized_mlp` 两个 architecture、各 5 个 seed、每 case 8 个 outcome roots；10/10
case 通过 renderer consistency、realized RGB latent availability 和 projected successor
comparison gate。

80 个 outcome roots 的 candidate-level predicted-better fraction 为 `0.6656`，two-level
case/root bootstrap CI 为 `[0.5977,0.7305]`；relative L2 improvement 为 `0.1421`，CI
为 `[0.0849, 0.1965]`。这是 realized-RGB successor verification 的跨 case positive
subgate，但不是 D1 teacher validation：单个 root 仍可为负，当前 bank 仍缺少同一
root/candidate pair 的多个独立 candidate-set contexts，且原 teacher 仍有
`teacher_score = -real_task_cost` 泄漏。因此 D2 继续保持 blocked。

### Repeated candidate-context structure repair (2026-10-15)

`experiments/gdd_wm/build_repeated_candidate_contexts.py` constructs the
missing D1 table by joining independent candidate banks on exact
root/action-sequence hash and assigning distinct context IDs. On two existing
PointMaze banks it produces 2 contexts, 40 roots, 410 matched root/action
pairs, and 820 records. The D1 audit consequently computes preference
stability (1,930 repeated pair-context comparisons; flip rate `0.0`). It
still rejects the data because the teacher is the leaked
`offline_environment_cost_v1` oracle and 164 outcome/test records lack full
runtime references. This is a structural D1 repair and sharper failure
localization; it is not D-JEPA teacher evidence and does not authorize D2.

### D1 package eligibility audit and D2 provenance hardening (2026-10-15)

The read-only package audit inspected all 24 existing candidate packages and
found `0/24` eligible. The failures are structural: the packages use
`offline_environment_cost_v1`, their teacher score is exactly negative realized
task cost, the original packages lack repeated root/candidate contexts, and
outcome/test rows have incomplete runtime references. The complete artifact is
`results/gdd_wm_teacher_candidate_audit_20261015/teacher_candidate_audit.json`.

D2 therefore remains blocked. The projection runner now requires a
provenance-bearing feature table with the exact candidate-dataset SHA-256,
canonical record-key schema, explicit exclusion of candidate-set context,
teacher rank, and held-out outcome inputs, plus a frozen-dynamics declaration.
It also rejects projection ranks above the train-sample rank. This is a
fail-closed interface repair; it does not convert the leaked package into a
valid teacher or authorize D2 training.

### Non-leaking train-only readout control (2026-10-15 continuation)

To isolate the D1 structure from teacher quality, the new
`experiments/gdd_wm/build_train_only_readout_teacher_package.py` joins two
checkpoint-stage contexts using exact root/action hashes and fits a frozen
standardized ridge readout on train roots only. It produces 2 contexts, 40
roots, 640 matched root/action pairs, 1,280 records, and complete RGB
successor references. The package audit passes (`1/1 eligible`), demonstrating
that the repeated-context and runtime-reference gates are executable on a
non-leaking control.

This does not validate the teacher: D1 reports flip rate `0.3081`, overall
execution consistency `0.4719`, and held-out outcome consistency `0.5800`.
The artifact is therefore a positive infrastructure control but a negative
teacher-quality result. Its claim boundary remains non-D-JEPA control only;
it cannot open D2 or support planning claims.

### Feature-family intervention (2026-10-16 continuation)

Five feature families were frozen before outcome evaluation under the same
two-context train-only ridge protocol: predicted terminal, source plus
terminal, source plus delta, source/delta/action, and action summary. The
root-bootstrap artifact is
`results/gdd_wm_nonleaking_teacher_control_20261015/feature_family_panel/summary.json`.

The action-only family has the strongest held-out outcome readout (pairwise
consistency `0.6179`, mean regret `0.1060`), while predicted terminal is
`0.5625` and `0.2116`; source/delta/action has no reliable consistency gain
(paired delta `-0.0620`, CI crossing zero). This identifies action-distribution
signal as a confound and does not validate latent decision geometry. It should
remain a required action-only baseline in future teacher and distillation
experiments, not be promoted as the proposed method.

### Action-residualized latent increment test (2026-10-16 continuation)

The teacher control now supports a second-stage train-only residual protocol:
an action-summary readout is fit first, and each latent family is fit only to
the remaining task-cost residual. On the held-out PointMaze roots,
residualized predicted-terminal reaches `0.5932` pairwise consistency versus
`0.6179` for action-only; the paired delta is `-0.0247` (root-bootstrap CI
crosses zero) and regret increases by `0.0250`. Residualized source/delta/action
is `0.5006`. Thus the current latent does not demonstrate incremental decision
information after action distribution is controlled. The artifact is
`results/gdd_wm_nonleaking_teacher_control_20261015/residualized_feature_panel/summary_vs_action.json`.

### Fixed nonlinear-capacity control (2026-10-16 continuation)

The residualized negative result was followed by a preregistered capacity
check rather than by outcome-driven tuning. The new
`experiments/gdd_wm/audit_train_only_teacher_capacity.py` keeps the
`source_delta_action` family and the train-only action baseline fixed, then
compares a standardized linear residual with one fixed CUDA shallow MLP. On
the held-out PointMaze roots, the linear residual reduces mean regret by
`0.1568` versus action-only (root-bootstrap 95% CI
`[-0.2602,-0.0678]`), whereas the MLP reduction is only `0.0631` with CI
`[-0.1797,+0.0360]`. Consistency gains are uncertain for both.

This result is a bounded positive linear-readout control, not a solution to
the latent transfer problem. It rejects the simple explanation that the
previous negative result was caused by insufficient nonlinear readout
capacity. The next interpretation remains support-sensitive and
architecture-conditioned: the current latent may expose a small linear
signal, but it has not shown robust incremental information after action
conditioning across environments or an independent planning consequence.
Artifact:
`results/gdd_wm_nonleaking_teacher_control_20261015/capacity_audit/capacity_audit.json`.

### Coordinate-frame and within-root estimand repair (2026-10-16 continuation)

The capacity control was stress-tested against the two protocol confounds
identified by the initial point estimate. On two independent canonical
goal-frame seed contexts, the original raw-action linear residual gave regret
delta `-0.0305` (95% CI crosses zero), while the fixed MLP gave `-0.0271`
(CI crosses zero). Replacing raw action summaries with exact goal-relative
summaries produced linear delta `0.0000` and MLP delta `-0.0265` (CI crosses
zero). Finally, centering the train target within each train root before
fitting, while retaining held-out root-wise ranking/regret, produced linear
delta `0.0000` and MLP delta `-0.0360` (CI crosses zero).

Thus the earlier absolute raw-action improvement is not stable under canonical
candidate coordinates and a within-root action-effect estimand. The repair is
now encoded in the runner as `--action-frame goal_relative` and
`--target-mode root_centered`; future decision-transfer audits must report
these controls before attributing gains to latent geometry. This strengthens
the negative diagnosis rather than creating a positive latent claim: current
evidence still lacks an independent, cross-environment latent increment.

### Canonical action-frame closure (2026-10-16)

The non-PushT candidate exporters were extended to include evaluator-only
`source_state_raw` and `goal_raw` fields, which closes the missing-data
condition for goal-relative action coordinates without exposing state to the
RGB model. Four new CUDA/EGL panels were generated under
`results/gdd_wm_cross_environment_20261016/*_goalframe/`.

The predeclared goal-relative/root-centered audit removes the apparent raw
action gain: TwoRoom linear/MLP regret deltas are `-3.0036`/`-3.3582` with
confidence intervals crossing zero, and Reacher linear/MLP deltas are
`+0.1087`/`+0.2677`, also with intervals crossing zero. Consequently the
current positive raw-action signal is classified as a coordinate/root-geometry
confound. The repair is successful at making the estimand valid, but it does
not create a positive latent result; future work must change the learned
action-effect representation or data support rather than increase readout
capacity.

### Context-safe capacity replication across non-PushT environments (2026-10-16)

The train-only capacity runner was corrected after finding that repeated
contexts had been merged before ranking. Each `context x root` is now an
independent ranking cell; root-level summaries macro-average unique roots and
bootstrap only unique outcome roots. The corrected CUDA artifacts are
`results/gdd_wm_cross_environment_20261016/tworoom_capacity_root_centered_context_safe/`
and
`results/gdd_wm_cross_environment_20261016/reacher_capacity_root_centered_context_safe/`.

The fixed MLP residual reduced held-out root-centered regret relative to the
action-only baseline in both environments (TwoRoom `-16.8480`, CI
`[-32.0209,-0.4949]`; Reacher `-1.9058`, CI `[-2.3356,-1.4063]`). Reacher's
linear residual also improves (`-1.6113`, CI `[-2.0892,-1.1483]`), while the
TwoRoom linear interval crosses zero. This is the first corrected
non-PushT cross-environment capacity signal, but it remains a diagnostic
replication over two history contexts, not the required architecture/seed
matched benchmark. The action frame is still raw, so the canonical
goal-relative replication remains mandatory before interpreting this as
independent latent transfer information.

### Goal-latent conditioning intervention (2026-10-16)

The panel exporters now retain frozen RGB goal embeddings, and the audit
supports `--goal-conditioning latent_relative` using only runtime-safe latent
features (`goal-source` and `predicted-goal`). On the canonical
goal-relative/root-centered protocol, TwoRoom gives linear/MLP regret deltas
`-4.3478`/`-1.2867` with intervals crossing zero; Reacher gives
`+0.2943`/`-0.2280`, with the linear interval excluding zero in the harmful
direction and the MLP interval crossing zero. This intervention does not
repair the negative result. It rules out missing goal latent as the sole cause
and focuses the next method change on action-effect support/identifiability,
not another readout-capacity increase.
