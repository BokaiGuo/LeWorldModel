# GDD-OPD：学生规划分布上的决策场蒸馏研究计划

**状态：Proposed / Not Yet Validated**
**日期：2026-10-24**
**拟议论文：** *From Relational Teachers to Predictive Decision Fields: On-Policy Geometry Distillation for World Models*

## 决策与研究边界

本文件将原有 GDD-WM 的未来方法定位升级为一条独立的、允许后训练的研究线：

> **GDD-OPD: Learning Intrinsic Decision Fields from Relational Teachers on Student-Generated Plans**

它研究的问题不是“冻结世界模型能否在测试时被更好地使用”，而是：在保持动作条件预测能力的约束下，world-model student 能否从更强的关系 teacher 在其**自身不断变化的规划分布**上学到可迁移的决策结构。

这项决定将 LeWorldModel 分为两条方法与证据均严格隔离的研究线：

| 研究线 | 核心问题 | 训练约束 | 不可借用的证据 |
| --- | --- | --- | --- |
| **D2P / Planning Compiler** | 如何从冻结视觉动力学中编译并搜索可执行动作程序？ | 严格 training-free；encoder、world model、policy、critic、adapter 均冻结 | 不得调用 GDD-OPD 训练过的 student，不能把后训练增益用于证明 training-free 性质 |
| **GDD-OPD** | 如何把关系 teacher 的、与 student 当前规划相关的决策知识内化为预测几何？ | 允许受约束的 student post-training | 不得把 Planning Compiler 的 test-time 收益写成蒸馏成功 |

两条线可共享预训练 checkpoint、环境、数据 schema、root-disjoint 划分和评估工具；不得共享最终方法组件、模型指纹、训练数据许可或成功判据。每个 artifact 必须明确其所属线及 frozen/trained model fingerprint。

## 现有证据的继承方式

本计划不重写或升级既有 GDD-WM 的结论。`GDD_WM_PROGRESS_20261019.md` 中的 teacher 控制、投影诊断、锚点/评分契约修复、支持与 phase 审计，都是本计划的工程与审计上下文，而不是 GDD-OPD 的实证证明。现有记录已经区分了代码/契约修复、诊断性结果、fail-closed 安全行为和独立闭环结论；这一分层必须保留。

特别是，当前证据不允许宣称：已验证的 D-JEPA relational teacher、普适 teacher preference、普适 OPD 收敛、GDD 专属闭环收益，或一般 world-model 优越性。任何新实验发现首先要排查数据泄漏、candidate identity/coordinate confound、goal anchor、horizon、支持集、fallback 和评估器契约，之后才可被解释为方法或理论结果。

## 科学问题与新颖性边界

核心问题为：

> Can a world-model student learn transferable decision structure from a stronger relational teacher on the student's own planning distribution, while preserving action-conditioned predictive dynamics and transferring beyond the supervised candidate set?

关系 teacher 可以在整个候选集合上作比较；部署时 student 的候选代价必须是 candidate-set-independent：

\[
c_\theta(h,U,g)=\frac{1}{d}\left\|F_\theta^{(H)}(E_\theta(h),U)-E_\theta(g)\right\|_2^2.
\]

因此，teacher 的全部比较不能被默认当作标量 cost 监督。相同动作对若在不同候选集合中被给出相反严格偏好，则不存在一个候选集合无关的标量代价能同时复现它们。这是本研究的出发点：先识别能被内化的关系，拒绝或单列不能内化的集合依赖关系。

以下都**不是**本工作的可单独主张：首次将偏好写入 future latent、首次做 student-induced OPD、首次蒸馏动作 Jacobian/tangent field、首次进行 pairwise-graph potential projection/HodgeRank，或将 contextual-bandit 的 OPD 收敛结论直接外推到 world-model MPC。潜在贡献只在于把下面四个约束共同闭合：student-induced candidate graph、可积决策势/非势残差、任务决策方向场、预测与既有成功行为的保护。

## 方法规格

### 1. Student-induced on-policy candidate graph

在 history \(h\)、目标 \(g\)、horizon \(H\) 下，当前 student 通过原生 MPC 生成
\(\mathcal C_k(h,g)\sim q_{\theta_k}(U\mid h,g)\)。候选应同时包含当前 elite、elite 局部扰动和少量远距离 bridge candidates；同一 root 的多个 candidate set 保留重叠 anchor actions。节点是具有明确动作序列的候选，边是冻结 teacher 对候选对的、经过尺度定义的关系。

这不是固定 action bank 的换名：每次模型更新后，训练根上都要重新生成候选图。只有在有限执行预算下比较固定 bank、初始-student replay、student-refreshed candidates 和 student-refreshed plus overlapping graph，才能声称 on-policy refresh 有贡献。环境 on-policy history 是更强的一层：定期在真实闭环中用当前 student 收集新 RGB histories；replay 不能被称为严格当前闭环分布。

### 2. 从关系 teacher 到可积 decision potential

对每条图边定义经校准的 teacher 差值 \(d^T_{ij}\)；仅有符号偏好时，使用排序不等式或经独立校准的 pairwise model，不能把所有胜负随意编码为 \(\pm1\) 后将残差解释为真实循环偏好。令 \(B\) 为 incidence matrix、\(W\) 为边权，学习零均值节点势：

\[
u_T^*=\arg\min_{\mathbf1^\top u=0}\left\|W^{1/2}(Bu-d_T)\right\|_2^2,\qquad
e_{\mathrm{nonpot}}=d_T-Bu_T^*.
\]

\(u_T^*\) 是可蒸馏的 candidate-set-independent decision potential；\(e_{\mathrm{nonpot}}\) 是不可由同一势函数解释的 teacher 关系，必须报告、隔离，不能硬塞进 scalar cost loss。图连通性和 \(\lambda_2(B^\top WB)\) 是查询与桥接候选的设计诊断，而不是论文装饰性统计。

### 3. 从势函数恢复受支持的局部 decision field

对 candidate \(U_i\) 的邻域，以 potential difference 拟合动作方向：

\[
g_T(U_i)=\arg\min_v \sum_{j\in\mathcal N(i)}w_{ij}
\left(v^\top(U_j-U_i)-(u^*_{T,j}-u^*_{T,i})\right)^2+\lambda\|v\|_2^2.
\]

这是一种**任务条件化的决策变化**，而不是动作造成的 future-world variation。它仅在局部扰动充分、势差可识别的子空间内有效；应记录估计条件数、方向覆盖和不确定性。没有支持的动作方向不得被伪装为连续真实 cost gradient。

### 4. 写入 student predictive geometry

student 代价的局部梯度为 \(g_S(U)=\nabla_U c_\theta(h,U,g)\)。为避免梯度对参数的混合二阶训练，首版采用受支持方向 \(v\in\mathcal S(U)\) 的有限差分：

\[
\mathcal L_{\mathrm{field}}^{FD}=
\sum_{(U,v)}w_{U,v}\left[
\frac{c_\theta(U+\epsilon v)-c_\theta(U-\epsilon v)}{2\epsilon}-g_T(U)^\top v
\right]^2.
\]

方向来自 elite differences、teacher 高决策影响方向和 bridge perturbations，而非无约束随机采样。完整目标为

\[
\mathcal L_{\mathrm{GDD\text{-}OPD}}=
\lambda_r\mathcal L_{\mathrm{relation}}+
\lambda_f\mathcal L_{\mathrm{field}}+
\lambda_p\mathcal L_{\mathrm{predictive}}+
\lambda_k\mathcal L_{\mathrm{keep}}.
\]

`relation` 匹配经 potential projection 后的合法关系；`field` 监督受支持的方向导数；`predictive` 保护真实 successor、多步预测与动作条件传播；`keep` 仅在独立真实执行证据确认 student 原有动作有效时保护该行为。realized 和 imagined 两条几何路径都必须评估。单一 terminal Euclidean cost 若不足以处理路径依赖、接触或不可逆成本，应作为明确的 function-class failure，另行测试 goal-conditioned/path-aware extension，不能暗改目标后仍声称原欧氏几何充分。

teacher 只在训练时查询；部署必须是 teacher-free 的 native student MPC，禁止 candidate ID、teacher rank、privileged state 或 fallback parity 充当 student gain。

## 可检验理论问题

1. **关系何时可内化为势？** 对可比较尺度的图边，投影残差给出任何单一势函数的最小拟合误差。这一最小二乘事实不是新定理；待研究的是图连通性、跨集合一致性、student cost function class 与未见候选误差之间的必要条件、充分条件或可证伪界。
2. **为什么 field supervision 可能转移到未见候选？** 若局部 teacher potential \(V_T\) 可微，则端点差的误差是路径上梯度误差的线积分。它只有在路径覆盖、平滑性和受支持方向成立时才提示泛化；端点上的少量 gradient matches 不能推出整条路径或全局 ranking 正确。理论应服务于有限 candidate sampling、distribution shift、unseen-action ranking 与 finite-horizon regret，而非给出脱离协议的 margin 不等式。

## 证据门与停止规则

| 阶段 | 需要锁定的事实 | 通过后才允许 | 失败的解释与行动 |
| --- | --- | --- | --- |
| **O0 Teacher provenance** | teacher/outcome 严格分离；root-disjoint split；candidate、goal、horizon、模型与查询预算的 fingerprint；无 outcome leakage | teacher-based pilot | 泄漏或无法独立审计时，只能作为 execution-supervised oracle pilot，不能声称 teacher distillation |
| **O1 Potential feasibility** | 重叠图连通、尺度合法、potential/non-potential decomposition、方向覆盖与条件数 | potential/field supervision | 高残差、断图或无支持方向时，拒绝对应关系；不靠加大 loss 强度掩盖 |
| **O2 On-policy coverage** | refresh 后候选分布漂移、elite/bridge 覆盖和 teacher query budget 的预注册比较 | student-refreshed training | 固定 bank 同样充分或 refresh 无收益时，限制为 offline distillation，不宣称 OPD 必要 |
| **O3 Student preservation** | realized 与 imagined alignment；多步 action-conditioned prediction；成功行为不被侵蚀 | 完整 co-training | 预测退化、成功动作被损害或仅 imagined 排序改善时，回退受限适配或拒绝方法 |
| **O4 Candidate-set transfer** | 未见 root、动作扰动、candidate set、goal/horizon 上的 pair ranking 和 elite regret；配对不确定性 | 闭环宣称候选泛化 | 仅训练候选改善时，field transfer 主张失败 |
| **O5 Independent deployment** | 新鲜、配对、root-disjoint、RGB-only、teacher-free native MPC；与强基线同预算；动作改变率、regret、success、latency/FLOPs | 方法论文的闭环主张 | 无独立 consequence、仅 fallback 对齐、或未超过强基线时，不称为更强 world model |

所有 learned-model 阶段必须在**主机 CUDA**上运行并记录 `torch.cuda.is_available()`、GPU 型号、driver/runtime、模型和数据 fingerprint；不得为完成门禁降级到 CPU。CPU 可以用于不加载学习模型的静态 schema、图代数和逻辑单测，但其结果不能替代 O0--O5 的学习或闭环证据。

## 首个决定性 pilot 与对照

先做 execution-supervised oracle pilot：在同一 PointMaze training split、同一 pretrained student、同一执行/teacher-query 预算和冻结测试候选协议下，比较 **pairwise GDD** 与 **potential plus decision-field GDD**。训练只读训练根的真实执行关系；测试根绝不被训练或阈值选择读取。输出包括 potential residual、方向覆盖、realized/imagined geometry、multi-step prediction、未见动作扰动与 MPC elite regret。

只有当 field 方法在保持预测的同时对未见候选有预注册的增量，才投入独立 relational teacher 和 environment-on-policy collection。随后用相同预算对照 pointwise score KD、pairwise KD、representation alignment、action-tangent distillation、fixed-bank/replay/student-refreshed sampling，以及不做 potential projection 的全关系蒸馏。对已验证成功动作，必须比较普通 OPD 与 success-preserving update，检查是否发生 distillation overshoot。

扩展顺序为 PointMaze（重置式候选对）→ Reacher（连续动作）→ PushT（接触条件）。PushT 进入前必须先有可信的原始闭环 baseline 与 oracle ceiling；零成功率不能区分蒸馏失败和 planner/action-interface 不可行。

## 实现与发布纪律

拟议独立入口位于 `le-wm-main/experiments/gdd_opd/`，可复用 `decision_benchmark` 的数据加载/评估接口，但不应由 Planning Compiler 导入。模块边界为 `teacher_evidence.py`、`decision_graph.py`、`potential_projection.py`、`decision_field_targets.py`、`train_gdd_opd.py`、`eval_gdd_opd.py`；具体实现仅在 O0 契约与 oracle pilot protocol 冻结后开始。

每项结果必须标为以下之一：`implementation/protocol repair`、`diagnostic-only`、`execution-supervised pilot`、`independent-teacher evidence`、`held-out transfer`、`independent deployment consequence` 或 `theory/method negative result`。代码问题、评估契约问题与真正方法负结果不能合并叙述。文章面向读者的主张只能使用已通过相应门的最弱充分表述。
