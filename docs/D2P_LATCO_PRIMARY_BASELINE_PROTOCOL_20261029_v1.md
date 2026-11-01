# D2P 强基线登记：LatCo 原论文与可比性

日期：2026-10-29。本文是独立新增登记，不修改旧清单、代码或结果；没有运行新实验。证据状态：`paper_reported/context_only`，不是本地复现或同协议效能结论。

## 覆盖核查与新增对象

[既有四论文清单](D2P_REPORTED_BASELINE_NUMBERS_HANDOFF_20261027.md)已登记 GRASP、Hi-LeWM、LeWM 和 PlanWAM 作者表；[覆盖复核](D2P_BASELINE_COVERAGE_PAULI_20261028_v2.md)记有四篇原表 registry 和本地 moving 六臂。LatCo 并非完全未出现：GRASP 的 Push-T 表已有 LatCo **改编版**数值，GRASP §4.1 明确说其设置不同于原版 MBRL。本次补充的是 **LatCo 原论文的作者实验及协议**，不是将原有 LatCo 行冒充新基线。核查范围为上述登记及 `docs` 中 LatCo/collocation 检索，不宣称仓库所有位置均不存在任何提及。

选择 LatCo 的原因：它直接联合优化中间 latent states 和动作、允许规划中暂时违反动力学再逐渐收紧，是检验动作空间 shooting/局部修复是否遗漏长程可行轨迹的实质竞争方案，而非换个优化器的同一动作搜索。原文见 Rybkin 等，*Model-Based Reinforcement Learning via Latent-Space Collocation*, ICML 2021，固定 [arXiv:2106.13229v2 PDF](https://arxiv.org/pdf/2106.13229v2)，Table 1–2、§5.1–5.3，PDF 物理页 6–7；方法和训练见 §3.5、Algorithm 1，物理页 5。此次实际读取 PDF，SHA-256 `6efcbf3546888e0c0ef89021bebf277a4c336505517fab31eb3450ffcf877288`。标题、作者与 arXiv 元数据一致；发表信息以原文首页为准，未核 DOI。

## 作者原文数值：不同任务，各自保留原表

Table 1（物理页 7）：视觉 Sparse MetaWorld，作者表中的百分数，± 统计含义及训练 seed 数在本次核查的表注和相邻正文中未确认，原样保留，不称 SD、SEM 或 CI。

| 方法 | Reaching | Button | Window | Drawer | Pushing |
|---|---:|---:|---:|---:|---:|
| LatCo | 91 ± 3% | 55 ± 4% | 49 ± 8% | 46 ± 3% | 38 ± 3% |
| PlaNet | 20 ± 1% | 13 ± 2% | 31 ± 2% | 22 ± 2% | 22 ± 3% |
| MPPI | 16 ± 1% | 10 ± 2% | 30 ± 2% | 21 ± 3% | 21 ± 4% |
| Shooting GD | 8 ± 1% | 7 ± 0% | 28 ± 1% | 18 ± 2% | 19 ± 3% |

Table 2（同页）为另一种预收集数据加在线数据的协议：Thermos 的 LatCo `37 ± 21%`、PlaNet `1 ± 1%`，Hammer 的 LatCo `13 ± 2%`、PlaNet `1 ± 0%`。不能将两个表视为相同训练数据或同一测试人群。论文 Table 1 同时有 DM Control 的密集奖励数值；它们不是成功率，不在此与百分数混排。

原文 §5.1 指出 MetaWorld 视觉观测、稀疏奖励（任务完成给 1，否则 0），机器人和物体位置随机初始化。除 Pushing、Thermos、Hammer 外 H=30、Tcache=30、Ttot=150；这三项为 H=50、Tcache=25、Ttot=150。Thermos/Hammer 动作 repeat=2，其他 MetaWorld 任务不 repeat。H、Tcache 与 Ttot 见原文定义，不能直接换算成本地 packet 或 predict calls。算法按 Tcache 执行动作后重规划并将新数据加入 replay buffer、更新动力学模型；§5.2 表 1 的 sparse tasks 从头在线学习，§5.3 的 Thermos/Hammer 先用含成功和失败轨迹的预收集数据初始化 buffer，再在线训练；后者评估时对 LatCo 和 Shooting GD 多次初始化选优。论文比较使用相同动力学**架构**，不等于各方法使用同一冻结权重或相同总算力。

优化器使用 Levenberg–Marquardt、潜状态及动作约束的乘子更新；论文比较也包括 MPPI、CEM/PlaNet 和 shooting GD。原论文的具体 checkpoint、逐任务数据与测试 seed 身份、完整前向/反向调用、总墙钟/硬件成本及表格 ± 的精确定义，本次未闭合，记 `unknown`；不据架构相同推断预算匹配。

## 既有作者复测：与 LatCo 原文分源

GRASP 论文 [arXiv:2602.00475v1 §4.1](https://arxiv.org/html/2602.00475v1#S4.SS1) 明言其 LatCo 设置不同于原方法；[Table 1](https://arxiv.org/html/2602.00475v1#S4.T1) 的 Push-T H40 `15.0% / 598.0 s` 是 GRASP 作者对改编方法的报告；成功率为 learned-world-model 内成功，时间为成功试验的中位数，非物理成功。既有[原表登记](D2P_REPORTED_BASELINE_NUMBERS_HANDOFF_20261027.md)包含其余 horizon 与分母限定。这里不把原 LatCo 的 MetaWorld 值、GRASP 的模型内值和本地 native success 合并或相减。

## 本地值与可比性登记

本轮检索未确认官方 LatCo 本地实现或本地 LatCo 评估收据；状态 `not_verified`，不是零分，也不是断言仓库中绝不存在。已有 [moving 六臂保存结果](../le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json) 是本地校准：6 方法各 2 roots，`comparison_recorded_not_admitted`；例如两个 `moving_gradient_open_loop` native_success 均为 false，model_Q 为 0.261319、0.072011，realQ_terminal10 为 0.454880、0.046496。它们是本地两条记录，不是 LatCo 值、不是概率估计，也不能将 model_Q 换作物理成功。其他本地方法和预算见 [执行矩阵](D2P_BASELINE_EXECUTION_MATRIX_JASON_20261028_v1.md)；本次仅只读记录，未重新运行或独立认证整个计算链。

| 对齐字段 | LatCo 原文 | 本地 moving / 待确认 | 判定 |
|---|---|---|---|
| 任务与观测 | Sparse MetaWorld，视觉，稀疏完成奖励 | moving PushT，本地 RGB/goal 与终端任务 | 不同任务、目标与 evaluator |
| 模型与训练 | PlaNet 类 latent dynamics/reward；在线训练；Table 2 另有离线初始化 | 冻结 LeWM；表级 checkpoint 和输入合同需锁定 | 模型及训练不等；不能称 planner-only 原样对比 |
| 控制与时间 | H=30/50；按 Tcache MPC 重规划；部分任务 repeat 2 | H2 与实际动作 packet、观察和重规划合同另行绑定 | 时间语义不等 |
| 测量与统计 | 作者任务百分数；± 定义此次未证实 | native_success、真实后继 realQ 和模型 model_Q 分别保存；两根校准 | 指标和独立单位不等；不作跨论文差值 |
| 规划预算 | LM、latent/action/dual 优化；完整计算账未确认 | 每臂 predict、反向、物理执行及墙钟需单独核账 | 未匹配，禁止等算力结论 |

若需要将 LatCo 作为本地强竞争对照，先核实作者代码版本与许可、模型/权重及必要依赖；预声明是 **原版完整管线**还是 **固定 LeWM 的 LatCo-style adaptation**。后者需明确 latent 参数化、目标成本与动力学约束、动作界限、初始化、LM/乘子更新、终止条件及最终可执行动作重放；不能将 restart-Adam、joint repair 或 latent waypoint 自行更名为官方 LatCo。固定同一 checkpoint、goal/roots、动作语义、观测频率和物理 evaluator，预先同时计量前向样本与调用、反向/Jacobian、优化与编码、物理反馈和墙钟/硬件，另设根独立的评估；若某轴无法匹配，报告资源–结果 trade-off 而非等总计算。需要单独登记约束残差和失败/不可行率，避免把不满足动力学的想象轨迹当可执行成绩。

允许引用：LatCo 原文在其在线视觉 MBRL Sparse MetaWorld 协议下报告上述结果；GRASP 作者另报告一种改编 LatCo 的模型内 Push-T 结果。禁止声称这两者为同一实现、原文结果已本地复现、D2P 优于 LatCo，或本地两根成功率代表总体性能。
