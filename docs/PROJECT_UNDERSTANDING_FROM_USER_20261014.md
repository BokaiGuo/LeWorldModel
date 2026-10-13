# LeWorldModel 项目理解记录（用户版）

> 记录日期：2026-10-14
> 来源：项目负责人对 LeWorldModel 原始研究文档、数学理论、实验代码、负结果记录和研究路线演进的综合理解。
> 文档性质：研究定位与问题理解记录，不是新的实验结果，也不替代仓库中的证据、协议和结果 artifact。

## 1. 项目的真正研究问题

本研究不是重新发明一个 LeWorldModel，不是单纯为 LeWM 添加谱分析指标，也不是只解决 PushT 的规划失败。真正的问题是：

> 一个从视觉数据中学到的 latent world model，究竟在什么条件下保留了足以支持真实物理动作、有限时域预测、跨任务迁移和闭环规划的信息？我们能否在执行动作之前，可靠识别这些条件，并通过合适的表示、记忆和动力学干预改善或限制它们？

这是一项横跨表征学习、动力系统、控制理论、可观测性、因果辨识、谱几何和模型预测控制的研究。

项目的核心转变是：早期试图用 covariance–predictor alignment 直接解释 world-model quality；在理论反例和 PushT 失败案例之后，主线转向更严格的 **action-to-task transfer sufficiency**。研究不再主要问 latent 是否“看起来学得好”，而是问 latent 中的动力学信息是否足以让 planner 做出真实有效的动作选择。

## 2. 决策充分性，而不只是预测能力

传统 world model 通常学习

\[
z_t=f_\theta(o_t),\qquad \hat z_{t+1}=F_\theta(z_t,a_t),
\]

并最小化 latent prediction loss。但本项目真正关心的是：

\[
\arg\min_U \widehat J(U)\stackrel{?}{\approx}
\arg\min_U J_{\mathrm{real}}(U).
\]

也就是说，模型预测出来的世界是否足以让 planner 在真实世界中选择正确动作。

一个 latent predictor 可以有低 MSE，却无法预测推动物体时的旋转方向；也可以准确预测大量与任务无关的视觉变化，却丢失少数决定任务成败的接触信息。因此，项目贯穿区分三个概念：

\[
\text{statistical predictability}
\neq
\text{physical identifiability}
\neq
\text{decision sufficiency}.
\]

## 3. 统一研究架构

```text
Frozen visual world model
  RGB -> latent representation -> learned predictor
        |
        v
Spectral geometry
  covariance, alignment, commutator, non-normality
        |
        v
Causal memory
  delay, Fourier, phase, contact-event history
        |
        v
Action transfer
  action sensitivity, identifiability, task readout
        |
        v
Physical adapter
  contact, pose, signed rotation, successor modes
        |
        v
Finite-horizon transfer sufficiency
  prediction error + action uncertainty + task sensitivity + risk
        |
        v
Selective closed-loop visual MPC
  trust/reject -> execute -> RGB verify -> replan
```

这些模块共同回答一个问题：如何让冻结的视觉表征从“能够预测”变成“能够被验证地用于控制”。

## 4. Latent 几何与有限时域传递

统一对象不是单独的 covariance 或 predictor spectrum，而是

\[
\mathcal W_\Phi=(C_\Phi,A_\Phi,B_\Phi,D_\Phi,M_{\mathrm{task}}),
\]

其中：

- \(C_\Phi\)：冻结表征或历史提升表征的统计协方差；
- \(A_\Phi\)：learned latent dynamics 的局部传播算子；
- \(B_\Phi\)：动作在 latent 中的响应；
- \(D_\Phi\)：latent 到外部任务变量的读出；
- \(M_{\mathrm{task}}\)：预先声明的任务度量，而不是模型自己定义的误差标准。

表示空间的几何不天然具有坐标不变性。改变等价 latent 坐标系可能改变普通 covariance、principal angle 或 commutator 数值，但不改变物理行为。因此必须使用 gauge-aware metric、whitening 约定和 task pullback metric，并明确哪些量具有表示不变性。

同时，eigenvalues 不能充分描述有限时域预测。非正规算子可以具有稳定特征值，却在短时间内产生 transient amplification；而 planner 工作在有限 horizon。高方差或谱增益方向如果不能被动作激发，或不能被任务读出，也不一定具有控制价值。

因此，研究对象逐步从 representation-centered geometry 转向

\[
G_h^\Phi=D_\Phi A_\Phi^{h-1}B_\Phi,
\]

即局部线性近似下，一次动作扰动经过 \(h\) 步对任务输出的影响。完整动作序列还需要由不同时间位置的 action-to-output operators 构成的有限时域传递矩阵。

## 5. Frozen latent 的非 Markov 性与记忆提升

单帧图像可以显示物体在哪里，却不一定说明物体运动方向、此前发生过什么碰撞、当前处于 stick/slip 哪种接触模式。因此仅用

\[
z_{t+1}=F(z_t,a_t)
\]

可能不存在准确的确定性映射：相同 \(z_t\) 可能对应多个隐藏物理状态。

这正是 Mori–Zwanzig closure、delay embedding、Hankel/Fourier memory、phase representation 和 event-gated GRU 等路线的研究动机。统一的历史提升为

\[
\eta_t=\Phi(z_{t-L:t},a_{t-L:t-1}),
\qquad
\eta_{t+1}\approx A_\Phi\eta_t+B_\Phi a_t+r_t.
\]

其中 \(r_t\) 不只是普通噪声，还包括历史截断误差、未观测接触模式、非线性项和离开训练流形后的误差。

PushT memory audit 中 event-gated GRU 没有稳定超过简单 Fourier baseline，说明：

> 增加可训练记忆容量，不等于找到正确物理状态，也不等于恢复可控性。

因此 memory lifting 必须被当作物理充分性机制来验证，而不是默认有效的网络模块。

## 6. PushT 的作用：物理因果诊断台

PushT 不只是 benchmark，而是暴露视觉 latent 物理缺陷的机制实验台。它同时检验：

- 视觉表示是否保留物体姿态；
- 动作是否产生正确的接触后果；
- 模型是否理解平移与旋转的耦合；
- 不同接触模式下的 action-to-task transfer 是否可辨识；
- 多步优化是否把小误差放大为真实控制失败。

把推杆移动到正确位置，不等于物体最终达到正确姿态。接触发生、接触持续、物体中心位移和物体角度闭合是不同的动力学事件。

这也是项目关注 \(SE(2)\)、contact mode、signed torque 和 successor consistency 的原因。若 latent 无法区分在相同动作下产生相反旋转的两种接触模式，换用更大的 MLP、GRU 或 Transformer 也不能补回缺失的信息；需要历史观测、动作干预、状态辨识甚至主动探测。

## 7. 自知不可信的 planner

项目最终目标不是让 MPC 在所有状态都盲目输出动作，而是形成一种有条件授予规划权限的控制协议：

\[
U^*=\arg\min_{U\in\mathcal U_{\mathrm{cert}}}
\left(\widehat J(U)+\lambda\widehat R_H(U)\right).
\]

\(\mathcal U_{\mathrm{cert}}\) 必须由独立证据证明在当前任务、状态和 horizon 下足够可靠。核心条件包括动作支持、动作效应可辨识性、真实与预测 successor 一致性和有限时域误差风险。

因此，规划能力应被看作一种 permission，而不是只要模型能输出预测就默认拥有的能力。当信息不足时，系统应识别缺失的是历史、动作激励、接触模式还是任务读出，然后通过新的观测或干预补足信息，或拒绝无根据的长时域规划。

## 8. 对已有结果与负结果的统一理解

合成 spectrum-matched 系统用于检验：控制 predictor spectrum 后，不同 covariance/predictor 几何是否仍导致不同 transfer behavior。这具有理论辨识价值，但不能直接证明真实 JEPA/DINO backbone 上存在同样规律。

PointMaze 用于把谱与动态诊断连接到 learned world-model 的离线决策代理任务。早期结果显示相对 fixed horizon=1 的改善，但更严格的独立 planning-gain 审计未通过统计门槛；这说明在记录轨迹上选择正确 horizon，不等于真正完成 goal-conditioned action planning。

JEPA/DINO adapter panel 检验诊断条件能否区分不同 frozen representation 上的学习结果，但 checkpoint 数量和匹配条件仍不足以支持普适模型判别规律。

这些负结果指向一个统一解释：

> 预测质量、表征几何、历史可观测性、动作效应可辨识性和决策可靠性不是同一个变量。任一代理指标都可能在某一层表现良好，却没有跨过下一层所需的信息边界。

## 9. 六个目标在统一框架中的位置

| 目标 | 研究作用 | 主要科学障碍 |
| --- | --- | --- |
| Alignment 预测规划性能 | 建立决策质量的可预测性 | 静态几何能否独立解释真实 action regret |
| Adapter 稳定姿态闭合 | 验证缺失动力学结构能否修复 | 接触模式、signed rotation、非 Markov 状态 |
| 物理因果 latent | 建立预测与真实干预后果的对应 | 动作辨识、支持集覆盖、隐藏混杂与状态混叠 |
| Frozen LeWM 视觉在线规划 | 将诊断落到闭环控制 | 目标条件化动作选择、误差积累与在线校准 |
| 普适 world model | 测试理论和模型的泛化范围 | 架构、动力学、观测和动作空间是否共享充分条件 |
| 通用 transfer certificate | 给出何时可信任模型的理论保证 | 单独 commutator 信息不足以认证 transfer |

这六项不是六个互不相关的论文主题：第 1 项研究预测风险，第 3 项确认信息是否具有物理意义，第 6 项给出可靠性边界；第 2 项尝试修复失败原因，第 4 项检验控制价值，第 5 项检验可推广范围。

必须区分“方法在不同模型上普适可用”和“同一个 pretrained world model 能控制所有环境”。前者是本项目自然的研究对象，后者不能由诊断框架自动推出。第 6 项也不能再以证明 commutator 无条件通用为目标，而应证明其适用条件和所需补充信息。

## 10. 最有潜力的原创贡献

项目最有潜力的原创问题是：

> **Decision Transfer Identifiability of Frozen Visual World Models**：给定冻结视觉表示、有限的独立动作干预数据和外部声明的控制任务，能否判断哪些真实动作后果能够从 latent 中被辨识、能以多大精度预测，以及误差何时足够小到可以支持可靠在线动作选择？

它包含三个层次：

1. **信息是否存在**：若两个物理状态编码为同一 latent，却在相同动作下产生不同任务后果，则任何只读取该 latent 的确定性 predictor 都无法同时正确预测。
2. **信息是否可辨识**：即使 latent 保留相关信息，训练动作缺少方向覆盖时，也不能可靠辨识 \(B_\Phi\) 或 action-to-task response；低预测误差不能替代 persistent excitation。
3. **信息是否足以决策**：即使动作响应有有限误差，还需判断误差是否改变候选动作排序，以及 planner 是否会把小偏差放大为大 regret。

三者共同定义 transfer sufficiency。信息不足时，系统不应只依靠更大模型猜测，而应识别缺口、增加观测或干预，或拒绝无依据的长时域规划。

## 11. 当前最关键的证据缺口

最大的瓶颈不是缺少新的数学工具，而是尚未完全连通以下关系：

\[
\underbrace{\text{Latent diagnostics}}_{\text{表征与谱结构}}
\not\Rightarrow
\underbrace{\text{Identified action effects}}_{\text{物理干预}}
\not\Rightarrow
\underbrace{\text{Correct action ranking}}_{\text{反事实决策}}
\not\Rightarrow
\underbrace{\text{Closed-loop success}}_{\text{在线控制}}.
\]

每一个箭头都需要独立证据。最关键的实验必须在相同初始状态执行真正不同的动作序列，使用真实后果评估 action ranking，再把诊断、干预校准、风险门控和 RGB-only MPC 放进同一套预先固定的测试协议。

## 12. 项目身份与成熟度

项目不应被定位为 “Spectral Analysis of JEPA World Models”，因为这会忽略物理辨识、部分可观测性和控制可靠性；也不应被缩小为 “A Better World Model for PushT”。更准确的定位是：

> **LeWorldModel 研究冻结视觉世界模型的决策充分性：通过研究 latent representation、可观测性、动作因果响应与有限时域动力学之间的关系，识别模型何时具有可靠的物理规划能力、何时缺少必要信息，以及如何通过历史提升、可辨识动力学适配和风险认证来恢复或限制其控制能力。**

最终追求不是建立一个预测得更准确的 world model，而是建立一种能够判断 world model 是否理解了与当前决策有关的世界、为什么理解或不理解，以及何时值得信任它执行动作的理论与方法。

当前成熟度应准确描述为：问题定义和理论结构已经比较完整；物理辨识、learned adapter、跨环境评价和真正在线规划之间，仍在建立决定性证据连接。这不是六项成果已经全部实现，而是六项目标已经形成一套相互关联、可被检验的研究框架。

## 13. 与仓库证据的关系

以上理解基于公开仓库中的研究文档、代码实现、发布实验记录和审计材料；它不是对所有 checkpoint 的独立重新运行。结果判断必须继续区分代码可确认事实、仓库报告的实验结论、代理/合成证据和尚未通过门禁的主张。

当前仍需保持的边界包括：alignment 尚未被证明能普遍预测 learned-world-model 规划性能，adapter 尚未稳定完成姿态闭合，latent 尚未被证明具备完整物理因果动力学，frozen LeWM 尚未完成纯视觉在线规划，普适 world model 尚未建立，commutator 不是通用 transfer certificate。
