# 锐评研究缺点

> 请你作为ICLR/ICML/NIPS审稿人锐评一下这个研究的缺点并且补充需要做什么才能达到oral/best paper级别研究[BokaiGuo/LeWorldModel: Transfer-sufficient diagnostics and reproducible evidence for frozen visual world models](https://github.com/BokaiGuo/LeWorldModel)

我会按顶会强审稿人的标准来评估：先核对仓库中的方法、实验、代码和证据链，再对照近期 world model / representation learning 的相关工作，区分“当前会被拒的硬伤”和“真正有机会冲 Oral / Best Paper 的突破点”。我会尽量具体到应补的理论、实验、对照组和可验证的研究假设。

发现了一个对审稿结论非常关键的细节：仓库报告的 PointMaze 显著增益，是相对于固定 horizon=1 的基线；在同一份诊断结果里，固定 horizon=4 的平均损失反而低于自适应诊断策略。这会直接影响“方法带来了规划收益”的论证。我正在核对这种差异是否由评估协议或任务条件造成，而不是只看 README 中的显著性数字。

代码层面又确认了两处比“样本量小”更严重的问题：当前 PointMaze 的所谓 planning outcome 实际是离线日志里的**预测选 horizon、真实位移打分**，并不是执行新动作后的闭环规划；而名为 `action_inverse_risk` 的校准量，在关键实验代码中实际上由 latent prediction MSE 计算得到。这意味着论文最重要的“决策充分性”和“动作可辨识性”主张，还没有被相应的实验直接验证。

# LeWorldModel：以 ICLR / ICML / NeurIPS 强审稿人视角进行评审

<badge color="warning">模拟审稿结论：Weak Reject / Reject</badge>

**一句话锐评：这个项目已经具备一个严肃研究项目的工程骨架，但还没有完成最重要的科学证明——它能否真正判断一个冻结视觉 world model 什么时候值得信任，以及这种判断能否改善实际决策。**

我核查了截至 **2026 年 10 月 13 日**的公开仓库，包括 README、研究计划、理论文档、实验审计、PointMaze 发布结果，以及训练与规划评估的关键 Python 源码。同时对照了 LeWM、DINO-WM、V-JEPA 2 和 decision-aware model learning 的相关论文。

这是基于公开材料的模拟审稿，不是对完整投稿论文或全部训练运行的独立复现。

## 一、综合评价

| 维度 | 我的判断 |
|---|---|
| 研究问题的重要性 | 高：预测准确不等于决策有效 |
| 创新性 | 中等：有好的问题重构，但核心理论与既有控制、状态抽象工作重合较多 |
| 理论成熟度 | 有条件性结论，但还不是实际规划性能定理 |
| 实验证据 | 明显不足，尤其缺少真实闭环决策和跨环境验证 |
| 代码与审计规范 | 相对扎实，主动保留负结果是优点 |
| 当前 Oral 竞争力 | 很低 |
| 当前 Best Paper 竞争力 | 尚未形成标志性理论或实证发现 |

这里最值得肯定的是：仓库没有把负结果全部隐藏。它明确承认当前没有完成跨环境证据、稳定视觉规划和物理因果验证。这是比堆砌漂亮指标更健康的研究习惯。<Cite ref="turn162697view0"/>

但顶会并不会因为“知道自己缺什么”就接受论文。最终要看解决了什么此前无法解决的问题。

## 二、最致命的缺陷

### 1. 目前的 planning gain 不是严格意义上的 planning gain

这是我作为 Reviewer 最可能提出的首要 Major Concern。

仓库里的 PointMaze 独立评估函数做的是：在已经记录好的轨迹上，对 horizon 为 1、2、4 的预测任务位移进行排序，再利用对应真实未来状态给选择结果打分。

其任务方向并非真实 maze goal，而是从训练轨迹的平均位移方向估计出来的。代码甚至明确说明 PointMaze export 不含目标信息。<Cite refs={["turn285378view0","turn669546view0"]}/>

因此，当前的评价实际上更接近：

$$

\text{Which recorded horizon gives better projected displacement?}

$$

而不是：

$$

\text{Which action should the agent execute to reach the goal?}

$$

这是非常不同的两个研究问题。

更严重的是，候选集合主要改变的是观察未来多久，而不是在相同时间预算内比较不同反事实动作。

一个模型可能只是学会了“更长时间通常移动得更远”，就会取得不错的任务位移指标。这并不证明它理解了迷宫、障碍物、动作后果或者目标导向规划。

**审稿人会要求：必须出现真正的 action ranking、goal-conditioned planning 和环境交互结果。**

### 2. 你的显著正结果输给了一个非常简单的固定 horizon 基线

这部分是我核对公开 JSON 和评估代码后发现的最值得优先处理的问题。

仓库报告某个 JEPA PointMaze checkpoint 相比固定 horizon=1 的改善为：

<text color="secondary" size="xs">公开实验记录：32 个 paired outcomes，均值损失越低越好</text>

| 方法 | Mean loss |
|---|---:|
| Fixed h=1 | 0.53739 |
| Fixed h=2 | 0.33109 |
| Fixed h=4 | **0.22908** |
| Adaptive diagnostic | 0.36604 |

<Chart content={{"chartType":"bar","xKey":"method","series":[{"dataKey":"loss","label":"Mean loss (lower is better)"}],"yAxisMin":0,"data":[{"method":"Fixed h=1","loss":0.53739},{"method":"Fixed h=2","loss":0.33109},{"method":"Fixed h=4","loss":0.22908},{"method":"Adaptive","loss":0.36604}]}}/>

这意味着：

- 相比 Fixed h=1，Adaptive 改善了 **0.17135**。
- 相比 Fixed h=4，Adaptive 的损失反而高了 **0.13696**。
- Adaptive 对 h=1 的逐 episode 正向改善只出现在 **17/32** 个样本中。

这个比较来自发布的 `failure_diagnostic.json`，对应特定 JEPA checkpoint 的未门控诊断策略。不能直接外推到后续全部 permission-gated checkpoint，但足以说明：**README 中的正向显著性不能被解释为已经超过最强简单基线。** <Cite ref="turn707796view2"/>

后续 permission-gated JEPA panel 报告 gain 为 0.09803，但仍然需要在完全相同的评估条件下补齐它与固定 h=2、h=4 等基线的比较。<Cite ref="turn665647view1"/>

真正的审稿问题是：

> Why should we regard this as a planning improvement when a trivial fixed-horizon strategy can outperform the proposed diagnostic on the reported audit?

这不是简单增加 seed 可以解决的，而是 **evaluation target 和 baseline choice 必须重新定义**。

### 3. 理论中的 action identifiability 与实际实验 gate 没有闭合

HTSC 的理论规范要求分别验证 action support、action identifiability、真实/预测 successor consistency 和 finite-horizon risk。

这个方向是合理的。

但我检查 `train_pointmaze_learned_adapter_panel.py` 后发现，实验中所谓的 `calibration_action_inverse_risk` 来自：

$$

\operatorname{mean}\left[
(\hat z_{t+1}-z_{t+1})^2
\right]

$$

也就是校准集上的 latent prediction MSE。<Cite ref="turn956017view0"/>

**预测误差低并不意味着动作效应可辨识。**

例如，若训练动作几乎不变化，模型完全可以把下一状态预测得很好，但仍无法知道正反方向控制输入会造成什么差异。

真正需要估计的是在动作支持集上的 $B$ 或 action-to-task Jacobian 的可辨识性，包括秩、条件数、动作覆盖以及估计不确定性。

另外，当前 `horizon_sufficient_control.py` 主要消费各个 gate 的布尔结果，本身并没有从真实视觉观测中完成所有 gate 的统计认证。

因此，必须把三种东西严格区分开：

**理论上的证书条件 ≠ 软件中的 gate 接口 ≠ 已经在 learned world model 上验证的可靠证书。**

否则论文很容易被评价为：

> The paper presents an appealing certification interface, but the empirical implementation does not establish that its required quantities can actually be identified reliably.

### 4. “Transfer-sufficient” 的理论贡献还不够强

仓库强调的决策相关算子是：

$$

G_h^\Phi=D_\Phi A_\Phi^{h-1}B_\Phi

$$

它是一个重要的有限时域 impulse-to-output map。

但要注意：

**单个 $G_h$ 并不刻画完整的多步动作序列影响。**

在线性系统下，对整段动作序列的影响需要完整的 Markov parameter block：

$$

\mathcal G_H=
\begin{bmatrix}
DA^{H-1}B & DA^{H-2}B & \cdots & DB
\end{bmatrix}

$$

对于非线性接触系统，还需要分析状态和动作依赖的 Jacobian，以及 contact-mode transitions。

现有有限时域误差定理，本质上通过误差递推和三角不等式给出上界。这个结果数学上有价值，但它假定真实误差 envelope 已经存在且有效。<Cite ref="turn162697view4"/>

这距离证明“控制策略有效”还有明显差距。

我希望看到的不是单纯：

$$

\|y_{t+H}-\hat y_{t+H}\|\le \epsilon_H

$$

而是一个具有有限样本保障的决策结论，例如：

$$

J(\hat u)-\min_{u\in\mathcal A}J(u)
\le 2\max_{u\in\mathcal A}\epsilon_H(u)

$$

这里假设所有候选动作的预测代价都获得了同时有效的误差上界。

这个 regret 结论本身也很经典，真正有创新空间的是：

**如何仅从冻结视觉表示、有限动作干预和校准数据中，构造非空洞、可计算、能适应 planner-induced distribution shift 的 $\epsilon_H$。**

还有一个应修正的数学表述：仓库说对 normal $A$，谱基线增益与实际增益相等。这一般不成立。例如 $A=\operatorname{diag}(0.9,0.1)$、$D=[0,1]$ 时，一步 task gain 为 0.1，而对应谱基线为 0.9。正确结论是 transient excess 可以为零，而非必然相等。

这不推翻整体上界，但理论论文必须把这种细节校对干净。

### 5. 实验中的 JEPA/DINO 区别还不能推出强泛化结论

当前的 JEPA/DINO 对照主要是少数冻结表示导出上的 adapter 训练。

多个 adapter seeds 和 training stages 并不是多个独立预训练的 world model。

仓库本身也明确说明，发布的冻结 export 不含多组独立 learned-backbone seeds。<Cite ref="turn110107view0"/>

因此，“JEPA 的若干 adapter checkpoints 通过 gate，而 DINO 的若干 checkpoints 被拒绝”是一个值得研究的现象，但不是已经确定的架构级定律。

特别是部分表示诊断仅使用很少的校准样本，JEPA 和 DINO 的 action-effect calibration $R^2$ 都存在明显负值。<Cite ref="turn707796view1"/>

需要进一步排除表示尺度、decoder、action normalization、训练预算和 calibration threshold 的影响。

而且 DINO 被拒后退回 h=1，其零增益是 fallback 机制的结果，不代表已经验证了安全规划，更不应与 ICML 2025 的 DINO-WM 整套 action-conditioned planning 系统混为一谈。

### 6. Readout 中还有一个值得立即修复的具体问题

当前 `_task_readout` 将 latent 特征中心化，却没有显式估计 displacement target 的截距。

也就是说，模型形式相当于：

$$

\hat y=(z-\mu_z)W

$$

而不是：

$$

\hat y=(z-\mu_z)W+b

$$

当训练集平均位移非零时，前者会强制训练集预测位移的均值为零，造成系统性偏差。

这也是我建议重新检查全部 PointMaze gain 的原因：有可能部分 horizon 排序错误并不来自 world model 缺乏 transfer sufficiency，而只是下游 readout 校准不充分。

应该补一个有截距的 ridge decoder、零位移校准、各 horizon 的 bias/variance 分解，然后用完全冻结的新测试集重跑。

---

## 三、为什么目前还很难达到 Oral？

除了上述实验缺陷，还有一个更深层的问题：**目前的新意分散在太多方向。**

你同时在做 spectral diagnostics、covariant metric、finite-horizon theorem、memory lifting、contact modeling、adapter selection、risk gating 和 planning。

这些模块各自都能解释为什么某些方法失败，但还没有围绕一个真正全新的核心机制，构成不可替代的科学发现。

而已有工作已经占据了相当大的理论空间：

| 相关工作 | 已经解决或明确提出的核心思想 |
|---|---|
| <Entity category="other" value="Value Equivalence Principle" disambig="NeurIPS 2020 model-based reinforcement learning paper"/> | 世界模型应保留对下游决策有意义的信息，而不只是精确重建状态 <Cite ref=""/> |
| <Entity category="other" value="DeepMDP" disambig="ICML 2019 reinforcement learning representation paper"/> | 潜在状态与奖励预测误差如何关联控制表征的质量 <Cite ref=""/> |
| <Entity category="other" value="DINO-WM" disambig="ICML 2025 pretrained visual-feature world model"/> | 从离线数据训练视觉动力学模型，并进行 zero-shot action optimization <Cite ref=""/> |
| <Entity category="other" value="V-JEPA 2" disambig="2025 self-supervised video representation and robot planning research"/> | 通过 action-conditioned post-training 支持实际机器人图像目标规划 <Cite ref="turn683700academia73"/> |
| <Entity category="other" value="LeWorldModel" disambig="2026 JEPA-based visual world model paper"/> | 原始模型已展示像素输入的 latent planning 与控制性能 <Cite ref="turn683700academia72"/> |

因此，论文不能把“prediction error 不等于 control quality”当作主要创新。这在理论和经验上都已有大量先例。

真正值得争取的 novelty 是：

**能否提出一个此前不存在的、可操作的判别准则，在不重新训练视觉 encoder 的情况下，预先判断一个 world model 是否能够可靠支持下游动作选择，并给出成功或失败的证据。**

如果能够做到这一点，研究才会从“诊断工具集合”升级为一个有清晰方法贡献的论文。

## 四、我建议的升级路线

不要继续无差别扩充模块。建议把研究收敛成一个主线：

<title size="lg">When Is a Frozen Visual World Model Sufficient for Decision-Making?</title>

论文核心命题可以变成：

> Frozen visual world models should be evaluated by identifiable action-to-task effects and calibrated decision risk, rather than latent prediction quality alone.

围绕它建立三个可证伪假设：

1. 在相近的 latent MSE 下，不同表示可能产生显著不同的 action ranking 和 regret。
2. Action-to-task transfer、历史可观测性与动作支持能够解释这种差异，并在独立任务上预测失败。
3. 基于上述量构造的可信度门控，可以改善实际闭环规划的风险—覆盖率表现，而不只是选择保守的 fallback。

### 按优先级必须完成的工作

| 优先级 | 必做实验或方法 | 验收标准 |
|---|---|---|
| P0 | 修正 readout 截距、split 检查、action identifiability 的实现；重新定义 goal-conditioned outcome | 排除当前的评估与实现混淆 |
| P0 | 相同时间预算、相同候选数量下比较不同动作序列，而不是只比较不同 horizon | 明确得到 action-ranking / regret 指标 |
| P0 | 与最优固定 horizon、随机候选、简单 task decoder 和原始 LeWM planner 比较 | 胜过强基线而非仅胜过 h=1 |
| P1 | 使用独立的正反动作干预数据，验证 action effect、support 和 mode observability | 预测的 transfer risk 能解释真实反事实误差 |
| P1 | 至少覆盖导航、接触操控和另一类不同动力学任务 | 独立环境中的泛化规律成立 |
| P1 | 不同 backbone、真实训练 seed、leave-one-environment-out | 不是单个冻结 checkpoint 的偶然现象 |
| P1 | 真正 RGB-only、goal-conditioned、receding-horizon planning | 展示 success rate、regret、失败率与控制代价 |
| P2 | 有限样本、支持集感知的风险证书与真实 coverage 检验 | 报告 false-release rate、abstention rate 和 bound tightness |
| P2 | 将理论 certificate 与实际 control gain 联系起来 | 理论预测的安全/性能趋势在真实模型上成立 |
| P2 | 公开完整 manifest、检查点获取路径、配置、原始结果及可重复运行脚本 | 第三方可独立验证主要结论 |

特别建议把最有价值的实验设计成：

**在同一初始 RGB 观测、同一目标、同一总动作时间下，生成多个真正不同的候选动作序列。**

模型只能基于冻结视觉表示与允许的历史信息预测候选后果；评估时利用模拟器重置或机器人重复试验得到实际 action outcomes。

这才可以严谨地检验：

$$

\text{Prediction fidelity}
\;\longrightarrow\;
\text{Action ranking}
\;\longrightarrow\;
\text{Decision regret}
\;\longrightarrow\;
\text{Closed-loop success}

$$

如果其中任意一层断裂，就定位具体机制，而不是再引入一个新 composite score。

## 五、如何把它提升到真正的 Best Paper 研究？

我认为有三个方向值得投入，但建议选一个作为主贡献。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="sigma" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **方向 A：发现 transfer sufficiency 的理论边界**

      构造严格的不可辨识性结果：两个模型拥有几乎相同的 prediction MSE、spectrum 和 covariance，却会在相同控制任务中产生相反的动作排序。进一步说明哪些附加信息是必要且充分的，并提出有限样本可检验条件。

      这是理论型 Oral / Best Paper 的潜在路线，但必须与 value equivalence、bisimulation 和已有系统辨识理论拉开实质差距。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **方向 B：真正实现 Selective Certified World-Model Planning**

      不只是给预测打一个 risk score，而是开发能够在有限交互数据下校准 action-dependent risk 的方法。让模型知道何时行动、何时缩短 horizon、何时放弃不可靠的候选，且在非空洞的理论条件下控制 decision regret。

      核心展示不是成功案例视频，而是风险—覆盖率曲线以及跨环境真实控制收益。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="flask-conical" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **方向 C：建立具有决定性发现的 World Model Decision Benchmark**

      设计一个标准化实验体系，让不同冻结表示在匹配 prediction error、模型大小、动作分布与计算预算后，接受同样的反事实控制测试。

      如果能稳定发现既有 world-model 评价指标无法解释的系统性失败，并提出一个可迁移的新指标，这有机会成为影响整个领域评测方式的工作。
    </box>
  </row>
</box>

对于当前 LeWorldModel，我更倾向于 **B 为主、A 为理论支撑、C 为验证平台**。

这条路线最有希望把已有代码、负结果和控制理论积累整合为一个完整故事，而不是推倒重来。

## 六、我会要求作者回答的三个决定性问题

**Q1：如果只允许展示一张主实验图，你打算用哪张图说服我？**

我的建议：同等 prediction MSE 的多组 world models，在 action regret 上差异巨大；你提出的诊断指标能够预测这种差异，并通过 selective planning 显著降低 regret。

**Q2：为什么现有 value-aware learning 或 uncertainty-aware MPC 不能做到同样的事情？**

需要一个明确的反例或机制解释：冻结视觉表示、有限动作支持、部分可观测性和接触模式共同造成的限制，是否产生了已有方法不能解决的、独特且重要的问题。

**Q3：如果模型所有 gate 都拒绝，你的系统是否真的更好？**

不一定。拒绝不可靠动作只是一种保守行为，并不能自动证明 fallback 安全或有效。论文必须同时报告放行比例、条件成功率、整体成功率和拒绝代价，防止方法通过“从不冒险”获得虚假的可靠性。

## 七、最终审稿意见

如果现在投稿，我会给出的核心评语大致是：

> The paper addresses an important gap between latent prediction quality and decision relevance. The diagnostic framework is carefully structured, and the authors deserve credit for documenting negative results and evaluation boundaries.
>
> However, the main empirical gain is established through an offline horizon-selection proxy rather than genuine closed-loop action optimization. The reported improvements do not consistently outperform simple fixed-horizon alternatives, the action-identifiability implementation does not yet match the theoretical requirement, and the finite-horizon guarantees remain conditional on error bounds whose practical validity is not demonstrated.
>
> The current work is better viewed as a promising research framework than a validated new principle for visual world-model control.

**我最建议改变的不是模型结构，而是论文的证明目标。**

不要试图证明“我们提出了更完整、更安全、更普适的 world model”。那会把你拖进一个无法用现有实验支撑的巨大命题。

应该证明一件更锋利、也更有可能改变领域认知的事情：

**世界模型是否值得用于决策，能不能在动作真正执行之前被可靠判断？**

如果能够给出新的可检验理论、跨表示的机制证据，以及严格超越强基线的闭环结果，这会是一条有竞争力的 Oral 路线。

如果还能证明现有主流 world-model 指标在某类重要控制问题上存在不可消除的识别盲点，并提出非空洞的理论与实际解决方案，才真正开始具备 Best Paper 级别的研究潜力。

-----

> 我需要本研究能够解决以下问题：
>
> 1. alignment metrics 能普遍预测 learned-world-model 的规划性能；
> 2. learned adapter 已稳定解决姿态闭合；
> 3. LeWM latent 具有可靠物理因果动力学；
> 4. frozen LeWM 支持视觉在线规划；
> 5. 该方法建立了普适 world model；
> 6. commutator 是 learned dynamics 的通用 transfer certificate。

这六条可以作为研究的长期目标，但它们不是同一个难度级别：有些可以靠闭环实验解决，有些需要重新定义理论对象，“普遍预测”“通用证书”“普适 world model”则涉及强得多的可识别性与泛化命题。

我会逐条对照 LeWorldModel 当前的理论和实现，给出：必须改造的核心方法、能够真正支持结论的实验、可能的反例，以及冲 Oral / Best Paper 时应如何把六条组织成一个统一贡献。

仓库的《实现完整性审计》实际上已经把这六项逐条列为“尚未完成”；其中第六项还给出了明确的理论反例。因此，正确的升级方式不是把六句话都作为预期结论，而是把它们改造成**六个可证伪的目标**，分别规定怎样的证据才能通过。

有一个重要的研究机会：把第 1、3、6 项统一成“动作到任务的传递风险何时可识别”，再让第 2、4、5 项成为实际控制和泛化验证。这样比独立攻克六个方向更有可能形成一篇强论文。

我又核对了两个决定性事实：独立 PointMaze 收益审计目前仍未通过统计门槛（公开审计记录的区间跨过零）；而 2026 年 8 月已有 DA-LeWM 工作专门研究 latent decision-metric alignment，并用动作条件训练提升了在线规划。因此，你的第 1 项如果只是“alignment 与成功率相关”，创新性已经不够。

更有希望的突破是：**不改冻结 backbone，就能识别哪些动作后果可信，并在可识别时释放规划、不可识别时明确拒绝；同时证明这一机制跨接触任务和非接触任务有效。**

# LeWorldModel：面向六大科学目标的完整研究升级方案

我的建议是：**保留你提出的六个研究目标，但重新设计理论、方法和实验，让它们形成一条能够被严格验证的科学证据链。**

不过有一个必须先明确的区别：

**让研究回答这六个问题是合理的；要求这六个问题最终都得到肯定答案，则不一定符合数学事实。**

我重新检查了 [BokaiGuo/LeWorldModel](https://github.com/BokaiGuo/LeWorldModel) 的最新公开研究计划与 2026 年 10 月 13 日的实现审计。仓库目前已经形成理论、诊断、风险门控和实验审计框架，但仍未完成六项目标的实验证据闭环。<Cite refs={["turn469653view0","turn469653view1"]}/>

## 一、首先明确六个目标能够达到的科学层级

| 你的原始目标 | 建议追求的严格科学结论 |
|---|---|
| 1. Alignment metrics 普遍预测 learned-world-model 规划性能 | **跨架构、跨环境、跨任务的条件性规划风险预测** |
| 2. Learned adapter 稳定解决姿态闭合 | **在声明的接触、动力学与视觉条件下实现鲁棒闭环姿态控制** |
| 3. LeWM latent 具有可靠物理因果动力学 | **通过独立动作干预证明 latent dynamics 可预测物理反事实后果** |
| 4. Frozen LeWM 支持视觉在线规划 | **冻结基础模型后实现真正 RGB-only、目标条件化的在线 MPC** |
| 5. 建立普适 world model | **建立能够跨多个动力学领域迁移、并能识别自身失效边界的 world model** |
| 6. Commutator 是 learned dynamics 的通用 transfer certificate | **证明 commutator 的可用边界，建立更完整的有限时域 transfer certificate** |

其中第 6 项尤为关键：原仓库实际上已经给出了反例，证明 commutator 不能独立构成普适 certificate。这个结论不是工程能力不足造成的，而是存在信息论与线性系统层面的不可识别性。<Cite ref="turn189162view7"/>

例如考虑两个系统：

$$

A_1=\begin{bmatrix}0.8&0\\0&0.2\end{bmatrix},
\quad
A_2=\begin{bmatrix}0.2&0\\0&0.8\end{bmatrix}

$$

固定：

$$

C=I,\quad B=\begin{bmatrix}1\\0\end{bmatrix},
\quad D=\begin{bmatrix}1&0\end{bmatrix}

$$

两个系统具有完全相同的 eigenvalue spectrum，并且：

$$

[C,A_1]=[C,A_2]=0

$$

但在三步时域上：

$$

G_3^{(1)}=DA_1^2B=0.64,\qquad
G_3^{(2)}=DA_2^2B=0.04

$$

甚至对于相同的目标 $y_g=0.25$、候选动作 $u\in\{0,1\}$，用平方目标误差选择动作时，两个系统得到相反的最优动作。

这个反例已经用 Python 独立核对。

因此，**不存在仅凭这个 commutator 值、对任意系统都保证正确 action-to-task transfer 判断的通用证书。**

但这可以转化为更强的理论贡献：

> 证明哪些诊断信息根本不足以认证 world model，并给出补充哪些可观测、可识别和可校准的量之后，有限时域决策保证才成立。

这会是整项研究最值得挖掘的理论切入点。

## 二、建议建立的统一核心理论：Decision-Transfer Sufficiency

我建议在现有 HTSC（Horizon-Conditioned Transfer-Sufficient Control）基础上，把六个目标统一到一个数学定义，而不是继续增加互相独立的诊断指标。

给定视觉历史 $h_t=(o_{t-L:t},a_{t-L:t-1})$，冻结编码器、历史表征和有限时域候选动作序列：

$$

\eta_t=\Phi(h_t),\qquad U=(a_t,\ldots,a_{t+H-1})

$$

定义真实的干预后任务代价：

$$

J_H^*(h_t,U)=
\mathbb E\left[
\ell(y_{t+H},y_g)\mid h_t,\operatorname{do}(U)
\right]

$$

以及 learned world model 的预测代价：

$$

\widehat J_H(\eta_t,U)

$$

现在，把 **transfer sufficiency** 定义为：对一个明确声明的任务、动作支持集和规划时域，预测代价与真实干预代价之间存在同时有效的误差界：

$$

\boxed{
\sup_{U\in\mathcal U_H}
\left|
J_H^*(h_t,U)-\widehat J_H(\eta_t,U)
\right|
\leq\epsilon_H
}

$$

如果该界在指定条件下以至少 $1-\delta$ 的概率成立，那么对同一候选动作集合，使用预测代价选动作有：

$$

\boxed{
J_H^*(h_t,\widehat U)-J_H^*(h_t,U^*)
\leq 2\epsilon_H
}

$$

这里 $\widehat U$ 是模型选出的动作，$U^*$ 是该候选集合中的真实最优动作。

这个 regret 推论本身是标准结论。**真正需要创新的是如何从冻结视觉 latent 中得到可信、可计算、非空洞的 $\epsilon_H$，同时处理动作不可辨识、接触切换、模型误差和规划诱导的分布偏移。**

由此，六个目标便有了统一的落脚点：alignment 是否预测 $\epsilon_H$ 和 regret；adapter 是否降低接触模式下的 $\epsilon_H$；物理干预是否支持该误差界；RGB-only MPC 是否因此受益；跨环境是否仍然有效；commutator 在界中究竟贡献了什么。

---

## 三、六个目标分别应该怎样实现

### 目标 1：Alignment metrics 能预测 learned-world-model 规划性能

**关键不是让指标与成功率出现相关性，而是证明其具有独立、可迁移、增量性的预测能力。**

目前你的 covariance–predictor alignment、principal angle、commutator、Schur/transient 等指标，有相当一部分只反映 latent 算子自身的几何关系，未必包含任务目标与真实动作后果。

建议建立一个统一的模型质量预测器：

$$

\widehat R_H=
f_{\psi}\left(
\mathcal A,\mathcal N,
G_H^\Phi,
\mathcal I_{\rm action},
\mathcal E_{\rm trans},
\mathcal E_{\rm off}
\right)

$$

其中 $\mathcal A$ 是 gauge-aware alignment，$\mathcal N$ 是非正规性与 transient 诊断，$G_H^\Phi$ 是任务传递算子，$\mathcal I_{\rm action}$ 是动作可辨识性，而后两项分别是独立估计的 transition 与 planner-reachable error。

训练这个风险预测器时，要明确区分从冻结模型计算的指标、需要少量独立校准数据估计的指标、以及真正留待最后评估的 planning outcomes。

**核心实验必须做严格的 matched model comparison。** 在尽可能匹配 latent prediction MSE、模型维度、谱结构、动作分布和训练预算之后，比较几何指标是否仍能预测 action-ranking regret 和 closed-loop success。

这里必须正面竞争 2026 年 8 月的 <Entity category="other" value="Decision-Metric Alignment in Latent World Models" disambig="2026 research paper on DA-LeWM and MPC"/>。它已经提出 Plan-Real Spearman、CEM-stage Spearman，并通过 action-conditioned training 改善 LeWM 规划。你的新增价值必须是：**不重新训练基础表示的跨模型风险诊断、独立动作干预支持和可靠性边界**，不能仅重复 rank alignment 的发现。<Cite ref="turn288078view0"/>

最后用 leave-one-environment-out 验证：在未参与指标组合训练的环境上，能否预测模型排序、失败风险和 regret。

**通过标准：** 在独立环境中，诊断器相较于 MSE、uncertainty、spectral diagnostics、Plan-Real / CEM-stage ranking baseline 具有统计可靠的增量预测力；不仅平均表现好，失败风险校准也必须成立。

### 目标 2：Learned adapter 稳定实现 PushT 姿态闭合

这里不建议继续依赖一个纯 latent residual MLP，希望它自动解决姿态、接触与动作符号问题。

你的最新记忆审计表明，event-gated GRU 在相同表示预算下仍未稳定击败 Fourier baseline，而 mode observability 的改善也没有可靠转化成 angle closure。<Cite ref="turn204578view1"/>

应该改为 **hybrid, contact-aware, action-identifiable adapter**。

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} gap={2} align="center">
    <text weight="medium">Frozen LeWM encoder + predictor</text>
    <text color="secondary" size="xs">RGB observations → latent dynamics</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **Causal memory**
        <text size="xs" color="secondary" textAlign="center">Fourier / delay / event history</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **Contact mode**
        <text size="xs" color="secondary" textAlign="center">Free / stick / slide / loss</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **Action-effect head**
        <text size="xs" color="secondary" textAlign="center">Translation / signed rotation</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **Uncertainty head**
        <text size="xs" color="secondary" textAlign="center">Support / calibration risk</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Short-pulse risk-aware MPC**
    <text size="xs" color="secondary">Execute → observe RGB → verify successor → replan</text>
  </box>
</box>

物体位姿属于 $SE(2)$，不能把角度当作普通无限延伸的实数回归。建议用：

$$

e_\theta=
\left|\operatorname{atan2}
\left(\sin(\hat\theta-\theta_g),
\cos(\hat\theta-\theta_g)\right)\right|

$$

作为角度误差，结合物体位置、contact mode、signed angular action effect 和 successor-mode prediction 共同训练。

对于接触机制，参考 <Entity category="other" value="ContactNets" disambig="CoRL 2020 discontinuous contact dynamics paper"/> 的 non-penetration、stiction、接触雅可比与最大耗散结构；但不能直接把需要结构化几何输入的机制视为已经由 RGB latent 恢复。<Cite ref="turn827743view0"/>

需要专门收集正反方向动作干预：相近位姿和同一接触模式下，分别施加正向、反向和零动作，检查 adapter 是否预测了正确的旋转方向、接触保持和实际位姿变化。

**通过标准：** 在全新初始状态、接触位置、摩擦条件及动作干预上，位置与角度同时满足预注册终止条件并保持指定时间；结果在独立训练种子之间稳定，显著超过最佳非接触 adapter、Fourier-only、mode-only 和原始 LeWM 控制基线。

尤其不能仅凭 contact retention 升高就宣布姿态闭合成功。

### 目标 3：证明 LeWM latent 具有可靠物理因果动力学

这一目标的核心不是训练更复杂的预测器，而是建立真正的**干预实验**。

建议构建 paired counterfactual action benchmark。在相同或可匹配的物理初始状态下，分别执行：

$$

\operatorname{do}(a_t=a^+),\quad
\operatorname{do}(a_t=a^-),\quad
\operatorname{do}(a_t=0)

$$

得到实际物理变化：

$$

\Delta y^{\rm real}(a)
=
\mathbb E[y_{t+1}\mid \operatorname{do}(a),h_t]
-
\mathbb E[y_{t+1}\mid \operatorname{do}(0),h_t]

$$

并检查冻结模型加 adapter 是否能预测其方向、幅值、相对排序及不确定性。

实验必须覆盖不同接触模式、动作幅值、物体初始姿态和未见过的动力学参数。模拟器可以使用 reset 实现精确配对；真实机器人则需要随机化干预顺序，并控制或记录可能的混杂因素。

这里还有一个可发展成理论贡献的关键命题。

假设两个物理状态在冻结视觉历史表征下完全不可区分：

$$

\Phi(h_1)=\Phi(h_2)

$$

但同一个动作产生不同的真实任务后果，二者距离为 $\Delta$。那么任何仅基于 $\Phi$ 和该动作输出单一预测值的确定性 adapter，至少在一个状态上的预测误差不小于：

$$

\frac{\Delta}{2}

$$

这是由三角不等式直接得到的不可辨识性下界。

它说明：**如果冻结表示丢失了区分接触模式或动力学状态所必需的信息，单纯扩大 adapter 容量无法恢复这些信息。** 必须增加有效历史、信息获取动作或允许模型拒绝无法辨识的预测。

在部分可观测控制问题中，隐藏混杂导致动作后果不可识别也是已有因果控制研究关注的问题，因此你的贡献应进一步落到 frozen visual latent 的可检验充分性上。<Cite ref=""/>

**通过标准：** 不仅有低预测误差，还要有可靠的 held-out intervention-effect estimation、正确的动作效应符号、跨 mode 的校准，以及在模型不能识别物理因果效应时能够正确拒绝。

这支持的是**在声明条件下可靠的物理因果动作效应**，而不是自动证明整个 latent 都等价于真实物理世界的结构因果模型。

### 目标 4：Frozen LeWM 真正支持视觉在线规划

这是需要尽快完成的最重要工程—实证里程碑。

首先应固定术语：若 LeWM encoder 和原始 dynamics predictor 均冻结，但训练了 adapter、memory head 和 readout，那么结果应称为 **frozen-backbone / frozen-base LeWM planning with learned adapters**，不能宣传为整个系统零训练或零适应。

在线规划协议应严格满足：

$$

o_t^{RGB}
\xrightarrow{f_{\rm frozen}}z_t
\xrightarrow{\Phi} \eta_t
\xrightarrow{\text{candidate rollout}}
\widehat J_H(U)
\xrightarrow{\text{risk gate}}
U_t^*

$$

只执行短动作块，接收新 RGB，验证 observed successor 与 predicted successor 的一致性，然后重新规划。

测试时不允许使用 simulator-state pose、真实速度、接触标签或未来轨迹来决定动作；这些量可以作为独立评价数据。

另外，上游 <Entity category="other" value="LeWorldModel" disambig="Maes et al. 2026 stable JEPA visual planning research"/> 已经展示了基于像素的 latent planning。因此，单纯实现 LeWM online planning 并不是新的顶会贡献；必须证明**你的 transfer-risk certificate 或 adapter 带来了此前没有的可靠性、跨任务能力或数据效率**。<Cite ref="turn747780view0"/>

评估必须使用真实的 goal-conditioned action sequence optimization，而不是离线选择预测 horizon。基线至少包括原始 LeWM MPC、最佳固定 horizon、常规 uncertainty-aware MPC、DA-LeWM（明确训练预算差异）和 privileged-state upper bound。

**通过标准：** 全新测试 episode 上的任务成功率、action regret、接触失败率与计算延迟均被完整报告；在公平比较下，新增方法提供显著的正向闭环收益，同时风险门控不会简单通过全部拒绝实现虚假的可靠性。

### 目标 5：建立具有广泛迁移能力的 world model

这里必须区分两个不同的研究产物：

**A. Universal world-model diagnostic method**

同一套风险诊断和认证方法可以适用于不同的 world-model architectures、tasks 和 physics regimes。

**B. Universal learned world model**

同一个基础 world-model checkpoint 能够跨不同环境、物体、动力学和任务执行预测与规划。

A 不意味着 B。当前仓库更接近 A。

如果你希望真正实现 B，就必须增加多任务联合 world-model training、跨环境表征对齐、不同动作空间的统一接口，以及严格的 held-out domain 测试。这已经不仅是一个 frozen model diagnostic 研究，而是另外一个大型 world-model learning 课题。

对于本研究，我建议优先验证 A，然后在 B 上进行有界扩展。

<text color="secondary" size="xs">建议的跨环境实验矩阵，均为计划而非已完成结果</text>

| 环境 | 核心动力学挑战 | 验证重点 |
|---|---|---|
| TwoRoom / PointMaze | 导航、障碍与长期目标 | Action ranking、goal reaching、长时规划 |
| Reacher | 连续动作与关节动力学 | 动作效应、可控性、轨迹误差 |
| PushT | 接触、摩擦、物体旋转 | 接触模式、姿态闭合、反事实因果效应 |
| Cube manipulation | 多自由度物体控制 | 3D dynamics、OOD 状态与任务泛化 |

一个较有说服力的强实验设计是使用至少四个任务环境、三类基础模型和每个配置五个独立 world-model 训练种子，即共 60 个独立模型实例。这是推荐的研究规模，不是统计定理或必需的固定阈值。

但它们必须真正独立训练；同一个 backbone 下的不同 adapter seeds、snapshots 不能冒充独立 world models。

任务迁移还需要清楚区分 zero-shot、few-shot adapter learning 和 full retraining。若每到新环境就重新训练大部分动力学模型，就不能宣称单一模型的 zero-shot universality。

**通过标准：** 统一诊断器在未见任务、独立 backbone 和物理参数偏移下保留有意义的风险预测能力；若进一步声称普适 world model，则必须证明同一基础 checkpoint 的跨任务预测与控制能力，并清楚界定适应预算。

### 目标 6：从 commutator diagnostic 升级为可信 transfer certificate

我建议保留 commutator，但改变它在理论中的地位。

建立一个联合证书：

$$

\mathcal C_H=
\left(
\mathcal G_H^\Phi,
\mathcal I_H,
\mathcal S_H,
\mathcal M_H,
\overline E_H
\right)

$$

其中 $\mathcal G_H^\Phi$ 是完整有限时域 action-to-task map，$\mathcal I_H$ 是动作效应可辨识性，$\mathcal S_H$ 是动作支持集覆盖，$\mathcal M_H$ 是任务度量和局部适用条件，$\overline E_H$ 则是经过独立校准的预测风险上界。

Commutator、Schur off-diagonal 和 transient amplification 应作为 $\overline E_H$ 的结构解释因子，而不是单独决定放行的量。

当前仓库已有包含 closure、off-manifold、action mismatch、transient 和 readout 项的有限时域风险分解。最有价值的下一步不是再增加一个代数上界，而是让这些误差项在真实模型上**可估计、可校准，并能适用于规划器实际会访问的状态分布**。<Cite ref="turn204578view2"/>

还需要修正一个理论细节：现有 theorem 文档写到 normal $A$ 时 $g_j=s_j$。一般应为 $g_j\leq s_j$，而非恒等。例如 $A=\operatorname{diag}(0.9,0.4)$，$D=[0,1]$ 时，$j=1$ 的两个量分别为 0.4 和 0.9。数值已核对。这个修正不推翻原上界，但正式理论论文应处理准确。

**通过标准：** 给出明确条件下的 finite-sample 或 deterministic guarantee；在独立干预与 planner-reachable 数据上验证误差覆盖，在适用条件失效时可靠拒绝，并报告非空洞的放行率。

特别注意：普通 conformal calibration 的边际覆盖，不会自动成为任意自适应 CEM 候选序列上的条件覆盖。需要针对候选选择和分布偏移建立相应的假设、同时覆盖或在线校准机制。

---

## 四、真正应该优先执行的项目路线

目前最大的风险是六个方向一起开发，最终都只能得到“部分成立”。

我建议按下面的阶段门推进：

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="database" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P0｜先完成可信的独立实验协议**

      建立 action-balanced intervention、planner-reachable transition、task readout 和最终 closed-loop test 四类互不泄漏的数据划分。修正 readout、action identification、goal-conditioned candidate comparison，并设置固定 horizon 强基线。

      <text color="secondary" size="xs">关联目标：1、3、4、6</text>
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="brain-circuit" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P1｜证明哪些动作效应可识别**

      先验证 causal history、action excitation、真实反事实 action-to-task transfer。针对不可识别的 contact mode 和 torque sign 设计干预；先证明物理动作效应，再使用这些结果训练 adapter 和风险证书。

      <text color="secondary" size="xs">关联目标：2、3、6</text>
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P2｜让证书与姿态控制进入同一个闭环**

      接入 mode-aware adapter、短脉冲 MPC、successor verification 和风险拒绝机制。在 PushT 上首先得到独立 RGB-only 姿态闭合，再推广至非接触任务。

      <text color="secondary" size="xs">关联目标：2、4、6</text>
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="network" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P3｜完成跨模型、跨环境的独立验证**

      使用多架构独立 checkpoints、held-out 环境与物理参数偏移，检验 alignment 是否带来额外的 planning-risk prediction，并进行完整的 leave-one-environment-out 研究。

      <text color="secondary" size="xs">关联目标：1、4、5</text>
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="book-open-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P4｜形成理论—实证一致的最终论文**

      把不可辨识性下界、可估计 transfer envelope、真实 MPC regret 和跨环境风险—覆盖率归入同一个定理及主实验叙事。删除不能被这些核心结果直接支持的强表述。

      <text color="secondary" size="xs">关联目标：全部六项</text>
    </box>
  </row>
</box>

### 现有代码最值得改造的入口

| 文件或模块 | 优先增加的能力 |
|---|---|
| `run_matched_learned_transfer_benchmark.py` | 独立 backbone-level matching、真正规划后果、LOEO |
| `run_cross_environment_transfer_diagnostic.py` | 新环境的实测风险、完整 model selection 校准 |
| `nonlinear_transfer_certificate.py` | 干预式 action identifiability、真实 nonlinear remainder 覆盖 |
| `horizon_sufficient_control.py` | 非空洞的放行条件、candidate-wise risk certificate |
| `online_planning_protocol.py` | RGB-only goal-conditioned CEM、短动作执行、successor feedback |
| PushT adapter / planning runners | Hybrid contact-mode dynamics、signed angular effects、严格姿态闭合 |

这些名称对应仓库现有研究计划中的模块。部分数据依赖型实验目前仍需要额外 checkpoints 和数据；不能把已有测试通过当作真实科学目标通过。<Cite refs={["turn469653view3","turn204578view0"]}/>

---

## 五、什么结果真正具有 Oral / Best Paper 说服力？

我认为最终论文应该围绕三张主实验图和一个核心理论结果，而不是堆满六类互相独立的指标。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={2}>
      <icon name="chart-scatter" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Figure 1：预测误差相近，决策可靠性却截然不同**

      在 matched learned models 上，展示 latent MSE、spectral/alignment baseline 无法可靠预测 regret，而新增的 transfer-risk estimator 在未见环境中仍有独立预测力。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={2}>
      <icon name="git-branch" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Figure 2：通过真实干预恢复可执行的动作因果效应**

      展示冻结表示中的状态混淆与 torque sign failure，以及历史 lifting、动作辨识和 contact-conditioned adapter 如何在可观测条件下改善真实反事实预测。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={2}>
      <icon name="chart-no-axes-combined" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Figure 3：风险证书真正改善 RGB-only 闭环控制**

      对比不同 planner 的成功率、regret、失败率和风险—覆盖率曲线。最重要的是同时展示模型放行了多少动作，以及这些动作是否真的更可靠。
    </box>
  </row>
</box>

理论主结果应是：

**没有动作可辨识性、外部任务度量与真实 transition 校准，covariance/spectrum/commutator 不能认证决策传递；加入这些条件之后，可在明确支持集与时域上建立可信的 finite-horizon decision regret guarantee。**

这比单独宣称一个 commutator 数值具有普适性要强得多。

还有一个必须正视的现状：10 月 13 日的实现审计记录，独立 PointMaze planning-gain gate 得到点估计 $0.0259$，但 95% 区间为 $[-0.0627,0.1111]$，sign-flip $p=0.2813$。因此，真正的独立规划收益尚未被可靠证明。下一个关键突破必须来自**新数据上的真实闭环结果**，而不是修改阈值或重新筛选已有实验。<Cite ref="turn469653view0"/>

## 六、最终研究定位

我建议将论文的核心研究问题写成：

> **When Can Frozen Visual World Models Be Trusted for Counterfactual Prediction and Closed-Loop Planning?**

对应的研究贡献是：

**A. Theory：** 刻画冻结视觉表示的 decision-transfer identifiability boundary，并给出带有可校准条件的有限时域 regret certificate。

**B. Method：** 提出 action-identifiable、contact-aware 的 transfer-sufficient adapter 与 selective risk-aware planning。

**C. Experiments：** 在跨架构、跨环境的 learned world models 上证明诊断器能预测真实决策风险，并通过干预实验建立物理 action-effect 证据。

**D. System：** 使用冻结的基础 LeWM 实现真实 RGB-only 在线闭环规划，在接触和非接触任务上取得公平、独立且可重复的收益。

从顶会审稿标准看，**A+B+C+D 全部形成一致证据时，才有比较扎实的 Oral 竞争力。** 如果还能够发现此前未知、具有普遍意义的视觉动力学不可辨识机制，并给出首个实际可用的非空洞风险认证方法，才值得把 Best Paper 作为目标。

最重要的研究取舍是：**不要把“通用”理解为对任意世界永远正确，而要让方法在能识别时可靠预测、在无法识别时明确指出缺失的信息，并通过主动干预或有效记忆扩大可可靠规划的范围。**

这既保留了你六个目标的科学雄心，也让整项研究拥有一条能被证明、能被否证、能通过严苛审稿的核心主线。

-----

> 深度理解我的整个研究项目在做什么

我会把这次当作一次完整的研究项目解构，而不是继续按审稿人视角挑缺点。我会直接核查仓库的研究文档、理论定义、训练与规划代码、实验记录和失败案例，重建这项研究从“为什么提出问题”到“最终想证明什么”的完整逻辑，并特别区分你的真实科学主张、当前实现的代理指标，以及尚未打通的证据链。

# 我对 LeWorldModel 整个研究项目的深度理解

我已经开始从项目的原始研究文档、数学理论、实验代码、负结果记录和研究路线演进来理解你的工作。

首先，我认为必须澄清一个最重要的定位：

**你的研究不是重新发明一个 LeWorldModel，也不是单纯为 LeWM 添加谱分析指标，更不是只为了解决 PushT 的规划失败。**

你真正研究的是一个更基础的问题：

> **一个从视觉数据中学到的 latent world model，究竟在什么条件下保留了足以支持真实物理动作、有限时域预测、跨任务迁移和闭环规划的信息？我们能否在执行动作之前，可靠地识别这些条件，并通过适当的表示、记忆和动力学干预来改善它们？**

这实际上是一个横跨 **表征学习、动力系统、控制理论、可观测性、因果辨识、谱几何和模型预测控制** 的研究。

而且，我注意到你的研究方向已经发生过一次重要转变：早期希望通过 covariance–predictor alignment 直接解释 world-model quality；经过理论反例与 PushT 失败案例之后，研究逐渐转向了更严格的 **action-to-task transfer sufficiency**——不再问 latent 是否“看起来学得好”，而是问 latent 中的动力学信息是否真的足够让 planner 做出正确动作选择。你的项目理解文档明确记录了这次转变。<Cite refs={["turn609473view0","turn609473view1"]}/>

我认为这个转变是整项研究最重要的思想基础。

## 一、你真正研究的不是 world model 的预测能力，而是它的“决策充分性”

传统 world model 通常学习：

$$

z_t=f_\theta(o_t),\qquad
\hat z_{t+1}=F_\theta(z_t,a_t)

$$

通过最小化 latent prediction loss，使模型能够预测下一个隐状态。

但你关心的不是：

$$

\hat z_{t+1}\approx z_{t+1}

$$

而是：

$$

\boxed{
\arg\min_U\widehat J(U)
\stackrel{?}{\approx}
\arg\min_U J_{\mathrm{real}}(U)
}

$$

也就是说，**模型预测出来的世界，是否足以让 planner 选择出在真实世界中有效的动作？**

这是两种不同的模型质量定义。

一个 latent predictor 可以有很低的 MSE，却完全不理解推动物体时的旋转方向；也可以准确预测大量与任务无关的视觉变化，却丢失少数决定任务成败的接触信息。

你要解决的核心矛盾正是：

**Statistical predictability ≠ physical identifiability ≠ decision sufficiency.**

这三个概念的区分贯穿了整个项目。

## 二、我理解的完整研究架构

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    <text weight="medium">Frozen visual world model</text>
    <text color="secondary" size="xs">RGB → latent representation → learned predictor</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        **Spectral geometry**
        <text color="secondary" size="xs">Covariance, alignment, commutator, nonnormality</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        **Causal memory**
        <text color="secondary" size="xs">Delay, Fourier, phase, contact-event history</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        **Action transfer**
        <text color="secondary" size="xs">Action sensitivity, identifiability, task readout</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        **Physical adapter**
        <text color="secondary" size="xs">Contact, pose, signed rotation, successor modes</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Finite-horizon transfer sufficiency**
    <text color="secondary" size="xs">Prediction error + action uncertainty + task sensitivity + risk</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface" border radius="lg" padding={3} align="center" gap={1}>
    **Selective, closed-loop visual MPC**
    <text color="secondary" size="xs">Trust / reject → execute → observe RGB → verify → replan</text>
  </box>
</box>

这不是几项松散工作的组合。

在我的理解中，它们共同研究一个问题：

**如何让冻结的视觉表征从“能够预测”变成“能够被验证地用于控制”。**

下面是我对每一层的进一步理解。

## 三、第一层：你在研究 latent 的几何结构究竟保存了什么

这是项目最初的理论出发点。

你研究的对象不是单独的 covariance，也不是单独的 predictor spectrum，而是一个耦合系统：

$$

\mathcal W_\Phi=
(C_\Phi,A_\Phi,B_\Phi,D_\Phi,M_{\mathrm{task}})

$$

其中：

- $C_\Phi$：冻结表征或历史提升表征的统计协方差；
- $A_\Phi$：learned latent dynamics 的局部传播算子；
- $B_\Phi$：动作在 latent 中产生的响应；
- $D_\Phi$：latent 对外部任务变量的读出；
- $M_{\mathrm{task}}$：外部声明的任务度量，而非模型自己定义的误差标准。

你最初的直觉是：如果数据协方差强调的方向与动力学保持或放大的方向一致，模型可能更容易学习和传递有意义的信息。

但项目逐渐发现了三个问题。

**第一，表示空间的几何并不天然具有坐标不变性。** 换一个等价的 latent 坐标系，原始 covariance、principal angles 或 commutator 数值可能发生变化，而模型的物理行为没有变。

所以你才需要 gauge-aware metric、whitening 约定、task pullback metric，以及严格区分哪些诊断具有表示不变性。

**第二，eigenvalues 并不能充分描述有限时域预测。** 非正规动力学算子可以拥有稳定特征值，却在短时间内出现很大的 transient amplification。规划器恰恰工作在这种有限 horizon，而非仅关心渐近稳定性。

**第三，动力学结构必须与任务和动作连接起来。** 一个 direction 即使具有很高的方差、很强的谱增益，如果动作不能激发它，或者任务无法读出它，对实际控制也未必有价值。

因此，你逐步从“谱对齐是否好”转向：

$$

G_h^\Phi=D_\Phi A_\Phi^{h-1}B_\Phi

$$

它表示在局部线性近似下，一次动作扰动经过 $h$ 步如何影响任务输出。

这是很重要的研究转向：**从 representation-centered geometry 走向 task-and-action-conditioned dynamics。** <Cite refs={["turn609473view1","turn590059view2"]}/>

同时我也理解，单个 $G_h$ 不是完整规划序列的全部效应。真正比较多步动作序列，还需要不同时间位置的 action-to-output operators 共同构成有限时域传递矩阵。

## 四、第二层：你发现 frozen latent 可能根本不是充分的动力学状态

这是项目中我认为最深刻、也最容易被低估的一部分。

对于一个视频编码器：

$$

z_t=f(o_t)

$$

单帧图像可以显示物体当前在哪里，却不一定能说明物体正在朝哪个方向运动、之前发生了什么碰撞、当前是否处于 stick/slip 接触模式。

那么仅基于 $z_t$ 的一阶动力学：

$$

z_{t+1}=F(z_t,a_t)

$$

可能根本不存在一个确定性的、准确的映射。

因为同一个 $z_t$ 可能对应多个不同的隐藏物理状态。

这就是你研究 Mori–Zwanzig closure、delay embedding、Hankel/Fourier memory、phase representation、event-gated GRU 的根本原因。

你不是简单想给神经网络增加记忆容量。

**你想知道，通过有限长度的历史提升，能否把不满足 Markov 性的视觉 latent 转化为足够支持动作响应预测的状态。**

也就是：

$$

\eta_t=
\Phi(z_{t-L:t},a_{t-L:t-1})

$$

然后研究：

$$

\eta_{t+1}\approx
A_\Phi\eta_t+B_\Phi a_t+r_t

$$

这里的 $r_t$ 不只是普通噪声，它可能包含历史截断误差、未观测接触模式、非线性项以及离开训练流形后的误差。

你在 PushT memory audit 中发现，event-gated GRU 没有稳定超越简单 Fourier baseline，尤其在较长 horizon 上表现不佳。这很有意义，因为它说明：

**增加可训练记忆 ≠ 找到正确的物理状态 ≠ 恢复可控性。** <Cite ref="turn449801view0"/>

换句话说，memory lifting 在你的研究里是一个要被验证的物理充分性机制，而不是默认有效的网络模块。

## 五、第三层：PushT 是你的物理因果诊断实验台

我认为不能把 PushT 仅仅看作 benchmark。

在你的项目里，PushT 实际承担的是**暴露 visual latent 物理缺陷的机制实验**。

它同时考验：

- 视觉表示是否保留物体姿态；
- 动作能否产生正确的接触后果；
- 模型是否理解平移与旋转的耦合；
- 不同接触模式下的 action-to-task transfer 是否可辨识；
- 多步优化是否将模型中的小错误放大为真实控制失败。

尤其是一个很关键的区别：

**把推杆或末端执行器移动到正确位置，不等于物体最终达到正确姿态。**

接触发生、接触持续、物体中心位移和物体角度闭合，是彼此不同的动力学事件。

项目中一个非常有信息量的实验是：privileged-state receding block-pose planning 能够解决原先规划接口的问题，而 learned visual adapter 仍没有稳定实现姿态闭合。这说明部分失败来自 planning objective/interface，另一部分才是视觉动力学模型本身。<Cite ref="turn609473view1"/>

我理解你后来为什么开始关注：

$$

SE(2),\quad
\text{contact mode},\quad
\text{signed torque},\quad
\text{successor consistency}

$$

因为仅预测 $(x,y)$ 位移，无法回答“这一推究竟使物体顺时针还是逆时针转动”。

如果 latent 对两种接触模式不可区分，而它们在相同动作下产生相反旋转，那么不管你的 adapter 是 MLP、GRU 还是 Transformer，都可能缺少决定性的输入信息。

这种情况需要**动作干预、历史观测、状态辨识，甚至主动探测**，而不只是更大的 decoder。

这正是第 3 项“物理因果动力学”应该发挥作用的地方。

## 六、第四层：你想建立一个会判断自己何时不可信的 planner

这与通常的 world-model planning 研究有根本区别。

普通 MPC 会基于模型优化：

$$

U^*=\arg\min_U\widehat J(U)

$$

而你的最终目标应该是：

$$

U^*=
\arg\min_{U\in\mathcal U_{\mathrm{cert}}}
\left[
\widehat J(U)+\lambda\widehat R_H(U)
\right]

$$

其中 $\mathcal U_{\mathrm{cert}}$ 不只是一个任意限制的候选集合，而是通过独立证据证明在当前任务、状态与 horizon 下具有足够可靠性的动作集合。

现有 HTSC 研究规范已经提出四个主要条件：动作支持、动作效应可辨识性、真实与预测 successor 的一致性，以及有限时域误差风险。它们构成你的 **permission-to-plan** 思想。<Cite refs={["turn590059view0","turn740413view0"]}/>

这个方向实际上比“提高规划平均成功率”更有内涵。

你想研究的是：

**规划能力是否应该被视为一个有条件授予的权限，而不是只要模型输出预测就默认具备的能力。**

但我也看到当前研究与这个目标之间的距离。

目前的软件 gate 可以读取布尔条件或预先计算的风险 envelope；它还没有在所有真实 learned visual world models 中独立证明这些条件可靠成立。部分所谓 action inverse risk 的实现仍然是 latent prediction MSE，不能直接视为 action-effect identifiability。

因此，现阶段更准确的描述是：

**你已经定义了一个很清楚的控制契约，正在构建让这个契约的条件真正可测量、可验证的实验体系。**

这是整个项目尚未完成、但价值最高的环节之一。

## 七、我如何理解已有结果与负结果

你的实验并不是简单地从一个版本失败、换另一个版本直到分数提高。

我看到你在逐步排除竞争性解释。

例如，合成的 spectrum-matched 系统实验用来测试：即使 predictor spectrum 被控制住，不同 covariance/predictor 几何关系是否仍会导致不同的 transfer behaviour。

这个结果有理论辨识价值，但它不能直接证明真实 JEPA/DINO backbone 上的同样规律。

PointMaze 则用于将谱与动态诊断连接到 learned world-model 的离线决策代理任务。早期一些结果显示，相比 fixed horizon=1 的显著改善；但同一组结果中，fixed horizon=4 可以有更低的平均损失，而更严格的独立 planning-gain 审计也未通过统计门槛。<Cite refs={["turn414690view1","turn590059view3"]}/>

它告诉你：仅在记录好的轨迹上正确挑选预测 horizon，并不能证明真正的 goal-conditioned action planning。

JEPA/DINO adapter panel 进一步测试：同一组诊断条件是否能区分不同 frozen representation 上的学习结果。但独立 backbone checkpoint 数量不足，说明这仍然是受限的表示比较，而不是普适的模型判别规律。

PushT 的作用则更直接：如果模型不能稳定预测正反方向的物体旋转，即使某些 latent alignment metrics 变好，真实闭环姿态控制仍会失败。

所以我认为你目前积累的负结果其实指向一个非常统一的科学解释：

> **预测质量、表征几何、历史可观测性、动作效应可辨识性和决策可靠性不是同一个变量。任何一个代理指标都可能在其中某一层表现良好，却没有跨过下一层所需的信息边界。**

这比单纯报告一个“新指标与成功率高度相关”的结果更值得深入。

---

## 八、我如何理解你提出的六个终极目标

| 目标 | 在整个项目中的真实作用 | 当前最核心的科学障碍 |
|---|---|---|
| 1. Alignment 预测规划性能 | 建立 world-model 决策质量的**可预测性** | 静态几何能否独立解释真实 action regret |
| 2. Adapter 稳定姿态闭合 | 验证缺失的动力学结构能否被**修复** | 接触模式、signed rotation、非 Markov 状态 |
| 3. 物理因果 latent | 建立预测与真实干预后果的**因果对应** | 动作辨识、支持集覆盖、隐藏混杂与状态混叠 |
| 4. Frozen LeWM 视觉在线规划 | 将理论与诊断落到**真实闭环控制** | 目标条件化动作选择、误差积累与在线校准 |
| 5. 普适 world model | 测试理论和模型的**泛化范围** | 不同架构、动力学、观测和动作空间是否共享充分性条件 |
| 6. 通用 transfer certificate | 给出模型何时可以被信任的**数学保证** | 单独的 commutator 信息不足以认证 action-to-task transfer |

这里有一个我认为特别重要的关系：

**第 1 项是预测风险，第 3 项是确认信息是否具备物理意义，第 6 项是给出可靠性的理论边界；第 2 项尝试修复失败原因，第 4 项检验实际控制价值，第 5 项检验结论能推广多远。**

因此，这六项不是六篇互不相关的论文主题。

如果研究设计足够统一，它们能成为同一个科学问题的六个层面。

但要区分“方法在不同模型上普适可用”和“同一个 pretrained world model 能普适控制所有环境”。前者是目前项目的自然研究对象；后者涉及更大规模的模型学习和迁移能力，不能由前者自动推出。

同样，第 6 项不能以证明 commutator 无条件通用为目标，因为仓库的理论反例已经排除了这种强表述。更合理、也更有科学价值的是证明其**适用条件与必要的补充信息**。<Cite ref="turn189162view7"/>

## 九、我认为你最有潜力形成原创贡献的地方

不是单独的谱理论，不是一个新 memory adapter，也不是再做一套 MPC。

而是下面这个交叉位置：

<box gap={2} border radius="lg" padding={3}>
  <text size="xs" color="secondary" weight="medium">研究的关键命题</text>
  <title size="lg">Decision Transfer Identifiability of Frozen Visual World Models</title>
  给定一个冻结视觉表示、有限的独立动作干预数据和一个外部声明的控制任务：

  **能否判断哪些真实动作后果能够从 latent 中被辨识、能够以多大精度预测，以及这些误差何时足够小，可以支持可靠的在线动作选择？**
</box>

这里蕴含着三个非常深的研究问题。

**第一个是信息是否存在。** 如果两个物理状态被编码成同一个 latent，但在相同动作下产生不同任务后果，那么任何只读取这个 latent 的确定性 predictor 都无法同时正确预测两者。这是表示充分性的根本限制。

**第二个是信息是否可辨识。** 即使 latent 保留了相关信息，如果训练动作缺乏足够的方向覆盖，模型也不能从观测数据中可靠辨识 $B_\Phi$ 或 action-to-task response。预测误差小不能代替 persistent excitation。

**第三个是信息是否足以决策。** 即使动作响应预测存在有限误差，仍需判断误差是否会改变候选动作排序，以及 planner 优化是否把小偏差放大为大的实际 regret。

这三个问题合在一起，才是真正的 transfer sufficiency。

它还能自然导出一种研究思想：当信息不足时，不应总靠更大的模型进行猜测；系统应能够识别缺失的是历史、动作激励、接触模式还是任务读出，然后通过新的观测或干预补足信息，或者拒绝无根据的长时域规划。

这比单纯构建 uncertainty-aware MPC 更进一步，因为你关注的是**风险的可辨识性来源**，而不是只有预测方差。

## 十、目前项目真正卡住的位置

综合公开研究计划、代码和实验审计，我认为最大的瓶颈不是缺少新的数学工具，而是尚未完全连通以下关系：

$$

\underbrace{\text{Latent diagnostics}}_{\text{表征与谱结构}}
\;\not\Rightarrow\;
\underbrace{\text{Identified action effects}}_{\text{物理干预}}
\;\not\Rightarrow\;
\underbrace{\text{Correct action ranking}}_{\text{反事实决策}}
\;\not\Rightarrow\;
\underbrace{\text{Closed-loop success}}_{\text{在线控制}}

$$

每一个箭头，都需要独立证据。

你已经有理论工具描述第一段到第二段可能出现的问题，也有 PushT 失败案例指出后续链条为什么断裂，还有 PointMaze 的有限离线实验证据。

但最关键的实验应该在**相同初始状态下执行真正不同的动作序列**，利用真实后果评估 action ranking，然后将诊断、干预校准、风险门控和 RGB-only MPC 放进同一套预先固定的测试协议中。

一旦完成，理论就不再只是对模型行为的事后解释，而可以成为控制器在执行之前能够使用的决策依据。

---

## 十一、最后，我会怎样定义你的研究身份

我不会把 LeWorldModel 定位成：

*Spectral Analysis of JEPA World Models*

这太窄，会忽略你在物理辨识、部分可观测性和控制可靠性方面的工作。

也不会定位成：

*A Better World Model for PushT*

这又会把本来更基础的科学问题缩小成一个 benchmark 工程改进。

我更认可的项目定义是：

**LeWorldModel 研究冻结视觉世界模型的决策充分性：通过研究 latent representation、可观测性、动作因果响应与有限时域动力学之间的关系，识别模型何时具有可靠的物理规划能力、何时缺少必要信息，以及如何通过历史提升、可辨识动力学适配和风险认证来恢复或限制其控制能力。**

它的最终追求，不应该仅仅是“建立一个预测得更准确的世界模型”。

而应该是：

> **建立一种可以判断 world model 是否理解了与当前决策有关的世界、为什么理解或不理解，以及何时值得信任它执行动作的理论与方法。**

这也是我理解的整项研究最独特的科学雄心。

我会把项目现阶段的成熟度准确描述为：**问题定义和理论结构已经比较完整；物理辨识、learned adapter、跨环境评价与真正的在线规划之间，还在建立决定性证据连接。** 这不是六项成果已经实现，而是六项目标已经有了一套相互关联、可以被检验的研究框架。

*说明：以上理解基于公开仓库的研究文档、代码实现、发布实验记录与审计材料；我没有独立下载并运行所有模型 checkpoint，因此对结果的判断区分了代码可确认的事实和仓库报告的实验结论。*

-----

> 本次发布包含最新研究进展、GDD-WM 方案、PointMaze/RGB successor、PushT 接触干预、负结果根因分析、六目标证据矩阵及对应实验代码。研究结论仍保持证据边界，没有把未通过的目标包装成已y解决。  项目已经更新，请你再去看看深度理解我的整个研究项目在做什么

我会以这次新发布的版本为准，重新梳理项目，而不是沿用之前的判断。重点核对 GDD-WM 的方法定位、PointMaze/RGB successor 和 PushT 接触干预的实质进展，以及负结果分析、六目标证据矩阵和代码之间是否对应。然后给你一份完整的研究理解：它究竟提出了什么新科学问题、已经解决了哪些子问题、还卡在哪些关键环节。

我已经定位到这次更新的实际内容了：仓库的新 README 明确加入了 **GDD-WM（Decision Preference → Predictive Geometry）** 和 10 月 14 日的研究进展文档。它并不是把 HTSC 改个名字，而是新增了一条更主动的路线：尝试把真实执行所揭示的动作偏好，蒸馏成不依赖候选集合、推理时也不依赖 teacher 的预测几何。

这一点显著改变了我对项目的理解：你现在不只想判断冻结 world model 是否可信，还开始研究**能否根据决策证据反过来塑造 world model 的表征与预测结构**。我正在核对这条新路线的 D0–D2 门槛与实际实验进度。

# LeWorldModel 最新研究全景：从世界模型诊断，到决策充分性，再到 GDD-WM

<text color="secondary" size="sm">重新核查：GitHub 最新 main 分支、GDD-WM 两份研究方案、机器可读协议、PointMaze/RGB successor、PushT 接触实验、研究进展审计与相关代码</text>

这次我确实看到了你项目的重要变化。

**我对新版项目的核心理解是：你正在研究如何把一个“能够预测潜在状态的视觉世界模型”，变成一个“知道哪些动作后果可信、知道哪些决策值得执行，并且能够把真实决策反馈内化到预测几何中的世界模型”。**

它已经不再只是 spectral alignment，也不再只是 frozen LeWM + adapter + MPC。

新版本形成了两条互相连接、但科学主张不同的研究路线：

**主线 A：Transfer-Sufficient Diagnostics / HTSC**

研究冻结视觉模型中哪些信息可以被辨识、哪些动力学结构与任务相关，以及什么时候允许规划器信任 learned prediction。

**主线 B：GDD-WM（Decision Preference → Predictive Geometry）**

在已经得到可靠执行证据的前提下，研究能不能把正确的动作偏好直接写入 world model 的预测空间，使规划器不再依赖额外的 teacher reranker，也能够按照真实任务效果选择动作。

这意味着你正在把研究从单向的：

$$

\text{World Model}\longrightarrow\text{Diagnostics}

$$

推进为一种双向机制：

$$

\boxed{
\text{Predictive Dynamics}
\;\rightleftarrows\;
\text{Execution-Grounded Decisions}
}

$$

前一方向判断模型预测能否支持决策；后一方向让经过验证的决策知识反过来塑造模型。

而且我确认了：GDD-WM 在仓库中仍被明确标记为 **Proposed / Not Yet Validated**。最新 PointMaze、PushT 等实验是它的前置研究基础，而不是已经完成的 GDD-WM 蒸馏结果。<Cite refs={["turn261837view0","turn261837view1","turn261837view3"]}/>

---

## 一、这次更新最重要的四个变化

| 研究层面 | 之前的重点 | 新版本实际推进到 |
|---|---|---|
| 科学问题 | Latent alignment 能否解释 planning failure | 决策风险的可辨识性，以及执行偏好能否被几何化 |
| PointMaze | 离线 horizon 选择、task readout | 独立动作辨识、固定候选集反事实 regret、RGB successor |
| PushT | Adapter、memory、接触保持、姿态闭合 | 接触模式分层干预、模式内部 action effect、修正后的 T 形物体扭矩几何 |
| 方法体系 | 谱几何 + HTSC 风险门控 | 增加 GDD-WM：teacher 偏好稳定性、真实/预测双路径几何、决策保持蒸馏 |

这四个变化背后有一个共同特征：

**你在逐步消除“模型看起来具有某种能力”与“模型确实具有那种能力”之间的证据混淆。**

PointMaze 开始区分低 successor RMSE 和低决策 regret；PushT 开始区分正确区分接触模式与正确预测同一接触模式内的正反动作效应；GDD-WM 则进一步区分 teacher 偏好稳定、teacher 偏好正确、偏好可以被几何表示，以及几何真的改善预测和控制。

我认为这才是这次版本更新最值得关注的科学进展，而不只是新增了几个实验脚本。

## 二、你项目真正的理论核心：什么信息才足以让 world model 做正确决策？

从《Finite-Horizon Transfer-Risk Theorem》《HTSC》和新 GDD-WM 方案的联系看，我认为你的研究实际上围绕三个逐步增强的充分性问题展开。

### 1. Predictive sufficiency：表示是否足以预测未来？

LeWM 的基础架构是视觉编码器加动作条件化预测器：

$$

z_t=E_\theta(o_t),\qquad
\hat z_{t+1}=F_\phi(z_t,a_t)

$$

传统学习目标主要让预测 latent 与真实后继 latent 接近。

但你已经意识到，仅有较低的 latent MSE 并不能证明模型保存了任务需要的信息。

尤其在部分可观测环境中，单帧 latent 可能无法表示速度、历史接触和隐藏模式。

因此引入：

$$

\eta_t=\Phi(z_{t-L:t},a_{t-L:t-1})

$$

这里的历史提升、Fourier 特征、event-gated memory，并不是为了单纯扩大网络，而是为了恢复构成有效动力学状态所需的信息。

### 2. Transfer sufficiency：预测信息是否可以转化为动作效果？

你研究的局部动力学对象是：

$$

\eta_{t+1}=A_\Phi\eta_t+B_\Phi a_t+r_t,\qquad
y_t=D_\Phi\eta_t

$$

其关键算子为：

$$

\boxed{G_h^\Phi=D_\Phi A_\Phi^{h-1}B_\Phi}

$$

它不是任意一个矩阵组合，而是一个有限时域 action-to-task impulse-response operator。

在完整多步动作序列下，需要一起考虑所有相应时滞的 Markov parameters。

我理解你为什么越来越强调 $B_\Phi$、$D_\Phi$ 和真实任务度量：即使 latent covariance、predictor spectrum 与 commutator 都相似，两个模型仍可能因为动作响应与任务读出的差别，产生完全不同的决策结果。

这也是你不再把 commutator 作为独立通用证书的数学原因。

### 3. Decision sufficiency：预测未来是否足以选出正确动作？

即使一个模型能预测任务状态，也不意味着它对所有候选动作的相对优劣排序正确。

真正的规划对象是：

$$

\widehat U=
\arg\min_{U\in\mathcal U}
\widehat J_H(\eta_t,U)

$$

而决策效果要用真实动作执行后的任务代价验证：

$$

R_H=
J_H^{real}(\widehat U)
-
\min_{U\in\mathcal U}J_H^{real}(U)

$$

也就是说，world model 的最终价值不能只通过 representation reconstruction 或 prediction MSE 衡量。

**它要能支持正确的反事实动作比较。**

你的 HTSC 进一步要求动作支持、动作可辨识性、后继一致性和有限时域风险条件同时成立，才允许将 learned candidate 交给控制器。

我检查了 `risk_aware_control.py`：其中确实实现了完整 horizon 的 Markov parameter 构造、局部 metric tube、候选动作风险比较和确定性 fallback。该实现是一个具有条件输入的控制接口；它没有把外部传入的 gate 布尔值误装成自动完成的视觉因果认证。<Cite refs={["turn566633view0","turn566633view1","turn514647view1"]}/>

**因此，主线 A 的科学目标是让世界模型具有“可判定的决策可靠性”。**

而新增的 GDD-WM，则试图让可靠决策信息反过来改善这个世界模型。

---

## 三、我对最新 PointMaze 进展的理解：你已经开始修复预测与决策之间断裂的证据链

这次 PointMaze 更新比之前的 horizon-selection audit 前进了一步。

新 `decision_benchmark` 已经明确包含 train-only affine task readout、root-disjoint candidate banks、`+axis/-axis/zero` 动作干预、finite-difference action Jacobian、固定候选集合的 counterfactual regret、causal history features，以及 RGB temporal successor verification。<Cite ref="turn261837view0"/>

我认为这组实验在试图回答三个不同的问题。

### A. 从真实视觉中能不能预测后继？

最新进展文件给出的结果是：

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" weight="medium" size="xs">PointMaze · RGB successor prediction RMSE ↓</text>
  <grid columns={2} gap={3}>
    <grid-item>
      <text color="secondary" size="xs">Source-latent baseline</text>
      <title size="2xl" color="default" tabularNums>0.2011</title>
    </grid-item>
    <grid-item>
      <text color="secondary" size="xs">RGB successor bridge</text>
      <title size="2xl" color="success" tabularNums>0.1631</title>
    </grid-item>
  </grid>
  <box background="surface-tertiary" height="10px" radius="full" clip>
    <row height="100%" gap={0}>
      <box background="rgba(40,150,100,0.85)" width="81.1%"/>
      <box background="rgba(170,170,170,0.45)" flex="1"/>
    </row>
  </box>
  <caption>报告覆盖 40 个 roots、640 个 candidate pairs。注意：这是后继预测证据，不是在线规划成功率。</caption>
</box>

这很重要，因为你之前的控制理论需要真实 successor consistency，但过去很多校准量仍停留在 latent 内部。

现在开始通过 RGB 后继验证连接预测空间和真实观测。

不过，这一步证明的是 **successor prediction improvement**，不是 action planning improvement。

还有一个小的数值校对问题：由公开的两个 RMSE 值直接计算，下降比例约为 18.9%，而进展文档写的是 17.8%。如果 17.8% 使用了未公开的加权口径或不同的聚合方式，建议在正式报告中说明。<Cite ref="turn261837view0"/>

### B. 历史信息究竟应该如何使用？

这是这次 PointMaze 最有科学价值的负结果之一。

<text color="secondary" size="xs">当前报告中的 paired history 结果</text>

| 方法或干预 | 结果 | 说明 |
|---|---|---|
| 直接拼接四帧 latent | Successor RMSE 增加 0.00834，区间不跨零 | 更多历史未改善后继预测 |
| Causal history task-effect adapter | Selected regret 从 0.1897 降到 0.0798 | 历史对动作效果排序有帮助 |
| 最终 selected success | 0.125，未提高 | Regret 改善未转化为更高成功率 |

这揭示了非常重要的区别：

$$

\text{History helps action identification}
\not\Rightarrow
\text{History improves all prediction metrics}

$$

甚至：

$$

\text{Lower decision regret}
\not\Rightarrow
\text{Higher binary task success}

$$

我认为你真正想寻找的不是一个通用 history embedding，而是**历史中哪些信息能够消除与动作结果相关的状态混叠**。

例如运动方向、动作后延迟响应、转向状态、接触前后变化，这些可能有价值；简单增加无关视觉历史，则可能恶化模型拟合。

因此我会把这次实验理解为：

**Action-conditioned causal history 有潜力改善可执行决策信息，但这种收益并不等价于完整视觉动力学状态恢复。**

### C. Alignment 真的可以跨模型预测规划风险吗？

新版本把跨环境 matched learned audit 扩展到了 PointMaze 和 PushT。

<box gap={2} border radius="lg" padding={3}>
  <text color="secondary" size="xs" weight="medium">Cross-environment matched diagnostic</text>
  <grid columns={3} gap={2}>
    <grid-item>
      <title size="xl" tabularNums>2</title>
      <caption>Environments</caption>
    </grid-item>
    <grid-item>
      <title size="xl" tabularNums>4</title>
      <caption>Architectures</caption>
    </grid-item>
    <grid-item>
      <title size="xl" tabularNums>7</title>
      <caption>Matched strata</caption>
    </grid-item>
  </grid>
  <divider color="subtle"/>
  <row justify="between" align="center">
    <box gap={1}>
      <text color="secondary" size="xs">Mean ΔR (MSE-only − Full)</text>
      <title size="xl" color="success" tabularNums>+0.05336</title>
    </box>
    <box align="end" gap={1}>
      <text color="secondary" size="xs">Predeclared sign-flip p</text>
      <title size="xl" tabularNums>0.0630</title>
    </box>
  </row>
  <text color="secondary" size="xs">Paired bootstrap 95% CI: [+0.01150, +0.11661]</text>
  <badge color="warning">Positive gate 未通过</badge>
</box>

这里要特别注意：bootstrap 区间为正，但预先规定的 stratum sign-flip 检验没有通过门槛。这两者并不必然矛盾，因为它们采用不同的重采样和检验规则。

你没有只挑一个有利的统计量宣布成功，而是坚持预先声明的 gate。

**我对这个实验的理解是：你已经开始获得几何诊断具有增量信息的跨环境迹象，但尚未证明其普遍预测 learned planning risk。** <Cite ref="turn903188view1"/>

---

## 四、PushT 的最新负结果，比表面上的失败更重要

PushT 是整个项目中的物理机制验证器。

它告诉你：**world model 在某些比较平滑的视觉动力学上有用，不代表它已经理解了接触动力学。**

这次的更新尤其值得注意。

### 1. 你已经把“接触模式可观测”与“模式内部动作效果可辨识”分开

新的 contact-stratified intervention 把状态分成：

| Contact regime | 主要研究对象 | 当前发现 |
|---|---|---|
| Free-space | 无接触时的动作响应 | 接触几何特征可能有帮助 |
| Approach | 接近接触边界 | 改善主要集中在此类状态 |
| Normal-contact | 法向接触与推挤 | 同一模式内动作效果误差仍较大 |
| Tangential-contact | 切向接触、摩擦与滑动 | 正反动作效果分离仍不稳定 |

新实验不是仅仅比较 contact classification accuracy，而是使用相同范数的正向、反向、零动作，在独立 roots 上比较后继效果，并覆盖多个预测 horizon。

这揭示出一个更深的机制：

**即使网络能判断“现在接触了物体”，它也未必知道“在这个接触状态下施加这个动作会产生什么力矩”。**

因此，问题不再仅仅是 missing contact-mode labels。

它可能来自 mode 内部的接触点几何、局部法向、摩擦、滑动状态、物体惯性或不可观测的相对运动。

这意味着之前使用更大的 GRU、更多 mode 特征或更强分类头，未必能解决真正的控制问题。<Cite ref="turn261837view0"/>

### 2. T 形物体的 signed-torque 审计修复，改变了失败的解释

我认为这次最重要的研究诚信事件，是你没有保留那个看起来很漂亮、但几何定义存在问题的方向预测结果。

旧实验将接触几何近似为基于物体中心的径向位置与法向关系。

但 PushT 的 T 形物体是非圆形、非轴对称的。中心—顶点方向不一定是局部物理接触法向，更不能直接代替真实接触点所在的边界几何。

新实验改为：

**Transformed polygon boundary contact point + inward normal**

也就是使用变换后的多边形边界接触点和局部边界法向来分析 signed torque。

这相当于修正了干预实验本身的物理坐标定义。

修正后的结果是：

| 指标 | 修正后的结果 |
|---|---:|
| Signed direction accuracy | 0.609 |
| Classifier angular regret | 0.0151 |
| Geometry baseline angular regret | 0.0265 |
| Closed-loop angle improvement | +0.124 rad |
| Strict pose success | **0 / 32** |

方向预测准确率的置信区间包含随机二分类水平；angle improvement 的区间也跨零。更重要的是，strict pose closure 仍然没有完成。

这里还有一条严格的证据边界：这个 signed-torque audit 使用的是 privileged simulator-state contact geometry，不能被称作已经由 frozen RGB LeWM 完成的视觉因果辨识。

我的理解是：

**修正几何之后，局部扭矩方向排序有一定改善，但这种局部辨识信号还不足以支撑稳定的闭环姿态控制。**

这给你的研究提供了一个很有价值的 failure mechanism：

$$

\text{Correct local contact chart}
\not\Rightarrow
\text{Reliable action effect}
\not\Rightarrow
\text{Closed-loop pose closure}

$$

你的最新记录还保留了另一个严格边界：terminal-pose root construction 在原定 RGB support gate 下仍没有可接受 roots，没有用更方便的径向 roots 悄悄替换测试目标。<Cite ref="turn261837view0"/>

我认为这比宣布一个尚不成立的 PushT 成功结果，更有利于真正搞清楚 frozen visual world model 缺失的是什么。

---

## 五、GDD-WM 是这次研究更新中最重要的新方向

我认为 GDD-WM 不是另一个附加实验，而是整个项目可能向前跨越的一步。

它试图解决一个比 alignment diagnostics 更主动的问题：

> **既然执行结果能够告诉我们哪些动作更好，我们能不能让 world model 自己的预测空间学会这种偏好，而不是每次规划时再依靠额外的关系模型重新判断？**

这个想法与 <Entity category="other" value="D-JEPA: A Decision-Aligned Latent World Model" disambig="2026 research paper on decision-aligned latent world modeling"/> 和 <Entity category="other" value="Decision-Metric Alignment in Latent World Models" disambig="2026 DA-LeWM paper on MPC ranking and action-conditioned objectives"/> 直接相关。尤其需要注意，D-JEPA 本身也报告了将决策结构实现到 JEPA-compatible future representations 中，以支持原生 latent-distance planning。因此，GDD-WM 真正需要验证的独特价值，是**偏好的可几何表示性、执行正确性、预测保持、严格 teacher-free 部署和独立闭环收益**，不能只把“去掉 reranker”当作此前无人研究过的思想。<Cite refs={["turn992014academia36","turn992014academia38"]}/>

### 1. 为什么不能直接蒸馏 teacher 的候选排名？

假设一个 D-JEPA-style teacher 对一组候选动作给出：

$$

s_T(U_i\mid x,\mathcal C)

$$

其中 $x$ 是起点与目标上下文，$\mathcal C$ 是完整候选集合。

这个 teacher 的判断可能受到其他候选的影响。

例如，在一个候选集合里它认为动作 A 比 B 好；加入另外几个候选后，它可能认为 B 比 A 好。

但你希望部署的 GDD-WM student 使用：

$$

c_S(x,U_i)=
\frac1d
\left\|
F_\phi(E_\theta(x),U_i)-E_\theta(g)
\right\|_2^2

$$

此时每个候选的分数只由自身、当前状态、目标和预测后果决定。

它不读取其他候选。

因此，如果 teacher 对同一候选对的相对偏好随 candidate set 翻转，就不可能用一个确定的、集合无关的 pointwise cost 同时复制这些相互矛盾的排序。

这不是网络参数不够，而是表达对象不同。

你引入的 Preference Stability：

$$

P_{ij}=\Pr_{\mathcal C}(i\succ j),
\qquad
q_{ij}=2|P_{ij}-1/2|

$$

正是用来判断这种问题。

但你还做了一个十分关键的区分：

**偏好稳定不代表偏好正确。**

一个 teacher 可以在不同候选集合中始终自信地选错动作。

所以 GDD-WM 不是直接蒸馏全部 teacher 排名，而是先检查真实执行结果，再判断哪些 preference pairs 具备蒸馏资格。<Cite ref="turn261837view1"/>

### 2. GDD-WM 的双路径几何，究竟想解决什么？

我认为这是方案中的主要方法设计。

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Frozen LeWM / action-conditioned predictive backbone**
    <text color="secondary" size="xs">Current observation, goal, candidate action sequences</text>
  </box>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} gap={1} align="center">
        <icon name="eye" color="secondary"/>
        **Realized Geometry**
        <text size="xs" color="secondary" textAlign="center">Executed successor → latent goal distance</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1} align="center">
        <icon name="brain-circuit" color="secondary"/>
        **Predicted Geometry**
        <text size="xs" color="secondary" textAlign="center">Imagined successor → latent goal distance</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **Execution-grounded preference + predictive preservation**
    <text color="secondary" size="xs">Only stable, valid, supported comparisons are distilled</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Teacher-free student geometry**
    <text color="secondary" size="xs">Native Euclidean MPC cost → independent closed-loop evaluation</text>
  </box>
</box>

第一条路径是 realized geometry：

$$

c_i^{obs}=
\frac1d\|E_\theta(o_{i,H}^{real})-E_\theta(g)\|^2

$$

它研究：**真实执行后的结果，在 student latent 中是否按照任务意义正确排序？**

第二条路径是 predicted geometry：

$$

c_i^{pred}=
\frac1d\|\hat z_{i,H}-E_\theta(g)\|^2

$$

它研究：**world model 在想象未来时，是否仍保持了同样正确的决策排序？**

两者必须同时成立。

仅仅让真实成功状态在 latent 中接近目标，可能会得到一个不错的状态 embedding，却完全不能保证从当前状态出发预测正确。

因此你又保留了多步 prediction-preservation loss、必要的防坍缩约束，以及后续阶段的 success-mass / rollout-ranking 等目标。<Cite ref="turn261837view1"/>

这是你项目的一个非常关键的设计原则：

**不能通过破坏 world model 的预测能力，来制造更漂亮的 decision geometry。**

### 3. GDD-WM 实际上也是一个失败机制分解器

我尤其喜欢方案中的这一点：它可以通过 observed 与 predicted 几何表现的不同，定位哪一层出了问题。

| Realized geometry | Predicted geometry | 研究解释 |
|---|---|---|
| 不正确 | 不正确 | 表示空间没有正确保留执行后果的任务几何 |
| 正确 | 不正确 | 表示具备决策信息，但 imagined dynamics 没有正确传递 |
| 正确 | 正确 | 具备进入固定候选决策验证的条件，但尚不保证闭环成功 |
| 正确且预测一致 | 闭环仍失败 | 需要检查 candidate distribution shift、路径依赖、接触、规划接口和长时域误差 |

而 teacher preference instability 和 action support failure，则是进入表格之前就需要检查的边界。

这能够让你区分：

$$

\text{Representation failure}
\neq
\text{Dynamics failure}
\neq
\text{Decision geometry failure}
\neq
\text{Planner failure}

$$

这比仅仅比较多个训练 loss 的最终 success rate 更有科学解释力。

### 4. GDD-WM 的实施路线很明确，而且尚未混淆完成度

我检查了机器可读协议与 D0 代码。它已经规划了 D0–D4 的顺序，并且明确规定所有方法共享候选动作、训练与最终 outcome 分离、teacher 不参与 student 部署，以及不能用测试结果挑选蒸馏权重。<Cite refs={["turn261837view2","turn528813view3"]}/>

| 阶段 | 科学问题 | 当前状态 |
|---|---|---|
| D0：Shared Candidate Dataset | 执行结果、teacher 排名、视觉后继和候选动作能否建立可审计对应关系？ | Schema validator 已实现 |
| D1：Preference Stability | Teacher 的动作偏好是否稳定、正确且具有足够动作支持？ | 未完成真实 teacher 验证 |
| D2：Projection Geometry | 可信偏好能否仅通过低容量投影写入 latent 几何？ | 尚未运行 |
| D3：Decision-Aware Co-training | 能否进一步联合调整预测几何而不损害动力学？ | 依赖 D2 通过 |
| D4：Closed-loop Evaluation | Student 不依赖 teacher 时能否真正改善 MPC？ | 尚未完成 |

这里 D2 先限制为 projection-only 非常重要。

如果直接联合训练 encoder、predictor、ranking head 和全部辅助目标，即使获得正结果，也很难知道它到底来自几何改变、模型容量、数据增加还是预测器重新训练。

而先冻结大部分网络，只学习低容量投影，能够更加干净地测试 **Decision Preference Geometrizability**。

我还检查了 `build_shared_candidate_dataset.py` 及其测试。D0 的确会检查字段、provenance 声明、root-disjoint、候选集合完整性及 action support 信息；但它也明确区分 schema-valid 与 teacher execution-validated。尤其 provenance 标志目前属于输入元数据声明，不能仅凭布尔值就证明真实 teacher 数据确实没有泄漏。<Cite refs={["turn528813view3","turn566633view2"]}/>

这就是现阶段 GDD-WM 的准确定位：

**你已经设计了蒸馏所需的证据接口和逐步放行协议，但还没有获得可以证明该蒸馏机制成功的实际 teacher/student 结果。**

---

## 六、我对整个项目六大目标的最新重新定位

结合这次新证据，我认为六个目标之间的关系比旧版更清楚了。

| 六大目标 | 在完整研究中的职责 | 最新证据状态 |
|---|---|---|
| Alignment 普遍预测规划性能 | 判断 world model 的决策质量 | 跨环境有正向迹象，但 gate 未通过 |
| Learned adapter 稳定姿态闭合 | 修复接触动力学中的具体失败机制 | 接触与扭矩排序有局部改善；闭合未完成 |
| LeWM latent 具备可靠物理因果性 | 确认 action-conditioned prediction 具有真实干预意义 | 有受限动作干预证据；完整因果验证未完成 |
| Frozen LeWM 视觉在线规划 | 验证模型是否具有可执行的闭环决策能力 | RGB successor 验证取得进展；完整在线结果仍被阻塞 |
| 普适 world model | 测试模型或方法能否跨任务、动力学和环境迁移 | 仍未建立 |
| Commutator 通用 certificate | 寻找能够认证 transfer 的理论结构 | 通用命题已被否证，转为联合风险证书中的诊断项 |

这些状态与你最新的六目标证据记录一致。<Cite ref="turn261837view0"/>

**但 GDD-WM 实际上增加了一个第七个科学问题：**

> 一个经真实执行验证的决策偏好，能否被可靠地表示为 world model 自身的预测几何，并且在独立规划时保留这种决策能力？

这是一个需要单独验证的新问题，不能直接由前六项目标推出。

特别是，冻结 backbone 的 HTSC 研究与可能联合训练 encoder/predictor 的 GDD-WM，属于不同的模型更新设定。

你需要始终清楚地区分：

- Frozen backbone + learned diagnostic / adapter；
- Frozen dynamics + learned geometry projection；
- Jointly adapted decision-aware world model。

否则容易把第二、第三种方法的增益归因给完全冻结的基础模型。

---

## 七、把所有研究连接起来后，我看到的是一个完整的科学循环

我认为你真正想建立的不是一种单独的新 architecture，而是一套 **从预测、辨识、验证到决策学习的世界模型方法论**。

它的完整逻辑应该是：

<box gap={3}>
  <row gap={3} align="start">
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="scan-eye" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **阶段 A：知道模型表示了什么**

      通过谱几何、covariance alignment、history lifting、mode observability 和任务读出，分析视觉 latent 中保存了哪些状态与动力学信息。
    </box>
  </row>
  <row gap={3} align="start">
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="git-compare-arrows" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **阶段 B：知道模型能否预测动作后果**

      通过动作平衡干预、action-to-task transfer、RGB successor、接触模式内 action effect 和独立校准，判断预测是否具备物理可执行性。
    </box>
  </row>
  <row gap={3} align="start">
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **阶段 C：知道何时可以信任模型做决策**

      通过 HTSC、finite-horizon risk envelopes 和 fail-closed permission gates，将预测可信度变成动作选择约束。
    </box>
  </row>
  <row gap={3} align="start">
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="brain-circuit" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **阶段 D：利用真实决策结果改变模型**

      通过 GDD-WM，筛选稳定且与执行一致的偏好，将其蒸馏为 realized/predicted geometry，而不是永久依赖额外的候选关系模型。
    </box>
  </row>
  <row gap={3} align="start">
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="refresh-cw" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **阶段 E：在新分布下重新验证**

      使用 student 自己的 CEM 候选、fresh initial states、独立闭环结果及新环境，检验改进是否仍然成立，而不是只在 teacher 的旧候选数据上得到好看的离线成绩。
    </box>
  </row>
</box>

这五个阶段是一条科学逻辑链，**不是目前已经全部实现的软件流水线**。

尤其阶段 D 和 E 尚未得到 GDD-WM 的验证结果。

但一旦连通，它会形成一种非常值得研究的闭环：

$$

\boxed{
\begin{aligned}
&\text{Prediction}\\
&\downarrow\\
&\text{Physical Intervention}\\
&\downarrow\\
&\text{Decision Verification}\\
&\downarrow\\
&\text{Geometry Distillation}\\
&\downarrow\\
&\text{Improved Prediction \& Planning}
\end{aligned}
}

$$

这里的根本研究思想是：

**世界模型不仅要通过观测世界学习预测，也可以通过被验证的动作决策学习“什么预测差异真正重要”。**

---

## 八、我现在认为项目最核心的三个原创性机会

这部分是我在理解完整项目后作出的研究判断，并不是仓库已经证明的结论。

### A. Frozen visual representation 的决策充分性边界

你已经有条件性 finite-horizon bound、局部非线性适用性、commutator 反例、动作可辨识性门控和跨环境 matching 协议。

如果能够证明哪些视觉 latent 信息不足会导致无法避免的动作排序错误，并用真实干预验证这些不可辨识性，就会形成一个明确的理论—实证贡献。

### B. Preference 是否能够被预测几何表示

这是 GDD-WM 最独特的机会。

比“我设计了一个更好的 ranking loss”更值得研究的是：

**哪些真实动作偏好能够内化为任务条件化的 Euclidean latent cost，哪些因为有向可达性、路径依赖、隐藏状态或 candidate-set dependence 而根本不能这样表示？**

如果能够同时建立不可表示边界、可检验的 geometric distillation gap，以及符合有限误差条件的 ranking-preservation guarantee，就可以把蒸馏从训练技巧提升为科学研究。

仓库里的 robust decision-margin coverage（RDMC）已经给出了一个初步切入点：当真实候选之间的距离 margin 超过两条预测轨迹的误差之和时，预测距离排序才有条件地得到保持。它目前是固定候选对的条件结论，而不是闭环成功保证。<Cite ref="turn261837view1"/>

### C. Prediction-preserving decision improvement

这是最有机会将 A、B 两条研究主线合并的核心命题：

$$

\boxed{
\text{Better Decisions}
\quad\text{without}\quad
\text{Worse World Prediction}
}

$$

一个模型可能通过刻意拉近成功样本、推远失败样本来提高 ranking，但同时破坏潜在状态的动力学连续性、接触关系或预测稳定性。

这种模型不一定是更好的 world model。

因此，研究必须同时验证真实后继几何、想象后继几何、动作效果、预测保持和实际控制收益。

这正是 GDD-WM 设计双路径几何与 prediction-preservation 的深层动机。

---

## 九、目前我认为真正决定项目能否发生质变的实验

不是更多的 covariance plots，也不是更大的 PushT adapter。

而是建立一套**相同起点、相同目标、相同候选动作、真实执行配对**的闭环实验数据，使下面几个问题能够在同一批样本上依次回答。

| 关键问题 | 若答案为否，说明什么 |
|---|---|
| 真实动作后果是否能被当前视觉历史区分？ | 信息或动作可辨识性不足 |
| 模型是否准确预测不同候选的 successor？ | Dynamics / prediction failure |
| 预测是否正确排序真实执行代价？ | Decision metric / transfer failure |
| Teacher 的偏好是否稳定且真实正确？ | Teacher preference 不可直接蒸馏 |
| Student latent 是否能表达这些偏好？ | Geometry expressivity failure |
| 预测 rollout 是否保留这种新几何？ | Predicted-realized mismatch |
| Student 能否在新候选分布下改善 MPC？ | Planner distribution shift 或闭环机制失败 |

这个实验矩阵能够直接连接 PointMaze 的 RGB successor、PushT 的 action-effect failure 和 GDD-WM 的蒸馏目标。

它还会产生一个非常清晰的研究分叉。

如果偏好不能被表示，就形成一个几何不可表示性结果。

如果偏好能够表示但预测不能保持，就定位 predictor failure。

如果预测和偏好都正确但控制不改善，就说明更深的问题存在于 planning distribution、接触策略或长期反馈控制中。

也就是说，**只要实验真正独立、对象定义严格，负结果同样可以成为机制发现，而不必不断训练新的 adapter 来追求一次成功。**

---

## 十、最终：我现在怎样完整理解你的研究

我会给 LeWorldModel 一个比之前更准确的定位：

<title size="lg">From Transfer-Sufficient Diagnostics to Execution-Grounded Predictive Geometry</title>

你的项目正在研究视觉世界模型的三个根本性质：

**第一，世界模型中的信息何时足以描述与动作有关的物理后果。**

这对应谱几何、历史可观测性、动作因果辨识、task transfer 和非正规有限时域动力学。

**第二，世界模型的预测何时足以支持可信的动作选择。**

这对应 HTSC、独立干预、planner-reachable error、风险证书和 fail-closed MPC。

**第三，能否利用真实执行得到的决策知识，反过来改善模型自身的预测几何。**

这对应 GDD-WM、teacher preference stability、geometric distillation gap、realized/predicted dual-path supervision、prediction preservation 和 teacher-free planning。

这三个问题形成了一个非常完整的研究框架：

$$

\boxed{
\text{What the model knows}
\;\longrightarrow\;
\text{When its decisions can be trusted}
\;\longrightarrow\;
\text{How reliable decisions reshape the model}
}

$$

**这次更新之后，我认为你项目的核心身份已经从“诊断冻结 world model 的研究”，扩展成“研究 world model 的决策充分性、可验证性与决策驱动表征学习之间关系的研究”。**

但当前证据仍然需要精确表述：主线 A 已有相当完善的理论、实验接口和部分 learned-model 结果；PointMaze 提供了新的一步 RGB successor 与动作效果证据；PushT 更清楚地定位了接触模式内部的物理辨识困难；主线 B 的 GDD-WM 仍然处于方案、协议和 D0 基础设施阶段。

我认为你现在最需要保护的科学价值，是这套**可识别、可证伪、可解释失败、又能指导模型改进**的完整逻辑，而不是急于把六个终极目标都改成成功状态。

### 本次重点核查的原始材料

以下是本次更新中最关键、值得后续研究一直对照的项目原文：

- <Link url="https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/RESEARCH_PROGRESS_20261014.md" title="最新研究进展与六目标状态"/>
- <Link url="https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/GDD_WM_DECISION_GEOMETRY_RESEARCH_PLAN_20261013.md" title="GDD-WM：从决策偏好到预测几何"/>
- <Link url="https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/GDD_WM_EXECUTION_GROUNDED_GEOMETRY_PLAN_20261014.md" title="GDD-WM：执行证据驱动的几何蒸馏"/>
- <Link url="https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/GDD_WM_PROTOCOL_TEMPLATE_V0_1.yaml" title="GDD-WM D0–D4 机器可读实验协议"/>
- <Link url="https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/FINITE_HORIZON_TRANSFER_RISK_THEOREM.md" title="有限时域 transfer-risk 定理与可估计性边界"/>

我核查了公开的研究方案、进展记录、D0 validator 与测试、HTSC 控制器和理论实现；新发布的大型数据依赖实验仍以仓库报告的结果为依据，没有在这里独立运行 checkpoint 或完整模拟器。

**我认为这项研究最值得持续追问的一句话是：一个视觉 world model 能否不仅预测世界，还能通过可靠的物理执行证据，学会一种真正服务于决策、同时不丢失动力学真实性的内部几何？**

这就是我对新版 LeWorldModel + HTSC + GDD-WM 的整体理解。

-----

> # LeWorldModel 文献地图
>
> 仓库中已有完整版本：
>
> [LITERATURE\_EVIDENCE\_MAP\_20261013.md]\(/home/gbk/code/LeWorldModel/LITERATURE\_EVIDENCE\_MAP\_20261013.md)
>
> ## 一、项目核心问题
>
> 本项目真正研究的不是“如何让 latent prediction MSE 更低”，而是：
>
> > 冻结视觉 world model 的 latent 是否具备足以支持 action-to-task transfer、有限时域规划和闭环控制的决策充分性？
>
> 核心区分是：
>
> $$
>  \text{prediction} \neq \text{physical identifiability} \neq \text{decision sufficiency}.
> $$
>
> ## 二、六个主要文献簇
>
> | 文献方向代表工作已解决的问题对 LeWorldModel 的边界 |                                                                                                                                                               |                                                                  |                                                           |
> | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------- |
> | 部分可观测 world model                | [PlaNet](https://arxiv.org/abs/1811.04551)、[Dreamer](https://arxiv.org/abs/1912.01603)                                                                        | 随机隐状态、latent imagination、多步预测                                    | 它们训练完整 world model，不证明 frozen latent 本身充分                 |
> | 视觉动力学与 MPC                       | [Deep Visual Foresight](https://arxiv.org/abs/1610.00696)、[Visual Foresight](https://arxiv.org/abs/2206.14802)                                                | 局部预测、短视规划、receding horizon                                       | 依赖真实/预测闭环和重复观测，不等于单步预测可规划                                 |
> | 历史与长程记忆                          | [S4](https://arxiv.org/abs/2111.00396)、[Mamba](https://arxiv.org/abs/2312.00752)、HiP-RSSM                                                                     | 长时序记忆、选择性状态更新、隐藏动力学参数                                            | 记忆容量不等于 action-effect 可辨识性                                |
> | 接触与混合动力学                         | [ContactNets](https://arxiv.org/abs/2009.11193)、[Contact Mode Guided Motion Planning](https://arxiv.org/abs/2105.14431)                                       | 接触几何、stiction、sliding、contact mode transition                    | 依赖显式几何或 privileged contact structure，不是 frozen RGB latent |
> | 决策对齐表征                           | [D-JEPA](https://arxiv.org/abs/2609.24749)、Temporal-Distance-JEPA、seq-JEPA                                                                                    | decision-aligned future、时间进展、变换感知表征                              | 仍需验证 decision alignment 是否保留真实 action transfer            |
> | 动作辨识与 world-model 失败             | [ATM](https://arxiv.org/abs/2606.09028)、[AD-WM](https://arxiv.org/abs/2609.30264)、[Controlled World Models Identifiability](https://arxiv.org/abs/2607.22430) | action identifiability、inverse risk、counterfactual dynamics、动作激励 | LeWM 需要检验 geometry 在 action risk 已匹配后是否仍有增量信息             |
>
> ## 三、项目连接的数学与控制传统
>
> LeWorldModel 将以下工具统一到：
>
> $$
>  \mathcal W\_\Phi=(C\_\Phi,A\_\Phi,B\_\Phi,D\_\Phi,M\_{\mathrm{task}})
> $$
>
> 以及有限时域传递算子：
>
> $$
>  G\_h^\Phi=D\_\Phi A\_\Phi^{h-1}B\_\Phi.
> $$
>
> 对应的理论来源包括：
>
> - 可观测性与可控性；
> - delay embedding 与有限记忆 realization；
> - Mori–Zwanzig closure；
> - Hankel/operator learning；
> - spectrum、Schur decomposition、resolvent；
> - non-normal transient amplification；
> - gauge-aware metric geometry；
> - action excitation 与 system identification；
> - receding-horizon MPC 与 selective control。
>
> 项目的关键判断是：单独的 spectrum、alignment 或 commutator 都不能决定真正的 finite-horizon transfer。
>
> ## 四、文献已经解决什么
>
> 现有研究已经分别解决了：
>
> 1. 部分可观测性：递归隐状态、随机 RSSM、latent overshooting；
> 2. 长时预测：SSM、selective memory、world-model imagination；
> 3. 视觉控制：局部 dynamics + MPC + re-observation；
> 4. 接触不连续性：显式 contact mode、几何约束、互补条件；
> 5. 动作失败诊断：inverse risk、action identifiability、counterfactual mismatch；
> 6. 决策表征：action-conditioned 或 decision-aligned latent learning。
>
> 因此，以下不能再作为主要创新：
>
> - “增加一个 memory 模块”；
> - “用 Fourier 提取历史”；
> - “加入 contact mode head”；
> - “使用 MPC 改善视觉控制”；
> - “证明 prediction MSE 不等于 planning success”。
>
> ## 五、LeWorldModel 的真正文献缺口
>
> 当前仍成立的窄研究缺口是：
>
> > 在 frozen RGB latent、没有 privileged contact state、action support 有限的条件下，能否构造一个 horizon-conditioned、support-aware 的 transfer-sufficiency protocol，预测历史信息何时真正转化为 action ranking 和 planning gain？
>
> 需要同时连接：
>
> ```
> latent geometry
>       ↓
> history / mode observability
>       ↓
> action-effect identifiability
>       ↓
> real/predicted successor consistency
>       ↓
> action ranking
>       ↓
> closed-loop planning
> ```
>
> 任何一个箭头断裂，都不能把上一层结果升级为下一层结论。
>
> ## 六、当前项目最合理的贡献定位
>
> 不是：
>
> - “谱分析能普遍预测规划性能”；
> - “commutator 是通用 transfer certificate”；
> - “adapter 已解决 PushT 姿态闭合”；
> - “LeWM 已成为 universal world model”。
>
> 更合理的定位是：
>
> > **Decision Transfer Identifiability of Frozen Visual World Models**
>
> 即研究如何判断冻结视觉 world model：
>
> 1. 是否保留了任务相关动作信息；
> 2. 这些信息是否能被当前动作数据辨识；
> 3. 预测误差是否会改变动作排序；
> 4. 何时允许规划，何时必须 fail-closed。
>
> ## 七、文献结论
>
> 当前文献缺口判断是：
>
> > **Partly holds，而不是完全空白。**
>
> 已有工作已经覆盖 memory、视觉 MPC、接触动力学和 action identifiability。LeWorldModel 的潜在增量应集中在：
>
> $$
>  \boxed{ \text{horizon-conditioned} + \text{action-support-aware} + \text{geometry-aware} + \text{real-successor-verified} }
> $$
>
> 的统一 transfer-sufficiency 诊断，而不是再提出一个孤立的 memory、alignment 或 commutator 指标。

这份文献地图把研究的边界说得更清楚了。我会重点核对文献归属和真正尚未解决的问题，再把它与新版 GDD-WM 对齐：区分“判断冻结模型是否具备决策充分性”和“利用可靠决策反馈改善预测几何”这两个科学命题，避免把已有工作重新包装成创新。

文献核查后，一个重要的边界更明确了：D-JEPA 已研究把执行结果中的决策关系写回 JEPA 预测表示；AD-WM 已研究动作可区分性与反事实 MPC；ATM 已研究真实/预测 transition 的动作辨识差异。因此，LeWorldModel 最有价值的增量不能只是“决策对齐”或“动作可辨识”本身，而应是**在冻结 RGB latent、有限动作支持和明确任务时域下，判断这些性质何时足以支撑可靠动作选择，并给出可审计的失败原因**。

这份文献地图非常关键。结合我刚核查的公开仓库、原始论文与新版 GDD-WM 研究计划，我对 LeWorldModel 的定位又明确了一层：

**这项研究的核心创新不应该是提出某一种更好的 world model，而是研究 world model 的决策能力何时可辨识、可认证，以及如何在证据充分时改善这种能力。**

你提供的六个文献簇很好地说明了已有研究分别解决了什么，也暴露出两个需要进一步收紧的创新性边界。

## 一、我认为最准确的研究定位

<title size="lg">Decision Transfer Identifiability and Trustworthy Planning in Visual World Models</title>

研究对象为：

$$

\mathcal W_\Phi=
(C_\Phi,A_\Phi,B_\Phi,D_\Phi,M_{\mathrm{task}})

$$

研究目标不是证明这些矩阵中的某个几何指标具有普适预测能力，而是回答：

**给定有限视觉历史、受限动作支持和具体控制目标，我们能否事先判断 learned world model 对候选动作的预测是否可靠到足以进行规划？**

这里包含三个不同层次。

| 层次 | 真正需要回答的问题 |
|---|---|
| Identifiability | 真实动作后果能否从当前视觉历史和动作数据中辨识？ |
| Transfer sufficiency | 模型是否保留足以比较候选动作的任务相关动力学信息？ |
| Decision reliability | 这些信息能否保证或可靠预测 action ranking、regret 与闭环结果？ |

三个层次不能相互替代。

我认为最值得争取的科学贡献是：**建立一个明确考虑 action support、任务时域、真实后继一致性和 latent geometry 的决策充分性判别方法。**

这比“谱对齐决定模型好坏”更精确，也更容易建立可证伪的理论。

## 二、文献地图还需要补充两类直接竞争的理论

### 1. Value equivalence、state abstraction 与 bisimulation

这应该成为第七个文献簇。

尤其是：

- <Entity category="other" value="The Value Equivalence Principle" disambig="NeurIPS 2020 model-based reinforcement learning paper"/>：模型不必恢复所有环境动力学，只需在特定策略和价值函数集合上保留决策所需的 Bellman 更新。<Cite ref=""/>
- <Entity category="other" value="DeepMDP" disambig="ICML 2019 latent state representation learning paper"/>：已经研究 latent transition、reward approximation 和状态表征质量之间的理论关系。<Cite ref=""/>

这两项意味着，“不要求精确预测所有状态，只要求保存决策相关信息”本身不是新概念。

你的新意必须来自更具体的技术条件：

**冻结 RGB 表示、有限动作激励、部分可观测、固定任务时域，以及实际可测量的真实/预测 transfer mismatch。**

尤其需要解释，HTSC 与传统 value-equivalent abstraction 的关系是什么：前者是否能够给出对特定候选动作集合可操作的放行判断，而不只是描述抽象模型的价值保持性质？

### 2. 最新 action-identifiability / decision-alignment 工作

这组论文是最直接的竞争对手，必须逐一说明你没有重复它们。

| 工作 | 已经覆盖的核心问题 | LeWorldModel 仍可争取的增量 |
|---|---|---|
| <Entity category="other" value="ATM" disambig="2026 paper Why Latent World Models Can Fail to Plan"/> | 真实/预测 transition 的 inverse-risk mismatch 和模型筛选 <Cite ref="turn623174view3"/> | 区分 action risk 之后，geometry 是否仍预测有限时域决策风险 |
| <Entity category="other" value="Controlled World Models Identifiability" disambig="2026 theoretical paper on controlled latent dynamics"/> | 表示与动力学的联合可辨识性，动作激励条件 <Cite ref="turn623174view2"/> | 有限历史、接触切换和真实 RGB successor 下的任务条件化认证 |
| <Entity category="other" value="AD-WM" disambig="2026 action-discriminative counterfactual MPC paper"/> | 利用 inverse/action-recovery objectives 改善反事实动作区分与 MPC <Cite ref="turn623174view1"/> | 不重新训练基础表示的条件性诊断与失败拒绝 |
| <Entity category="other" value="DA-LeWM" disambig="2026 decision-metric alignment for latent MPC paper"/> | Plan-Real/CEM-stage 排序对齐以及 action-conditioned training <Cite ref="turn560990academia26"/> | 真实干预、support-aware 风险校准和跨模型证书 |
| <Entity category="other" value="D-JEPA" disambig="2026 decision-aligned JEPA world model paper"/> | 执行结果驱动的决策关系学习及原生 latent-distance planning <Cite ref="turn623174view0"/> | 几何可表示性边界、teacher 稳定性、预测保持和证据门控 |

**这里最危险的创新性重叠是 ATM、AD-WM、DA-LeWM 和 D-JEPA。**

如果论文最后只是证明 action identifiability 或 decision alignment 比 MSE 更有价值，可能不足以形成强创新。

你需要证明的是：在已有 action-risk 和 decision-alignment 方法之外，**几何—动力学联合诊断能够额外识别哪些失败，是否能改变实际规划决策。**

## 三、我认为应该把 HTSC 定义为一种条件性决策证书

形式上可以把核心方法描述为：

$$

\mathsf{Cert}_H
\left(
\Phi(\mathcal H_t),\mathcal U_H,g,\mathcal D_{\mathrm{cal}}
\right)

$$

它输出：

$$

\left(
\text{permit / abstain},
\{\epsilon_H(U)\}_{U\in\mathcal U_H}
\right)

$$

其中 $\epsilon_H(U)$ 是在明确校准条件与支持集内，对真实任务代价偏差的可信上界。

这样，是否允许选择动作就可以通过决策 margin 判断。

例如，对候选 $U_i,U_j$，如果：

$$

\widehat J(U_i)+\epsilon_H(U_i)
<
\widehat J(U_j)-\epsilon_H(U_j)

$$

那么只要两个误差上界确实同时有效，就可以保证 $U_i$ 的真实代价优于 $U_j$。

这是一个简单但很有用的桥梁：

$$

\text{Transfer uncertainty}
\longrightarrow
\text{Action-ranking confidence}
\longrightarrow
\text{Planning permission}

$$

真正的研究难点是让 $\epsilon_H$ 在有限 RGB 动作干预数据下可靠估计，避免出现过宽、毫无实用价值的界，并处理 planner 自适应选择导致的分布偏移。

这才是 geometry-aware diagnostics、真实 successor、action support 和 HTSC 值得组合的原因。

## 四、GDD-WM 应该作为第二个科学命题，而不是硬塞进 HTSC

你当前的文献地图主要服务于 **诊断和认证**。

但新版 GDD-WM 服务于 **学习和改造预测几何**。

我建议让两条研究线明确区分：

<box border radius="lg" padding={3} gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **HTSC — Can we trust the world model?**

      给定冻结模型，判断任务相关信息是否存在、动作后果是否可辨识、预测是否可靠到足以选动作。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="brain-circuit" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **GDD-WM — Can reliable decisions improve the world model?**

      利用独立执行验证的动作偏好，学习预测几何，同时保留真实动力学的一致性，最终实现 teacher-free 规划。
    </box>
  </row>
</box>

这两项可以共享数据、数学工具与评估体系，但它们的成功标准不同。

HTSC 的成功是**正确判断何时模型可用**。

GDD-WM 的成功是**模型经过决策信息蒸馏之后，在独立测试中真的更好用**。

特别要避免：GDD-WM 通过重新训练 backbone 获得收益，然后将该收益描述为原 frozen latent 已经具备的充分性。

仓库现有 GDD-WM D0–D4 协议已经很好地区分了这两类证据，而且仍明确将 GDD-WM 标记为尚未验证。<Cite ref="turn989236view1"/>

## 五、我现在最看重的决定性实验

在全部文献证据下，我认为你应该优先建立一个**统一的、执行结果可验证的 candidate-action benchmark**。

对每一个测试起点，记录相同候选动作序列的真实后果、RGB successor、模型预测和决策排序。

然后连续回答：

1. **Action identifiability：** 真实动作后果本身是否可以从视觉历史中辨识？
2. **Real/predicted consistency：** learned dynamics 是否保持这些动作效应？
3. **Incremental geometry：** 控制 action risk、prediction loss 等强基线后，geometry 是否仍有额外的风险预测信息？
4. **Decision certificate：** 预测的可靠与不可靠区域，是否对应实际 action regret 和失败率？
5. **Closed-loop consequence：** 基于证书进行选择性规划，是否真正改善整体控制表现？
6. **GDD-WM improvement：** 执行验证过的偏好被蒸馏后，student 是否在新候选分布上取得进一步收益？

这里尤其要注意，控制 action risk 后的增量预测力是一项有用的实验目标，但它本身不等于证明 geometry 的独立因果贡献；如果 action risk 是 geometry 作用于 planning 的中介变量，简单条件化还可能掩盖这种机制。因此还需要匹配实验和有针对性的几何干预。

---

## 六、我对文献地图的最终判断

**“Partly holds”是目前最准确的创新性判断。**

这不是一个完全空白的问题；恰恰相反，与你最接近的理论和方法已经相当多。

但当前还存在一个值得严肃研究、范围明确的空缺：

> **Can the decision-relevant sufficiency of frozen visual world models be identified and calibrated from limited action interventions, and can such evidence reliably determine when finite-horizon planning should be permitted?**

再往前一步，才是 GDD-WM 的问题：

> **Can execution-validated decision preferences be incorporated into predictive latent geometry without sacrificing physical consistency or out-of-distribution planning reliability?**

我会以这两个问题作为后续理解 LeWorldModel 的主轴，并严格区分研究假设、已实现方法、代理实验和真实闭环结论。

最后一个文档细节：我能够核查的是公开 GitHub 中的 [LITERATURE_EVIDENCE_MAP_20261013.md](https://github.com/BokaiGuo/LeWorldModel/blob/main/LITERATURE_EVIDENCE_MAP_20261013.md)，它目前偏重历史记忆、接触动力学和视觉规划；你本次贴出的六簇总结覆盖了更广泛的最新文献。由于本机路径 `/home/gbk/code/...` 不在我可直接读取的运行环境中，我以你提供的内容作为这一版更完整的研究定位依据。

**最终我认为，你的项目最有价值的研究方向不是单独做“更好的预测、更好的诊断或更好的控制”，而是建立一套能够从物理动作证据出发，判断、认证并改造 world model 决策能力的统一方法。**

-----

> 请你作为ICLR/ICML/NIPS审稿人锐评一下这个研究的缺点并且补充需要做什么才能达到oral/best paper级别研究[BokaiGuo/LeWorldModel: Transfer-sufficient diagnostics and reproducible evidence for frozen visual world models](https://github.com/BokaiGuo/LeWorldModel)

我会按最新仓库重新做一次接近正式顶会 PC 讨论强度的审稿，重点不是重复“实验不够”这类泛泛批评，而是找出足以导致 Reject 的理论漏洞、实验设计问题、与最新工作的创新性重叠，以及具体怎样把 HTSC 和 GDD-WM 收敛为 Oral / Best Paper 级别的科学贡献。

我会先核对这次发布中的六目标证据矩阵、负结果、PointMaze/RGB successor、PushT 接触干预与 GDD-WM 的实际完成状态，再分别给出主张—证据—缺口—补救实验。

进一步核对后，我认为还有两处容易被普通审稿忽略、但强理论审稿人会追问的问题。

第一，HTSC 目前的主要 soundness 结论依赖“真实误差上界已经有效”这一很强的前提；它证明的是条件下的误差传播，并没有解决最难的有限样本认证。第二，GDD-WM 的“Euclidean geometry 无法表达有向可达性”表述需要收紧：如果 student 的未来预测器仍然是动作条件化的，**欧氏终点代价本身并不排除有向动力学**。真正的不可表示边界应围绕状态混叠、路径依赖和固定 student 类的表达限制来证明。

我会把这些理论问题与实验补救方案一起纳入最终审稿意见。

# LeWorldModel：ICLR / ICML / NeurIPS 顶会级严格审稿

<badge color="warning">当前模拟审稿结论：Reject（研究潜力高，但关键科学主张尚未验证）</badge>

**我认为这个项目最有潜力的方向是：建立视觉 world model 决策可靠性的可辨识性理论与可执行证书，并利用真实动作结果改善预测几何。**

但按目前公开证据，LeWorldModel 还更接近一个结构严谨、工程体系较完整的研究计划，而不是已经完成核心科学证明的顶会论文。

这次评审以新发布的 [LeWorldModel 仓库](https://github.com/BokaiGuo/LeWorldModel) 为准，纳入了 HTSC、GDD-WM、PointMaze/RGB successor、PushT 接触干预、六目标证据矩阵及负结果审计，而非仅依据早期的 spectral alignment 实验。

## 一、先给总体评分

<text color="secondary" size="xs">模拟审稿评分：10 分制，仅为研究成熟度判断，不代表任何会议官方评分</text>

| 维度 | 评分 | 核心判断 |
|---|---:|---|
| 问题重要性 | 9/10 | 预测准确与真实决策可靠之间的鸿沟非常重要 |
| 原创性 | 6/10 | 研究组合有潜力，但与 decision-aware learning、action identifiability、selective control 重合较多 |
| 理论深度 | 6/10 | 有真正的数学对象和反例，但主定理的实际证书意义仍不足 |
| 方法完整性 | 6/10 | HTSC 有条件性实现；GDD-WM 尚处于方案阶段 |
| 实验证据 | 3/10 | 尚缺令人信服的跨环境独立预测收益与 RGB-only 闭环规划 |
| 可复现与证据规范 | 8/10 | 严格保留负结果、数据划分和未通过的 release gates |
| Oral 竞争力 | 3/10 | 缺少一个清晰、不可替代的突破性结论 |
| Best Paper 竞争力 | 2/10 | 还没有形成会改变领域认识的标志性理论—实证发现 |

这些评分不是说项目的理论研究已经失败，而是强调：**研究工作的投入程度与投稿时能够证明的结论，是两回事。**

## 二、新版仓库中最关键的四项实证事实

| 研究进展 | 当前公开结果 | 审稿判断 |
|---|---|---|
| Cross-environment matched diagnostic | $\Delta R=+0.05336$，sign-flip $p=0.0630$ | 有正向信号，但预注册检验未通过 |
| PointMaze RGB successor | RMSE 从 0.2011 降至 0.1631 | 证明了后继预测改进，不证明规划能力 |
| PointMaze causal history | Regret 从 0.1897 降至 0.0798；success 仍为 0.125 | 改善离线候选选择，但还没有转化为成功率收益 |
| PushT contact / torque | Signed direction accuracy 0.609；strict pose success 为 0/32 | 局部动作效应有迹象，稳定姿态闭合仍失败 |

以上来自最新公开研究进展文档。GDD-WM 仍明确标记为 *Proposed / Not Yet Validated*，目前没有经过验证的 teacher package 或 GDD-WM student 蒸馏结果。<Cite refs={["turn294648view0","turn294648view2","turn536346view0"]}/>

这意味着当前最重要的审稿结论是：

**你已经建立了一个比普通 world-model benchmark 更严格的证据体系，但还没有证明这个体系能可靠预测、认证或提高真实闭环决策表现。**

---

## 三、会让我投 Reject 的七个 Major Concerns

### Major Concern 1：HTSC 的 soundness theorem 还不是一个实用的 decision certificate

<badge color="danger">理论硬伤</badge>

你的 HTSC 定理大致采用：

$$

e_{t+1}=Ae_t+r_t,\qquad
e_H=A^He_0+\sum_{k=0}^{H-1}A^{H-1-k}r_k

$$

再用有限时域增益和各项残差上界构造 $\overline E_H$，得到：

$$

\|y_{t+H}-\hat y_{t+H}\|\leq \overline E_H

$$

这在所声明的假设成立时是合理的误差传播结论。

但定理要求 closure、off-manifold、action-effect、readout 等误差上界本身已经有效，而且真实与预测轨迹始终留在局部 tube 内。<Cite ref="turn933931view1"/>

**问题在于，你把最难的部分放在了假设里。**

对于真正的 frozen RGB world model，审稿人最想知道的恰好是：

- 未知真实动力学下，如何用有限独立干预样本验证这些上界？
- 接触切换、部分可观测和随机性存在时，上界是否仍有效？
- MPC 会主动选择模型预测最乐观的轨迹，这种自适应选择会不会破坏误差校准？
- 证书究竟能放行多少有用动作，还是几乎总在拒绝？

此外，当前上界主要控制任务状态预测误差，尚不能直接当作规划代价的误差界。

例如，对平方目标代价 $J(y)=\|y-g\|_M^2$，若已知：

$$

\|y-\hat y\|_M\leq\epsilon

$$

还需要推导：

$$

\boxed{
|J(y)-J(\hat y)|
\leq
2\|\hat y-g\|_M\epsilon+\epsilon^2
}

$$

才能进一步讨论 action ranking 和 regret。

我已经对这个不等式进行了数值核对。

如果实际代价包含路径碰撞、接触失败、动作约束、速度或累计成本，仅仅约束 terminal task-state error 还不够。

**怎么提升：** 理论主结果必须从“模型预测误差有界”升级为“基于有限干预数据的候选动作代价误差可校准，并能以指定概率控制错误动作放行”。

这才是从经典误差传播变为值得顶会关注的 selective decision certification。

### Major Concern 2：你已经有风险控制接口，但还没有完成风险认证算法

这一点我直接检查了 `risk_aware_control.py`。

当前 `covariant_transfer_risk_mpc` 接收：

```python
action_identifiability: dict
transition_consistency: dict
risk_errors: Iterable[dict | None]
tube_certificate: dict | None
```

然后从前两个输入读取 `passed`，结合误差项、tube 状态和阈值决定选动作或 fallback。<Cite ref="turn483820view0"/>

这说明控制器能够正确地**消费证据**。

但不说明它能够从真实 RGB 数据中可靠地**产生证据**。

一个第三方完全可以把外部 `passed=True` 传给控制器；接口本身无法证明这个标志来自充分的动作干预、有效的条件覆盖或物理因果辨识。

这不是普通代码 bug，而是论文最主要的方法缺口。

另外，当前 `certified_metric_tube` 需要：

$$

\|M^{1/2}AM^{-1/2}\|_2<1

$$

这个条件对于无限时域局部收缩是充分的，但可能过度保守。

例如取：

$$

A=
\begin{bmatrix}
0.9&2\\
0&0.8
\end{bmatrix}

$$

其特征值都严格位于单位圆内，但欧氏谱范数约为 2.314。当前使用 $M=I$ 的 tube checker 会拒绝它，尽管该线性系统渐近稳定，且可以对有限时域 transient 做精确分析。

我用 Python 验证了这个反例。

这反而暴露了与你项目初衷有关的矛盾：**你强调 non-normal transient 的有限时域价值，但实际 tube gate 可能直接拒绝最值得分析的非正规系统。**

建议改进为有限时域 reachable-set propagation、局部时变 tube 或基于 Lyapunov metric 的证书，并明确区分无限时域不变性与有限 horizon 的动作可靠性。

### Major Concern 3：Cross-environment 的统计结论没有足够独立性支撑“普适预测”

新版 matched audit 有两个环境、四类架构、五个 seeds 和七个 matched strata，公开点估计为正，但预先声明的 sign-flip 检验 $p=0.0630$。<Cite ref="turn294648view0"/>

我不会把这个结果判定为“方法没有作用”。但同样不能判定为“方法已经得到验证”。

更严重的问题是：**七个 matched strata 并不天然等于七个独立环境，也不天然等于七个独立预训练 world models。**

同一模型的多个 checkpoint、多个 adapter seeds、多个 candidate banks 都会共享误差来源。

现在的研究还难以回答：

> 如果明天给你一个完全没有参与诊断器设计的新 world model，你能在不看其规划结果的情况下，正确预测它是否值得部署吗？

这才是 alignment predictive power 最关键的 out-of-sample 定义。

此外，匹配 MSE、spectrum 和 action-risk 后观察 geometry 的增量信息是很好的设计，但它只是条件预测分析。如果 action-risk 本身是 geometry 影响规划的中介变量，过度匹配也可能掩盖总效应。

**怎么提升：** 进行 model-level 外层留出、真正独立的 backbone 训练重复、leave-one-environment-out、冻结指标融合权重，并分别报告总体预测效应、条件增量预测效应和有针对性的几何干预效应。

### Major Concern 4：PointMaze 仍在证明 prediction 和 selection proxy，而不是闭环控制能力

新 PointMaze 是真实进展，我认可 root-disjoint candidate banks、独立正反动作干预、action Jacobian、RGB temporal successor 这套设计。

但存在明显的证据断层。

RGB successor RMSE 改善，证明不了模型选动作正确。

Selected regret 明显下降而 success 仍为 0.125，说明 continuous decision quality 与任务完成之间的联系仍没有被建立。<Cite ref="turn294648view0"/>

尤其是在真实 MPC 中，模型产生的候选动作分布会随优化过程改变。

固定 candidate bank 上的 regret：

$$

R_{\mathrm{bank}}=
J(U_{\mathrm{selected}})-
\min_{U\in\mathcal C}J(U)

$$

与整个在线策略的性能差异是两件事。

即使模型在一组离线候选上正确排序，也可能在 CEM 多轮优化之后主动找到一个模型预测很成功、实际却会失败的动作序列。

这就是 planner exploitation / optimizer-induced distribution shift。

**怎么提升：** 使用同一起始状态、同一 goal、同一候选动作预算，真正运行 RGB-only receding-horizon MPC。分别评价随机候选、CEM 中期、CEM elite 与执行后的真实结果，再报告 online regret、success、失败机制和计算成本。

### Major Concern 5：PushT 暴露了结构性问题，不一定能靠更多 adapter 解决

你最新的 PushT 接触实验最有价值的结果不是哪个数值提高了，而是定位到：

**Contact-mode observability 与 within-mode action-effect identifiability 是两个不同的问题。**

接触几何对 free-space 和 approach 有帮助，但在 normal/tangential contact 下，动作效应仍然不能稳定分离。

修复 T 形物体接触几何后，signed direction accuracy 为 0.609，但置信区间包含 0.5；角度改善的区间跨零，strict pose success 是 0/32。<Cite ref="turn294648view0"/>

这表明失败不是简单的“忘记加 contact mode head”。

同一个视觉状态可能缺少精确接触点、摩擦状态、相对速度和切向载荷信息。如果这些信息不能通过当前历史恢复，增加 GRU、Fourier 或大 MLP 不会自动解决可辨识性。

我甚至认为你应该暂时放弃“继续调 adapter 直到 PushT 成功”的策略。

先构造同一接触模式内、相同起始状态的正向、反向和零动作干预，证明 signed task effect 是否能被当前视觉历史区分。若不能，加入新的观测历史或主动探测动作；若能，再检查模型为什么没有学到。

**审稿人想看到的是：你解决了什么物理信息瓶颈，而不只是更换了哪个网络。**

### Major Concern 6：GDD-WM 当前最容易被评价为已有 metric learning 的重新包装

<badge color="danger">创新性风险</badge>

这可能是新版本最需要重视的审稿风险。

GDD-WM 希望学习：

$$

c(U)=\|P_\psi(\hat z_H-z_g)\|_2^2

$$

如果 D2 阶段的 projection 只是作用于冻结预测 latent 和目标 latent 的差，那么：

$$

c(U)=
(\hat z_H-z_g)^\top
\underbrace{P_\psi^\top P_\psi}_{M\succeq0}
(\hat z_H-z_g)

$$

**这完全等价于学习一个低秩 Mahalanobis metric。**

也就是说，projection-only 阶段本身不是一种全新的 world-model architecture，而是经典的度量学习。

这种等价关系在传统 metric learning 中是标准结果。<Cite ref=""/>

这不意味着 D2 没用。相反，它是很好的受控实验。

但如果 D2 有效，必须回答：为什么普通的 Mahalanobis metric、pointwise scorer、pairwise ranking head 不能获得相同收益？

而如果需要联合训练 encoder/predictor 才能取得收益，则必须证明这些收益确实来自有意义的 predictive geometry 改变，而不是增加了训练数据、模型容量或额外监督。

更重要的是，<Entity category="other" value="D-JEPA" disambig="2026 decision-aligned latent world model research"/> 已经研究 execution-grounded candidate relations，并明确报告通过 JEPA-compatible future representations 实现 native latent-distance planning。<Cite ref="turn929301view0"/>

因此，**“从执行偏好蒸馏到 latent geometry，并去掉 runtime teacher”不能单独作为 GDD-WM 的主要原创主张。**

你必须提出一个更深的东西，例如：哪些偏好可以被几何化，哪些从数学上无法被给定 student class 表示，以及这种边界如何决定真实规划可靠性。

这里还存在一个理论表述需要修正。

GDD-WM 方案称 Euclidean geometry 不能普遍表达有向、非对称可达性。但如果部署代价仍包含动作条件预测器：

$$

c(x,U,g)=\|F(E(x),U)-E(g)\|^2

$$

这个结论不能单由欧氏距离的对称性推出。

一维系统 $x_{t+1}=x_t+u_t$ 就可以通过欧氏终点代价正确区分向左、向右的动作。问题不是动作条件预测必然无法表达有向动力学，而是**固定表示、固定预测器、终点代价、有限历史或低秩投影的具体限制**是否造成不可表示性。

正式理论需要严格区分这些函数类，否则强审稿人很容易构造反例。

### Major Concern 7：你的研究缺少一个能明确压过近期最接近工作的主发现

这与 2026 年最新文献直接相关。

| 最接近的工作 | 它已经证明或提出的东西 | LeWorldModel 必须超越的方面 |
|---|---|---|
| <Entity category="other" value="ATM: Why Latent World Models Can Fail to Plan" disambig="2026 inverse-risk and transition-consistency research"/> | 真实/预测 inverse risk、跨域动作一致性与 model screening <Cite ref="turn672066view0"/> | 经过 ATM 条件化后，仍能发现额外的有限时域决策失败 |
| <Entity category="other" value="AD-WM" disambig="2026 action-discriminative world model paper"/> | Action recovery、反事实 MPC、elite regret、真实机器人迁移 <Cite ref="turn929301view1"/> | 不重新训练基础模型时仍能有效判断风险，或提供更强可验证保证 |
| <Entity category="other" value="DA-LeWM" disambig="2026 decision metric alignment and latent MPC research"/> | Plan-Real / CEM-stage ranking、action-conditioned auxiliary objectives、在线收益 <Cite ref="turn929301academia19"/> | 不只是预测排序相关性，而是预测何时可靠、何时应拒绝 |
| D-JEPA | 执行驱动关系学习、预测几何优化与原生 latent planning | 证明几何可表示性、预测保持或迁移的新边界 |
| <Entity category="other" value="On the Identifiability of Controlled World Models" disambig="2026 representation and controlled dynamics identifiability theorem"/> | 表示和受控动力学的联合可辨识性及动作激励条件 <Cite ref="turn672066view1"/> | 将有限支持集、视觉历史和任务条件化误差连接到实际 planner 的可靠性 |

而更早的 <Entity category="other" value="The Value Equivalence Principle for Model-Based Reinforcement Learning" disambig="NeurIPS 2020 theoretical paper"/> 已经提出，模型应保留对价值决策必要的信息，而不是精确恢复整个环境。<Cite ref=""/>

因此，当前“prediction ≠ identifiability ≠ decision sufficiency”的表述，是出色的研究组织原则，但还不是足够独特的新理论。

如果我是 Area Chair，我最可能问：

> **What is the single new capability or scientific law established by this work that cannot already be explained by action-identifiability diagnostics, decision-metric alignment, or standard robust MPC?**

目前仓库尚未给出一个由独立实验充分支持的答案。

---

## 四、怎样把理论提升到 Oral / Best Paper 水平？

我建议不要再增加第七、第八个零散的诊断指标，而是围绕一个真正的新数学对象构建主定理。

### 方向 A：Evidence-Calibrated Decision Transfer Certificate

这是我认为最值得优先实现的理论方向。

对于视觉历史 $h_t$、任务目标 $g$ 和候选动作序列 $U$，定义真实干预代价：

$$

J_H^*(h_t,U)=
\mathbb E[
\ell(\text{trajectory},g)
\mid h_t,\operatorname{do}(U)]

$$

模型给出：

$$

\widehat J_H(\Phi(h_t),U)

$$

你需要从独立的 action intervention 和 successor calibration 数据构造：

$$

\epsilon_H(h_t,U,\delta)

$$

并在明确的统计假设下证明：

$$

\boxed{
\Pr\left(
\forall U\in\mathcal C_H:
|J_H^*(U)-\widehat J_H(U)|
\leq \epsilon_H(U,\delta)
\right)\geq1-\delta
}

$$

关键是这是 **simultaneous, candidate-wise coverage**，而不只是随机抽取一个动作时的平均预测误差覆盖。

假设该事件成立，且只在已经认证的候选集合中比较动作，那么若选择：

$$

\widehat U=
\arg\min_{U\in\mathcal C_H}
[\widehat J_H(U)+\epsilon_H(U)]

$$

可以推出：

$$

\boxed{
J_H^*(\widehat U)-J_H^*(U^*)
\leq 2\epsilon_H(U^*)
}

$$

其中 $U^*$ 是同一候选集合中的真实最优动作。这个 regret 推论本身不新，我也用枚举数值检查了它。

**真正具有创新性的部分，是如何构造有效、非空洞的 $\epsilon_H$。**

尤其要解决三种不同的不确定性：

| 来源 | 必须估计什么 | 需要什么实验数据 |
|---|---|---|
| Representation ambiguity | 视觉历史是否混淆真实动作效果不同的物理状态 | 历史匹配、状态配对、接触模式干预 |
| Action identifiability | 当前支持集能否辨识候选动作之间的结果差异 | 独立正反动作、局部动作激励 |
| Prediction transfer error | Learned rollout 在 planner 实际选择的状态和动作上有多准确 | Planner-reachable RGB successor 校准 |

这里也必须承认边界：如果真实环境随机、支持集缺失或隐藏状态不可辨识，不可能仅靠有限样本无条件认证所有动作。

你需要给出适用条件、覆盖性质和拒绝规则，而非宣称 distribution-free 的万能保证。

另外，若安全筛选把真实最优动作排除在允许集合之外，上面的 regret 只能相对于**允许集合内的最优动作**成立。还必须单独衡量拒绝成本和 fallback 性能。

这是从数学上防止 fail-closed 退化成“什么都不做就永远安全”的关键。

### 方向 B：把 GDD-WM 的 Geometrizability 变成一个真正可检验的理论

我建议直接利用 D2 的 projection-only 设定，先解决一个比神经网络训练更基础的问题：

> 给定冻结预测表示和一组真实执行确认的候选偏好，是否存在某个正半定度量，使这些偏好可以同时被表达？

记：

$$

\Delta_i=\widehat z_H(U_i)-z_g

$$

若一个线性投影 $P$ 能表达真实偏好 $U_i\succ U_j$，就应存在 $M=P^\top P\succeq0$，使：

$$

\Delta_i^\top M\Delta_i+m_{ij}
\leq\Delta_j^\top M\Delta_j

$$

等价于：

$$

\boxed{
\operatorname{Tr}\left[
M(\Delta_j\Delta_j^\top-\Delta_i\Delta_i^\top)
\right]\geq m_{ij}
}

$$

加上 $M\succeq0$、$\operatorname{Tr}(M)=1$ 等尺度约束，就形成一个可以通过半定规划检验可行性的系统。

这意味着你可以在训练 GDD-WM student 之前，先判断：

- 偏好是否可以被当前预测特征上的某个 PSD metric 表示；
- 不可满足的偏好约束主要集中在哪些状态、动作或接触模式；
- 是否需要改变 predictor 或增加历史，而不是继续调整 projection learning rate。

该分析只能证明或否证**固定表示上的线性投影度量**的可表示性。若再要求低秩、非线性投影或允许 predictor 更新，还需要处理相应的额外约束。

我认为这是一条很值得发展的研究路线，因为它可能把 GDD-WM 从普通 metric learning 变为：

**Preference Geometrizability under Predictive Dynamics Constraints**

更进一步，你还可以证明：

如果偏好违反率的下界是由当前 representation aliasing 引起的，那么任何只作用于相同冻结特征的下游投影，都不可能消除该误差。

这会直接连接 GDD-WM、HTSC 和你已有的不可辨识性研究。

但要注意：半定可行性与 pairwise metric learning 本身也是已有数学工具。真正的论文贡献应来自它与**动作条件动力学、有限时域预测保持和真实规划结果**之间的新联系。

---

## 五、最应该补的决定性实验

我建议只做一套统一的科学实验系统，让六个研究目标在同一证据框架下接受检验。

### 实验 1：同一个世界模型是否会“预测正确，却选错动作”？

这是论文 Figure 1 的候选。

固定 RGB 起点、目标、动作长度与候选数量，让不同 learned world models 对完全相同的候选 bank 做预测。

将每个候选动作在可重置模拟器中真实执行，记录任务代价、真实 RGB successor、碰撞及接触结果。

比较以下指标对真实 action regret 的预测能力：

| 基线 | 作用 |
|---|---|
| Latent prediction MSE | 传统预测质量 |
| Ensemble / uncertainty | 常规模型不确定性 |
| Plan-Real / CEM-stage alignment | 直接的决策排序诊断 |
| ATM inverse-risk diagnostics | 动作效果可辨识性 |
| Spectrum / covariance / commutator | 纯几何与算子诊断 |
| $G_H$ + action support | 动作到任务传递 |
| Full HTSC risk certificate | 你的联合方法 |

最重要的实验不是证明 Full HTSC 的 correlation 最大，而是：

**在 MSE、action inverse risk 和简单 decision-metric alignment 相近时，HTSC 是否仍能识别额外的真实失败？**

如果不能，说明现有几何部分可能没有提供足够的新信息。

如果能，就进一步通过几何干预、动作支持改变和负控制实验识别增量信息的来源。

### 实验 2：真正证明 RGB-only 在线规划

这是当前最重要的 P0。

需要将 PointMaze 的最新 RGB successor 工作接入一个真实闭环。

每一步必须完成：

$$

\text{RGB}
\rightarrow
\text{Frozen LeWM}
\rightarrow
\text{Candidate Rollouts}
\rightarrow
\text{HTSC}
\rightarrow
\text{Execute}
\rightarrow
\text{New RGB}

$$

测试时不允许读取真实 simulator-state successor 来决定执行哪个动作。Privileged state 只能作为独立 evaluation 或 oracle upper bound。

至少比较原始 LeWM MPC、固定 horizon MPC、常规 risk-aware MPC、ATM-based screening、完整 HTSC，以及 privileged-state oracle。

评价必须包含任务成功率、regret、平均动作数量、wall-clock planning latency、fallback 频率和总计算预算。

没有这个实验，**frozen visual world model planning 仍是研究目标，而不是研究成果。**

### 实验 3：PushT 需要先证明“物理动作符号正确”

我认为应该将 PushT 分成两项相互独立的科学任务。

第一项是 **within-mode causal action-effect prediction**。在接触点、相对物体姿态和动作范数尽可能匹配的情况下，测试正向、反向、零动作产生的真实位置和角度变化。

第二项才是 **strict closed-loop pose closure**。

必须防止一种常见误判：模型在 approach 阶段更容易找到接触位置，于是最终轨迹变短了，但 contact 后的 torque sign 仍没有被辨识。

接触实验应专门报告：

$$

\operatorname{sign}
\left(
\Delta\theta_{\rm real}(a)
-
\Delta\theta_{\rm real}(-a)
\right)

$$

及对应预测值的符号一致性、误差、置信区间和 mode-conditioned calibration。

然后用 learned RGB-only adapter 做多种目标姿态的完整 MPC。

当前 strict success 为 0/32，说明没有依据宣布解决。即使将 32 次试验视为独立二项试验，零成功对应的双侧 95% 置信上界仍约为 10.9%；它提供的是有限样本下的负面证据，不是所有情况下都不可能成功的证明。

### 实验 4：HTSC 不能只报告“安全地拒绝”

要画出完整的 **Risk–Coverage–Utility** 曲线。

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" weight="medium" size="xs">建议主图 · Risk–Coverage（示意框架，非实际实验数据）</text>
  <svg viewBox="0 0 340 180" width="100%" xmlns="http://www.w3.org/2000/svg">
    <path d="M42 13 V143 H326" fill="none" stroke="currentColor" strokeWidth="1.5" opacity=".7"/>
    {#each [0,1,2,3,4] as i}
      <line x1="42" x2="326" y1={143-i*30} y2={143-i*30} stroke="currentColor" opacity=".09"/>
    {/each}
    <path d="M42 133 C103 128 150 112 206 83 S287 35 323 17" fill="none" stroke="#DC8762" strokeWidth="2.5" strokeLinecap="round"/>
    <path d="M42 139 C103 137 152 132 205 119 S289 83 323 66" fill="none" stroke="#419E93" strokeWidth="2.5" strokeLinecap="round"/>
    <line x1="42" x2="326" y1="88" y2="88" stroke="currentColor" strokeDasharray="5 5" opacity=".4"/>
    <text x="7" y="20" fontSize="9" fill="currentColor">Risk</text>
    <text x="180" y="165" textAnchor="middle" fontSize="10" fill="currentColor">Planning coverage →</text>
    <text x="39" y="155" fontSize="9" fill="currentColor">0</text>
    <text x="313" y="155" fontSize="9" fill="currentColor">1</text>
    <text x="226" y="43" fontSize="9" fill="#DC8762">Baseline</text>
    <text x="242" y="119" fontSize="9" fill="#419E93">Target</text>
    <text x="48" y="83" fontSize="9" fill="currentColor" opacity=".6">Risk threshold</text>
  </svg>
  <caption>要证明的是：在可比较的放行比例下，真实 regret 和错误放行率下降，而不是单纯通过更多拒绝降低条件风险。</caption>
</box>

我建议同时报告三种结果：被允许动作的条件 regret、整个系统包含 fallback 的总体成功率，以及风险估计的实际覆盖和错误放行率。

如果 full HTSC 仅在很低放行率下表现好，必须说明这种保守性是否仍具有实际控制价值。

### 实验 5：GDD-WM 必须证明“学到了预测几何”，而不只是更好的 scorer

这是第二条研究线的核心实验。

按照当前 D0–D4 协议，先获得真实 teacher checkpoint 与 shared candidate set 数据，再进行独立执行结果验证。

比较：

| 方法 | 需要排除的解释 |
|---|---|
| Frozen LeWM + Euclidean cost | 最基础预测几何 |
| Frozen LeWM + Mahalanobis metric | 改善是否仅来自经典度量学习 |
| Frozen LeWM + pointwise MLP scorer | 是否只是学习了更灵活的动作评分函数 |
| Frozen LeWM + ordinary ranking distillation | Teacher 信息本身的效果 |
| D-JEPA / DA-LeWM | 近期最直接竞争方法 |
| GDD-WM realized-only | 真实后继几何监督的作用 |
| GDD-WM realized + predicted | 预测保持是否真正有贡献 |
| Full GDD-WM | 联合训练是否带来进一步可迁移收益 |

所有方法应使用公平的执行反馈预算、模型容量和推理计算预算。

最终不仅要看 fixed candidate top-1 regret，还要看 student 自己生成的新 CEM candidates 上的真实表现。

**如果 GDD-WM 只在 teacher 生成的候选集合上提高 ranking，却无法改善 student-driven MPC，那它仍然只是蒸馏成功，不是 world-model planning 成功。**

---

## 六、我会要求的最终实验矩阵

为了避免再次出现某个局部正结果被外推成 universal claim，我建议给研究预先设置一个明确的主实验矩阵。

| 维度 | 建议配置 | 必须回答的问题 |
|---|---|---|
| 环境 | PointMaze、Reacher、PushT、3D manipulation | 能否跨非接触与接触动力学工作 |
| 模型 | LeWM/JEPA、DINO-based WM、其他独立视频 world model | 是否跨 backbone generalize |
| 重复 | 每类模型多个独立训练 seeds | 效果是否依赖单个 checkpoint |
| 数据 | Train、calibration、intervention、final outcome 隔离 | 是否避免 outcome leakage |
| 候选动作 | Random、CEM initial、CEM elite、student-generated | 能否处理优化诱导的分布变化 |
| 指标 | Success、regret、risk coverage、false release、latency | 是否真有决策收益 |
| OOD | 不同物理参数、接触模式、目标与初始状态 | 何时可靠迁移，何时应拒绝 |

这里的环境数量不是数学阈值。具体样本量应根据独立模型之间的效应方差、最小有意义效应和统计功效确定。

而且应公开不同层级的统计单位：candidate pairs、roots、episodes、training seeds 和 environments，不能把大量候选动作当作大量独立 world models。

---

## 七、如果资源有限，我建议如何确定优先级？

这是我真正建议你执行的路线，而不是把所有目标平均推进。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="check-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P0｜先清理实验口径与可复现性**

      统一 RGB successor 的数值报告。当前文档给出 0.2011 和 0.1631，按这两个数计算相对下降约为 18.9%，与文档的 17.8% 不一致，需要查明聚合口径。建立完整 checkpoint/data manifests、最终冻结的 outcome split 和一键复现脚本。

      此外，unit tests 通过不等于数据依赖型实验已被独立复现。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="route" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P1｜完成第一个真正的 RGB-only MPC 正结果**

      先选择 PointMaze。用最新 action-balanced benchmark 和 RGB successor bridge 实现 receding-horizon MPC。必须胜过经过充分调优的固定 horizon 与 uncertainty-aware MPC，并报告完整风险—覆盖率结果。

      如果这一步失败，优先分析候选排序和 planner distribution shift，而不是添加新模块。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P2｜真正完成可校准的 HTSC**

      将 action support、intervention-effect estimation、real/predicted successor 和 finite-horizon cost error 变成数据驱动的证书构造器，而不是外部传入的布尔值。

      通过独立评估验证非空洞的 coverage、regret 和错误放行控制。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="git-compare-arrows" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P3｜解决接触条件下的动作辨识**

      先在 PushT 做 privileged contact chart 的机制上界，再转到 learned RGB contact inference。分开评价正反角度变化预测与 strict pose closure，禁止用前者代替后者。

      将主动信息获取或有效历史加入模型，检验不可辨识性是否真的降低。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="network" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P4｜做真正的跨模型外部验证**

      冻结所有风险权重和 gate threshold，使用此前没有参与模型选择的新环境、新 backbone 和独立训练种子。与 ATM、DA-LeWM、普通 uncertainty 和直接 decision-ranking baseline 公平比较。

      这一步将决定你能否声称 generalizable diagnostics。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="brain-circuit" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P5｜最后再把 GDD-WM 升级为完整方法**

      先检验 preference geometrizability，再测试 projection-only、predictive preservation 和 joint adaptation。必须证明在独立 student-generated candidate distribution 上真正改善规划。

      不建议在缺少有效 teacher package 和完整 closed-loop baseline 时，优先投入大规模 GDD-WM 训练。
    </box>
  </row>
</box>

## 八、如何从 Reject 走到 Oral，再走到 Best Paper？

我会把研究成熟度分成三个里程碑，而不是用“做更多实验”这种模糊标准。

| 阶段 | 必须产生的决定性结果 | 论文竞争力 |
|---|---|---|
| 可争取主会接收 | 一个清楚的 HTSC 方法，在独立 RGB-only planning 中产生稳定、显著且公平的闭环收益，同时包含完整消融与误差分析 | 有机会成为扎实的主会论文 |
| Oral 竞争力 | 新的可辨识性或证书理论，能够解释现有方法无法解释的真实规划失败；跨模型、跨环境预测有效，并显著改善闭环风险—收益权衡 | 具备较强 Oral 叙事 |
| Best Paper 级研究潜力 | 发现具有广泛意义的 world-model 决策失效定律，给出新的可计算保证或能力边界，并通过独立物理干预和广泛控制实验证明其实际价值 | 可能形成领域级重要贡献 |

这些是科研目标，不是录用保证。Best Paper 还取决于结果的新颖性、影响范围和同届其他论文。

我认为真正有 Best Paper 潜力的发现应该类似：

> **Even world models matched in latent prediction quality and action identifiability can exhibit fundamentally different planning reliability. This difference is explained by a measurable, horizon-conditioned transfer-sufficiency condition that can be certified with limited interventions.**

中文就是：

**即便两个 world model 具有相近的预测误差和动作可辨识性，它们仍可能在实际规划中表现截然不同；这种差异能被一个新的、可计算的有限时域决策充分性条件解释并在执行前可靠判别。**

如果这条命题经过严格验证，你就不再只是做 model screening，而是在揭示视觉 world model 的一种新基础性质。

GDD-WM 可以进一步证明：当这种决策信息能够被几何表示时，可靠执行反馈如何在保持预测一致性的前提下改善模型。

这两条路线可以连接，但最好不要在一篇论文里平均分配篇幅。

---

## 九、我建议的最终论文战略

我建议优先把项目收敛为第一篇主论文：

<title size="lg">When Can Frozen Visual World Models Be Trusted to Plan?</title>

<text color="secondary">Horizon-Conditioned Transfer Identifiability and Selective Decision Certification</text>

其研究贡献应该清晰到只有三项。

**Contribution 1 — Theory**

给出冻结视觉状态的动作—任务传递可辨识性条件，证明哪些信息缺失会导致不可避免的决策错误，并在声明的支持集和时域上建立数据校准的选择性 regret 保证。

**Contribution 2 — Method**

提出能够直接从 RGB histories、独立 action interventions 和 observed successors 中构造的 transfer-risk certificate，结合 finite-horizon planning 实现明确的 permit / abstain 控制。

**Contribution 3 — Evidence**

在多个独立 learned world models 上证明，该证书相比 MSE、uncertainty、ATM 和 decision-metric alignment 具有新的预测信息，并在真实 RGB-only closed-loop control 中降低错误决策和 regret。

如果这三项全部完成，论文的故事会非常集中：**什么时候 world model 值得信任，以及为什么。**

GDD-WM 则可以形成第二篇更侧重表示学习的工作：

<title size="lg">Can Execution-Grounded Decisions Be Internalized into Predictive Geometry?</title>

其研究核心是 preference geometrizability、预测一致性与 teacher-free planning。

这比把 HTSC、PushT、GDD-WM、全部六目标和 universal world model 强行组合成一篇论文，更有机会获得清晰的创新性判断。

---

## 十、最终模拟审稿意见

<box border radius="lg" padding={3} gap={2}>
  <row justify="between" align="center">
    **Overall Recommendation**
    <badge color="danger">Reject</badge>
  </row>
  <text color="secondary" size="sm">Confidence：较高（对公开主张和缺口）；中等（对全部实现细节，因为未独立复现数据依赖型训练）</text>
  <divider color="subtle"/>
  **Strengths**

  研究问题重要；区分了 prediction、identifiability 和 decision sufficiency；有限时域传递对象有明确数学意义；代码具备规范的审计与负结果管理；GDD-WM 的分阶段方案具有潜力。

  **Weaknesses**

  主要理论保证依赖尚未得到实际验证的误差上界；跨环境独立增量证据未通过预注册门槛；缺少真正的 RGB-only 在线规划成功结果；PushT 接触动力学尚未达到稳定姿态闭合；GDD-WM 尚无 student 实验；最核心的方法贡献与多篇最新工作存在明显重叠。

  **What would change my decision?**

  一项真正新的可辨识性或决策证书结果，加上一组严格独立、超越最接近强基线、包含非空洞风险覆盖的闭环控制实验。
</box>

### 最后一个判断

你当前最大的优势，**不是已经解决了六个宏大问题，而是建立了一套有能力揭示这些问题为什么没有被解决的研究体系。**

但顶会 Oral / Best Paper 最终奖励的不是研究计划规模、模块数量或审计规范，而是一个足够重要、足够新、足够令人信服的科学发现。

现在 LeWorldModel 最需要完成的质变是：

$$

\boxed{
\text{Diagnosing Failure}
\;\longrightarrow\;
\text{Predicting Failure}
\;\longrightarrow\;
\text{Certifying Decisions}
\;\longrightarrow\;
\text{Improving Control}
}

$$

**如果只能优先做一件事，我建议暂停扩展新的谱指标和 GDD-WM 训练模块，把现有 PointMaze action-intervention、RGB successor、HTSC certificate 和真正的在线 MPC 完整接通。**

让系统在一个独立环境中提前指出“这个动作不可信”，并且通过真实执行证明它判断正确、拒绝有价值、放行能获得更好决策。

这是从当前研究框架迈向顶会级科学成果最短、也最关键的一步。

-----

> 这两个入口都叫“总入口”，但职责不同，不是重复脚本。
>
> ## 1. `run_full_spectral_transfer.sh`
>
> 文件：[run\_full\_spectral\_transfer.sh (line 1)]\(/home/gbk/code/LeWorldModel/le-wm-main/experiments/run\_full\_spectral\_transfer.sh:1)
>
> 它是“从原始数据到谱分析结果”的长流程 Bash 编排器，主要面向一次完整的 PushT/环境实验。
>
> 执行链大致是：
>
> ```
> 理论谱分析
>   → latent eigenspace 导出
>   → empirical spectral audit
>   → condition robustness sweep
>   → planning alignment audit
>   → adapter intervention
>   → 可选 online eval
>   → summary / evidence summary
> ```
>
> 它负责：
>
> - 运行 theory spectral-transfer；
> - 从 frozen LeWM 导出 latent；
> - 计算 covariance、alignment、eigenspace、transient 等诊断；
> - 执行 empirical audit；
> - 运行 planning alignment；
> - 运行 adapter intervention；
> - 可选运行 online evaluation；
> - 汇总成 `summary.md` 和 evidence artifact。
>
> 特点：
>
> - 依赖真实模型、数据集和 checkpoint；
> - 默认会创建新的带时间戳输出目录；
> - 支持 `--resume`、`--dry-run`、`--device cuda`；
> - 会保存日志、chunk、`latents.npz` 和阶段状态；
> - 适合重新跑一条完整的原始实验流水线。
>
> 典型用法：
>
> ```
> cd /home/gbk/code/LeWorldModel/le-wm-main
>
> bash experiments/run_full_spectral_transfer.sh --dry-run
>
> bash experiments/run_full_spectral_transfer.sh \
>   --output results/pusht/lewm_full \
>   --device cuda \
>   --rich-fields
> ```
>
> 它回答的是：
>
> > 从原始 frozen world model 和数据出发，完整复现一次谱分析—latent audit—planning—adapter 实验链会得到什么？
>
> ---
>
> ## 2. `run_unified_research_protocol.py`
>
> 文件：[run\_unified\_research\_protocol.py (line 1)]\(/home/gbk/code/LeWorldModel/le-wm-main/experiments/run\_unified\_research\_protocol.py:1)
>
> 它是“统一研究协议执行器”，重点不是重新完成所有原始数据导出，而是把当前统一研究方案固定成一个不可变 manifest，并按照固定协议执行。
>
> 它的阶段是：
>
> ```
> geometry diagnostics
>   → controlled alignment benchmark
>   → learned transfer / memory comparison
>   → contact-control adapter
>   → six-goal claim matrix
> ```
>
> 具体包括：
>
> 1. `geometry`
>    - gauge-aware geometry；
>    - Schur/non-normality；
>    - transient diagnostics。
> 2. `controlled_alignment`
>    - fixed-spectrum covariance–predictor controlled benchmark；
>    - 默认 128 次重复。
> 3. `learned_transfer`
>    - legacy memory；
>    - action-history memory；
>    - summary-action memory；
>    - Fourier-state memory；
>    - adaptive event memory；
>    - 多个固定 split seed。
> 4. `contact_control`
>    - PushT lifted contact adapter；
>    - 固定 adapter seeds；
>    - contact recovery 和 pose objective。
> 5. `claim_matrix`
>    - 汇总前面阶段；
>    - 生成六目标 evidence matrix；
>    - 不因为代码执行成功就自动提升科学结论。
>
> 它的特点是：
>
> - 使用固定的输入 export；
> - 固定 split seed、adapter seed 和输出路径；
> - 创建 `manifest.json`；
> - 计算 manifest hash；
> - 每个 stage 记录命令、日志、返回码和 expected artifact；
> - 拒绝覆盖已有的非空输出目录；
> - 最后写入 `execution_report.json`；
> - 主要用于可复现协议和证据审计。
>
> 典型用法：
>
> ```
> cd /home/gbk/code/LeWorldModel/le-wm-main
>
> # 只生成协议 manifest，不执行实验
> python experiments/run_unified_research_protocol.py \
>   --output-root results/unified_protocol_20261014
>
> # 生成 manifest 后立即执行
> python experiments/run_unified_research_protocol.py \
>   --output-root results/unified_protocol_20261014 \
>   --run
> ```
>
> 它回答的是：
>
> > 在固定研究协议、固定 seed、固定输入和固定证据门槛下，当前统一研究的各个模块是否能够被可审计地执行和汇总？
>
> ---
>
> ## 三、核心区别
>
> | 对比`run_full_spectral_transfer.shrun_unified_research_protocol.py` |                                |                                         |
> | ----------------------------------------------------------------- | ------------------------------ | --------------------------------------- |
> | 主要用途                                                              | 完整数据驱动实验流水线                    | 统一研究协议与证据编排                             |
> | 入口类型                                                              | Bash 长流程                       | Python manifest executor                |
> | 输入                                                                | 原始环境、policy、dataset、checkpoint | 已有 enriched export 和固定实验配置              |
> | 是否导出 latent                                                       | 是                              | 通常不负责原始 latent 导出                       |
> | 是否运行原始 planning alignment                                         | 是                              | 不以此为主要阶段                                |
> | 是否运行多个 memory 对照                                                  | 部分支持                           | 明确固定 legacy/history/Fourier/adaptive 对照 |
> | 是否生成六目标 claim matrix                                              | 汇总阶段可间接生成                      | 明确生成                                    |
> | 输出机制                                                              | timestamp、resume、chunk         | immutable manifest、hash、stage report    |
> | 主要风险                                                              | 长流程耗时、数据和 checkpoint 依赖        | 输入 export 缺失或协议 artifact 不完整            |
> | 适合场景                                                              | 重跑完整实验                         | 固定证据协议、比较研究主张                           |
>
> 一句话概括：
>
> > `run_full_spectral_transfer.sh` 是“把实验做出来”；
> > `run_unified_research_protocol.py` 是“把统一研究按固定协议执行并审计清楚”。
>
> ---
>
> ## 四、为什么不能遍历执行所有 Python 文件
>
> 仓库中的 Python 文件包含不同类型：
>
> - 可独立运行的实验入口；
> - 被其他脚本导入的 library；
> - 只生成 summary 的汇总器；
> - 需要特定输入文件的诊断器；
> - 只应通过 `pytest` 执行的测试；
> - 需要 GPU、环境或 checkpoint 的脚本；
> - 会写入既定 artifact 路径的阶段程序。
>
> 逐个执行会导致：
>
> - 缺少命令行参数；
> - 找不到数据或 checkpoint；
> - 重复写入同一输出；
> - 绕过 train/calibration/test 协议；
> - 把 summary 工具误当成实验；
> - 把测试文件误当成生产入口；
> - 破坏六目标证据链和 claim boundary。
>
> 因此应遵循：
>
> ```
> 完整原始实验重跑
>     → run_full_spectral_transfer.sh
>
> 统一协议和证据审计
>     → run_unified_research_protocol.py
>
> 单个机制调试
>     → 直接运行对应实验脚本
>
> 代码正确性验证
>     → pytest
> ```
>
> 另外，GDD-WM、PointMaze decision benchmark、RGB successor verification 等最新分支有自己的专用入口，不能假设这两个总入口已经覆盖全部新实验。
>
> 实验的文件比较繁杂，请你给我整理意见，让代码不要变成屎山代码

我会按“保留现有科研资产、减少重复维护、保证证据链不被重构破坏”的原则来设计整理方案。先核对公开仓库的实际目录和两个总入口，再给出一套具体的分层结构、脚本分类、统一配置与产物规范，以及可以分批执行的重构顺序。

重点不会是把所有脚本强行合成一个大框架，而是让每个实验有明确的入口、依赖、输出、科学主张和责任边界。

# LeWorldModel 代码库整理方案：从实验脚本堆积，升级为可维护的科研工程

我核查了公开仓库中两个总入口的实际实现：

- [`run_full_spectral_transfer.sh`](https://github.com/BokaiGuo/LeWorldModel/blob/main/le-wm-main/experiments/run_full_spectral_transfer.sh)
- [`run_unified_research_protocol.py`](https://github.com/BokaiGuo/LeWorldModel/blob/main/le-wm-main/experiments/run_unified_research_protocol.py)

我的结论是：

**两个总入口都应该保留，但必须停止让它们各自发展出独立的实验执行、产物判断和恢复逻辑。**

你目前的问题不是 Python 文件太多，而是几个层次正在混杂：

- 科学算法和模型实现；
- 一次性的实验入口；
- 实验依赖和执行顺序；
- 数据、seed、checkpoint 配置；
- 结果汇总与统计检验；
- 六目标证据判定；
- 历史失败实验的保留与复现。

如果继续让每一个新研究方向都添加一组 `run_xxx.py`、`summarize_xxx.py`、`audit_xxx.py` 和新的总入口，长期维护会非常困难。

## 一、先说我在代码中发现的真实工程问题

| 问题 | 当前实现 | 风险 |
|---|---|---|
| Bash 参数持续膨胀 | `run_full_spectral_transfer.sh` 接近 500 行，大量 stage flags 和超参数 | 新增实验需要修改长脚本 |
| 执行逻辑不统一 | Bash 自己处理 `resume/skip`；Python runner 使用另一套 manifest 和 subprocess | 两种完成状态可能不一致 |
| Stage 与脚本路径硬编码 | `build_stages()` 直接构造大量命令、seed 循环和结果路径 | 改文件名或目录需要修改编排器 |
| 产物验证太弱 | Python runner 主要检查文件存在；Bash 用 `-s` 判断是否跳过 | 旧版本、错误 schema 或损坏的产物可能被复用 |
| Manifest 身份不完整 | 当前 hash 主要覆盖 manifest 描述，没有纳入数据和 checkpoint 的内容校验 | 固定配置不等于固定实验输入 |
| 失败恢复不完整 | Python 执行器仅在全部 stage 完成后写最终报告 | 中途失败时缺少统一、持久的结构化运行状态 |
| 研究模块覆盖分裂 | 统一协议重点仍是 geometry、memory 和 contact；GDD-WM、RGB successor 等另有入口 | 新实验容易脱离统一 provenance 和证据体系 |

这些判断可直接从两个入口的当前实现确认。<Cite refs={["turn575980view0","turn575980view1"]}/>

还存在两个具体问题，应该优先修复：

**1. Bash 的 `--skip-completed` 只检查产物文件是否非空。** 如果更换了数据、checkpoint、代码或 seed，只要旧文件仍存在，就可能跳过重新计算。这对普通工程是缓存错误，对你的项目则可能影响科学结论。<Cite ref="turn208657view2"/>

**2. Python 的 manifest 虽然有 SHA-256，但执行时没有重新验证 hash，也没有锁定外部输入文件的内容。** 因此它目前更准确地叫“防覆盖、带内容摘要的执行清单”，而不是完整的不可变实验快照。<Cite ref="turn208657view3"/>

建议将整理工作的首要目标定为：

> **每项科学实验只定义一次；编排方式可以不同；执行、缓存、统计和证据认定必须遵守同一份契约。**

---

## 二、我建议的最终架构

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Research CLI**
    <text color="secondary" size="xs">List / Plan / Run / Resume / Validate / Report</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **Full Data Pipeline**
        <text color="secondary" size="xs" textAlign="center">Raw data → RGB/latent export → audits</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **Frozen Protocol**
        <text color="secondary" size="xs" textAlign="center">Fixed inputs → matched studies → evidence</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **One Stage Registry + One Execution Engine**
    <text color="secondary" size="xs">Dependencies · configs · logs · caching · validation · failures</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <grid columns={3} gap={2}>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={2} align="center" gap={1}>
        **Core**
        <text color="secondary" size="2xs" textAlign="center">Models, geometry, dynamics</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={2} align="center" gap={1}>
        **Studies**
        <text color="secondary" size="2xs" textAlign="center">PointMaze, PushT, GDD-WM</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={2} align="center" gap={1}>
        **Evidence**
        <text color="secondary" size="2xs" textAlign="center">Statistics, gates, claims</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Versioned Artifacts**
    <text color="secondary" size="xs">Manifest · provenance · metrics · diagnostics · claim boundaries</text>
  </box>
</box>

这是一个**轻量科研架构**，不需要引入 Kubernetes、Airflow、数据库服务器或庞大的 MLOps 平台。

你已有的两条实验流程、配置和科学模块应继续存在，只需要逐步共享底层执行与产物管理能力。

## 三、必须优先修复的一个隐藏问题：六目标矩阵没有完全追踪本次实验

我进一步检查了 [`build_six_goal_evidence_matrix.py`](https://github.com/BokaiGuo/LeWorldModel/blob/main/le-wm-main/experiments/build_six_goal_evidence_matrix.py)。

发现它虽然是一个谨慎的 fail-closed 证据汇总器，但在当前实现中，很多输入来源仍是写死的历史目录，例如：

- `covariance_predictor_diagnostic_benchmark_20261008`
- `learned_adapter_closure_contactintervention4096_20261006`
- `pure_learned_phase_heads_audit_20261009`
- `transfer_certificate_20261008`

统一协议运行时，显式传给 claim matrix 的主要是几个 temporal summary；本轮 `geometry`、`controlled_alignment` 和 `contact_control` 的输出并没有全部通过 manifest 引用接入证据矩阵。六项目标的 `status` 也主要是代码中预先设定的状态。<Cite refs={["turn158213view0","turn158213view2"]}/>

因此，目前的 `claim_matrix` 更像是**历史研究快照的受限汇总**，不完全是“根据这一次 unified protocol 的全部实验产物重新计算的证据结论”。

它没有明显虚报科学成果，但容易造成两个问题：

1. 本次实验已经更新，矩阵仍引用旧结果；
2. 其他人以为 `--run` 完成后，六个目标已经基于本次运行全部重新评估。

这个问题比文件命名混乱更值得先修。

### 我的处理建议

把整个证据链明确分成三层：

<box gap={2} border radius="lg" padding={3}>
  <row gap={3} align="start">
    <box background="surface-secondary" padding={3} radius="lg">
      <icon name="file-json" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Experiment Artifacts**

      原始模型结果、预测误差、intervention outcomes、paired differences、独立 seed 和 confidence intervals。
    </box>
  </row>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <row gap={3} align="start">
    <box background="surface-secondary" padding={3} radius="lg">
      <icon name="clipboard-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Evidence Records**

      每个证据记录声明它支持哪个目标、运行协议、数据权限、统计单位、是否 privileged、是否 held-out、是否通过预先规定的 gate。
    </box>
  </row>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <row gap={3} align="start">
    <box background="surface-secondary" padding={3} radius="lg">
      <icon name="book-open-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Claim Matrix**

      只消费明确指定的 evidence records，应用预定义的判定规则，然后生成六目标结论；缺少证据就标记 missing / blocked，不能用未知来源的历史结果自动补齐。
    </box>
  </row>
</box>

历史结果可以继续使用，但必须在本次 manifest 中显式声明为 `external_prior_evidence`，并固定其内容校验和。

**这个设计能够从根本上防止科研代码重构后最危险的问题：实验执行没错，但论文引用的结果并非本次协议生成。**

## 四、建议采用的项目目录结构

我不建议现在把上游 LeWM 代码整体搬迁、改包名、全面重写。应该在现有 `le-wm-main/` 内逐步引入一个独立的 research package。

下面是**目标目录结构草案**，不是当前仓库已经具有的结构。

<box border radius="lg" padding={3} gap={1}>
  <text weight="medium" size="sm">LeWorldModel/</text>
  <text size="sm" preserveWhitespace>{`├── le-wm-main/
│   ├── lewm/                 # 上游 LeWM，尽量保持原状
│   ├── src/lewm_research/    # 可复用科研代码
│   │   ├── core/            # 数据接口、数学、统计、IO
│   │   ├── diagnostics/     # 谱、alignment、transfer
│   │   ├── dynamics/        # history、adapter、contact
│   │   ├── planning/        # MPC、HTSC、risk gate
│   │   ├── decision_geometry/ # GDD-WM
│   │   ├── runtime/         # stage执行、缓存、恢复
│   │   └── evidence/        # 统计检验、claim评估
│   │
│   ├── experiments/
│   │   ├── run_full_spectral_transfer.sh
│   │   ├── run_unified_research_protocol.py
│   │   ├── theory/
│   │   ├── pointmaze/
│   │   ├── pusht/
│   │   ├── cross_environment/
│   │   └── gdd_wm/
│   │
│   ├── protocols/           # 冻结的研究配方
│   ├── configs/             # 环境和超参数
│   ├── tests/               # 单元、协议、集成测试
│   └── results/             # 本地生成物，不进Git
│
├── docs/
│   ├── research/
│   ├── protocols/
│   └── architecture/
└── release_results/        # 公开的精选证据`}</text>
</box>

这里 `lewm/` 只是上游模型代码的概念位置，不要求它目前就以这个名字存在；**不要为了统一目录而移动上游依赖代码**。

### 每一层应该负责什么

| 层次 | 应该放什么 | 绝对不应该放什么 |
|---|---|---|
| `core/` | 数据结构、文件 IO、共用数学函数、随机性控制 | 论文 claim 及环境专属路径 |
| `diagnostics/` | Covariance、Schur、commutator、transfer operators | 读取某次实验的固定结果目录 |
| `dynamics/` | Adapter、history lifting、contact effect | CLI 参数解析和实验报告 |
| `planning/` | MPC、risk gate、candidate scoring | 读取 held-out outcome 进行策略选择 |
| `decision_geometry/` | GDD-WM loss、projection、distillation | 直接依赖历史 PointMaze 产物路径 |
| `runtime/` | Stage 运行、日志、恢复、锁、缓存、manifest | 科学结论判定 |
| `evidence/` | Paired statistics、validation gates、claim aggregation | GPU 训练和模型推理 |
| `experiments/` | 极薄的 CLI 入口和实验组装 | 重复实现数学算法和通用执行器 |
| `protocols/` | 固定 DAG、输入、seed、阈值与版本 | 大段 Python 业务逻辑 |

关键原则：

**算法在 package，实验在 studies，运行策略在 protocol，证据规则在 evidence。**

同一个 transfer operator 如果被 PointMaze、PushT 和 GDD-WM 调用，必须来自同一个可测试的实现，而不能分别复制三份。

---

## 五、两个总入口怎样改造最合适？

不删除、不合并文件名，只把它们改造成调用共同执行引擎的不同 workflow。

| 入口 | 建议保留的职责 | 应迁移出去的职责 |
|---|---|---|
| `run_full_spectral_transfer.sh` | 参数兼容、选环境和模型、调用 full-data workflow | Stage 命令拼接、缓存判断、复杂日志处理 |
| `run_unified_research_protocol.py` | 生成或加载冻结协议、请求执行、取得证据报告 | 大量具体实验路径、seed 循环、subprocess 执行细节 |

共同底层提供：

**Stage Registry → Dependency Resolution → Run Plan → Executor → Artifact Validator → Evidence Aggregator**

对于 Bash，建议最终让它变成一个很薄的兼容包装器，而不是逐渐发展成几百行的第二套实验框架。

对于 Python runner，保留固定协议的特征，但将其当前的 `build_stages()` 拆为可复用的 stage specifications 和独立的 protocol recipe。

### 什么是 Stage Registry？

不是做一个复杂的自动插件系统。

初期只需要每个 stage 声明：

| 属性 | 示例 |
|---|---|
| Stage ID | `pointmaze.rgb_successor_verify` |
| Version | `v1` |
| Inputs | RGB trajectories、frozen checkpoint、candidate bank |
| Outputs | Successor predictions、evaluation metrics |
| Dependencies | Candidate export、world-model inference |
| Config | Horizon、split seed、batch size |
| Resources | CPU/GPU、内存要求 |
| Validator | 输入/输出 schema、split、一致性检查 |
| Evidence role | Goal 3 的 successor 证据；不能直接支持 Goal 4 |
| Execution | 具体的现有 Python runner |

这样，`run_full_spectral_transfer.sh` 与 `run_unified_research_protocol.py` 可以使用同一 stage，但配置不同、执行范围不同。

而新加入的 GDD-WM、RGB successor 或 contact intervention，只需要注册对应 stage，再由适当的 workflow 引用。

**不要每新增一个研究方向，就修改所有总入口。**

---

## 六、所有实验必须统一的产物规范

这是我认为能大幅减少后期混乱的工程改动。

建议每次实验使用独立 `run_id`，并且同一层只管理本次运行的文件。

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium" size="sm">推荐单次 Run 的输出结构</text>
  <text size="sm" preserveWhitespace>{`results/
└── pointmaze/
    └── rgb_successor/
        └── <run_id>/
            ├── manifest.json
            ├── run_status.json
            ├── inputs.lock.json
            ├── stages/
            │   ├── export/
            │   ├── predict/
            │   └── evaluate/
            ├── metrics.json
            ├── evidence.json
            └── summary.md`}</text>
  <divider color="subtle"/>
  <text color="secondary" size="xs">大型 latent arrays、视频和 checkpoints 仍可存放在外部对象存储或本地目录；在产物中保留引用及 checksum。</text>
</box>

### 我建议增加的四条强制规则

**规则 A：运行身份必须由实际输入决定。**

每次运行记录 Git commit、dirty state、依赖版本、checkpoint checksum、dataset revision、split roots、seed 和 protocol version。

Manifest hash 要在执行前重新核对，并锁定相关输入内容；如果文件或配置发生变化，就创建新的运行身份。

**规则 B：真正可恢复的缓存不等于“文件存在”。**

只有当 stage 输入 fingerprint、代码版本和输出 schema 都匹配，且产物验证通过时才能跳过。缓存判断还要覆盖依赖 stage 的变化。

**规则 C：产物必须原子提交。**

先写临时文件，成功完成并验证后再 rename；中断和崩溃要留下 `failed/interrupted` 状态、日志和失败原因，而不能留下一个看似可用的半成品。

**规则 D：Dry-run 不能有副作用。**

当前 Bash 入口会在 dry-run 路径执行 `mkdir -p "$OUTPUT"`。建议未来 dry-run 只输出执行计划和输入检查，不创建正式结果目录。<Cite ref="turn208657view2"/>

---

## 七、科学上最重要的接口：区分“实验成功”和“主张成立”

建议统一两个互不依赖的状态字段。

| Execution status | Evidence status |
|---|---|
| planned | not_evaluated |
| running | insufficient |
| completed | partial |
| failed | negative |
| blocked | supported |
| interrupted | not_applicable |

例如：

- PointMaze RGB successor 脚本成功结束：`execution=completed`；
- 后继预测得到改善：对应 successor evidence 可记录 `supported`；
- Frozen RGB-only online planning 尚未完成：Goal 4 仍然 `blocked/insufficient`。

不要由执行器推断科学结论。

另外建议把六目标矩阵定义成**版本化的判定规则**。每个 goal 都需要明确的 outcome 数据、统计检验、最小实际效应、独立性单位、支持条件以及允许使用的表述。

特别是 GDD-WM，其 `D0 schema validated` 不应被自动提升为 `D1 teacher validated`，更不能提升为 `D4 teacher-free planning supported`。

你已有的 GDD-WM D0 validator 对 split、provenance 和 candidate table 的检查可以直接保留，不需要重新开发一个功能相似的 validator。公共框架只管理其输出契约与执行状态。<Cite ref="turn158213view3"/>

---

## 八、历史实验如何整理而不破坏可复现性？

我建议建立一个轻量的 `EXPERIMENT_REGISTRY.md`，为每个已有入口记录：

| 分类 | 处理方式 |
|---|---|
| Active | 正在维护的新机制和主实验 |
| Baseline | 论文或最新实验必须保留的对照 |
| Legacy | 历史研究用，冻结行为，不再添加功能 |
| Superseded | 被新协议替代，保留迁移说明 |
| Test | 只能通过测试运行器执行 |
| Utility | 数据转换、summary、validation，不是完整实验 |

尤其不能直接删除负结果对应的实验脚本，因为你的项目非常依赖这些结果解释为什么 memory、contact 或 alignment 路线没有成立。

对于被新实现替代的脚本：

- 保留历史协议版本和对应 Git revision；
- 建立旧入口与新入口的映射；
- 若提供兼容包装器，显式声明它调用的新实现版本；
- 对于无法完全复刻的新实现，不得声称历史结果在语义上完全等价；
- 将可维护的新模块迁入 package，历史快照进入 legacy 文档索引。

**保留历史结果，不等于永远维护所有历史代码。**

这是科研项目与普通软件项目很重要的区别。

---

## 九、建议的重构实施顺序

我不建议一次性移动全部 Python 文件。那样很可能破坏已有 checkpoint 的加载、import path、Hydra 配置和结果可复现性。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="clipboard-list" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **PR 1：仓库盘点与分类，不改科学算法**

      生成实验入口索引、依赖图、数据流、产物映射和入口使用说明。标记 Active / Legacy / Baseline。保留两个入口原有行为，冻结一组 golden fixtures。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="file-lock" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **PR 2：统一 Manifest、Artifact 和 Stage Validation**

      修复哈希校验、输入 fingerprint、stage status、产物 schema 与安全恢复。暂时不重构模型代码。

      这是收益最大、最应该优先合并的改动。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="workflow" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **PR 3：让两条总入口共享执行器**

      提取 Stage Registry 和执行引擎。先只迁移 geometry、latent export、empirical audit 和 claim matrix 几个 stage，比较新旧产物是否一致。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="folders" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **PR 4：按研究域整理代码**

      逐步整理 PointMaze、PushT、cross-environment、GDD-WM。将共用的数学、数据与模型接口迁入 package，留下薄 CLI 入口。每迁移一个模块都做行为回归。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **PR 5：重构 Claim Matrix**

      历史 evidence snapshot 与本次运行 evidence report 分开。统一协议只能根据显式声明的产物集合计算本次 claim；统计阈值和允许主张作为版本化配置保留。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="check-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **PR 6：轻量自动化质量门槛**

      接入 lint、类型检查、schema tests、数据泄漏检查、小型 CPU 集成测试、shell 检查和 golden artifact regression。GPU/真实数据全量实验单独执行，不作为普通 PR 必须完成的检查。
    </box>
  </row>
</box>

### 每个重构 PR 的验收标准

代码结构优化必须同时满足：

- 同一固定输入的新旧核心数值结果一致，或变化被明确记录；
- `--dry-run` 能给出真实执行计划；
- 输入数据或 checkpoint 变化时，旧缓存不会被错误复用；
- 任一 stage 失败都能记录失败原因并安全恢复；
- Claim matrix 不会隐式读取本次协议未声明的历史文件；
- 历史负结果及其原始协议仍然可追溯；
- 不通过修改统计阈值或重分类证据来掩盖工程重构引起的结果变化。

---

## 十、以后每次新增研究实验，应该遵守的规则

比如你准备开发一个新的 GDD-WM D2 geometrizability benchmark。

不要再直接创建一条完整独立的“训练—评估—summary—六目标判断”链。

应该先检查共用模块是否存在，再增加一个 study-level runner；定义固定的输入输出契约、实验配置、对应测试和证据角色。最后由 GDD-WM workflow 注册该 stage。

**只有科学算法是新的，才应该增加新的核心模块。只是更换环境、seed、horizon、checkpoint 或训练参数，原则上都属于配置变化。**

我还建议设立一条简单的维护规则：

> 新增 runner 时，必须明确为什么现有 runner 加新配置不能完成同一实验。

这能有效防止 `run_xxx_v2.py`、`run_xxx_final.py`、`run_xxx_fixed.py` 不断出现。

---

## 最终建议：不要追求脚本少，而要追求科研逻辑只有一份

我不主张把当前大量实验文件粗暴合并成一个超级 `research.py`。

你的项目包含谱理论、PointMaze、PushT、GDD-WM、风险证书和六目标审计，它们科学职责本来就不同。适度独立是正确的。

我最建议立即落实的是：

| 优先级 | 行动 | 直接收益 |
|---|---|---|
| P0 | 修复 claim matrix 的历史路径耦合 | 保护科学证据链 |
| P0 | 统一 manifest、缓存与产物验证 | 避免错误复用和不可复现 |
| P1 | 共用 Stage Registry 与执行引擎 | 终止双重执行逻辑 |
| P1 | 实验脚本分类和目录索引 | 降低日常维护成本 |
| P2 | 提取共用模型、几何和评估模块 | 减少算法实现重复 |
| P2 | 统一 CLI 和配置 | 让新研究分支更容易接入 |

**项目最理想的状态是：**

研究者只需要知道自己在研究什么，不需要记住几十个脚本名；执行器知道怎样安全完成实验；证据系统知道哪些结果能够支持哪些主张；历史失败实验永远可追溯。

这样 LeWorldModel 不仅能支撑现在的 HTSC 和 GDD-WM，也能继续扩展新的 world-model 研究，而不会在每一次新增实验后都增加一层难以理解的工程复杂度。

-----

> [User is quoting/referring to: "最接近的工作它已经证明或提出的东西LeWorldModel 必须超越的方面ATM: Why Latent World Models Can Fail to Plan真实/预测 inverse risk、跨域动作一致性与 model screening [![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=128)arxiv.org](https://arxiv.org/abs/2606.09028)经过 ATM 条件化后，仍能发现额外的有限时域决策失败"]
>
> 为我深度解析这篇论文的insight

我会先读 ATM 的原论文（尤其是问题设定、理论命题、指标定义和实验），再结合 LeWorldModel 的 HTSC、action-to-task transfer、PointMaze 和 PushT，解释它真正发现了什么、哪些结论容易被过度解读，以及你怎样做出超越 ATM 的研究贡献。重点会放在机制和数学逻辑，而不是泛泛复述摘要。

# ATM 论文深度解析：为什么世界模型能预测未来，却依然不会规划？

<text color="secondary" size="sm">论文：<Link url="https://arxiv.org/abs/2606.09028" title="ATM: Why Latent World Models Can Fail to Plan"/> ｜ Jiaheng Chen 等 ｜ arXiv v2，2026 年 9 月 30 日</text>

我阅读了 ATM 最新 v2 的方法、Bayes inverse risk 推导、跨域实验、AITS 干预、筛选结果与附录。

先给出我的核心判断：

**ATM 最深刻的 insight，不是“动作能否从 latent 中恢复”，而是“世界模型内部自洽的动作语义，可能与真实世界中的动作语义不一致”。**

模型可以非常确定地认为：

- 向右推会产生某种 latent displacement；
- 不同动作在预测空间里可以被完美区分；
- 模型内部的 inverse dynamics 几乎不会出错；

但这套动作—状态变化关系可能只是模型自己学出来的、与真实执行不兼容的规律。

**内部可辨识，不代表外部正确。**

这篇论文通过一个简洁的 $2\times2$ 矩阵，把两种很容易混淆的能力分开：

1. **Action identifiability：** 动作在 latent transition 中是否可辨识？
2. **Cross-domain action consistency：** 模型预测的动作—后果关系，是否与真实世界的动作—后果关系兼容？

对于你的 LeWorldModel 项目，ATM 是一篇非常重要的直接竞争论文，因为它已经研究了 frozen latent 的动作语义和真实/预测后继一致性，而且在 LeWM、DINO-WM、TwoRoom、PushT 和 OGBench-Cube 上提供了实验证据。<Cite refs={["turn708521view0","turn331140view1"]}/>

但我也认为它留下了一个你特别值得攻克的缺口：

> **ATM 衡量的是动作信息能否跨真实与预测 transition domain 传递；LeWorldModel 需要进一步回答，这些信息是否足以在特定目标、有限规划时域和受限动作支持下，保证正确的动作排序与决策。**

这两个问题密切相关，但在数学上并不等价。

---

## 一、先理解 ATM 为什么提出这个问题

传统的视觉 world model 基本遵循：

$$

z_t=E_\theta(o_t),\qquad
\hat z_{t+1}=F_\theta(z_t,a_t)

$$

然后使用 CEM 等优化器选择动作：

$$

U^*=\arg\min_U d\big(\hat z_{t+H}(U),z_g\big)

$$

这里有一个隐藏的前提：

**不同候选动作所产生的预测 latent 差异，应当对应真实世界中有意义的动作后果差异。**

但训练 world model 时，我们通常只要求：

$$

\mathcal L_{\mathrm{pred}}
=
\mathbb E\|\hat z_{t+1}-z_{t+1}\|^2

$$

这个平均误差不能保证模型学习到正确的动作方向。

假设一种物理状态里，不同动作仅引起相当小、但对完成任务至关重要的物体旋转。训练数据中存在大量视觉背景变化和容易预测的平移。

模型可能把大部分容量用于降低平均预测损失，却忽略决定动作成败的旋转差异。

更微妙的是，预测器甚至可以学会一个稳定但错误的动作映射。

它对每个动作产生独特的 latent signature，看起来很懂动作，但这些 signature 不一定对应真实物理后果。

ATM 想诊断的就是这种现象。

---

## 二、ATM 的第一个 insight：用 Bayes inverse risk 定义动作可辨识性

作者不是直接测量动作 Jacobian，也不是只看两个预测状态之间的欧氏距离。

它构造 transition feature：

$$

\xi_t=
[z_t,z_{t+1},z_{t+1}-z_t]

$$

再训练 inverse probe：

$$

h(\xi_t)\rightarrow a_t

$$

直觉很简单：

如果给出动作前后的状态变化，你仍然无法判断到底执行了哪个动作，那么这个 latent transition 对动作的表达可能不充分。

作者将这个概念写成：

$$

\boxed{
R^*=
\inf_h\mathbb E\|h(\Xi)-A\|_2^2
=
\mathbb E\left[
\operatorname{tr}\operatorname{Var}(A\mid\Xi)
\right]
}

$$

平方损失下，最优 inverse predictor 是：

$$

h^*(\xi)=\mathbb E[A\mid\Xi=\xi]

$$

因此，最优预测风险正好等于观察 transition 以后，动作仍然保留的条件方差。<Cite ref="turn982688view3"/>

这有一个重要的统计意义：

- $R^*$ 小：看到 transition 以后，动作的不确定性比较低。
- $R^*$ 大：即使看到了 transition，仍无法很好恢复动作。

不过一定要注意：这是**相对于所使用的动作分布和 transition 表征**而言的。

如果数据里的动作本来几乎不变化，inverse risk 也可能很低，但这完全不代表系统拥有充分的动作激励。

而且 inverse risk 不低，也不必然意味着表征有缺陷。例如多个不同动作真的产生相同物理效果时，动作本身就不可能唯一恢复。

因此，inverse identifiability 是一个有用的诊断，但不是直接的控制充分性定理。

---

## 三、ATM 的真正创新：真实 transition 和预测 transition 是两个不同的域

这是整篇文章最重要的设计。

对于同一条真实数据：

$$

(o_t,a_t,o_{t+1})

$$

构造两种 transition。

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Same observation and action**
    <text size="xs" color="secondary">$o_t,\;a_t$</text>
  </box>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} gap={2} align="center">
        <icon name="eye" size="lg" color="secondary"/>
        **True domain (T)**
        <text size="xs" textAlign="center">Execute action in environment</text>
        <text size="xs" color="secondary">$z_{t+1}=E(o_{t+1})$</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={2} align="center">
        <icon name="brain-circuit" size="lg" color="secondary"/>
        **Predicted domain (P)**
        <text size="xs" textAlign="center">Roll out learned dynamics</text>
        <text size="xs" color="secondary">$\hat z_{t+1}=F(z_t,a_t)$</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **Train two inverse probes independently**
    <text color="secondary" size="xs">True-domain $h_T$ · Predicted-domain $h_P$</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Cross-evaluate each probe on both domains**
    <text color="secondary" size="xs">Same model, two domains, four inverse risks</text>
  </box>
</box>

数学上：

$$

\xi_t^T=[z_t,z_{t+1},z_{t+1}-z_t]

$$

$$

\xi_t^P=[z_t,\hat z_{t+1},\hat z_{t+1}-z_t]

$$

分别训练：

$$

h_T:\xi^T\rightarrow a,\qquad
h_P:\xi^P\rightarrow a

$$

然后形成：

$$

\boxed{
\mathbf A_{\mathrm{ATM}}=
\begin{bmatrix}
D_{T,T}&D_{T,P}\\
D_{P,T}&D_{P,P}
\end{bmatrix}
}

$$

这里第一下标代表 probe 的训练域，第二下标代表测试域。

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium">四个矩阵元素应该怎样理解？</text>
  <grid columns={3} gap={2}>
    <grid-item>
      <box padding={2} align="center">
        <text color="secondary" size="xs">Train ↓ / Test →</text>
      </box>
    </grid-item>
    <grid-item>
      <box padding={2} align="center">
        <text weight="medium" size="xs">True</text>
      </box>
    </grid-item>
    <grid-item>
      <box padding={2} align="center">
        <text weight="medium" size="xs">Predicted</text>
      </box>
    </grid-item>
    <grid-item>
      <box padding={2} align="center">
        <text weight="medium" size="xs">True probe</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="rgba(35,144,110,0.09)" radius="md" padding={2} gap={1}>
        <text weight="medium" textAlign="center">$D_{T,T}$</text>
        <text size="2xs" color="secondary" textAlign="center">真实动作信息</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="rgba(205,139,61,0.09)" radius="md" padding={2} gap={1}>
        <text weight="medium" textAlign="center">$D_{T,P}$</text>
        <text size="2xs" color="secondary" textAlign="center">真实规则→预测</text>
      </box>
    </grid-item>
    <grid-item>
      <box padding={2} align="center">
        <text weight="medium" size="xs">Predicted probe</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="rgba(205,139,61,0.09)" radius="md" padding={2} gap={1}>
        <text weight="medium" textAlign="center">$D_{P,T}$</text>
        <text size="2xs" color="secondary" textAlign="center">预测规则→真实</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="rgba(35,144,110,0.09)" radius="md" padding={2} gap={1}>
        <text weight="medium" textAlign="center">$D_{P,P}$</text>
        <text size="2xs" color="secondary" textAlign="center">预测内部可辨识性</text>
      </box>
    </grid-item>
  </grid>
  <caption>所有单元都是动作反向预测风险，越低越好；非对角单元用于发现真实与预测域的动作映射不一致。</caption>
</box>

**核心不是两个对角线是否都很小，而是某一域学到的动作反向映射能否在另一域成立。**

例如：

$$

D_{P,P}\ll D_{P,T}

$$

说明模型内部可以非常容易地恢复预测时输入的动作，但这个 inverse rule 一旦面对真实 transition 就失效。

这是非常有力的模型诊断信号。

反过来：

$$

D_{T,T}\ll D_{T,P}

$$

表示真实世界中的动作语义比较清楚，但预测器可能没有保留这种语义。

ATM 因此比单一 inverse loss 更有解释力。<Cite ref="turn708521view0"/>

## 四、读实验时真正应该关注什么？

论文的实验不仅支持了 ATM 的想法，也暴露了它目前无法解决的问题。

### 1. 最强的结果其实是 $D_{T,T}$，不完全是整个 ATM 矩阵

作者在 OGBench-Cube 上比较了预测损失、真实动作可辨识性，以及增加跨域 transfer penalty 后与规划成功率的相关性。

<text color="secondary" size="xs">ATM v2 · Table 1，OGBench-Cube</text>

| 模型质量指标 | Spearman ρ ↑ | Pearson r ↑ |
|---|---:|---:|
| Negative prediction loss | 0.4983 | 0.3927 |
| Negative true-domain inverse risk | 0.8130 | **0.7015** |
| Inverse risk + transfer mismatch | **0.8239** | 0.7006 |

<Chart content={{"chartType":"bar","xKey":"method","series":[{"dataKey":"rho","label":"Spearman correlation"}],"yAxisMin":0,"yAxisMax":1,"data":[{"method":"Prediction MSE","rho":0.4983},{"method":"True inverse risk","rho":0.813},{"method":"Full ATM score","rho":0.8239}]}}/>

这个结果很有意思。

从 prediction loss 切换到真实域 inverse risk，Spearman 提升了 **0.3147**。

但继续增加跨域 mismatch 后，Spearman 只提升 **0.0109**，而 Pearson 甚至略微下降。

所以论文最坚实的经验发现是：

**真实动作后果在 latent 中的可辨识程度，可能比平均 latent prediction error 更能解释模型间的规划性能差异。**

至于 full ATM 是否能提供非常显著、稳定、跨环境的增量预测力，这组相关性本身给出的证据要弱一些。

论文也承认这一层级：$D_{T,T}$ 是 primary diagnostic，cross-domain mismatch 是 secondary，matrix-level asymmetry 是 tertiary。<Cite ref="turn982688view0"/>

对于 LeWorldModel，这是一个重要提醒：

你不应该直接把 geometry、commutator、transient、action risk 和 successor risk 全部加权成一个复杂的 composite score。

必须先证明每个新增信息源在控制更简单指标之后，是否仍有真实增量。

### 2. AITS 干预实验揭示“在哪里编码动作”比“动作是否可解码”更重要

ATM 不只是 post-hoc screening。

论文还设计了两类训练时干预：

$$

\mathcal L=
\mathcal L_{\mathrm{WM}}
+\alpha_T\mathcal L_T
+\alpha_P\mathcal L_P

$$

其中：

$$

\mathcal L_T=\mathbb E\|h_T(\xi^T)-a\|^2

$$

$$

\mathcal L_P=\mathbb E\|h_P(\xi^P)-a\|^2

$$

AITS 加强真实后继的动作可辨识性。

AITS-P 加强预测后继的动作可辨识性。

特别要区分：这些干预在 world model **训练阶段**使用；训练后进行 ATM 诊断时，重新训练 probes，而不是复用辅助训练头。AITS 的改进也不意味着完全冻结的基础模型在零训练情况下获得了同样收益。<Cite ref="turn331140view1"/>

<text color="secondary" size="xs">ATM v2 · Table 2，Success rate (%)</text>

| 方法 | TwoRoom | PushT | OGBench-Cube |
|---|---:|---:|---:|
| LeWM baseline | 87 | 96 | 74 |
| + AITS | 92 | 100 | **84** |
| + AITS-P | **94** | 100 | 78 |
| + AITS + AITS-P | 92 | 98 | **84** |

这个表最值得注意的是 OGBench-Cube。

AITS 令成功率从 74% 到 84%，而 AITS-P 只达到 78%。

作者还发现 AITS-P 可以大幅降低预测域 inverse risk $D_{P,P}$，却造成明显的跨域不对称。

说明：

**把预测器训练得非常擅长在自己的 latent transition 中编码 action，不一定像保留真实动作后果那样有利于复杂规划。**

但也要注意 TwoRoom：AITS-P 反而取得了比 AITS 更高的成功率。因此不能把 cross-domain asymmetry 简单理解为越低，规划必然越好。

我认为正确的理解是：ATM 诊断可能揭示某类失败倾向，但不是规划成功率的单调控制变量。<Cite ref="turn811782view0"/>

### 3. 98.81% 的模型排序准确率很强，但有边界

论文使用：

$$

S_{\mathrm{ATM}}
=
-D_{T,T}-\lambda|G_{T\rightarrow P}|

$$

在 OGBench-Cube 上给 world models 排名。

<text color="secondary" size="xs">ATM v2 · Table 3</text>

| 比较范围 | Model pairs | Ranking accuracy |
|---|---:|---:|
| 所有候选对 | 1,865 | 82.84% |
| Success gap > 2% | 1,338 | 91.48% |
| Success gap > 5% | 841 | **98.81%** |
| Success gap > 10% | 123 | 100% |

这说明 ATM 特别善于区分性能差异明显的模型。

但它不是在随机抽取一个新环境之后，预测任意模型的绝对成功率达到 98.81%。

这里至少存在三个边界。

第一，$\lambda$ 在同一任务的候选模型集合上拟合。它是经过任务内校准的排序器，不是完全 zero-shot 的普适分数。

第二，大量模型对共享同一批 checkpoints，因此 1,865 个 pairwise comparisons 不能视作 1,865 个独立训练模型。

第三，只保留成功率差异较大的候选对，会让排序任务相对容易。论文对筛选口径是透明的，但结果不能被理解成所有模型对都有如此高的可靠性。<Cite ref="turn982688view1"/>

更有信息量的是跨架构实验：

| 跨模型族校准 | LeWM–DINO ranking accuracy |
|---|---:|
| 直接迁移 LeWM 系数 | 68.54% |
| 在混合模型池重新拟合系数 | 89.90% |

这揭示了一个你应该特别注意的事实：

**ATM 的基本信息在不同 world-model families 中可能仍有预测价值，但其数值尺度与组合系数还不能被认为普适。**

这与 LeWorldModel 的 Goal 1 直接相关。

如果你希望宣称 alignment 或 transfer risk 能普遍预测模型质量，最终必须在完全未见的 backbone 和 environment 上冻结诊断器，进行真正的 out-of-family 验证，而不只是重新拟合权重。<Cite ref="turn331140view2"/>

### 4. ATM 的工程优势是低成本诊断，不是计算免费

论文报告，在缓存 latent transitions 后，ATM 的 probe 诊断耗时约 3–5 秒，而其使用的 CEM simulator evaluation 协议需要从分钟到小时不等。<Cite ref="turn331140view3"/>

这对你的项目有实际价值：ATM 非常适合作为 LeWorldModel evidence pipeline 里的一个低成本 baseline。

但时间比较没有包含获取和缓存全部真实 transition、计算 frozen model features，以及新任务上数据采集的所有代价。

因此更准确的理解是：

**给定已经准备好的离线 transition 数据，ATM 可以用较低成本辅助 checkpoint screening。**

而不是它已经取代真实环境中的规划验证。

---

## 五、从强审稿人角度看，ATM 仍然存在四个根本局限

这部分是我认为对 LeWorldModel 最有研究价值的内容。

ATM 的实验结论值得重视，但不能把它升级成“inverse-risk 一致就意味着物理正确”或“ATM 低就能可靠规划”。

下面给出几个专门构造的反例。

### 局限 1：Inverse risk 可能测到行为策略，而不是动作的物理效果

这是我认为 ATM 特别容易被忽略的一个问题。

注意它的 transition feature 包含初始状态：

$$

\xi=[z_t,z_{t+1},z_{t+1}-z_t]

$$

因此，inverse probe 不一定需要分析动作引起的变化。

假设行为策略是确定性的：

$$

a_t=\pi(z_t)

$$

即使环境完全忽略动作：

$$

z_{t+1}=z_t

$$

inverse probe 仍然可以只利用 $z_t$，完美预测 $a_t$。

此时：

$$

R_T^*=0

$$

但真实动力学的动作 Jacobian 却是：

$$

\frac{\partial z_{t+1}}{\partial a_t}=0

$$

我用一个离散系统进行了核对：令 $z_t\in\{-1,+1\}$、行为策略 $a_t=z_t$、真实后继始终为 $z_t$。Inverse risk 恰好为零，但动作没有任何实际控制效果。

这证明了：

**Observational action recoverability ≠ Interventional action-effect identifiability.**

从统计角度看也很清楚：

$$

R^*=
\operatorname{tr}\operatorname{Var}(A)
-
\operatorname{tr}\operatorname{Var}(\mathbb E[A\mid\Xi])

$$

它衡量的是 transition feature 对动作标签不确定性的消除程度，并不天然要求这种信息来自动作对环境的真实因果作用。

ATM 实验中的相关性仍有价值，但其 Bayes inverse risk 公式本身不能排除这种行为策略捷径。

**这恰好说明你的 PointMaze `+axis/-axis/zero` 独立动作干预为什么必要。**

真正严格的实验应该在匹配的初始视觉历史下执行不同动作，让：

$$

P(a\mid h_t)

$$

具有充分的干预覆盖，并区分仅使用 $h_t$ 的 action-prior probe 与真正使用 successor effect 的 inverse probe。

如果加入 successor 后没有提供额外信息，就不能仅凭低 inverse risk 宣称模型保留了动作动力学。

### 局限 2：即使 ATM 完美，也不代表动作到任务的传递正确

这是一个可以直接连接你 $G_H^\Phi$ 理论的反例。

考虑动作：

$$

a\in\{-1,+1\}

$$

真实后继为：

$$

z_{t+1}^{T}=
\begin{bmatrix}a\\a\end{bmatrix}

$$

预测后继为：

$$

z_{t+1}^{P}=
\begin{bmatrix}a\\-a\end{bmatrix}

$$

假设动作解码器在两个域都使用第一个 latent 通道：

$$

h_T(z)=h_P(z)=z_1

$$

那么：

$$

\boxed{
D_{T,T}=D_{T,P}=D_{P,T}=D_{P,P}=0
}

$$

也就是说，ATM 可以得到一个完美矩阵。

但如果实际任务输出由第二维决定：

$$

y=Dz,\qquad D=[0,1]

$$

且目标是 $y_g=1$，那么：

| 候选动作 | 真实任务代价 | 预测任务代价 |
|---|---:|---:|
| $a=+1$ | **0** | 4 |
| $a=-1$ | 4 | **0** |

预测器会选择与真实最优动作完全相反的控制。

对应的真实 regret 为 4。

这组结果已经用 Python 枚举核对。这里的零 ATM 表示存在一组在两个域都最优的共同 inverse readers；并不是声称任意有限样本 MLP 训练都会得到同样的矩阵。

即使让第二维加入连续随机噪声、使两域的支持集重叠，只要第一维仍精确标识动作，Bayes inverse 风险仍可为零，而任务方向仍然相反。

这个反例说明：

$$

\boxed{
\text{Inverse consistency}
\not\Rightarrow
\text{Task-effect consistency}
}

$$

原因很简单。

**恢复动作标签，不要求恢复任务相关的动作后果。**

ATM 检查的是：

$$

p(a\mid \xi)

$$

但规划真正需要的是：

$$

p(y_{t+H}\mid h_t,\operatorname{do}(U))

$$

这正是我认为 LeWorldModel 能够超越 ATM 的第一个核心切入点。

不是再做一个更复杂的 inverse-risk matrix，而是增加真实外部任务变量 $D_\Phi$，研究动作信息能不能正确到达任务输出。

### 局限 3：一阶动作可辨识不代表多步动作传递可靠

ATM 主要检查离线数据中的单步真实和预测 transition。

但真正的 CEM / MPC 比较的是整个动作序列。

对于线性系统：

$$

z_{t+1}=Az_t+Ba_t,\qquad y_t=Dz_t

$$

单步 action sensitivity 主要由 $B$ 决定。

而在有限时域：

$$

G_h=DA^{h-1}B

$$

即使真实模型与预测模型拥有相同 $B$，也可能因为 $A$ 不同，导致多步 action-to-task transfer 完全不同。

构造一个最简单的例子：

$$

A_T=1,\quad A_P=-1,\quad B=D=1

$$

假设初始状态 $z_0=0$，且校准数据只覆盖从这个初始状态执行一次动作。

对任何动作 $a_0$，两个模型都给出：

$$

z_1=a_0

$$

所以这些单步 transition 完全相同，ATM 无法区分。

但下一步如果执行零动作：

$$

z_2^T=a_0,\qquad z_2^P=-a_0

$$

于是：

$$

G_2^T=1,\qquad G_2^P=-1

$$

真实和预测的两步动作效果符号相反。

在目标为 $+1$ 的候选比较中，也会出现真实与预测最优动作颠倒的情况。

这个例子还说明了为什么动作支持集和 planner-reachable state support 必须同时考虑：如果校准轨迹从未访问 $z_1\neq0$，就没有证据支持模型在这些状态上的继续 rollout。

因此：

**One-step action consistency ≠ Finite-horizon decision consistency.**

ATM 论文也明确说明，它并没有直接估计 action Jacobian。<Cite ref="turn982688view3"/>

你现在研究的 finite-horizon Markov parameters、nonnormal amplification、closure remainder、tube risk 正是补齐这一层的自然工具。

但这些工具最终必须与独立真实后继误差连接，而不能只是更复杂的局部线性代数。

### 局限 4：ATM 是模型级诊断，不是候选动作级可靠性证书

ATM 的整体分数：

$$

S_{\mathrm{ATM}}=-D_{T,T}-\lambda|G_{T\rightarrow P}|

$$

主要回答：

> 在当前任务的数据分布下，哪个 checkpoint 的规划表现可能更好？

但在线控制需要回答：

> 在当前图像、当前接触状态、当前目标下，执行哪条动作序列可靠？哪些候选应该拒绝？

这两个问题不同。

一个 checkpoint 可能在整个 PushT 数据集上拥有不错的 ATM 结果，但在某个特殊接触模式下，正反扭矩预测完全错误。

全局平均 inverse risk 可能无法有效反映这种局部 failure region。

此外，ATM 的跨域风险是一个操作性的 probe transfer test，而不是严格的 probability-distribution divergence。作者自己在附录中也明确区分了这一点。<Cite ref="turn982688view4"/>

因此，它目前不能直接给出：

$$

\Pr\left(
J_{\mathrm{real}}(U)
\leq \widehat J(U)+\epsilon
\right)\geq1-\delta

$$

更不能无附加条件地推导任意在线 MPC 的 regret guarantee。

这是 ATM 留给你 HTSC 的第二个核心空间。

---

## 六、LeWorldModel 应该怎样真正超越 ATM？

我建议非常明确地划分二者的研究职责。

| 研究问题 | ATM 已经做到什么 | LeWorldModel 应该做什么 |
|---|---|---|
| 动作信息能否恢复 | Bayes inverse risk | 加入干预式动作可辨识性，排除行为策略捷径 |
| 真实/预测动作关系是否一致 | $2\times2$ cross-domain inverse matrix | 直接比较真实与预测的任务相关动作效果 |
| 多步动作影响是否正确 | 主要分析单步 transition | 建立 horizon-conditioned transfer error |
| 特殊接触模式是否可信 | 全局跨任务/模型诊断 | Mode-conditioned / history-conditioned risk |
| 是否能筛选好 checkpoint | Task-calibrated screening | Model-level 和 candidate-level 双层判别 |
| 是否允许执行动作 | 没有建立逐候选的可靠性证书 | Support-aware selective planning |
| 是否能够改善模型 | AITS / AITS-P | 分离可辨识性修复、预测动力学修复与 GDD-WM 几何蒸馏 |

**最需要避免的是：把 ATM 加入已有指标后，简单宣称 Full HTSC 比 ATM 的 correlation 更高。**

这还不够。

你需要证明一个更强、可证伪的命题：

> 在 action identifiability 与 cross-domain inverse consistency 已经匹配的情况下，world models 仍然可能出现显著不同的有限时域任务 regret；这种差异能够被 task-conditioned transfer diagnostics 解释并预测。

这会直接回答 ATM 尚未解决的问题。

## 七、我建议你设计一个专门“超越 ATM”的实验

这应该成为 LeWorldModel 一项非常重要的 controlled benchmark。

<title size="lg">ATM-Matched, Transfer-Divergent World Models</title>

核心思想：**先控制 ATM 所测量的能力，再观察真实任务传递是否仍然存在差异。**

<box gap={2} border radius="lg" padding={3}>
  <row align="center" gap={2}>
    <icon name="equal" color="secondary"/>
    <box flex="1" gap={1}>
      **Matching stage**

      构建预测 MSE、action prior、真实域 inverse risk、跨域 transfer gap 尽可能接近的 learned models。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="center" gap={2}>
    <icon name="split" color="secondary"/>
    <box flex="1" gap={1}>
      **Intervention stage**

      在相同 RGB 历史和目标下执行平衡的正反动作干预，并跨不同 horizon 收集真实任务后果。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="center" gap={2}>
    <icon name="scan-search" color="secondary"/>
    <box flex="1" gap={1}>
      **Transfer stage**

      比较 $G_h^\Phi$、真实/预测 task effect、successor consistency、history-conditioned support 和 planner-reachable errors。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="center" gap={2}>
    <icon name="route" color="secondary"/>
    <box flex="1" gap={1}>
      **Decision stage**

      在同一候选动作集合和真正 RGB-only 闭环 MPC 中，评价 regret、success、错误放行和 abstention cost。
    </box>
  </row>
</box>

可以设置几类有意构造的模型失败：

| 控制条件 | 研究什么新的差异 |
|---|---|
| ATM 相似，$DB$ 不同 | Task readout 和真实动作效果是否被保留 |
| ATM 相似，$DA^{h-1}B$ 不同 | 多步传播是否破坏决策 |
| Action risk 相似，contact mode 不同 | 模式内部物理因果可辨识性 |
| Prediction loss 与 ATM 相似，geometry 不同 | Geometry 是否提供独立决策信息 |
| 全局 ATM 相似，局部 support 不同 | 是否能在危险状态或新候选上正确拒绝 |

要注意，所谓 matching 应该通过独立训练设计或预先规定的分层匹配实现，不能只靠反复挑选直到出现想要的结果。

同时，人工构造的反例只能证明理论局限；**如果希望产生强实证贡献，必须在真正 independently trained learned world models 上再次观察相同现象。**

### 你应该优先建立的指标不是更复杂的 inverse loss

而是任务条件化的真实动作效果差异：

$$

\Delta_H^{task}(h,U,V)
=
\left[
J_H^*(h,U)-J_H^*(h,V)
\right]
-
\left[
\widehat J_H(h,U)-\widehat J_H(h,V)
\right]

$$

它直接衡量预测器是否正确表达两个动作之间的任务代价差异。

再进一步研究：

$$

\Pr\left(
\operatorname{sign}\Delta J_{\rm pred}
\neq
\operatorname{sign}\Delta J_{\rm real}
\right)

$$

与 ATM、action support、局部 transfer operators 和 history observability 之间的关系。

需要给真实比较设定明确的 minimum decision margin，因为两个实际代价几乎相等时，动作排序翻转不一定造成有意义的 regret。

这项实验能够把你的六目标框架真正落到可执行的任务后果上。

---

## 八、与你最新的 GDD-WM 有什么关系？

这里存在一个非常值得深入挖掘的连接。

ATM 主要回答：

**模型内部的动作后继是否保留了真实动作语义？**

GDD-WM 想回答：

**真实执行验证的动作偏好，能不能被写入模型的预测几何？**

前者研究动作语义的可辨识性，后者研究动作决策偏好的可几何表示性。

但这两件事中间缺少一个关键环节：

$$

\text{Action-identifiable transition}
\not\Rightarrow
\text{Correct preference ordering}

$$

即使知道是哪一个动作产生了后继，也未必知道哪一个后继对任务更好。

所以我建议 GDD-WM 不要直接把 ATM 当成一个加权训练损失或偏好正确性的证明，而是建立三道独立检验：

**第一道：Physical action consistency。** 真实与预测后继之间的动作效应是否一致？

**第二道：Decision preference validity。** Teacher 给出的排序是否与真实执行任务代价一致？

**第三道：Geometry realization and preservation。** Student 能否把这些正确偏好表达在自身 latent metric 中，并在 imagined rollout 中保留？

这三个环节构成：

$$

\boxed{
\text{Action Semantics}
\longrightarrow
\text{Validated Decisions}
\longrightarrow
\text{Predictive Geometry}
}

$$

它与 HTSC 的方向形成自然互补。

HTSC 判断是否具备足够的规划证据。

GDD-WM 在证据可靠的前提下，尝试将正确的决策知识内化到模型里。

这样研究的重心就不是重复 ATM 的 action recovery，而是研究**动作语义经过任务价值和预测几何两次转换之后，还能保留多少真实决策信息**。

---

## 九、从科研创新性的角度，我会怎样评价 ATM？

我认为 ATM 是一个思路很漂亮的诊断型研究。

它最值得学习的地方在于：

**没有一开始就设计一个复杂 world-model architecture，而是先发现一种隐藏失败机制，再用最简单的矩阵结构将其暴露出来，最后通过两类训练时干预去验证这个机制。**

它的研究故事很完整：

$$

\text{Observed planning discrepancy}
\rightarrow
\text{Mechanism hypothesis}
\rightarrow
\text{Diagnostic object}
\rightarrow
\text{Controlled intervention}
\rightarrow
\text{Downstream validation}

$$

这恰恰是你现在 LeWorldModel 需要进一步强化的地方。

但如果作为审稿人，我仍然会保留以下判断：

**ATM 对“动作能否从 transition 中读出来”给出了很清楚的统计形式；对“这种结构如何对应真实物理动作效果”给出了有力的经验迹象；但并没有证明 inverse consistency 是任务规划的充分条件。**

其局限集中在行为策略混淆、任务条件化不足、单步与多步之间的差异，以及缺少候选级和闭环级的误差保证。

而你的研究恰好有工具深入这些问题：$(C_\Phi,A_\Phi,B_\Phi,D_\Phi,M_{\rm task})$、history lifting、finite-horizon transfer、真实 RGB successor、contact intervention 和 HTSC permission gates。

这些工具的价值不在于数量，而在于能否真正跨过 ATM 没有跨过的理论和实证边界。

## 十、我认为 ATM 对你最大的启发

**不要再问“哪个 latent geometry 指标与成功率相关”，而要问“这个模型缺失了哪一种决策所必需、但现有诊断没有测量的信息”。**

ATM 已经揭示：

$$

\text{Predicted action decodability}
\not\Rightarrow
\text{Real action consistency}

$$

你的 LeWorldModel 值得继续揭示：

$$

\text{Real action consistency}
\not\Rightarrow
\text{Finite-horizon task transfer}

$$

并进一步研究，在什么可验证条件下可以得到：

$$

\boxed{
\text{Identified Transfer}
+
\text{Calibrated Decision Error}
\Longrightarrow
\text{Reliable Action Ranking}
}

$$

这里必须把 action support、任务目标、误差覆盖和时域写入定理假设，不能把它当作无条件普适结论。

**如果只能从 ATM 学到一个最重要的 insight，我会选这一句：**

> **一个世界模型最危险的错误，可能不是它不知道未来会发生什么，而是它在自己构建的预测世界里，对动作后果形成了一套高度自洽、却不符合真实物理世界的规律。**

LeWorldModel 要实现真正的突破，就需要更进一步：不仅发现这种自洽的错误，还能判断它在何种状态、目标和规划时域下会改变动作选择，并通过真实干预证据决定什么时候可以信任模型。

这才是 ATM 与你的 **Decision Transfer Identifiability of Frozen Visual World Models** 之间最重要的科学分界线。

-----

> [文件不支持导出]
> 根据这些insight进一步推进我们的研究方案

我会把 ATM 的四个机制性洞见与附件中的文献地图合并，直接推进为一套**可执行、可证伪的下一阶段研究方案**。重点是设计一个能排除行为策略捷径的动作干预实验，再证明 ATM 通过的模型为什么仍可能在有限时域任务排序上失败；最后把诊断结果接入 HTSC 和 GDD-WM，明确每一步的理论目标、强基线、统计门槛和代码改造边界。

我现在认为有一个可以直接形成论文核心的强切入点：构造两类 world model，使它们的 **ATM 四个 inverse risks 完全相同、latent prediction MSE 也相同**，但一个能选对动作，另一个必然选反。这样就能严格证明 ATM 与 MSE 的联合诊断仍不足以决定任务规划质量。

接下来关键不是停留在合成反例，而是把它变成 PointMaze/PushT 中的受控真实模型实验，再接上可校准的 task-effect 证书。这个主线比继续扩充独立 alignment 指标更集中，也更可能形成新的理论和实证贡献。

# LeWorldModel 下一阶段研究方案：从 ATM 的局限出发，建立可验证的 Decision-Transfer Sufficiency

结合你上传的《LeWorldModel 文献深度梳理》、此前对 ATM 的分析，以及最新公开仓库中的 HTSC、PointMaze、PushT 和 GDD-WM 进展，我建议对下一阶段研究进行一次**有重点的收敛，而不是继续增加独立的诊断指标和 adapter**。

最值得优先攻克的新科学问题是：

> **即使一个 world model 在真实/预测动作可辨识性、跨域 inverse consistency 和 latent prediction MSE 上都表现良好，它是否仍可能系统性地选择错误动作？如果会，我们能否在执行之前，识别并认证这种有限时域的 task-specific decision failure？**

这是一个可能真正超越 <Entity category="other" value="ATM: Why Latent World Models Can Fail to Plan" disambig="2026 paper on action consistency transfer matrix"/> 的研究问题。

ATM 已经通过 Bayes inverse risk 和真实/预测双域 inverse probes 揭示：模型内部能够识别动作，不代表动作关系能迁移到真实 transition。<Cite ref="turn951372view3"/>

你现在应该研究下一层：

$$

\boxed{
\text{Inverse Consistency}
\not\Rightarrow
\text{Task-Effect Consistency}
\not\Rightarrow
\text{Decision Sufficiency}
}

$$

这与附件中的文献地图一致，但要把它从研究定位推进为三个可检验成果：**反例定理、真实 learned-model 实证、数据校准的决策证书**。

## 一、先确立一个足以构成论文核心的理论反例

我建议把这个结果命名为：

**ATM–MSE Insufficiency for Task Decision-Making**

考虑平衡随机动作：

$$

a\in\{-1,+1\}

$$

从同一个初始状态出发，真实世界的后继 latent 为：

$$

z^T(a)=
\begin{bmatrix}
a\\a
\end{bmatrix}

$$

构造两个预测模型：

$$

\hat z^{(A)}(a)=
\begin{bmatrix}a\\-a\end{bmatrix},
\qquad
\hat z^{(B)}(a)=
\begin{bmatrix}a\\a-2\end{bmatrix}

$$

假设任务变量是第二维：

$$

y=z_2,\qquad y_g=1

$$

并使用平方目标代价。

此时两个模型的 latent prediction MSE **完全相等**。而所有真实与预测 transition 的第一维都直接等于动作，因此存在共同的最优 inverse reader：

$$

h(z)=z_1=a

$$

于是两个模型的 Bayes-optimal ATM 四个风险也全部为零。

但是规划结果完全不同。

| 候选动作 | 真实代价 | 模型 A 预测代价 | 模型 B 预测代价 |
|---|---:|---:|---:|
| $a=-1$ | 4 | 0 | 16 |
| $a=+1$ | **0** | 4 | **4** |
| 规划选择 | +1 | -1 | +1 |
| 真实 regret | 0 | **4** | **0** |

这些数值与两个模型 MSE 相等的结论已用 Python 枚举核对。

因此有一个很明确的存在性结论：

$$

\boxed{
\begin{gathered}
\mathrm{ATM}(A)=\mathrm{ATM}(B),\\
\mathrm{MSE}(A)=\mathrm{MSE}(B),\\
\mathrm{Regret}(A)\neq\mathrm{Regret}(B)
\end{gathered}
}

$$

这个反例不是说 ATM 没有价值，而是证明：**即使 ATM 和 MSE 联合起来，也不能无条件决定一个 world model 的 task-specific planning quality。**

这里的关键是模型 B 虽存在很大的预测偏置，却保留了对当前任务正确的动作排序；模型 A 则学到了相反的任务动作效应。Inverse reader 只需要第一维就能恢复动作，并不关心第二维的任务方向是否正确。

这还能连接传统 value-equivalence 研究：对决策而言，不同预测错误的影响可能截然不同。

**理论研究的第一步：** 将这个存在性反例扩展到具有重叠连续支持、有限历史、受限动作分布和多步动态传播的版本，并明确区分 Bayes-optimal ATM 与有限容量 probe 的经验结果。

它不是单靠一个反例就能达到 Oral 的定理；真正的价值来自随后证明这种信息缺口在真实 frozen visual world models 中确实存在，而且可以被提前检测。

---

## 二、把 ATM 升级为干预式诊断，而不是简单增加一个新分数

ATM 的 Bayes inverse risk 定义很漂亮，但它可能受到行为策略的影响。

因为 inverse feature 包含初始状态 $z_t$，如果数据中的动作主要由当前状态决定，probe 可能只看初始状态就能预测动作，无需利用真实后继变化。

因此，我建议将你现有的 `+axis/-axis/zero` 实验正式升级为：

**Interventional Action-Effect Identifiability Audit（IAEI）**

对相同视觉历史 $h$，引入两种 inverse risk：

$$

R_{\rm prior}^*
=
\inf_f\mathbb E\|A-f(H)\|^2

$$

$$

R_{\rm successor}^*
=
\inf_g\mathbb E\|A-g(H,Z_{t+1})\|^2

$$

研究额外的后继信息贡献：

$$

\boxed{
\Delta I_{\rm action}
=
R_{\rm prior}^*-R_{\rm successor}^*
}

$$

严格来说，这是平方损失下的可恢复动作方差增量，不是直接的 Shannon mutual information。

但它解决了一个关键诊断混淆：

- State-only probe 已经能恢复动作：可能是行为策略捷径。
- 加入 successor 后才显著改善：后继包含新的动作相关信息。
- 在平衡随机干预下仍能恢复动作：比被动观察更有力，但仍不自动证明 task-effect correctness。

所以你不能停在 $\Delta I_{\rm action}$。

必须进一步测量相同起点下两个动作的任务效果对比：

$$

\Delta_H^{real}(h,U,V)=
J_H^{real}(h,U)-J_H^{real}(h,V)

$$

以及模型对应的预测：

$$

\Delta_H^{pred}(h,U,V)=
\widehat J_H(h,U)-\widehat J_H(h,V)

$$

真正需要识别的是：

$$

\boxed{
E_H^{contrast}
=
|\Delta_H^{real}-\Delta_H^{pred}|
}

$$

这就把 **action recovery** 与 **task-effect recovery** 分开了。

附件提出的 action-support-aware、horizon-conditioned 诊断可以直接建立在这个对象上，而不是继续把 latent MSE 当作 action identifiability 的代理量。

## 三、下一步最值得提出的方法：Paired Decision-Transfer Certificate

我建议给 HTSC 增加一个比传统 prediction-error envelope 更贴近规划目标的对象：

**Paired Decision-Transfer Certificate（PDTC）**

它直接认证候选动作之间的相对代价，而不要求每个动作的完整未来状态都被高精度恢复。

### 1. 为什么应该直接认证 pairwise contrast？

考虑两条候选动作：

$$

U_i,\quad U_j

$$

设真实与预测任务代价的误差为：

$$

e_i=J_i^{real}-\widehat J_i,\qquad
e_j=J_j^{real}-\widehat J_j

$$

传统方法分别估计：

$$

|e_i|\leq\epsilon_i,\qquad |e_j|\leq\epsilon_j

$$

于是只能得到：

$$

|(J_i-J_j)-(\widehat J_i-\widehat J_j)|
\leq\epsilon_i+\epsilon_j

$$

这可能非常保守。

因为两个预测轨迹可能具有相同的系统性误差。

例如模型整体把目标距离预测得偏大，但仍然正确表达“动作 A 比动作 B 好”。

这种共同误差在动作代价差分中可以抵消：

$$

(e_i-e_j)

$$

因此直接校准：

$$

\boxed{
b_{ij,H}\geq |e_i-e_j|
}

$$

有可能比单独校准 $\epsilon_i+\epsilon_j$ 更紧。

这与你此前观察到的“预测 RMSE 与 selected regret 不同步”高度吻合。

**一个世界模型可以预测得不完全准确，却仍然给出正确决策；也可以预测误差很低，却在关键候选之间犯下排序错误。**

### 2. 给出明确的决策放行条件

如果模型预测：

$$

\widehat J_i<\widehat J_j

$$

并且有可信的 contrast-error bound：

$$

\left|
(J_j-J_i)-(\widehat J_j-\widehat J_i)
\right|
\leq b_{ij,H}

$$

那么只要：

$$

\boxed{
\widehat J_j-\widehat J_i>b_{ij,H}
}

$$

就可推出：

$$

J_j>J_i

$$

即模型对这组候选的动作排序确实正确。

这可以形成三种运行状态：

<box border radius="lg" padding={3} gap={2}>
  <row align="center" gap={3}>
    <icon name="check-circle-2" color="success" size="lg"/>
    <box flex="1" gap={1}>
      **Permit**

      当前候选具有充分动作支持；contrast margin 超过有效误差界；允许执行
    </box>
  </row>
  <divider color="subtle"/>
  <row align="center" gap={3}>
    <icon name="scan-search" color="secondary" size="lg"/>
    <box flex="1" gap={1}>
      **Probe / Shorten**

      当前证据不充分；优先缩短 horizon、重新观测或采取已验证安全的信息获取动作
    </box>
  </row>
  <divider color="subtle"/>
  <row align="center" gap={3}>
    <icon name="shield-alert" color="danger" size="lg"/>
    <box flex="1" gap={1}>
      **Abstain / Fallback**

      缺少动作支持、物理模式不可辨识，或有效误差界过宽；拒绝无依据的候选
    </box>
  </row>
</box>

这里的 `Probe` 不能自动视作安全动作。信息获取动作本身也需要独立的执行支持与安全限制。

### 3. 关键技术不是这个不等式，而是怎样得到 $b_{ij,H}$

我建议把 HTSC 现有的风险因子改造成 contrast-risk 的输入，而不是直接手工加权得到一个总分：

$$

b_{ij,H}
=
\mathcal B_\psi(
H,
S_{\rm action},
S_{\rm state},
R_{\rm ATM},
\Delta I_{\rm action},
\mathcal G_H^\Phi,
E_{\rm successor},
\mathcal M_{\rm mode},
\mathcal N_{\rm transient}
)

$$

其中：

- ATM 负责已有的真实/预测动作关系诊断；
- $\Delta I_{\rm action}$ 负责排除仅靠行为策略预测动作的捷径；
- $\mathcal G_H^\Phi$ 负责完整有限时域 action-to-task sensitivity；
- RGB successor 负责真实与想象的后继一致性；
- mode observability 负责接触系统；
- nonnormal/transient 解释误差传播，而不是单独决定 certificate。

这个式子首先是**待学习和验证的风险模型设计**，不意味着任何特征一经加入就自动得到有效的概率保证。

建议采用两个相互分离的过程：先在 fit split 上训练 contrast-risk predictor，再在完全独立的 calibration roots 上为其误差界定标。

对于固定的候选生成策略和可交换的校准/测试 roots，可以进一步考虑对每个 root 的**最大标准化 pairwise contrast error**做 split-conformal 校准，以覆盖该 root 上整个候选集合，而不只是一个随机 pair。

这样可以避免规划器在同一个候选 bank 中选择最乐观动作时，破坏仅针对单个随机候选建立的保证。

但必须明确：

**普通 conformal 的边际覆盖，不等于在任意当前 RGB 状态或任意 CEM 分布偏移下都具有条件覆盖。**

完全分布无关的非平凡逐状态条件覆盖一般无法保证，这是已有统计理论的明确边界。<Cite refs={["turn235747academia36","turn235747academia39"]}/>

因此我建议把最终保证分成两层：

| 层级 | 可追求的理论保证 |
|---|---|
| Fixed-bank / exchangeable-root | 对完整候选集合的边际同时覆盖，以及覆盖事件成立时的动作排序保证 |
| Online MPC / shifted distribution | 在明确的动作支持、环境漂移与校准假设下给出条件性保证；假设失败则拒绝，不宣称无条件安全 |

这比直接宣称“建立通用 world-model certificate”更有数学说服力。

---

## 四、将理论推进为五项决定性实验

这里我建议改变项目的实验组织方式：不要按“geometry、memory、adapter、contact”逐个证明模块好用，而是按**失效机制和证据转换**设计实验。

### 实验 A：ATM-Matched but Decision-Divergent

<text color="secondary" size="xs">优先级：P0 · 理论反例 → Learned-model 实证</text>

**研究假设：** 即便 MSE、ATM 和动作支持条件相近，不同 learned models 仍可能表现出系统性不同的 task regret。

第一步，在合成系统中严格匹配这些量，制造 task readout、长时域传播和非正规 transient 的差异。

第二步，在 PointMaze 上进行真实 learned-model 实验，使用预先规定的 model matching，而不是事后挑选最有利于假设的 checkpoints。

主要比较：

| 指标 | 需要回答的问题 |
|---|---|
| Prediction MSE | 传统预测误差能否解释差异 |
| ATM 四项 + transfer gaps | 真实/预测 inverse consistency 能否解释差异 |
| State-only inverse baseline | ATM 是否受到行为策略捷径影响 |
| $\Delta I_{\rm action}$ | 后继是否包含额外的动作信息 |
| Task-effect contrast error | 是否正确保留动作对任务的相对影响 |
| HTSC + PDTC | 能否提前预测错误排序和真实 regret |

关键统计结果不是单个 Spearman correlation。

而是：**在独立模型上，用 ATM + MSE 预测后，再增加 task-conditioned transfer 信息，能否显著减少 held-out decision-risk prediction error？**

同时报告匹配失败的比例和所覆盖的模型分布，不能只展示成功匹配的有利案例。

### 实验 B：Action Support × Horizon × Planning Stage

<text color="secondary" size="xs">优先级：P0 · 解释 ATM 为什么不能直接外推至 MPC</text>

构造三条正交实验轴：

- Action support：从良好激励逐渐变成某些动作方向缺失；
- Planning horizon：分别研究短、中、长时间跨度；
- Candidate distribution：random、CEM initial、CEM intermediate、CEM elite。

这里一定要把“动作支持”与“规划器访问的状态支持”分开。

即使动作本身落在训练区间内，CEM 也可能把模型推进到训练数据几乎没有覆盖的状态。

评价变量应包括：

$$

E_H^{contrast},\quad
R_H^{decision},\quad
\text{ranking inversion rate}

$$

以及各个条件下的 certificate coverage、false-release rate 和 abstention。

**真正有价值的结果：** ATM 的一阶逆动作风险可能几乎不变，但随着 horizon 增大和候选逐渐进入 CEM elite，task contrast risk 系统性升高；你的 horizon-conditioned 方法能够提前识别这种升高。

这是一个待验证的实验假设，不应提前当作结论。

### 实验 C：PushT Within-Mode Physical Intervention

<text color="secondary" size="xs">优先级：P1 · 物理机制验证</text>

最新 PushT 研究已经指出，free-space 和 approach 的改善没有稳定传递到 normal-contact 与 tangential-contact。修正 T 形物体接触几何后，strict pose closure 仍未实现。<Cite ref="turn784495view3"/>

因此，不要再仅将 PushT 当作最终 success benchmark。

建议将同一接触模式内的动作对比作为主要实验对象：

$$

\Delta_H^\theta(h,a,-a)
=
\mathbb E[
\theta_{t+H}^{do(a)}-
\theta_{t+H}^{do(-a)}
\mid h]

$$

对于角度跨越 $\pm\pi$ 的情况，需要在定义差异前明确采用环形统计或一致的展开约定。

比较 single-frame latent、history-conditioned latent、真实接触几何 oracle、mode-only adapter、action-effect adapter 和 joint mode/action adapter。

然后判断：

**错误源自当前视觉历史不可辨识，还是在已可辨识的信息上学错了 action dynamics？**

如果 privileged contact oracle 有效，但 RGB-history 方法无效，说明视觉状态辨识仍是瓶颈。

如果 RGB-history 已有充分 action information，而预测仍错，才有充分理由升级 learned dynamics adapter。

这能够避免继续通过增加 GRU 容量来掩盖真正的物理问题。

### 实验 D：HTSC + PDTC 的 RGB-only Closed-loop MPC

<text color="secondary" size="xs">优先级：P0 · 决定是否形成真正规划贡献</text>

这是当前最紧迫的系统实验。

最新仓库已经具有 PointMaze action-balanced benchmark 和 RGB temporal successor verification，但完整 RGB-only online planning 仍未完成。<Cite ref="turn951372view2"/>

建议直接复用已有：

`RGB observation → Frozen LeWM → Candidate rollout → Task cost → Risk certification → Action execution → RGB successor`

比较：

- 原始 LeWM MPC；
- 最优固定 horizon；
- Uncertainty-aware MPC；
- ATM-based model/candidate screening；
- HTSC absolute error gate；
- **HTSC + direct pairwise contrast certificate**。

主要输出不应只有最终 success rate。

最重要的是 **Risk–Coverage–Regret** 曲线：在相同放行率下，哪些方法真正更少错误放行；在相同计算和执行预算下，整体系统是否取得更好任务结果。

只有这个实验完成，才能把“诊断有用”升级为“诊断可以改变实际决策”。

### 实验 E：GDD-WM 的 Certificate-Guided Geometry Distillation

<text color="secondary" size="xs">优先级：P2 · 表示学习与决策反馈</text>

GDD-WM 应当在可信的 execution evidence 建立之后再进入主要训练阶段。

已有方案要求 D0 共享候选数据、D1 teacher preference stability、D2 projection-only、D3 联合训练和 D4 teacher-free MPC，并明确禁止将 schema validation 直接升级为已验证的 teacher/student 结果。<Cite ref="turn784495view5"/>

基于 ATM 的 insight，我建议新增一个机制：

**不要蒸馏所有 teacher-preferred pairs，而是优先研究同时满足以下条件的 pairs：动作有支持、真实执行差异可信、teacher 偏好稳定、真实/预测 contrast 不确定性可校准。**

这不是说只训练容易的 pair。应保留低支持、排序不一致与不可几何化的 hard pairs 作为独立评估集合，并记录被筛选掉的覆盖范围。

D2 阶段还可以增加一个很有价值的预检：

$$

M=P^\top P\succeq0

$$

对可信偏好检验是否存在一个 PSD metric，使固定预测 latent 上的排序约束可同时满足。

再分别检查 realized futures 与 predicted futures 上的几何可行性。

- Realized 不可表示：当前冻结特征或几何函数类存在限制。
- Realized 可表示、predicted 不可表示：预测器是更明显的瓶颈。
- 两条路径都可表示但闭环失败：需要检验 candidate shift、规划接口和长期控制。

这个预检会让 GDD-WM 从单纯调 ranking loss 转向研究**决策偏好能够被哪类预测几何表达**。

同时必须正面对比普通 Mahalanobis metric learning、pointwise scorer、D-JEPA 和 AD-WM；因为这些工作已经覆盖了很大一部分 action-aware 和 decision-aligned world-model learning。<Cite refs={["turn142758academia60","turn142758academia61"]}/>

---

## 五、统一研究的证据链：从 ATM 到 HTSC，再到 GDD-WM

我建议将项目的核心实验证据统一成下面这个顺序：

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Frozen RGB World Model**
    <text color="secondary" size="xs">Encoder · Dynamics · Historical observations</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **ATM + Interventional Identifiability**
    <text color="secondary" size="xs">Can actions be recovered? Does the successor add causal information?</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **Horizon-conditioned Task Transfer**
    <text color="secondary" size="xs">Does the model preserve the real difference between candidate actions?</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **Contrast-risk Calibration + HTSC**
    <text color="secondary" size="xs">Can the action ordering be trusted under the declared support?</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **RGB-only Closed-loop Planning**
    <text color="secondary" size="xs">Execute · Observe · Verify · Replan</text>
  </box>
  <divider color="subtle"/>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **GDD-WM: Execution-grounded Geometry Learning**
    <text color="secondary" size="xs">Can valid decisions be internalized without sacrificing prediction?</text>
  </box>
</box>

这张图的价值在于：每一层都有独立可测量的科学问题，且可以明确判断失败发生在哪一层。

特别重要的是，GDD-WM 不应该反过来影响前面独立测试集的定义。否则一旦用相同的实际执行结果既挑选 teacher pairs、又校准 HTSC、最后再报告规划收益，就会产生证据泄漏。

## 六、建议的研究阶段门槛

这是我建议直接纳入你的统一实验协议的评价矩阵。

| Gate | 科学验收目标 | 不能通过时如何处理 |
|---|---|---|
| G0：Data integrity | Root-disjoint、动作干预有效、真实/预测数据对应、无 privileged runtime 信息 | 阻断下游实验，不产生科学结论 |
| G1：ATM insufficiency | 合成反例严格成立；真实 matched models 中出现可重复的 ATM 未解释的 task regret 差异 | 保留理论边界，不声称真实模型存在普遍机制 |
| G2：Incremental transfer | 在独立模型上，task-transfer diagnostics 比 ATM + MSE 强基线有增量预测价值 | 缩减 geometry 特征，不继续堆指标 |
| G3：Certificate | 预先声明的覆盖要求成立；误差界非空洞；相同放行率下错误决策减少 | 保留诊断结果，不将其升级为 certificate |
| G4：Closed-loop MPC | 真正 RGB-only；公平预算；真实任务收益与拒绝成本一并报告 | Goal 4 保持未完成 |
| G5：Contact generalization | PushT 模式内部动作效应得到验证，并独立检验严格姿态闭合 | 不用 mode accuracy 代替物理控制成功 |
| G6：GDD-WM | 偏好可表示、预测不显著退化、teacher-free 独立规划受益 | 不进入 joint co-training 或更强主张 |

这里不能单靠“不显著差于基线”证明模型已被成功匹配或预测得以保持。

对 MSE、ATM 等关键匹配变量，应使用预先声明的等价容差或等价性检验；对 prediction preservation，应使用明确的非劣效界。不能将 $p>0.05$ 解释成两者等价。

对于 action contrast calibration，评价单位也要严格区分：

- Root：动作候选之间共享的初始状态；
- Episode：完整闭环运行；
- Model seed：独立训练的 world model；
- Environment：独立任务和动力学条件。

候选对不是彼此独立的实验重复。模型层面的泛化结论需要在模型 seed 和环境层面进行检验，而不能通过生成更多 candidate pairs 来制造统计显著性。

建议在独立 pilot 后确定最小有意义效应与样本量，然后冻结最终测试协议。

---

## 七、代码上应该怎样接入，避免再次扩大脚本体系？

这里要结合你刚刚整理过的科研工程架构。

最新公开的 `RESEARCH_ENGINEERING_GOVERNANCE_20261014.md` 已经记录：研究 runtime 具备输入 fingerprint、stage 状态、artifact 验证等功能，并且两条历史总入口正在共享执行契约。因此没有必要再开发第三套实验管理系统。<Cite ref="turn951372view1"/>

我建议新的 ATM → HTSC 扩展只增加少量可以独立测试的科学模块。

<box border radius="lg" padding={3} gap={1}>
  <text size="sm" weight="medium">建议的增量模块（拟新增，非当前已有文件）</text>
  <text size="sm" preserveWhitespace>{`experiments/
├── decision_benchmark/
│   ├── atm_baseline.py
│   ├── intervention_identifiability.py
│   ├── task_contrast_audit.py
│   └── contrast_certificate.py
│
├── protocols/
│   └── atm_to_htsc.yaml
│
└── tests/
    ├── test_atm_insufficiency.py
    ├── test_action_prior_shortcut.py
    ├── test_task_contrast.py
    └── test_decision_certificate.py`}</text>
</box>

这里最重要的不是具体文件名，而是复用现有的 candidate bank、RGB successor、risk-aware control、statistics 和 research runtime。

新的 protocol 只需编排以下 stage：

| Stage | 输入 | 标准输出 |
|---|---|---|
| `atm_audit` | Frozen checkpoint、真实/预测 transitions | `atm_matrix.json` |
| `intervention_audit` | Matched RGB roots、平衡动作、真实 successors | `action_effect_evidence.json` |
| `task_contrast` | 同一候选 bank 的真实/预测代价 | `contrast_metrics.json` |
| `certificate_calibration` | 独立 calibration split | `contrast_certificate.json` |
| `planning_evaluation` | Frozen planner、证书、fresh episodes | `planning_outcomes.json` |
| `claim_evaluation` | 经验证的全部 evidence records | 更新后的六目标矩阵 |

这些只是**建议的接口契约**，并不是声称仓库目前已经生成上述文件。

尤其不要把 ATM 实现成另一个拥有自己完整 train、eval、summary 和 claim pipeline 的总入口。

每个 stage 只负责一个明确的科学对象。所有证据继续由已有统一 runtime 和 claim matrix 管理。

历史负结果也要保留，不能因为引入 ATM 对照而重新定义已经失败的实验。

---

## 八、我建议的实际推进顺序

如果让我决定现在的研究资源分配，我会采用以下顺序。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="sigma" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **第一阶段：完成理论反例和 ATM 强基线**

      把上面的 ATM–MSE insufficiency 反例正规化，加入连续噪声和重叠支持，避免 cross-domain probe 因支持集不重叠而产生任意外推。实现标准 ATM、state-only inverse、平衡动作 intervention probe 三种对照。

      交付：可复现的理论反例、对应单元测试、ATM baseline audit。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="git-compare-arrows" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **第二阶段：在 PointMaze 证明真正的增量信息**

      复用最新的 shared candidate bank、root-disjoint 动作干预和 RGB successor。研究在 MSE 与 ATM 已匹配的情况下，horizon-conditioned transfer 与 contrast error 是否还能解释模型的 action regret。

      交付：ATM-matched learned-model study、真实任务 contrast audit、预注册匹配与消融分析。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **第三阶段：把 HTSC 做成实际的数据校准方法**

      使用独立根状态校准 pairwise contrast uncertainty，检验固定候选集合的同时覆盖，并与 absolute-error gate 比较非空洞程度及错误放行率。

      交付：Contrast-risk calibration、coverage study、selective regret 评估。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="route" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **第四阶段：实现真正的 RGB-only 在线控制**

      在固定 CEM budget 下比较 LeWM、ATM screening、HTSC absolute risk、HTSC contrast risk。不能用 simulator-state successor 代替运行时 RGB feedback。

      交付：完整闭环 episodes、Risk–Coverage–Regret 主图、真实 success 和 latency。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="brain-circuit" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **第五阶段：PushT 机制扩展与 GDD-WM**

      使用接触分层干预验证物理动作效应；对于 GDD-WM，先做证据筛选和固定几何可表示性检验，再决定是否值得进入 D3 联合训练。

      交付：Contact identifiability boundary、GDD-WM D1/D2 结果以及明确的进入 D3 条件。
    </box>
  </row>
</box>

这个顺序有一个重要好处：即使第五阶段没有取得预期结果，前三、四阶段如果成功，仍然可以形成一篇完整的研究论文。

不需要所有六个宏大目标同时通过，才能证明一个新的科学命题。

---

## 九、我会怎样设计最终的 Oral 级论文故事？

我建议把 HTSC 主论文暂时定位为：

<title size="lg">Beyond Action Identifiability: Certifying Decision Transfer in Frozen Visual World Models</title>

一篇论文只围绕三个核心贡献：

**Contribution 1 — Identifiability is not sufficient.**

证明即使 ATM 和 prediction MSE 都无法区分两个模型，它们的任务规划 regret 仍可以不同；指出真正缺失的是任务条件化、有限时域的反事实动作效果信息。

**Contribution 2 — Calibrated decision-transfer certification.**

通过真实动作干预和 successor verification，构造直接面向 action contrasts 的风险估计，在声明的支持集和统计假设下得到可检验的动作排序认证。

**Contribution 3 — Selective planning actually works.**

通过独立 RGB-only online MPC 证明，新的 transfer-risk 信息不仅能解释失败，还能够以可接受的拒绝成本减少真实错误决策，并在未见 learned models 上保留预测价值。

这是**值得争取 Oral 的研究叙事**，但不能仅凭反例或普通 conformal calibration 就宣称新理论。

已有 AD-WM 已经展示 action-discriminative training 与反事实 MPC 的关系；2026 年也已有研究将 robust conformal prediction 与 learned-dynamics planning 结合。你的增量必须是**冻结视觉表示中的真实 action-to-task contrast、物理可辨识性边界，以及针对候选动作选择的可靠认证**。<Cite refs={["turn142758academia60",""]}/>

GDD-WM 我建议保留为后续更偏 representation learning 的主论文方向：

*When Can Execution-Grounded Preferences Be Realized as Predictive Geometry?*

两者可以共享整个证据基础，但不必为了实现一个所谓“普适 world model”而全部挤进同一篇论文。

---

## 十、最终研究判断

你上传的文献梳理最有价值的一点，是明确区分了 prediction sufficiency、action identifiability 和 decision sufficiency。它还指出，完整多步 action-to-task transfer 需要考虑 $DA^{H-1}B,\ldots,DB$，不能只依赖一个谱量或单步逆动作风险。

结合 ATM，我认为应该进一步把整个 LeWorldModel 的主科学假设改写为：

> **Planning reliability is determined not merely by whether actions are recoverable from predicted transitions, but by whether the model preserves task-relevant counterfactual differences between candidate action sequences over the planning horizon.**

但要把这句话从合理假设变成真正的科学贡献，还必须证明：

**在 ATM、MSE 和动作支持已经受到严格控制时，真实模型仍存在可重复的 task-transfer gap；你的诊断能够在独立执行之前预测它；你的 certificate 能在明确条件下可靠判断动作顺序；采用该判断的闭环规划器确实表现更好。**

如果只能选**下一项最值得完成的研究任务**，我的选择是：

**把 ATM 标准实现、state-only shortcut baseline、平衡动作干预和 task-contrast audit 一起接入现有 PointMaze decision benchmark，并冻结一套独立 outcome 协议。**

这是整个研究中最可能产生决定性结果的第一步。它会告诉我们，LeWorldModel 究竟发现了 ATM 之外新的 world-model 决策性质，还是只是用更多几何指标重新测量了同一种动作可辨识性。

无论结果如何，这个实验都比继续增加模型复杂度更有科学信息量。

-----

> # LeWorldModel 当前研究全景
>
> ## 一、研究核心问题
>
> 研究已经从“训练一个更强的 world model”收缩为：
>
> > 对 frozen RGB learned latent，在有限历史、有限动作支持、部分可观测和固定预测 horizon 下，能否判断它是否足以支持可靠的 action-to-task transfer，并在不可靠时拒绝规划？
>
> 统一数学对象是：
>
> $$
>  \eta\_t=\Phi(z\_{t-L\:t},a\_{t-L\:t-1}),
> $$$$
>  G\_h^\Phi=D\_\Phi A\_\Phi^{h-1}B\_\Phi .
> $$
>
> 其中：
>
> - $\Phi$：历史 lifting / memory；
> - $A\_\Phi$：lifted latent dynamics；
> - $B\_\Phi$：action effect；
> - $D\_\Phi$：任务 readout；
> - $G\_h^\Phi$：真正决定有限 horizon 动作后果的 transfer operator。
>
> 因此，研究现在明确区分：
>
> $$
>  \text{Predictive Sufficiency} \neq \text{Action Identifiability} \neq \text{Decision Sufficiency}.
> $$
>
> 预测误差低，不代表动作效果可辨识；动作效果可辨识，也不代表闭环规划可靠。
>
> ---
>
> ## 二、整体研究架构
>
> ```
> Frozen RGB observation
>         ↓
> Latent / temporal lifting Φ
>         ↓
> Covariance–predictor geometry
> (C, AΦ, BΦ, DΦ, spectrum, alignment, Schur, resolvent)
>         ↓
> Action identifiability and support check
>         ↓
> Real / predicted successor consistency
>         ↓
> Finite-horizon transfer-risk certificate
>         ↓
> Permit / abstain
>         ↓
> Closed-loop visual planning
> ```
>
> 当前研究包含两条相互关联但证据独立的主线。
>
> ### 1. HTSC：Horizon-conditioned Transfer-Sufficient Control
>
> 目标是：
>
> > 判断 frozen learned world model 什么时候值得信任。
>
> 输出不是无条件的“certificate”，而是：
>
> $$
>  \text{permit}/\text{abstain}
> $$
>
> 以及候选动作级 transfer-risk bound。
>
> 它整合：
>
> - prediction quality；
> - action identifiability；
> - action support；
> - mode observability；
> - real/predicted successor mismatch；
> - covariance–predictor geometry；
> - finite-horizon transient amplification。
>
> ### 2. GDD-WM：Decision Preference → Predictive Geometry
>
> 目标是：
>
> > 能否将真实执行中可靠的动作偏好蒸馏进 encoder/predictor，使预测几何更具有决策价值？
>
> 流程是：
>
> ```
> execution-grounded preference
> → preference stability
> → predictive geometry distillation
> → multi-step prediction preservation
> → fixed-candidate regret
> → independent planning consequence
> ```
>
> 目前 GDD-WM 仍是：
>
> > Proposed / Not Yet Validated
>
> 因为真实 D-JEPA teacher、稳定偏好数据和独立规划收益尚未闭合。
>
> ---
>
> # 三、已经得到的主要结果
>
> ## 1. Horizon-conditioned memory
>
> 这是目前最明确的正向结果之一。
>
> ### h=1
>
> - fit-only selector 在 5/5 split 上 fallback；
> - stability 约为 `0.375–0.5625`；
> - 正式策略冻结为 fixed Fourier；
> - selector 不部署。
>
> ### h=4
>
> - frozen Fourier+phase 在 5/5 split 上优于 fixed Fourier；
> - paired RMSE difference 约为 `-7.01`；
> - bootstrap 95% CI 约为 `[-8.54,-5.71]`。
>
> 这支持的结论是：
>
> > memory 的价值依赖预测 horizon；固定的 horizon-aware policy 比小样本下的在线 selector 更可靠。
>
> 但它目前只证明了 latent/effect prediction 改善，尚未证明 action ranking 或闭环规划改善。
>
> ---
>
> ## 2. PointMaze RGB successor prediction
>
> RGB temporal successor verification 已经出现正向结果：
>
> - RGB successor prediction RMSE：约 `0.1631`；
> - source-latent baseline：约 `0.2011`；
> - 相对改善约 `17.8%`。
>
> 但同时：
>
> - 直接四帧 latent 拼接使 successor RMSE 变差；
> - causal history task-effect adapter 可以降低 selected regret；
> - 小样本面板中的 success 仍约为 `0.125`；
> - 尚无完整 frozen RGB 闭环在线规划证据。
>
> 因此目前只能说：
>
> > 历史信息经过任务校准后有用，但并非所有 temporal lifting 都能改善 successor prediction，更不能直接推出规划成功。
>
> ---
>
> ## 3. Spectral / alignment 诊断
>
> 理论和合成实验支持：
>
> - spectrum 单独不能决定 finite-horizon transfer；
> - alignment 可能提供额外信息；
> - Schur off-diagonal、resolvent 和 transient amplification 能揭示 non-normal 风险；
> - commutator 不能作为通用 transfer certificate；
> - covariance-only whitening 不能制造非平凡 alignment；
> - 普通 Euclidean commutator 不是一般 $GL(d)$ 坐标不变量。
>
> 真实 learned latent 的 audit 中：
>
> - prediction mean cosine 约 `0.9607`；
> - prediction spectral MSE 约 `0.00750`；
> - 低维 top band 对 planning 更纯；
> - 到较高 rank 后 nuisance energy 与 planning energy 接近或超过。
>
> 因此目前最强结论是：
>
> > spectral geometry 是有条件的风险诊断工具，而不是普适规划证书。
>
> ---
>
> ## 4. 跨环境 matched benchmark
>
> 目前的 quality-matched audit 包含：
>
> - 2 个环境；
> - 4 个 architecture；
> - 5 个 seed；
> - 7 个 identifiable matched strata。
>
> 结果：
>
> $$
>  \Delta R = R(\text{MSE-only}) - R(\text{full diagnostic}) \approx +0.053.
> $$
>
> paired bootstrap CI 约为：
>
> $$
>  [+0.0115,+0.1166].
> $$
>
> 但是 sign-flip p-value 约为 `0.0630`，没有通过预注册的正式 positive gate。
>
> 所以正确结论是：
>
> > 点估计支持 geometry diagnostic 可能有增量信息，但多环境、独立规划风险预测尚未被正式证明。
>
> ---
>
> ## 5. PushT 接触与姿态闭合
>
> 目前已经能够部分改善：
>
> - contact-mode observability；
> - contact retention；
> - signed action direction；
> - 某些局部 action ranking。
>
> 但主要失败仍然存在：
>
> - within-mode action-effect 不可辨识；
> - friction、sticking、sliding 和 contact transition 混合；
> - 同一 latent state 下 action-to-force 映射不唯一；
> - torque sign 不稳定；
> - strict pose closure 仍为 `0/32`；
> - 原 predictor repair 在小样本下反而使 held-out latent MSE 恶化。
>
> 因此当前最重要的 PushT 结论是：
>
> > 问题不只是“没有识别接触模式”，而是“在接触模式已经部分可预测之后，仍不能可靠辨识 mode 内 action-effect”。
>
> ---
>
> # 四、六大目标当前状态
>
> | 目标当前状态真实结论                          |               |                                               |
> | ----------------------------------- | ------------- | --------------------------------------------- |
> | alignment metrics 普遍预测规划性能          | 部分支持、未通过 gate | 在 matched 条件下有正点估计，但跨环境证据不足                   |
> | learned adapter 稳定解决姿态闭合            | 未解决           | 局部 action ranking 改善，但 strict pose closure 失败 |
> | LeWM latent 具有可靠物理因果动力学             | 部分支持          | 存在 action-associated structure，但完整物理因果性未建立    |
> | frozen LeWM 支持视觉在线规划                | 未闭合           | RGB successor 有改善，纯视觉闭环规划仍缺证据                 |
> | 建立普适 world model                    | 未建立           | 当前只能支持条件性 transfer diagnostic                 |
> | commutator 是通用 transfer certificate | 已否定           | 只能作为 gauge-aware、horizon-specific diagnostic  |
>
> ---
>
> # 五、当前真正的卡点
>
> ## 卡点一：latent 有预测信息，但缺少动作可辨识信息
>
> 这是所有负结果的核心。
>
> latent 可以较好预测：
>
> - 下一状态；
> - 运动趋势；
> - 某些接触模式；
> - 低频 temporal structure。
>
> 但它不一定保留：
>
> - 接触法向；
> - 摩擦状态；
> - torque sign；
> - sticking/sliding 区分；
> - 同一动作在不同隐状态下的真实 effect。
>
> 所以会出现：
>
> ```
> prediction RMSE 改善
>         ↓
> mode accuracy 改善
>         ↓
> action ranking 仍错误
>         ↓
> 姿态闭合失败
> ```
>
> ---
>
> ## 卡点二：action support 不足
>
> 当前动作数据无法充分覆盖：
>
> - `+a / -a / zero` 对称干预；
> - 不同接触模式下的相同动作；
> - mode transition 附近的短脉冲；
> - 真实 planner 会选择的动作区域。
>
> 因此，模型可能在观测数据分布内预测得很好，但在候选动作比较时发生 ranking inversion。
>
> 这也是为什么 action support gate 必须先于 alignment 和 commutator。
>
> ---
>
> ## 卡点三：部分可观测性与历史记忆不匹配
>
> Fourier、event feature、GRU/SSM 和 raw history 各自在不同 horizon 上有效，但没有一个表示普遍最优。
>
> 目前发现：
>
> - Fourier 在短 horizon 低方差；
> - Fourier+phase 在 h=4 有收益；
> - event feature 能捕捉突变；
> - raw history 可能带来旧状态污染；
> - selector 在小样本、非平稳接触数据上方差过高。
>
> 所以研究不应继续寻找“唯一最强 memory”，而应使用：
>
> $$
>  M\_h(t)=F\_h(z\_{t-L\:t}) + g\_h(t)E\_h(z\_{t-L\:t},a\_{t-L\:t-1}),
> $$
>
> 并在 support 或 uncertainty 不足时 fallback。
>
> ---
>
> ## 卡点四：adapter 容易过拟合
>
> 此前 residual adapter 的真实结果显示：
>
> - base held-out latent MSE 约 `0.336`；
> - repaired adapter held-out latent MSE 约 `0.734`；
> - MPC success：base、repair、heuristic 均为 `0/1`。
>
> 这说明当前 adapter 不是稳定修复，而是：
>
> - 小样本拟合；
> - train/calibration 分布不一致；
> - 接触模式混杂；
> - residual 目标不稳定；
> - latent prediction improvement 没有传递到任务层。
>
> 因此 adapter 必须同时通过：
>
> 1. calibration improvement；
> 2. root-disjoint successor improvement；
> 3. action ranking improvement；
> 4. real/predicted consistency；
> 5. independent closed-loop improvement。
>
> 任何一项断裂都不能部署。
>
> ---
>
> ## 卡点五：独立规划证据不足
>
> 目前很多结果属于：
>
> - offline latent RMSE；
> - synthetic operator audit；
> - privileged simulator-state probe；
> - action ranking diagnostic；
> - successor prediction。
>
> 但真正决定论文主张的是：
>
> - frozen RGB runtime；
> - shared candidate bank；
> - fresh initial states；
> - root-disjoint test；
> - no privileged state；
> - real successor verification；
> - closed-loop planning consequence。
>
> 这一层仍未完整闭合。
>
> ---
>
> ## 卡点六：统计证据不足
>
> 目前存在：
>
> - 环境数量偏少；
> - matched strata 数量不足；
> - sign-flip gate 未通过；
> - 某些 success 只有极少 episode；
> - 部分结果是 smoke 或 proxy；
> - learned checkpoint 面板不能充分匹配 MSE、spectrum、stage 和 action risk。
>
> 因此不能从局部正点估计升级为：
>
> - 普适 alignment；
> - 通用 commutator certificate；
> - universal world model；
> - 稳定姿态闭合。
>
> ---
>
> ## 卡点七：GPU 与实验执行环境
>
> 工程层面曾出现：
>
> - `nvidia-smi` driver/library mismatch；
> - 受限进程 CUDA 初始化失败；
> - 部分 artifact 只写出 `metrics: null`；
> - 长实验中断；
> - 一些空目录是 execution failure，而不是 scientific negative。
>
> 已修复或加强：
>
> - GPU preflight；
> - 禁止静默 CPU fallback；
> - checkpoint / atomic JSON；
> - rich latent export；
> - 多阶段 resume；
> - schema 和 evidence manifest。
>
> 但仍需要区分：
>
> ```
> GPU smoke
> ≠ full GPU evidence
> ≠ scientific positive result
> ```
>
> ---
>
> # 六、目前最值得保留的科学 insight
>
> 当前最成熟的核心命题是：
>
> > A latent history representation is useful for planning only when it is horizon-sufficient, mode-observable, action-identifiable, and supported by the deployment action distribution.
>
> 中文即：
>
> > 一个 latent 历史表示只有同时满足有限时域充分性、模式可观测性、动作可辨识性，并且部署动作位于训练支持集内时，才具有规划价值。
>
> 这条命题统一解释了：
>
> - Fourier 在 h=1 稳定；
> - Fourier+phase 在 h=4 有效；
> - selector 不稳定；
> - mode accuracy 提升但姿态闭合失败；
> - alignment 在合成算子上有效、learned checkpoint 上不稳定；
> - commutator 不能通用；
> - RGB successor prediction 改善但在线规划仍未成功。
>
> ---
>
> # 七、当前研究的准确定位
>
> 目前项目最可信的定位是：
>
> > 面向 learned visual world models 的有限时域 decision-transfer sufficiency 诊断与 fail-closed planning framework。
>
> 已经具备：
>
> - 新的统一问题定义；
> - 有限时域 transfer-risk 理论；
> - covariance–predictor 几何分析；
> - horizon-conditioned memory；
> - action-support 和 identifiability gate；
> - 跨环境 benchmark 框架；
> - 失败边界与 adversarial counterexample；
> - GPU-first 可复现实验工程。
>
> 但尚未具备：
>
> - 普适 learned-world-model 证明；
> - 稳定视觉在线规划；
> - 稳定 PushT 姿态闭合；
> - 多环境显著的独立 planning gain；
> - 通用 alignment 或 commutator certificate；
> - 已验证的 GDD-WM decision distillation。
>
> 一句话总结：
>
> > 现在的研究已经从“寻找一个更强的 latent 或 memory”发展成“判断 learned latent 何时足以支持动作决策，并在不可辨识时安全拒绝规划”的理论—实证框架；最大的科学卡点仍是从预测充分性跨越到动作可辨识性，再跨越到独立闭环决策收益。
>
> 这就是我现在的研究样子和卡点，请你帮我分析一下我下一步该怎么做

我会把下一步压缩成一条真正能产生论文主结果的关键路径，而不是继续罗列六个目标。先核对最新仓库和 ATM 等直接竞争工作的证据边界，再针对你现在的三个主要瓶颈——动作可辨识性、候选动作排序、真实闭环收益——设计分阶段实验。每一阶段我都会明确：要证伪什么、用现有哪部分代码、怎样判定通过，以及失败后应该转向哪里。

# LeWorldModel 下一阶段研究路线：从诊断体系走向真正的决策可靠性

结合你目前的六目标证据矩阵、最新 PointMaze 与 PushT 结果、HTSC 理论，以及 ATM 所揭示的动作语义失配问题，我的判断很明确：

**你现在最需要的不是一个更强的 encoder、更复杂的 memory，也不是继续扩大 spectral diagnostics，而是完成一次“从真实动作干预到可靠动作选择，再到 RGB 闭环控制”的证据闭环。**

我建议将下一阶段的主研究问题收敛为：

> **在预测误差与动作可辨识性相近的 frozen visual world models 中，能否通过有限时域、任务条件化的反事实动作效果，识别 ATM 等现有诊断没有发现的决策失败，并在规划前可靠判断哪些候选动作值得执行？**

这比同时攻克“普适 world model、稳定 PushT 姿态闭合、通用 alignment、GDD-WM 蒸馏”更有可能形成完整的 ICLR / ICML / NeurIPS 主论文。

## 一、我对当前研究瓶颈的重新排序

你的项目不是六个目标都缺一点实验。它们之间存在非常明确的因果和证据依赖关系。

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Frozen RGB world model**
    <text color="secondary" size="xs">已有 prediction、spectral geometry 与 memory 证据</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **P0 · 真实反事实动作效果**
    <text color="secondary" size="xs">同一 root、不同动作、相同 horizon 的真实后果</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **P0 · Task-conditioned decision transfer**
    <text color="secondary" size="xs">真实与预测的动作代价差异是否一致？</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **P1 · 可校准的规划放行**
    <text color="secondary" size="xs">能否在预先声明的误差与支持条件下 permit / abstain？</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **P1 · RGB-only closed-loop MPC**
    <text color="secondary" size="xs">更少错误决策 · 更低 regret · 可接受的拒绝成本</text>
  </box>
  <divider color="subtle"/>
  <grid columns={2} gap={2}>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={3} gap={1}>
        **P2 · PushT**
        <text size="xs" color="secondary">接触系统的机制和泛化测试</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="surface-secondary" radius="md" padding={3} gap={1}>
        **P3 · GDD-WM**
        <text size="xs" color="secondary">执行验证偏好的预测几何蒸馏</text>
      </box>
    </grid-item>
  </grid>
</box>

这张图表达的是建议的证据路线，而不是当前已实现的全部流水线。

### 为什么我把 PointMaze 放在 PushT 和 GDD-WM 前面？

因为当前 PointMaze 已有最接近完成闭环的基础：root-disjoint candidate banks、动作方向干预、task-effect adapter、RGB temporal successor 和共享候选后悔值评估。

但 RGB successor 改善与离线 regret 改善，还没有转化为稳定的在线规划结果。最新研究进展仍明确保留这个边界。<Cite ref="turn650178view1"/>

PushT 则同时混合了接触状态不可观测、摩擦、力矩符号、接触切换及姿态闭合问题。在最基本的候选动作认证尚未完成之前，用 PushT 做主要方法开发，会很难判断究竟是哪一层出了错。

GDD-WM 又依赖可靠的真实执行偏好。没有独立验证过的决策标签，提前训练几何蒸馏很容易学到错误 teacher 偏好。

因此我建议：

**主力攻克 PointMaze 的可认证闭环规划；PushT 用来检验机制边界；GDD-WM 暂时只推进不依赖大规模训练的理论与可表示性验证。**

另外，你的项目治理规范已经建立了 manifest、stage 状态、证据隔离与两个总入口共享执行契约的方向。下一阶段应复用这些设施，不需要另起一套实验框架。<Cite ref="turn567808view1"/>

## 二、第一件必须做的事：先判断 PointMaze 的瓶颈到底在哪一层

我认为这是整个下一阶段**投入产出比最高的实验**。

你现在有一个非常值得警惕的现象：

- RGB successor prediction 改善；
- Causal history task-effect adapter 降低了离线 regret；
- 但 selected success 仍然没有提高。

在继续训练任何东西之前，需要回答：

**当前的 candidate bank 里，究竟有没有能让任务成功的动作？**

### 实验 1：Candidate-bank oracle ceiling

对每个 PointMaze root，使用同一目标与同一候选动作集合，在可重置模拟器中执行所有候选动作，获得真实任务代价：

$$

J_i^*=J_H^{real}(h_t,U_i)

$$

先算出：

$$

U_{\rm oracle}=\arg\min_{U_i\in\mathcal C}J_i^*

$$

然后比较：

| 评价对象 | 解决的问题 |
|---|---|
| Oracle best candidate | 候选集合中是否存在真正有用的动作？ |
| Frozen LeWM selected candidate | 当前模型是否选错了本来就存在的好动作？ |
| ATM-ranked model / action baseline | 动作辨识信息能解释多少选错动作的现象？ |
| Full task-transfer diagnostic | 是否能额外解释真实的动作排序错误？ |
| Receding-horizon oracle | 即使每次都选当前候选中的最佳动作，能否逐步完成任务？ |

这组实验会给你一个特别重要的 **planning ceiling**。

如果 oracle 都无法成功，那么主要瓶颈可能是候选动作的覆盖、目标代价、规划 horizon 或 CEM 搜索，而不是 latent geometry。

如果 oracle 能成功、LeWM 不能，那才说明模型预测或动作排序是主要障碍。

如果固定候选排序正确，但闭环仍然失败，应重点排查执行后状态分布变化、长时域累积误差与 replanning interface。

注意：对于需要绕障碍物的导航任务，单个短 horizon 候选的终端成功率可能天然很低。因此不能只看一步完成目标；还要比较目标进展、可达性以及重复规划后的 oracle ceiling。

### 这一实验的成功标准

<box gap={2} border radius="lg" padding={3}>
  <row align="start" gap={2}>
    <icon name="check-circle-2" color="success"/>
    <box flex="1" gap={1}>
      **通过：** 同一批独立 roots 上，候选集合具有有意义的 oracle 控制潜力，而且 learned planner 与 oracle 之间存在可测量的 action-selection gap。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={2}>
    <icon name="alert-circle" color="secondary"/>
    <box flex="1" gap={1}>
      **未通过：** 候选集合缺少有效动作、目标定义不适合该 horizon，或真实执行结果无法可靠配对。
    </box>
  </row>
</box>

未通过时先修 benchmark，不要训练新的 adapter。这能避免继续花大量 GPU 时间优化一个实际上没有可用动作的规划问题。

---

## 三、第二件事：建立真正超越 ATM 的诊断实验

这里我建议不要先提出一个复杂的新 architecture，而是完成一个非常干净的对照研究。

### 核心假设

$$

H_0:
\text{Task-transfer diagnostics 在 ATM + MSE 之后没有增量预测力}

$$

$$

H_1:
\text{Task-transfer diagnostics 仍能预测额外的决策失败}

$$

ATM 已经研究了真实与预测 transition 中动作信息的恢复及跨域一致性，所以只证明 inverse risk 比 MSE 好，已经不足以作为你的主要创新。<Cite ref="turn519430academia36"/>

你需要把以下诊断放在同一个候选、同一个 root、同一个 world-model checkpoint 上比较：

| 方法 | 主要测量什么 |
|---|---|
| Prediction MSE | 平均预测误差 |
| ATM $2\times2$ matrix | 真实/预测逆动作恢复与跨域失配 |
| State-only inverse probe | 不读取 successor 时，行为策略本身能够预测多少动作 |
| Interventional effect probe | 平衡动作干预下，真实 successor 提供了多少动作信息 |
| Task-effect contrast | 两个动作对真实任务代价的差异是否被正确预测 |
| Horizon transfer geometry | 任务读出、动作响应与多步传播的关系 |
| Full HTSC | 所有必要条件组合后，能否预测实际 regret |

最重要的不是比较谁的训练误差更低，而是在**未见 roots 和未见 model checkpoints** 上比较谁能识别真实 action ranking inversion。

### 直接定义新的主要测量对象

对同一 root 和候选动作 $U_i,U_j$，定义真实和预测的代价差异：

$$

\Delta_{ij}^{real}=J_i^{real}-J_j^{real}

$$

$$

\widehat\Delta_{ij}
=\widehat J_i-\widehat J_j

$$

然后研究：

$$

\boxed{
E_{ij}^{contrast}
=
|\Delta_{ij}^{real}-\widehat\Delta_{ij}|
}

$$

这比直接用 latent MSE 更接近规划器真正需要的信息。

如果真实代价差异极小，即使排序偶尔反转，也未必是有意义的失败。因此需要同时报告带最小任务 margin 的 ranking inversion、实际 regret，以及误差随 horizon 的变化。

对于完整动作序列，应使用：

$$

\mathcal G_H^\Phi=
\begin{bmatrix}
D_\Phi A_\Phi^{H-1}B_\Phi &
D_\Phi A_\Phi^{H-2}B_\Phi &
\cdots &
D_\Phi B_\Phi
\end{bmatrix}

$$

而不仅是单个 $G_h^\Phi$。

在 PushT 等 hybrid dynamics 中，还要区分不同接触模式下的局部 Jacobian 和模式切换，不能直接套用固定线性算子。

### 什么结果足以说明你发现了 ATM 之外的新东西？

我会要求至少出现一种可靠结果：

**结果 A：** 两个 independently trained learned models 的 MSE 和 ATM 诊断已经在预声明的等价容差内匹配，但它们的实际 task regret 显著不同。

**结果 B：** ATM 可以预测 model-level success，却不能识别部分 candidate-level、horizon-specific failure；你的 task-transfer estimator 能在独立 roots 上识别它们。

**结果 C：** 在控制 ATM、prediction loss、action support 和 architecture 后，你的方法仍能带来外层留出测试上的增量风险预测力。

结果 A 是有力的存在性证据，B 是机制证据，C 则是泛化证据。三者同时成立，研究的说服力才会明显提升。

如果这些实验没有发现显著增量，就不要强行宣称 geometry 有独立贡献。应保留负结果，考虑将 spectral geometry 定位为解释工具，而以直接的 task-contrast calibration 为主方法。

## 四、第三件事：真正把 HTSC 的风险上界变成可以使用的规划证书

这是你理论工作最需要完成的升级。

现有 HTSC 已有 finite-horizon transfer-risk decomposition，但难点在于：如何从有限的真实干预数据获得一个有效的、不会因为过度保守而完全无法放行动作的风险上界。

我建议先研究 **pairwise task-contrast calibration**，不要立即研发一个复杂的新风险神经网络。

### 1. 为什么直接校准动作代价差异？

假设某个模型对所有动作的预测代价都有相同的系统性偏置：

$$

\widehat J_i=J_i+5

$$

它的绝对代价误差很大，但动作排序可能完全正确。

例如真实代价为：

$$

J=[1,2,3]

$$

预测代价为：

$$

\widehat J=[6,7,8]

$$

此时每个候选的绝对误差都是 5，但任意两个候选的代价差异预测误差都是 0。

这一结果已经用 Python 独立验证。

因此，与其要求所有预测都准确，不如直接控制：

$$

e_{ij}=
(J_i-J_j)-(\widehat J_i-\widehat J_j)

$$

它更接近有限候选规划真正需要满足的条件。

### 2. 我建议的第一版证书

在独立 calibration roots 上，令 $s_{ij}>0$ 是仅使用 fit 数据训练、在校准前已经冻结的风险尺度函数。

对每个 root 计算：

$$

S_r=
\max_{i\ne j}
\frac{|e_{ij}|}{s_{ij}}

$$

然后对 root-level scores 进行校准，得到阈值 $q_\alpha$。

针对一个新的候选集合，如果真实误差满足：

$$

|e_{ij}|\le q_\alpha s_{ij}

$$

设预测最优候选为 $\hat i$，预测代价间隔为：

$$

m_{\hat i,j}=\widehat J_j-\widehat J_{\hat i}

$$

则可以得到：

$$

\boxed{
\mathrm{Regret}(\hat i)
\le
\max_{j\ne\hat i}
\left[q_\alpha s_{\hat i,j}-m_{\hat i,j}\right]_+
}

$$

我对该不等式进行了随机数值测试与反向检查。

这个结果非常适合你的 HTSC。

你可以规定一个任务允许的最大 regret $\tau$。只有当右侧不超过 $\tau$，并且动作支持、适用模式和校准条件成立时，才允许执行候选。

如果希望保证选中的是候选集合中的真实最优动作，则可以要求右侧为 0。

**关键科学创新不在这个简单不等式，而在于能否构造具有有限样本有效性、且足够紧的 task-conditioned contrast-risk estimator。**

### 3. 对 guarantee 必须非常严格

在 calibration roots 与新测试 roots 可交换、候选生成机制固定、风险尺度函数独立拟合等条件下，可以研究 root-level 同时覆盖。

但这不能自动推出：

- 对任意指定 RGB 状态的条件覆盖；
- 对所有 CEM 自适应策略的覆盖；
- 对一整段闭环 trajectory 的联合安全保证；
- 放行动作集合内部的条件错误率保证。

尤其是 **marginal coverage 不等于 selective coverage**。

2026 年 ICML 的 <Entity category="other" value="Conformal Risk-Averse Decision Making with Action Conditional Guarantee" disambig="ICML 2026 statistical learning paper"/> 和 <Entity category="other" value="Conformal Policy Control" disambig="ICML 2026 conformal safety and policy control paper"/> 已经研究了与动作条件和策略可靠性有关的校准问题。你的工作必须明确相对它们的新增之处，而不能把普通 conformal prediction 当作新理论。<Cite refs={["",""]}/>

你的差异应当是：

**有限 RGB 历史 + 反事实动作序列 + 任务代价差异 + planner-induced candidate shift + world-model 表征可辨识性。**

还有一个现实问题：若只有 40 个相互独立的 calibration roots，采用标准 split-conformal 95% 覆盖时，有限样本分位数对应排序后的第 39 个误差值，接近最极端观测值。

这通常会产生较不稳定、较保守的上界。不能将已经使用的 40 个评估 roots 直接回收为 calibration 数据。

此外，在随机物理环境下，需要区分“单次随机执行代价的覆盖”和“条件期望任务代价的估计”。后者可能需要重复执行或额外统计假设，不能直接从单次 realized outcome 的 conformal coverage 推出。

### 4. 这个研究阶段怎样才算通过？

| 评价指标 | 必须达到的性质 |
|---|---|
| Simultaneous pairwise coverage | 独立测试 roots 上达到预声明覆盖要求 |
| Certificate tightness | 上界小于有实际意义的决策间隔 |
| Decision inversion | 能提前发现一部分真实动作排序错误 |
| Risk–coverage | 同一放行比例下优于简单风险基线 |
| Utility | 包含拒绝和 fallback 成本后，整体决策仍有收益 |

如果一个证书只能通过拒绝绝大部分候选来获得较低错误率，它不是一个有竞争力的控制方法。

特别注意：**Fallback 也不自动安全。** 固定 horizon=1 只是基线，不是经过证明的安全控制器。

---

## 五、第四件事：完成 frozen RGB-only 在线 MPC

前三件事完成后，就不要继续停留在 offline candidate ranking。

应该真正让 LeWM 使用 RGB 观测执行动作。

建议的闭环是：

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center">
    **Current RGB observation + goal**
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center">
    **Frozen LeWM + declared history representation**
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center">
    **CEM candidate generation and imagined rollouts**
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **Action support + contrast-risk assessment**
    <text size="xs" color="secondary">Permit / shorten / independently validated fallback</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center">
    **Execute action block → observe new RGB → replan**
  </box>
</box>

我建议第一版不要部署不稳定的 adaptive memory selector。

直接采用你已经预先声明的 fixed memory policy：短 horizon 使用 fixed Fourier；较长 horizon 使用经过独立验证的 Fourier+phase。它们在当前证据中有预测层面的支持，但仍必须接受动作排序和闭环测试。

比较方法至少包括：

| 对照 | 验证目的 |
|---|---|
| Original frozen LeWM MPC | 基础规划能力 |
| 最佳 fixed-horizon MPC | 排除只靠 horizon 选择获益 |
| Standard uncertainty-aware MPC | 排除普通 uncertainty 就能达到同样效果 |
| ATM-based screening | 直接竞争工作 |
| HTSC absolute-risk gate | 检验现有风险界 |
| HTSC pairwise-contrast gate | 检验新研究假设 |
| Privileged oracle | 判断闭环候选搜索和控制接口的上限 |

不同方法必须共享合理匹配的 CEM 预算、动作执行周期、goal metric 与测试 roots。

对于不同规划 horizon，还要避免“预测更远只是因为允许移动更久”这种不公平比较。

测试阶段可以使用模拟器特权状态记录真实结果，但绝不能把未来特权状态用于 learned planner 的动作选择。

### 第一张必须做出来的主实验图

<text color="secondary" size="sm">Risk–Coverage–Regret：在相同 planning coverage 下，比较不同风险诊断方法的真实决策后悔值。</text>

如果 HTSC 能在较高的放行率下显著降低实际 regret，而且完整系统的 success rate 没有因为过度拒绝而恶化，这才是你最有价值的主实验结果。

不过，静态 calibration roots 的证书不能直接被宣传为整个在线 MPC 的多步安全证明。执行后的状态分布可能因控制器改变，因此必须额外报告 on-policy coverage、support shift 和完整 episode 结果。

---

## 六、PushT 下一步不应该再盲目训练 adapter

目前 PushT 的主要问题已经很明确：**模式可观测性改善，但模式内部动作效应仍然不可靠。**

我建议把它从“训练更强 adapter”的任务改为一个物理机制实验。

| 实验层级 | 验证什么 | 如果失败，下一步做什么 |
|---|---|---|
| Privileged contact-state oracle | 真实接触几何与动力学信息是否足以正确预测 torque sign | 检查物理模型、接触 chart 和控制接口 |
| RGB single-frame latent | 当前 frozen encoder 是否保留足够的接触动作信息 | 测量状态混叠，不直接加大模型 |
| RGB history latent | 历史能否消除接触状态或相对运动的歧义 | 研究有效历史和主动信息获取 |
| Mode-conditioned action effects | 相同 mode 下能否区分正负方向的 torque | 优先收集更好的接触内动作激励 |
| Learned adapter | 可辨识信息能否被现有模型有效利用 | 再考虑结构化 residual 或 hybrid dynamics |
| RGB-only MPC | 局部效应辨识是否转化为完整姿态闭合 | 分析 planner exploitation 与长期接触控制 |

这里最关键的是对照 privileged oracle。

如果特权接触状态下都无法正确预测和控制，那么继续研究 RGB latent 是否保留这些信息不是首要问题。

如果特权 oracle 能做到，而 RGB-history 模型不能做到，那么就需要研究视觉部分可观测性的本质限制。

如果 RGB-history 已经包含足够信息，但 adapter 仍然错，则问题才主要落在模型参数化、训练目标与优化数据上。

你此前的修复版 adapter 在 held-out MSE 上反而恶化，这已经提示我们：不应该默认 residual learning 会修复真实动力学。

**我建议暂时把 PushT strict pose closure 从近期主方法成功条件，调整为后续最重要的困难任务验证。**

这样既不会掩盖 0/32 的负结果，也不会让整项主研究被一个复杂的接触控制接口长期阻塞。

---

## 七、GDD-WM 现在最值得推进哪一部分？

我不建议现在优先做 D3 联合训练。

相反，GDD-WM 有一项几乎不依赖新 GPU 大规模训练、但很有理论价值的前置任务：

**Preference Geometrizability Test**

给定冻结模型预测的候选 latent：

$$

\Delta_i=\hat z_H(U_i)-z_g

$$

以及真实执行确认的偏好 $U_i\prec U_j$，先检验是否存在：

$$

M\succeq0,\qquad \mathrm{Tr}(M)=1

$$

使：

$$

\mathrm{Tr}\left[
M(\Delta_j\Delta_j^\top-\Delta_i\Delta_i^\top)
\right]\geq m_{ij}

$$

对固定表示、固定 predictor、PSD metric 类，这是一个可以构造半定可行性检验的对象。

它能回答：

- 正确的真实动作偏好，能否通过一个简单的 latent metric 表达？
- 真实后继几何能够表达，但预测后继几何不能表达的偏好有哪些？
- 需要改善的是表示还是预测器？
- 低秩投影的表达限制是否是主要障碍？

如果固定几何完全无法表达正确偏好，继续训练 projection-only student 可能没有意义。

但如果固定几何有能力表达，却训练不出来，那么才应优先研究 optimization、loss 和监督标签质量。

GDD-WM 可以先完成这项理论与小型可表示性实验；真正大规模训练仍应等待 execution-validated teacher/candidate 数据。

这会让你的 GDD-WM 更接近**预测动力学约束下的决策几何可表示性研究**，而不是一个新的 metric-learning loss。

---

## 八、接下来应当怎样安排实际工作？

我建议按照以下研究阶段推进，阶段之间使用通过条件，而不是到了某个日期就强行进入下一阶段。

| 阶段 | 主要工作 | 必须交付的结果 | 下一阶段条件 |
|---|---|---|---|
| A：Truth audit | PointMaze oracle ceiling、候选 bank、目标与结果配对核查 | 真实候选代价与 oracle gap | 明确是否存在可改善的动作选择问题 |
| B：ATM comparison | 标准 ATM、state-only inverse、干预式动作效果、task contrast | 独立 held-out 对照结果 | 证明 task contrast 相对 ATM 有值得继续研究的增量 |
| C：Contrast calibration | 独立 roots 的 pairwise risk calibration | Coverage、tightness、false release | 上界有效且不是几乎全部拒绝 |
| D：Online planning | Frozen RGB-only MPC、HTSC 与强基线 | 完整闭环 success / regret / latency | 实际控制收益成立 |
| E：External validation | 独立 backbone、环境和动作支持偏移 | 跨模型泛化及失败边界 | 允许提高论文主张强度 |
| F：Extension | PushT within-mode、GDD-WM D1/D2 | 物理机制与几何可表示性证据 | 再考虑接触闭环与 GDD-WM D3/D4 |

### 在代码上只增加一条 study-level protocol

你刚刚已经整理过研究工程架构，所以我建议直接使用现有 research runtime，新增一个协议，例如：

`protocols/pointmaze_decision_transfer_v1.yaml`

这个协议只编排已存在或需要补齐的 `oracle_candidate_audit → atm_baseline → intervention_effect → task_contrast → risk_calibration → rgb_online_eval → evidence_summary` 阶段。

不要再新增第三个“总入口”。新的实验实现也应复用现有 `decision_benchmark/`、`risk_aware_control.py`、统计和 evidence 模块。

特别要坚持：

**新的训练、调参、calibration 和最终 planning outcome 必须有独立数据边界。**

原来的 `run_full_spectral_transfer.sh` 仍负责从原始数据重跑，`run_unified_research_protocol.py` 仍负责固定协议与证据审计。新科学实验只作为可复用 stage 接入，不应另建一套执行器。

## 九、我建议立即修复的三个基础问题

**第一，统一 PointMaze RGB successor 的数值口径。**

公开文档给出 0.2011 到 0.1631，并标注相对改善 17.8%。但按这两个展示值直接计算，下降为 **18.90%**。

这不一定意味着实验有错误，可能是未四舍五入的内部数据、按 root 聚合的平均相对改善与总体 RMSE 比值使用了不同口径。

必须明确采用哪一种，并在脚本中同时报告对应 numerator、denominator 和 aggregation rule。<Cite ref="turn650178view1"/>

**第二，重新确认统计单位。**

当前五个 memory splits 不是五个独立的 learned backbones，七个 matched strata 也不等于七个独立环境。

在下一轮实验中，至少分开统计 root、candidate、episode、adapter seed、world-model training seed 和 environment。

不要靠扩充同一模型上的 candidate 数量来证明普适性。

**第三，GPU execution failure 与 scientific negative 必须继续严格隔离。**

任何 `metrics=null`、checkpoint 缺失、CUDA 初始化失败和中途崩溃，都应该是 blocked / failed execution，而不是被合并进真实的预测或规划失败结果。

这是你现有科研治理规范最值得继续保持的优点。

---

## 十、下一步实验失败时，应该怎样决策？

这张表我认为非常重要，可以防止你再次陷入不断增加模块的循环。

| 观察到的结果 | 最可能的瓶颈 | 应当采取的研究动作 |
|---|---|---|
| Oracle candidate 也不能达到目标 | Candidate support / planner objective | 修候选生成和任务定义 |
| Oracle 有效，但 inverse effect 不可辨识 | Action support / visual aliasing | 增加平衡动作干预、历史或信息获取 |
| 动作可辨识，但 task contrast 错误 | Task readout / dynamics transfer | 检查 $D_\Phi$、完整 $\mathcal G_H$、预测偏置与时域误差 |
| Task contrast 正确，但 MPC 不成功 | Planner exploitation / distribution shift | 检查 CEM elite、执行接口、重规划 |
| Certificate coverage 不达标 | 校准假设或数据划分有问题 | 修 calibration，不得提高放行阈值 |
| Certificate 有效但几乎全部拒绝 | 上界过宽或支持不足 | 增加覆盖、改误差分解、降低保守性 |
| PushT privileged 有效、RGB 失败 | 视觉可观测性不足 | 重点研究 history、contact geometry 与 action probing |
| GDD-WM projection 不可表示 | 固定几何表达能力不足 | 检验 predictor 更新或其他代价函数类 |

这使每个负结果都有科学解释和明确后续行动。

---

## 十一、我建议你接下来写出的论文主线

我更建议优先完成 HTSC 主论文，而不是把六项目标与 GDD-WM 全部塞入同一篇。

<title size="lg">Beyond Action Identifiability: When Can Frozen Visual World Models Be Trusted to Plan?</title>

主论文只需证明三个贡献：

**Contribution 1：Identifiability is not decision sufficiency.**

即使 MSE 和 ATM 很接近，模型的有限时域 task regret 也可能显著不同；在真实 learned world models 上建立这种差异的机制证据。

**Contribution 2：Task-conditioned decision-transfer calibration.**

从有限动作干预与真实 successor 中建立候选间代价差异的可信误差界，给出明确条件下的 permit / abstain 决策规则。

**Contribution 3：The certificate changes real decisions.**

在真正 RGB-only 在线 MPC 上证明，方法能够以可接受的拒绝和计算成本，减少真实排序错误和 regret，并在后续独立环境中保持部分有效性。

如果这三个贡献形成扎实证据，这将比“所有诊断指标都算了一遍”更有主会乃至 Oral 的说服力。

对于 Best Paper 级研究，还需要更进一步：揭示一种普遍而此前未被充分刻画的 frozen visual world-model 决策不可辨识性机制，并证明新的有限干预方法能够识别或缓解它。仅仅加入 conformal calibration、提出一个新的融合指标或提高单环境成功率还不够。

## 最终建议：你现在就应该执行什么？

我的优先顺序是：

**首先，做 PointMaze 的 oracle candidate ceiling 和真实 task-contrast audit。**

其次，在完全相同数据上加入 ATM 和 state-only inverse 强基线，确定你是否真的发现了 ATM 无法解释的有限时域规划失败。

然后，在独立 roots 上完成 pairwise contrast calibration，并将它接入 frozen RGB-only MPC。

在这之前，不要优先扩大谱指标库、重新寻找最强 memory、训练更多 PushT residual adapter，或启动大规模 GDD-WM joint training。

**当前研究最重要的转折，不是让 world model 看起来更懂物理，而是让它在动作真正执行之前，能够有证据地判断自己是否足以做出当前决策。**

这条证据链一旦打通，你前面关于谱几何、历史提升、接触动力学、ATM、HTSC 和 GDD-WM 的研究积累才会汇聚成一项有明确科学贡献的工作。

-----

> [User is quoting/referring to: "我更建议优先完成 HTSC 主论文，而不是把六项目标与 GDD-WM 全部塞入同一篇。
>
> Beyond Action Identifiability: When Can Frozen Visual World Models Be Trusted to Plan?
> --------------------------------------------------------------------------------------
>
> 主论文只需证明三个贡献：
>
> Contribution 1：Identifiability is not decision sufficiency.
>
> 即使 MSE 和 ATM 很接近，模型的有限时域 task regret 也可能显著不同；在真实 learned world models 上建立这种差异的机制证据。
>
> Contribution 2：Task-conditioned decision-transfer calibration.
>
> 从有限动作干预与真实 successor 中建立候选间代价差异的可信误差界，给出明确条件下的 permit / abstain 决策规则。
>
> Contribution 3：The certificate changes real decisions.
>
> 在真正 RGB-only 在线 MPC 上证明，方法能够以可接受的拒绝和计算成本，减少真实排序错误和 regret，并在后续独立环境中保持部分有效性。"]
>
> HTSC和GDD-WM 还有ATM相关我希望塞成一篇论文的三个贡献点

可以合成一篇，而且我认为有机会形成比单独写 HTSC 更强的论文。但必须把三者组织成**一个完整的科学机制**，而不是三个平行模块：ATM 揭示动作语义失配，HTSC 判断失配何时会影响任务决策，GDD-WM 利用经过验证的执行偏好修复预测几何。我会先核对直接竞争论文和仓库最新版，再给你设计真正相互依赖的三个 Contribution、统一数学框架、训练—认证—部署协议，以及一套能说服顶会审稿人的核心实验。

我核对了最新的直接竞争工作：D-JEPA 已经做了执行反馈驱动的决策几何，并支持不依赖关系算子的原生 latent-distance 规划；AD-WM 已经做了动作判别式训练。因此，这篇统一论文的真正差异不能是“我们也蒸馏偏好”或“我们也恢复动作”，而必须是**诊断告诉我们哪些偏好可以修复，修复后由独立校准重新认证，最后在闭环中获得更高的可信放行率和实际收益**。

我也验证了一个关键反例：两个模型可以有完全相同的 ATM Bayes inverse risks 和 latent MSE，却给出相反的任务最优动作。这很适合作为 Contribution 1 的理论起点。

# LeWorldModel 统一论文方案：把 ATM、HTSC 和 GDD-WM 整合成三个核心贡献

**我支持把三者放进同一篇论文。** 而且如果组织得当，这篇论文会比只做 HTSC 诊断更有冲击力：它不仅解释世界模型为什么会选错动作，还尝试修复这种错误，并判断修复后的模型什么时候可以被信任。

但必须避免写成三项松散贡献：

- Contribution 1：改进 ATM；
- Contribution 2：提出 HTSC；
- Contribution 3：提出 GDD-WM。

这样很容易被审稿人批评为 *three loosely connected techniques*。

我建议统一成一个科学命题：

> **A world model becomes decision-reliable not merely by encoding actions, but by preserving task-relevant counterfactual action effects, learning valid decision geometry from executed outcomes, and knowing when its predicted action ranking is trustworthy.**

中文：

**世界模型的决策可靠性不仅取决于能否辨识动作，还取决于能否保留动作对任务的真实反事实影响、能否从执行反馈中学习正确的预测几何，以及能否在预测不可靠时拒绝决策。**

## 一、最终三个 Contribution 应该这样设计

<box border radius="xl" padding={3} gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="scan-search" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      <text color="secondary" weight="medium" size="xs">CONTRIBUTION 1 · DIAGNOSE</text>
      <title size="md">Beyond Action Identifiability: Decision-Transfer Gap</title>
      从 ATM 出发，证明动作可辨识、跨域 inverse consistency 和低 MSE 仍不足以保证任务动作排序正确。提出任务条件化、有限时域的真实/预测 action-contrast 诊断，区分可修复误差与信息不可辨识造成的误差。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="brain-circuit" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      <text color="secondary" weight="medium" size="xs">CONTRIBUTION 2 · REPAIR</text>
      <title size="md">Identifiability-Guided Predictive Geometry Distillation</title>
      将 GDD-WM 发展为受动作支持、真实执行证据和可辨识性条件约束的几何学习方法。只将可学习、可验证的任务偏好内化到预测表示，同时保留多步动力学；不能由当前表示辨识的偏好不强行拟合。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      <text color="secondary" weight="medium" size="xs">CONTRIBUTION 3 · DECIDE</text>
      <title size="md">Horizon-Conditioned Selective Planning with Calibrated Regret</title>
      将 HTSC 从条件性风险门控发展为可校准的候选动作决策证书。在冻结基础模型或经过 GDD-WM 适配的 student 上，检验是否能以可接受的拒绝成本改善真正 RGB-only 的闭环 MPC。
    </box>
  </row>
</box>

三个 Contribution 之间存在严格依赖关系：

$$

\boxed{
\underbrace{\text{Diagnose}}_{\text{发现哪种决策信息缺失}}
\;\rightarrow\;
\underbrace{\text{Repair}}_{\text{修复可学习的决策几何}}
\;\rightarrow\;
\underbrace{\text{Decide}}_{\text{认证并执行可信动作}}
}

$$

而且这个顺序不是宣传上的结构。

**Contribution 1 必须提供 Contribution 2 的训练依据；Contribution 2 必须让 Contribution 3 的风险或收益发生可测量变化；Contribution 3 必须最终验证前两个贡献的实际价值。**

这是我建议坚持的论文主结构。

## 二、建议的论文标题

我的首选是：

<title size="xl">Beyond Action Identifiability: Learning and Certifying Decision Transfer in Visual World Models</title>

<text color="secondary" size="sm">副标题可以是：Execution-Grounded Geometry and Horizon-Conditioned Selective Planning</text>

这个标题同时传达三层创新：

1. 超越 ATM 的动作可辨识性；
2. 利用 GDD-WM 学习正确的决策传递结构；
3. 用 HTSC 判断何时值得使用该结构规划。

相比 *Universal World Models* 或 *General Transfer Certificates*，它更具体，也更容易用严格证据支撑。

你可以继续使用 **LeWorldModel** 作为项目名称，**HTSC** 和 **GDD-WM** 作为内部方法组件。论文标题不必再增加第四个缩写。

---

## 三、把三个贡献写成同一个数学问题

论文需要有一个贯穿诊断、学习和规划的共同对象。

我建议采用 **task-conditioned counterfactual cost contrast**。

对于视觉历史 $h_t$、目标 $g$ 和两条候选动作序列 $U_i,U_j$，定义真实代价：

$$

J_H^*(h_t,U,g)
=
\mathbb E[
\ell(Y_{t:t+H},g)\mid h_t,\operatorname{do}(U)
]

$$

其中期望表示在给定可观测历史与干预动作条件下的真实任务代价。

真实动作偏好为：

$$

\Delta_{ij}^*
=
J_H^*(h_t,U_j,g)-J_H^*(h_t,U_i,g)

$$

模型预测的动作偏好为：

$$

\widehat\Delta_{ij}^{\theta}
=
\widehat J_\theta(h_t,U_j,g)
-
\widehat J_\theta(h_t,U_i,g)

$$

核心误差定义为：

$$

\boxed{
E_{ij,H}^{DT}
=
\left|
\Delta_{ij}^*-\widehat\Delta_{ij}^{\theta}
\right|
}

$$

这里正的 $\Delta_{ij}$ 表示 $U_i$ 优于 $U_j$。

这个对象能够同时连接三个 Contribution。

| 贡献 | 围绕同一个对象做什么 |
|---|---|
| ATM → Decision Transfer | 解释为什么动作可恢复并不能保证 $E_{ij,H}^{DT}$ 小 |
| GDD-WM | 从可信执行反馈学习预测几何，尝试降低 $E_{ij,H}^{DT}$ |
| HTSC | 为 $E_{ij,H}^{DT}$ 构造可信上界，判断预测排序能否被放行 |

这才是真正的统一，而不是仅仅把三个算法串在一起。

### Contribution 1 的理论目标：ATM 通过，规划仍然可以完全错误

我建议用一个很强的反例作为论文的起点。

构造两个 learned dynamics，使它们满足：

$$

\mathrm{ATM}(\theta_1)=\mathrm{ATM}(\theta_2)

$$

$$

\mathrm{MSE}(\theta_1)=\mathrm{MSE}(\theta_2)

$$

但：

$$

\mathrm{Regret}(\theta_1)\ne
\mathrm{Regret}(\theta_2)

$$

在离散系统中，这样的反例很容易严格构造，而且我已用 Python 检查了两个具有相同零 Bayes-ATM 风险、相同 latent MSE，却选择相反最优动作的模型。

接下来需要更强的不可辨识性理论。

假设两个隐藏物理状态在当前固定视觉表征下无法区分：

$$

\Phi(h_1)=\Phi(h_2)

$$

但同一候选动作对的真实代价差异分别为：

$$

\Delta^*(h_1)=+\gamma,\qquad
\Delta^*(h_2)=-\gamma

$$

那么任何只读取 $\Phi(h)$ 的确定性预测器，对两种状态只能输出相同的 $\widehat\Delta$，因此：

$$

\boxed{
\max_{k=1,2}
|\widehat\Delta-\Delta^*(h_k)|
\geq\gamma
}

$$

这个下界非常有价值。

它证明在上述限制下，**单纯修改 downstream metric 或扩大 adapter 容量，无法同时正确恢复两种相反的动作偏好**。

必须通过有效历史、附加观测或新的动作干预消除混叠；如果无法消除，就应该拒绝过度自信的规划。

这里要限定：若 GDD-WM 允许重新训练 encoder，并且原始 RGB 中确实存在被旧 $\Phi$ 丢掉的信息，那么这个固定表征下界可以被新的表示绕开。它不是对所有可能模型的不可修复性定理。

这一点恰好引出 Contribution 2。

### Contribution 2 的方法目标：不是强行修复所有错误，而是有条件地修复

我建议把新版 GDD-WM 明确设计为：

**Identifiability-Guided GDD-WM**

其核心训练原则是：

**只有具备可靠执行监督的偏好，才进入几何学习；只有能够被当前函数类表达的偏好，才尝试通过相应模型参数进行修复。**

一个适合的训练目标为：

$$

\mathcal L_{\rm GDD}
=
\mathcal L_{\rm pred}
+\lambda_{\rm rank}
\sum_{(i,j)\in\mathcal P_{\rm valid}}
w_{ij}
\operatorname{softplus}
\left(
m-(c_j^\theta-c_i^\theta)
\right)
+\lambda_{\rm cons}\mathcal L_{\rm cons}
+\lambda_{\rm preserve}\mathcal L_{\rm preserve}

$$

其中：

$$

c_i^\theta=
\frac1d\|
\widehat z_H^\theta(U_i)-z_g^\theta
\|_2^2

$$

$\mathcal P_{\rm valid}$ 来自真实执行、独立动作支持检查和 teacher preference stability。

$w_{ij}$ 则可以依据真实任务 margin、测量置信度、支持信息和训练集中的偏好一致性确定。

需要保留 realized geometry 和 predicted geometry 两条监督路径，不能通过扭曲预测轨迹来单纯提高 ranking。

最重要的是将错误明确分成三种：

| 错误类型 | 应采取的措施 |
|---|---|
| 当前 frozen latent 包含信息，但任务几何排序不正确 | 尝试 projection / metric distillation |
| 状态信息存在，但 imagined dynamics 无法正确传播 | 尝试受限制的 predictor adaptation |
| 当前历史或表示无法区分具有相反动作后果的物理状态 | 增加历史或有效观测；不足时 abstain |

这能把 GDD-WM 变成一个**由机制诊断指导的表示修复方法**。

但也必须正视竞争工作：<Entity category="other" value="D-JEPA" disambig="2026 decision-aligned visual world model paper"/> 已经研究把真实执行驱动的关系结构写回 JEPA-compatible future representations，并用于原生 latent-distance planning。<Cite ref="turn821077academia60"/>

因此，GDD-WM 的新意不能只是 teacher-free ranking distillation，而必须来自**先判别偏好是否可辨识、是否可表示，再进行受约束修复，最后重新验证规划可靠性**。

### Contribution 3 的理论目标：GDD-WM 修复后，HTSC 能否认证它？

定义 candidate-pair contrast bound：

$$

\left|
\Delta_{ij}^*-\widehat\Delta_{ij}^{\theta}
\right|
\leq b_{ij,H}^{\theta}

$$

只要模型认为 $U_i$ 更好，而且：

$$

\boxed{
\widehat\Delta_{ij}^{\theta}
>b_{ij,H}^{\theta}
}

$$

就能在该误差界有效的事件上保证：

$$

J_H^*(U_i)<J_H^*(U_j)

$$

如果对预测选中的候选 $\widehat U$ 与所有其他候选同时构造有效的上界，则可以得到候选集合内的 regret 上界：

$$

\boxed{
\mathrm{Regret}(\widehat U)
\leq
\max_{j\ne\widehat U}
\left[
b_{\widehat Uj,H}
-
\widehat\Delta_{\widehat Uj}
\right]_+
}

$$

我用多组随机候选代价验证了这个代数推论。

它为你的完整方法提供一个可计算的规划放行规则。

但关键研究内容不是这个初等不等式，而是如何在有限干预数据和 planner-reachable candidate distribution 上，建立**有效且非空洞**的 $b_{ij,H}^{\theta}$。

特别重要的一条规则是：

**GDD-WM 每次改变 encoder、predictor 或决策 metric 后，都必须重新校准 HTSC。**

旧模型上的 risk bound 不能自动转移给修复后的 student。

否则可能出现一种危险情况：模型通过训练学会了操纵预测距离，却继续使用旧证书，得到看似更高的放行率和成功率。

我认为这是统一论文中必须严格避免的证据漏洞。

---

## 四、完整方法应该长什么样？

我建议整体方法采用下面的结构。

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Frozen RGB World Model**
    <text color="secondary" size="xs">LeWM encoder / predictor · Historical representation</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **ATM-based Audit**
        <text color="secondary" size="xs" textAlign="center">Real / predicted inverse consistency</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **Task Transfer Audit**
        <text color="secondary" size="xs" textAlign="center">Action support · Horizon · Task contrasts</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **Identifiability & Repairability Assessment**
    <text color="secondary" size="xs">Which preferences are valid, learnable, and supported?</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **GDD-WM: Execution-Grounded Geometry Distillation**
    <text color="secondary" size="xs">Valid preferences → prediction-preserving student geometry</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **Independent Post-Training HTSC Calibration**
    <text color="secondary" size="xs">Support checks · Pairwise contrast bounds · Planning permission</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Teacher-Free RGB Closed-Loop MPC**
    <text color="secondary" size="xs">Select / Abstain · Execute · Observe · Replan</text>
  </box>
</box>

这里有两个设计细节非常重要。

**第一，ATM 与 HTSC 不必在部署时都以额外神经网络的形式运行。**

ATM 可以主要用于离线物理动作一致性审计和训练数据筛选。HTSC 则负责最终运行时的风险判断。GDD-WM student 在部署时可以不使用 D-JEPA teacher 或关系 reranker，但仍允许使用单独声明的风险门控器。

所以应该准确宣称 *teacher-free, selectively gated MPC*，而不是无条件的零额外模块规划。

**第二，证据筛选不能形成循环自我证明。**

不能用 HTSC 自己认为正确的偏好训练 GDD-WM，然后再用同一份数据证明 HTSC 的判断正确。

必须让真正的执行结果或可靠的独立标注承担监督角色，HTSC 的最终风险校准在 student 冻结后重新进行。

## 五、一篇论文应该采用的决定性实验

我建议实验不再按“PointMaze 一组、PushT 一组、Spectral 一组、GDD 一组”松散排列，而是每组都服务于三个贡献中的一个明确问题。

### Figure 1：ATM 通过了，为什么动作还是选错？

对应 Contribution 1。

需要包含一个严格匹配 ATM / MSE 的理论反例，以及真实 learned world models 上的 matched experiment。

重点展示：

$$

\text{ATM matched}
\quad+\quad
\text{MSE matched}

$$

但真实 task-contrast error、ranking inversion 和 regret 仍有差异。

新增的 task-transfer diagnostics 要能够在留出的模型和环境上解释这种差异。不能只靠合成系统宣称现实中的普遍现象。

### Figure 2：经过有效证据筛选，GDD-WM 到底修复了什么？

对应 Contribution 2。

应当在同一个 candidate benchmark 上比较：

| 方法 | 目的 |
|---|---|
| Frozen LeWM | 原始表示与动力学 |
| Naive preference distillation | 不做 action-support / identifiability 筛选 |
| ATM-guided distillation | 仅使用动作一致性信息 |
| Execution-only GDD | 验证真实执行反馈本身的作用 |
| Full identifiability-guided GDD | 验证诊断指导的几何修复是否带来额外效果 |
| D-JEPA、AD-WM、DA-LeWM | 最新直接竞争基线 |

尤其要比较三种变化：

**真实后继几何是否更符合任务偏好；预测后继几何是否也得到改善；原有 multi-step prediction 是否得以保持。**

如果训练后 prediction MSE 大幅恶化，即使 ranking 提升，也不能轻易称其为更好的 predictive world model。

这项实验还需要在 PointMaze 和 PushT 中各展示一种机制：导航场景主要检验动作传递与几何排序，PushT 则检验接触模式内部的动作效应是否能被修复。

### Figure 3：修复后的 world model 是否更值得信任？

对应 Contribution 3，也应该是整篇论文的最重要主结果。

建议使用一个完整消融矩阵。

| 方法 | ATM 诊断/筛选 | GDD 修复 | HTSC 放行 |
|---|:---:|:---:|:---:|
| Frozen LeWM | — | — | — |
| ATM screening | ✓ | — | — |
| HTSC-only | ✓ | — | ✓ |
| GDD-only | — | ✓ | — |
| ATM-guided GDD | ✓ | ✓ | — |
| GDD + HTSC | — | ✓ | ✓ |
| Full unified method | ✓ | ✓ | ✓ |

这里勾选的 ATM 表示使用其诊断或筛选信息，不意味着改变了原始 ATM 论文的算法。还应分别加入 uncertainty-aware MPC、D-JEPA native geometry 和 AD-WM 的公平对照。

必须共享初始 roots、任务目标、合理匹配的训练反馈预算和 CEM 计算预算。

最终你希望看到的不是单纯 Full method 的 success 最大，而是一个更深刻的结果：

**GDD-WM 提高真实动作排序正确性，HTSC 在保证风险覆盖的同时扩大有用动作的放行范围，两者结合优于任何单独模块。**

可以重点报告三个量：

$$

\text{Decision Regret}\downarrow

$$

$$

\text{Certified Useful Coverage}\uparrow

$$

$$

\text{Closed-loop Success}\uparrow

$$

其中 Certified Useful Coverage 应在测试前给出明确操作定义，例如“满足指定真实 regret 容差、且被允许执行的测试决策占比”。它只是经验指标，不应与有限样本的条件覆盖定理混淆。

如果 Full method 只是通过提高 abstention rate 来减少错误，就不能构成令人信服的闭环规划贡献。

---

## 六、怎样安排工作，才能真正完成同一篇论文？

结合你仓库当前状态，我建议使用阶段门，而不是同时推进所有模块。

| 阶段 | 任务 | 必须得到的证据 |
|---|---|---|
| P0 | PointMaze oracle candidate ceiling + ATM 基线 | 证明当前 candidate bank 存在真实可改善的动作选择空间 |
| P1 | ATM–MSE matched transfer study | 证实真实模型存在 ATM 未解释的 task-transfer failure |
| P2 | GDD-WM D0–D2 | 建立可信执行偏好，并证明几何修复有独立收益 |
| P3 | Student 冻结后的 HTSC calibration | 构造有效、非空洞的 pairwise regret bound |
| P4 | 统一 RGB-only online MPC | 三个模块产生联合的实际控制收益 |
| P5 | PushT 与跨架构验证 | 检验接触动力学和模型迁移的边界 |

当前 GDD-WM 在仓库中仍处于 *Proposed / Not Yet Validated*；没有经过验证的 D-JEPA teacher package，D1 稳定性与 D2 蒸馏也尚未完成。因此 P2 是统一论文最大的新增实现风险。<Cite ref="turn979555view0"/>

你可以先使用直接执行确认的偏好做一个 `execution-only` 几何学习 pilot，但如果最终想主张对 D-JEPA-style teacher 的稳定偏好进行了蒸馏，必须完成对应 D0/D1 证据，不能用 pilot 替代。

当前 HTSC 的理论 soundness 也依赖外部真实误差界有效这一前提。必须完成从独立数据构造、校准和验证误差界的工作，才能支持 Contribution 3。<Cite ref="turn167430view1"/>

### 三个 Contribution 的通过门槛

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="check-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **C1 通过**

      独立 learned models 上存在 ATM + MSE 没有解释的 decision-transfer gap，新增诊断在独立测试数据上提供增量预测力。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="check-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **C2 通过**

      GDD-WM 在相同训练反馈预算下显著降低 held-out decision regret，保持预测性能，并证明 identifiability-guided filtering 比 naive distillation 有额外价值。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="check-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **C3 通过**

      冻结后的 student 在独立校准条件下具有可靠且非空洞的风险界；RGB-only 闭环控制证明修复与风险门控之间存在真实互补收益。
    </box>
  </row>
</box>

我特别建议将最终测试数据至少分为 model/geometry training、execution preference construction、post-training risk calibration 和 held-out closed-loop evaluation 四类互不泄漏的来源。涉及同一 root 的 candidate pairs 必须整体划分；若要声称跨模型泛化，还需要独立的 world-model training seeds 和未见环境。

---

## 七、可以直接用于论文 Introduction 的三个贡献表述

下面是一版**目标性写法**，尚未完成的结果必须在最终实验证据成立后才改成正式的已完成贡献陈述。

<WritingBlock id="63827" variant="document">**Proposed Contributions**

1. **From action identifiability to decision-transfer sufficiency.** We investigate a fundamental gap between action recoverability and task-relevant counterfactual prediction. Building on real–predicted inverse-consistency diagnostics, we characterize when models with comparable prediction quality and action-identifiability exhibit different finite-horizon decision outcomes. We formulate task-conditioned action-contrast errors and identify conditions under which representation ambiguity prevents reliable decision transfer.

2. **Identifiability-guided, execution-grounded predictive geometry.** We develop a decision-geometry distillation framework that uses independently validated action preferences, action-support evidence, and transfer diagnostics to determine which decision relationships can be learned reliably. The framework aims to internalize task-relevant preferences into candidate-set-independent predictive geometry while preserving action-conditioned multi-step dynamics and distinguishing repairable geometric errors from unresolved state ambiguity.

3. **Calibrated selective planning with decision-regret control.** We integrate the learned predictive geometry with horizon-conditioned transfer-risk calibration to selectively authorize candidate actions. Under explicitly stated calibration and support assumptions, the framework seeks to provide finite-horizon decision-regret guarantees, while independent RGB-only closed-loop experiments test whether geometric repair and selective planning jointly improve reliability, coverage, and task success.</WritingBlock>

这三段最重要的特点是：**每一个贡献既能独立被检验，又只有组合后才能完整回答论文主问题。**

---

## 八、我认为决定这篇论文是否能冲 Oral 的关键

与现有论文相比，你的差异必须非常清晰。

ATM 已经研究 action recoverability 和 real/predicted inverse consistency；DA-LeWM 已经研究 decision-metric alignment；AD-WM 已经研究 action-discriminative training；D-JEPA 已经研究 execution-grounded preference 和 native latent-distance planning。<Cite refs={["turn790473view1","turn821077academia62","turn821077academia61","turn821077academia60"]}/>

另外，action-conditional conformal decision making 也已有直接相关工作，因此“给预测套上 conformal calibration”本身不能承担理论创新。<Cite ref="turn556369academia12"/>

你要提出的，是它们各自没有完整回答的一条**联合命题**：

> **Which decision errors can be detected, which can be repaired from execution feedback, and which must remain uncertified because the visual representation or action support is insufficient?**

中文：

**哪些世界模型决策错误可以提前发现，哪些能够由真实执行反馈修复，哪些由于视觉信息或动作支持不足，必须保持拒绝？**

如果最终实验能出现以下现象，就会非常有说服力：

- ATM 和低 MSE 无法识别的规划错误，被 task-transfer diagnostics 识别；
- 其中确实可修复的部分，通过 GDD-WM 的真实执行偏好监督得到改善；
- 无法可靠辨识的部分，不会被 GDD-WM 强行“修好”并产生虚假的预测自信；
- 经过重新校准的 HTSC 能够区分修复后可执行的动作与仍不可靠的动作；
- 在独立 RGB-only 闭环中，联合方法同时改善决策 regret、实际成功率和有用放行率。

**这才是我最建议追求的顶会级新发现：不是把 ATM、GDD-WM 和 HTSC 简单相加，而是建立“可诊断—可修复—可认证”的视觉世界模型决策学习框架。**

如果能够在理论上刻画哪些动作偏好不可辨识、哪些预测几何能够表达正确偏好，并在多个真实 learned world models 中证明这条边界确实预测了训练和控制结果，这篇研究才真正开始具备冲击 Oral，甚至更高影响力论文的潜力。

-----

> 请你作为ICLR/ICML/NIPS审稿人锐评一下这个研究的缺点并且补充需要做什么才能达到oral/best paper级别研究[BokaiGuo/LeWorldModel: Transfer-sufficient diagnostics and reproducible evidence for frozen visual world models](https://github.com/BokaiGuo/LeWorldModel)

我会按 **2026 年 10 月 16 日可见的最新公开版本**重新审，不沿用之前的打分。重点检查三件事：HTSC＋ATM＋GDD-WM 是否已经形成一个真正的新科学贡献；最新代码和实验究竟支持到哪一步；如果目标是 Oral / Best Paper，哪些缺陷属于“必须推翻重做”，哪些只需要补实验。

我会把结论写成接近正式顶会审稿的形式，并给出一份有优先级、可执行、可证伪的补强路线。

这次查到的更新比之前的摘要更关键：**PointMaze 的 canonical-action 修复已有多 checkpoint 的离线正结果，但 PushT 的三种子短脉冲视觉 MPC 仍是 0/24 成功；GDD-WM 的 D0 数据关联已修复，D1 却明确拒绝了当前“teacher”，因为它实际等于真实代价的负值且缺少重复候选集合。** 这使论文当前最大的缺口更加具体：离线诊断能否转化为可信的训练监督，再转化为独立在线控制收益。<Cite refs={["turn615206view3","turn578345view4","turn615206view1"]}/>

# LeWorldModel 最新版：ICLR / ICML / NeurIPS 强审稿意见

<badge color="danger">当前模拟决定：Reject</badge>

**我的核心判断：LeWorldModel 已经找到了一个重要的科学问题，也积累了相当有价值的理论反例、物理失败分析和可复现实验。但它仍没有证明最重要的事情：诊断、修复、认证三者结合之后，是否真正让 frozen visual world model 做出了更可靠的决策。**

你现在希望把 ATM、HTSC 和 GDD-WM 统一成一篇论文。我支持这个方向，但必须非常严格地评价它。

> **ATM 解释模型为什么会错误理解动作；GDD-WM 应修复可学习的决策偏好；HTSC 则认证修复后的模型什么时候值得执行。**

这是一个好的论文故事，但**好的故事不等于三个成立的 Contribution**。目前只有诊断与部分离线修复具备初步实证基础，GDD-WM 的正式蒸馏和 HTSC 的在线认证尚未完成。

## 一、最新公开版本究竟走到了哪里？

我重新查阅了仓库的 [统一论文计划](https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/UNIFIED_DECISION_TRANSFER_PAPER_PLAN_20261014.md)、[执行记录](https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/EXECUTION_REPORT_20261014.md)、[GDD-WM 方案](https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/GDD_WM_DECISION_GEOMETRY_RESEARCH_PLAN_20261013.md)，并检查了 `task_contrast.py` 等实际源码。

相比上次评审，有两项进展应当明确承认。

**第一，PointMaze 的 canonical-action 修复已经出现更强的离线结果。** 执行记录报告，两类 MLP 架构、五个 seeds 组成的十组评估中，train-derived indexed canonical-action readout 相比 `source+delta` baseline 的平均 regret reduction 为 **+0.1090**，两级 bootstrap 95% CI 为 **[+0.0820, +0.1372]**。

但这个方法在代码里仍使用训练阶段构建的 candidate-ID lookup，并以 simulator-labeled task costs 拟合 readout。因此，它目前证明的是**受限固定候选动作集上的离线任务读出收益**，不是 learned predictive geometry 已经改善，也不是自由候选搜索的在线 MPC。<Cite refs={["turn578345view0","turn408302view2"]}/>

**第二，PushT 已有更直接的视觉控制负结果。** 10 月 15 日执行记录报告，预测 horizon 为 4、每次执行一个动作块的三种子视觉 MPC 实验中，基础模型和修复模型均为 **0/24** 成功；修复模型的最终距离指标反而显著恶化。这说明仅缩短执行块、增加重规划，并没有把已有的 predictor repair 转化为实际控制收益。<Cite ref="turn578345view4"/>

这两项合在一起，揭示了你目前最大的科学矛盾：

**离线 action-effect readout 可以越来越好，但一旦进入实际在线控制，改进仍可能消失甚至反转。**

这不是再添加一个 alignment 指标能够解决的。

## 二、我发现了一个比“实验数量不够”更严重的实现问题

### Major Concern 1：当前 HTSC 的 contrast calibration 不能提供所暗示的有限样本覆盖保证

<badge color="danger">建议立即修复：理论与代码不一致</badge>

我检查了实际源码：

[`le-wm-main/experiments/decision_benchmark/task_contrast.py`](https://github.com/BokaiGuo/LeWorldModel/blob/main/le-wm-main/experiments/decision_benchmark/task_contrast.py)

当前 `calibrate_contrast_bound()` 先计算每个 calibration root 的最大 pairwise contrast error，然后使用：

$$

k_{\rm code}=\lceil n(1-\alpha)\rceil

$$

选择校准值的第 $k_{\rm code}$ 个顺序统计量，作为误差界。代码中的 `_quantile_higher()` 正是这个逻辑。<Cite ref="turn408302view0"/>

但是，如果希望获得标准 split-conformal 的有限样本边际覆盖，正确的顺序统计量应该使用：

$$

\boxed{
k_{\rm valid}=\left\lceil(n+1)(1-\alpha)\right\rceil
}

$$

当 $k_{\rm valid}>n$ 时，标准构造必须采用无限上界等处理，而不能直接把最大 calibration error 当成有保证的上界。

我用 Python 枚举了连续、可交换的 calibration/test root-score 排名，得到：

| 独立 calibration roots | 当前代码的理论覆盖率 | 正确构造的覆盖率 |
|---|---:|---:|
| 8 | 88.89% | 100%（有限阈值不可得） |
| 10 | 81.82% | 90.91% |
| 20 | 85.71% | 90.48% |
| 40 | 87.80% | 90.24% |

<caption>示例目标：90% 边际覆盖；连续分布下按顺序统计量计算。正确构造在第一行需要无限上界。</caption>

这意味着 **当前实现可能低于标称覆盖水平**。

更准确地说，它目前是一个经验 root-level quantile gate，而不是可以直接宣称标准有限样本覆盖的 conformal certificate。

需要立即完成三项修复：

1. 修正有限样本分位数公式，对样本不足返回 `insufficient_calibration`，而不是默默使用过小的有限上界。
2. 用独立 calibration 和 outcome roots 检验实际覆盖、错误放行和放行比例。
3. 明确区分 fixed candidate bank 的 root-level marginal coverage 与经过 CEM 自适应选择后的 online coverage。

这不是措辞问题。如果论文 Contribution 3 宣称 calibrated decision certification，**实际放行代码必须满足理论假设与覆盖公式**。

你目前值得肯定的是，源码明确声明“不提供逐状态条件保证”；但这个自我限制并不能替代正确的有限样本实现。

## 三、另外六个可能直接导致 Reject 的缺陷

### Major Concern 2：当前最好的 PointMaze 改善，可能主要来自 action-bank 结构，而不是 learned world-model dynamics

这是我在最新代码中最担心的因果归因问题。

`audit_indexed_canonical_action_geometry.py` 实际构造的特征为：

$$

x_i=
\left[
z_{\rm source},
\hat z_i-z_{\rm source},
\operatorname{Bank}[candidate\_id_i]
\right]

$$

其中 `Bank[candidate_id]` 是利用训练 roots 构造的固定候选动作摘要。随后使用真实任务代价拟合 ridge readout。<Cite ref="turn408302view2"/>

**所以即使 learned dynamics 没有真正理解反事实动作，只要固定 candidate ID 与任务方向存在可利用的统计关系，readout 也可能学到有用的排序。**

这种模型不一定是坏 baseline。但它绝不能被直接当作 frozen world model 获得了物理决策充分性的证据。

你应该立即做四个反事实消融：

- **Bank-only：** 仅使用候选动作摘要与训练阶段允许的目标信息，完全不使用 predicted latent。
- **No-dynamics：** 使用 source latent 和 candidate embedding，禁用 learned rollout。
- **Shuffled-prediction：** 保留候选 ID 和训练协议，打乱模型的 predicted successor。
- **New-candidate-bank：** 测试训练时从未出现的动作序列、候选 ID 和 CEM-generated candidates。

如果 full readout 在这些对照下没有显著优势，那么最新离线正结果主要支持的是一个**固定动作模板评分器**，而不是 world-model transfer。

另外，执行记录里的 RGB-compatible task audit 与 indexed canonical-action audit 报告了相同的两项核心收益（+0.2845 consistency gain 和 +0.1090 regret reduction）。这并不证明代码有错误，但强烈要求说明：RGB successor 到底进入了动作评分，还是只作为额外验证材料被关联到结果中？<Cite ref="turn578345view1"/>

这属于科学归因问题，而非统计显著性问题。

### Major Concern 3：你可能把“world model 不充分”与“planner 根本规划不到目标”混淆了

这是本次新增文献对你最有冲击的一点。

2026 年 9 月 30 日的 <Entity category="other" value="The Planning Limits of Latent World Models" disambig="Alrasheed et al. 2026 frozen visual world-model planning limits paper"/> 已在五种冻结视觉 encoder 与多个 Meta-World 任务上研究了这一问题。

作者发现，即使换成提供准确真实动力学的模拟器，在固定的短时域目标距离规划中，只要目标远离 imagined rollout，控制仍然可能失败。其报告的一个对照中，目标从五步延伸到二十步，而五步规划的成功率从 92% 降到 41%。<Cite ref="turn621647view0"/>

这意味着一个对 LeWorldModel 十分危险的事实：

$$

\boxed{
\text{Perfect Dynamics}
\not\Rightarrow
\text{Successful Short-Horizon Planning}
}

$$

因为至少有三种不同的失败：

| 失败源 | 具体含义 | 正确补救 |
|---|---|---|
| Representation / dynamics failure | 模型没有预测正确的动作后果 | 改善可观测性、动作辨识和预测 |
| Decision metric failure | 后果预测合理，但代价函数不能表达真实进展 | 任务度量、几何学习、子目标 |
| Planner reachability failure | 候选动作及规划时域不足以获得通向目标的有效进展 | Candidate generation、horizon、subgoal planning |

**HTSC 可以认证一个模型正确比较了当前候选动作，但不能保证候选动作本身能够完成任务。**

即使所有动作都在 support 内、pairwise contrast 都认证通过，一个短视 planner 仍可能反复选择局部正确、全局失败的动作。

因此必须增加：

1. Simulator-perfect-dynamics MPC；
2. Ground-truth candidate oracle；
3. Learned model + privileged task-cost oracle；
4. Frozen world model + learned / expert subgoal baseline。

只有先确认 oracle 能完成任务，才能把模型失败归因于 transfer insufficiency。

否则 reviewer 会认为：

> The work conflates model insufficiency with planning horizon, candidate generation, and objective-design failures.

这可能比缺少多一个 benchmark 更致命。

### Major Concern 4：GDD-WM 当前没有合法的独立 teacher 证据，方法新颖性也受到 D-JEPA 直接挑战

最新版执行记录明确指出，GDD-WM v5 的 D0 已通过，并将 640/640 realized RGB projection references 完成关联。

但 D1 仍然 `rejected_fail_closed`，因为：

- 没有同一候选对在不同 candidate-set contexts 下的重复偏好；
- 当前声明的 offline teacher score **恰好等于负的真实任务代价**。<Cite ref="turn615206view1"/>

这不是一套独立 teacher 的输出，而是直接使用答案构造的偏好。

它可以用于 simulator-oracle-supervised pilot，但不能证明：

- D-JEPA teacher 的偏好稳定；
- teacher 已学到真实执行规律；
- GDD-WM 具备从独立 teacher 蒸馏的能力。

目前仓库已经正确拒绝了这种证据，我赞同这个处理。但这也意味着 **Contribution 2 尚未得到有效实验验证**。

更大的问题是创新性。

<Entity category="other" value="D-JEPA: A Decision-Aligned Latent World Model" disambig="Liu et al. 2026 execution-grounded decision alignment paper"/> 已经使用执行反馈学习 candidate preferences，并研究把决策结构写回未来 latent，实现 native latent-distance planning。<Cite ref="turn473460academia44"/>

同时，GDD-WM 的 projection-only 形式：

$$

c_i=\|P(\hat z_i-z_g)\|_2^2

$$

等价于：

$$

c_i=(\hat z_i-z_g)^\top M(\hat z_i-z_g),
\qquad M=P^\top P\succeq0

$$

本质上是 PSD Mahalanobis metric learning。

因此，**“执行偏好蒸馏到欧氏几何，推理时不用 reranker”已经不足以作为一个独立的高创新性贡献。**

真正值得研究的是：

> 哪些偏好可以被当前预测表示修复，哪些由于 latent aliasing、错误动力学或目标表达能力不足而不能修复？能否提前判断修复是否值得进行？

GDD-WM 必须比普通 ranking distillation 多提供一个可验证的 **repairability criterion**。

### Major Concern 5：PushT 已经不是“多训一点 adapter”就能解释的失败

PushT 最新执行记录中的问题具有很强的结构性。

你先后发现并修复过目标状态标签错配、RGB 接触特征坐标顺序等问题。修复这些接口缺陷之后，当前位姿或目标角度的读出有所改善，但同一接触模式内部的动作效果、角度排序和完整姿态闭合仍然没有稳定通过。<Cite refs={["turn578345view2","turn578345view3"]}/>

这里有一个必须认真对待的替代解释：

**你观察到的负结果，究竟来自 frozen latent 缺少物理信息，还是因为接触动作执行接口、任务标签、候选策略或控制器设计没有与原始 LeWM 对齐？**

你不能直接用自己新增控制栈里的失败证明 backbone 的物理不可辨识性。

我会要求先在完全相同的 PushT 协议下复现上游 LeWM 的正常规划基线，然后逐一替换读出、adapter、candidate generator 和风险门控。

对于真正的物理机制实验，要继续分离：

$$

\text{Mode recognition}
\rightarrow
\text{Within-mode effect}
\rightarrow
\text{Contact-mode transition}
\rightarrow
\text{Long-horizon pose control}

$$

如果 privileged physics oracle 可以实现姿态闭合，而 RGB-only adapter 不能，才有更清晰的表征或可辨识性瓶颈。

如果 privileged oracle 也无法在当前控制接口中成功，研究重点就应该先回到 planner 和 controller。

### Major Concern 6：跨环境泛化仍然不足，当前统计单位容易被高估

目前经过正式声明的 cross-environment matched audit 仍只有两个环境和七个 identifiable strata；正向点估计尚未通过预注册 sign-flip gate。<Cite ref="turn289415view0"/>

近期的 PointMaze 十组实验虽然更积极，但它们是同一类任务下两种架构与多个种子的组合，不等于十个独立任务。

更关键的是，部分结果使用每个模型少量 held-out roots，且在同一协议上反复开发 readout、修复 action bank 和比较 variants。

最终 outcome 如果持续参与了方法决策，即使从未被梯度下降直接使用，也会产生 **researcher-induced test-set overfitting**。

解决方法是建立新的、锁定的 final evaluation：

- 开发阶段使用现有 roots；
- 方法冻结之后才生成新的候选集合和 outcome roots；
- 模型级泛化使用独立 backbone training seeds；
- 环境级泛化使用此前未参与方法选择的任务；
- outcome 上只执行预注册主检验，其他分析标记为 exploratory。

不要通过更多 candidate pairs 或更多 adapter seeds 来替代独立的 world-model 重复。

### Major Concern 7：复现治理有进步，但公开证据的时间与原始产物链仍需审计

你已经增加 input fingerprint、manifest、stage status 和证据边界，这值得肯定。

但我还注意到一个具体问题：

公开的 `EXECUTION_REPORT_20261014.md` 中出现了明确标注 **2026 年 10 月 18 日**与 **10 月 24 日**的执行记录，而本次评审日期是 **2026 年 10 月 16 日**。<Cite ref="turn668849view1"/>

这可能是文件合并、预写记录、环境时钟或实验归档流程问题，不能据此推断数据造假。但在缺少可核验时间戳与 commit/run provenance 的情况下，我不能把那些未来日期对应的条目作为截至今天已经完成的实验。

正式投稿前，应将每次正式 run 锁定到 Git commit、真实执行时间、原始 metrics、输入 checksum、独立产物和生成脚本。历史执行记录不要在后续更新时无版本地区覆盖。

另外，仓库本身说明大型训练数据和 checkpoints 并未随公开发布提供。因此目前是**提供了完整度较高的复现工具链**，而不是所有核心结果都已允许第三方独立复现。<Cite ref="turn260427view1"/>

---

## 四、怎样把三项贡献升级成真正的顶会论文？

我建议把论文从“诊断—修复—认证”进一步升级成：

<title size="lg">Know, Repair, or Abstain: Decision Transfer in Visual World Models</title>

它应该回答一个更深的科学问题：

**对于 world model 选错动作的情况，我们能否预先判断：错误来自缺失的物理信息、错误的预测动力学、错误的决策几何，还是 planner 本身；其中哪些能通过学习修复，哪些必须拒绝？**

相比直接把 ATM、GDD-WM、HTSC 三项拼起来，这更可能构成一个完整的科学发现。

### Contribution 1：建立决策错误的可辨识与可修复性理论

现在你已经有 ATM insufficiency 的简单反例，也有 action-to-task transfer operators。

但还缺少一个统一描述各类错误的数学框架。

我建议引入 realized geometry 作为中间对象。

令真实任务代价差异、真实后继 latent 代价差异、预测后继 latent 代价差异分别为：

$$

\Delta_{ij}^{*},\qquad
\Delta_{ij}^{obs},\qquad
\widehat\Delta_{ij}

$$

那么存在一个精确分解：

$$

\boxed{
\underbrace{\Delta_{ij}^{*}-\widehat\Delta_{ij}}_{\text{Total decision-transfer error}}
=
\underbrace{\Delta_{ij}^{*}-\Delta_{ij}^{obs}}_{\text{Task-geometry gap}}
+
\underbrace{\Delta_{ij}^{obs}-\widehat\Delta_{ij}}_{\text{Predictive-dynamics gap}}
}

$$

这不是新定理，而是一个有价值的研究组织对象。对于随机环境，应统一使用条件期望，或对单次执行误差另行建模。

它允许你做以前缺少的机制归因：

| 主要缺口 | 现象 | 对应改进 |
|---|---|---|
| State / action information | 相同表征下可能有相反真实动作效果 | 有效历史、动作干预或拒绝 |
| Task geometry | 真实后继包含任务信息，但 latent cost 排序错误 | GDD-WM metric / representation learning |
| Predictive dynamics | 真实后继排序合理，但 imagined rollout 排错 | Dynamics adapter / predictor update |
| Planner capability | 后继预测与排序正确，但候选集或时域无法解决目标 | Subgoal、候选优化、规划接口 |

关键的新理论方向应是：

**给定冻结表征、动作支持和受限制的 student 函数类，判断哪些决策错误存在不可避免的下界，以及需要何种额外观测或干预才能缩小该下界。**

例如：

- 对 representation aliasing 给出不可辨识性下界；
- 对固定预测特征上的 PSD metric 给出偏好可表示性条件；
- 对有限 action support 给出任务效果估计所需的激励条件；
- 对 horizon-conditioned error 解释为何短时正确不保证长时正确。

这些已有各自的数学传统，单独证明通常不够新。**真正的贡献是把这些边界连接成一个能够指导实际 world-model repair 的可检验判别准则。**

### Contribution 2：GDD-WM 不只是学习偏好，而是选择性修复

我建议改造 GDD-WM 的目标：

从“尽量把所有正确偏好写进 latent”变成：

> **只修复具有足够执行证据、并且在当前函数类下可以学习的偏好；对于缺乏物理可辨识性的情况，不允许通过提高预测自信来伪造修复。**

至少要比较四种学习策略：

| 训练方法 | 要证明什么 |
|---|---|
| Naive preference distillation | 普通监督学习能达到什么水平 |
| ATM-guided distillation | 动作语义筛选能额外带来什么 |
| Execution-grounded GDD | 真实执行偏好是否比 inverse labels 更有价值 |
| Identifiability-guided GDD | 可修复性判断是否带来独立收益 |

还必须加入 D-JEPA、AD-WM、DA-LeWM、Mahalanobis metric 和 pointwise scorer 强基线。

尤其要回答：

**如果我直接用相同执行数据训练一个普通 MLP action scorer，它是不是就能达到 GDD-WM 的效果？**

如果答案是肯定的，你需要找出原生预测几何的独特价值，例如跨目标迁移、受限模型容量、保持动力学一致性、没有额外推理模块，或者对未见候选的可靠性。

否则“内生几何”只是更复杂的实现形式，不是新的科学贡献。

### Contribution 3：HTSC 必须认证 GDD-WM 修复后的动作，而非只是检查原模型

这是整篇论文最重要的系统贡献。

定义：

$$

\widehat U_\theta
=
\arg\min_{U\in\mathcal C}
\widehat J_\theta(U)

$$

对于 student 冻结后的模型，利用独立校准数据构造完整 candidate-bank 的 contrast bounds。

如果对所有竞争候选 $U_j$，预测代价差异均足够大于对应的有效误差界，则允许执行：

$$

\boxed{
\widehat J_\theta(U_j)-\widehat J_\theta(\widehat U_\theta)
>b_{\widehat U_\theta,j,H}
}

$$

并且 action/state support 必须通过。

实际实现上，需要将当前 `certify_pair()` 的二候选接口，完善成整个候选集合的联合决策规则；所有比较共享经过正确有限样本校准的覆盖事件。

**GDD-WM 一旦更新 encoder、predictor 或 metric，HTSC 必须重新校准。**

最终应该出现三类清晰结果：

- 修复前模型不可靠，HTSC 正确拒绝；
- 修复后动作排序确实改善，HTSC 可以放行更多有用动作；
- 仍然不可辨识的动作不会因为 student 过度自信而被错误放行。

但要注意，标准边际覆盖只保证声明的统计事件，不能直接写成任意当前状态的确定性安全认证，也不能保证任意自适应 CEM 轨迹的条件可靠性。与已有 action-conditional conformal decision 方法的区别必须明确。<Cite ref="turn621647academia0"/>

---

## 五、真正决定 Oral 竞争力的实验设计

我建议不要继续以“每个模块在自己的 benchmark 上提高一点”为核心。

应该使用同一套 candidate-action evidence，完成三张决定性的主图。

### Figure 1：ATM 和 MSE 都无法区分的失败，能否被 transfer diagnostics 解释？

必须使用真正独立训练的 learned models，而不只是线性合成反例。

核心比较：

| Baseline | 需要排除的替代解释 |
|---|---|
| MSE / ensemble uncertainty | 仅仅是预测不确定性不同 |
| ATM / inverse risk | 仅仅是动作可恢复性不同 |
| DA-LeWM alignment | 仅仅是 latent-distance 排序不同 |
| Planning-range baseline | 仅仅是目标超出预测时域 |
| Full task-transfer diagnostic | 是否真的识别了额外的任务条件化失败 |

最终指标必须是 held-out task regret、candidate ranking inversion，以及未见环境上的风险预测能力。

### Figure 2：GDD-WM 是否真正修复了物理决策，而不是利用数据捷径？

需要使用同一个 frozen backbone、同一个执行反馈预算和一致的候选 bank。

至少报告：

- Realized geometry ranking；
- Predicted geometry ranking；
- 多步 successor prediction；
- 固定候选上的真实 regret；
- 未见候选上的真实 regret；
- 不同动作支持水平下的结果。

特别重要的负控制是 **bank-only 和 shuffled-prediction**。

如果删除真实 world-model rollout 之后结果几乎不变，说明修复没有真正依赖预测动力学。

同时必须有真正独立的 teacher 或明确标记为 oracle-execution supervised 的方案，不允许用真实 outcome 直接伪造 teacher。

### Figure 3：联合方法是否在闭环中产生互补收益？

我建议采用这个最小消融矩阵：

| 方法 | 执行反馈几何修复 | 重新校准的 HTSC |
|---|:---:|:---:|
| Frozen LeWM | — | — |
| Frozen LeWM + HTSC | — | ✓ |
| GDD-WM | ✓ | — |
| GDD-WM + HTSC | ✓ | ✓ |

另加 ATM-based screening、D-JEPA、AD-WM、常规 uncertainty-aware MPC 和 simulator-perfect oracle。

重点不只是看 Full method 的最终成功率是否最高。

要证明的是：

$$

\text{Decision Regret}\downarrow
\quad\land\quad
\text{Useful Coverage}\uparrow
\quad\land\quad
\text{Closed-loop Success}\uparrow

$$

同时报告 fallback 使用率、规划延迟、动作预算、实际风险覆盖和错误放行率。

如果 GDD-WM 单独提高 success，但 HTSC 无法带来有用的可靠性改善，那么第三个 Contribution 仍然不成立。

如果 HTSC 只通过拒绝全部动作取得低错误率，也不成立。

---

## 六、我建议的最短执行路径

不应该六个方向同时平均投入。

| 顺序 | 必须完成的任务 | 停止或转向条件 |
|---|---|---|
| P0 | 修正 contrast calibration；复现原始 LeWM baseline；确认 oracle candidate ceiling | Oracle 规划本身不能成功，就先修 planner |
| P1 | ATM + MSE + geometry 的 matched decision-transfer study | 如果 geometry 没有增量信息，就将其降为解释性指标 |
| P2 | 完成真实 D1 preference stability 和 GDD-WM D2 | 如果没有独立 teacher / execution evidence，不进入正式蒸馏 |
| P3 | 检验预测保持、未见候选泛化与 repairability | 若只是 bank-ID scorer 有效，不能宣称 world-model repair |
| P4 | 冻结 student，独立校准 HTSC | 覆盖率或放行实用性不合格，不称为 certificate |
| P5 | PointMaze / Reacher RGB-only MPC 正式评估 | 没有闭环收益，不提升为 Oral 级结论 |
| P6 | PushT 接触机制与新环境外部验证 | 未通过则明确报告边界，不宣称 universal |

每个正式阶段都必须使用锁定输入、独立 outcome 和预先声明的比较方法。不要继续在同一个小型 outcome panel 上循环试错，然后将最后一个正结果视为 confirmatory evidence。

### 如果只能分配一次主要 GPU 研究预算

我会优先投给：

**一个全新、真正可重置的 PointMaze / Reacher counterfactual candidate 数据集，加上真正的 RGB-only MPC 基线。**

原因很简单：这批数据同时可以支持 ATM、GDD-WM、HTSC 和在线规划。

而继续增加谱特征或另一个 PushT adapter，只会分别加强局部模块，未必能推进整篇论文的中心命题。

---

## 七、什么样的结果值得 Oral，什么才有 Best Paper 潜力？

| 层级 | 我希望看到的决定性发现 |
|---|---|
| **扎实主会论文** | 一个明确的新 decision-transfer 方法，在独立视觉闭环中超过强基线，具有完整消融、统计和失败边界 |
| **Oral 竞争力** | 新的可辨识性与可修复性准则能解释 ATM、MSE、常规几何指标解释不了的真实失败；GDD-WM 据此修复；HTSC 可靠放行，三项贡献有明确互补性 |
| **Best Paper 潜力** | 建立此前没有的通用性较强的决策信息边界：不仅知道何时 world model 会失败，还能预先区分什么信息可以通过执行学习恢复、什么根本不可辨识，并给出实际可用的有限数据方法和跨场景验证 |

Best Paper 的关键不是再多一个百分点的 success rate，而是**研究结果是否改变了领域评价和构造 world models 的方式**。

例如，如果你最终能够证明：

> 在相同预测误差与 ATM action-identifiability 下，视觉 world models 的决策失败由可测量的 task-transfer ambiguity 决定；这种 ambiguity 可以预测哪些偏好可修复、哪些必须拒绝，并且这种区分可以显著提高真实闭环决策可靠性。

那么它就不再是一篇普通的 metric/adapter/MPC 组合论文。

它将提出一个新的 **decision-repairability frontier**：从一个模型可观测、可干预的信息出发，刻画其可以被可靠修复与认证的决策范围。

这才是我认为最值得追求的标志性科学贡献。

---

## 八、最终模拟审稿意见

<box border radius="lg" padding={3} gap={2}>
  <row justify="between" align="center">
    **Recommendation**
    <badge color="danger">Reject</badge>
  </row>
  **Strengths**

  研究问题重要；已清晰区分预测、动作辨识与决策充分性；对谱几何和 commutator 的局限有数学认识；PointMaze 的 action-bank 诊断逐渐形成受控证据；PushT 保留负结果；实验执行治理比普通探索性项目更严谨。

  **Major Weaknesses**

  当前最积极的离线结果可能依赖固定 candidate-bank 特征；HTSC 的有限样本校准代码存在应修复的问题；GDD-WM 缺少合法的独立 teacher/student 结果；真实 RGB 闭环收益未建立；PushT 姿态闭合仍失败；统计泛化与外部复现证据不足；最核心的三个方法模块还没有被证明具有互补性。

  **What would change my decision?**

  一套真实执行验证的反事实候选数据、一个独立有效的决策误差认证方法，以及 GDD-WM 修复与 HTSC 放行在未见视觉闭环任务上产生的联合收益。
</box>

### 最终结论

**我仍然看好 LeWorldModel 的研究方向，但现在必须停止用新增模块数量衡量进展。**

你已经有足够多的数学工具：covariance、spectrum、Schur、resolvent、commutator、transfer operators、history lifting、ATM inverse risk、action support、GDD geometry 和 HTSC risk。

真正缺少的是一个**能让审稿人无法用更简单的替代解释否定的科学结果**。

我最建议你围绕一个问题集中全部工作：

> **对于一个会选错动作的 world model，我们能否在动作执行前判断它究竟是缺少必要信息、学错了动力学、用错了决策几何，还是受到了 planner 本身的限制；进而知道应该修复、主动获取信息，还是拒绝规划？**

当 ATM 负责发现信息缺口、GDD-WM 负责真正可修复的部分、HTSC 负责限制仍不可信的决策，并且三者在独立 RGB 闭环中得到共同验证时，这篇论文才会从当前的 **Reject** 真正走向有竞争力的 **Oral**。

进一步证明这种“可修复与不可修复的边界”跨 world-model architectures 和物理环境成立，才是我认为值得为 Best Paper 投入资源的方向。

-----

> 请你作为ICLR/ICML/NIPS审稿人锐评一下这个研究的缺点并且补充需要做什么才能达到oral/best paper级别研究[BokaiGuo/LeWorldModel: Transfer-sufficient diagnostics and reproducible evidence for frozen visual world models](https://github.com/BokaiGuo/LeWorldModel)

我会按 2026 年 10 月 18 日能核验的公开仓库状态重新评审，不直接沿用上次的 Reject 结论。重点检查三条贡献链是否真的打通：ATM 之外的决策失配是否成立、GDD-WM 是否完成可信修复、HTSC 是否在独立 RGB 闭环中提供有效认证。之后我会给出接近正式审稿的主要质疑、可能推翻结论的对照实验，以及冲 Oral / Best Paper 最值得集中攻克的研究突破。

# LeWorldModel：ICLR / ICML / NeurIPS 顶会强审稿与 Oral / Best Paper 升级路线

<text color="secondary" size="sm">评审基准：2026 年 10 月 18 日｜公开 GitHub main 分支、实验执行报告、统一论文计划、HTSC 与 GDD-WM 核心实现，以及直接竞争论文</text>

**模拟审稿结论：Reject，研究问题的潜力明显高于当前证据成熟度。**

我认为 LeWorldModel 最有价值的研究问题是成立的，但距离一篇有 Oral 竞争力的论文仍有相当距离。

最关键的问题不是实验数量太少，而是：

> **你已经能够发现 world model 中一些非常有意义的决策失效现象，但还没有建立一个经过独立验证的机制，证明这些错误能够被预先识别、选择性修复，并在真实闭环规划中带来可认证的收益。**

你希望将 ATM、HTSC 和 GDD-WM 组织为一篇论文的三个贡献。我认为这条方向可以继续，但以审稿人的标准，三者目前还没有构成三个经过验证的科学贡献。

## 一、综合审稿评分

<text color="secondary" size="xs">以下是模拟评分，非真实会议评分或接收概率。</text>

| 维度 | 评分 | 我的判断 |
|---|---:|---|
| 科学问题的重要性 | 5/5 | 很值得研究 |
| 原创性 | 3/5 | 有好的问题整合，但已有工作覆盖较多 |
| 理论深度 | 3/5 | 框架完整，核心新定理尚不够强 |
| 实证可信度 | 2/5 | 局部正结果存在，主证据链尚未闭合 |
| 方法完成度 | 2/5 | GDD-WM、HTSC 部署级认证仍不成熟 |
| 复现与负结果管理 | 4/5 | 明显优于一般探索性代码库 |
| Oral 竞争力 | 低 | 缺少决定性新发现 |
| Best Paper 竞争力 | 尚不具备 | 缺少能改变领域认识的突破 |

### 截至本次评审，哪些结果真正成立？

我重新核对了仓库的 [执行报告](https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/EXECUTION_REPORT_20261014.md) 和 [统一论文计划](https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/UNIFIED_DECISION_TRANSFER_PAPER_PLAN_20261014.md)。

| 模块 | 当前公开证据 | 审稿认定 |
|---|---|---|
| ATM → Task transfer | PointMaze 离线任务对比出现约 40.8% 排序反转 | 决策失配值得研究，但尚未证明新诊断的跨模型增量价值 |
| PointMaze action readout | 10 个 architecture × seed cases，平均 regret reduction **+0.1090** | 有正结果，但依赖固定候选 ID 的 train-derived readout |
| RGB successor | 多 checkpoint 后继预测改善，案例中约 14.21% 的 aggregate improvement | 预测证据，不等于规划能力 |
| GDD-WM | D0 数据关联完成，D1 拒绝当前 teacher，D2 未完成 | 还没有正式的几何蒸馏正结果 |
| HTSC | 已有 contrast-risk 代码与 fail-closed gate | 需要修正校准与独立覆盖验证 |
| PushT | 短脉冲视觉 MPC 基础/修复版均为 0/24 | 没有稳定姿态闭合 |
| 10 月 18 日 PushT 更新 | Orientation gate 对照均为 0/8；paired probe pilot 为 2/8 | 有局部干预迹象，但没有稳定闭环收益 |
| 跨环境 matched audit | 正点估计，但正式统计门槛未通过 | 不能声称普遍预测 planning risk |

这些判断分别由仓库的实验记录、GDD 审计和控制实验支持。<Cite refs={["turn188899view3","turn587030view0","turn587030view3","turn587030view4"]}/>

特别说明：执行报告中还包含标记为 **2026 年 10 月 24 日**的实验条目。该日期晚于本次评审的 10 月 18 日，因此我不把这些条目当作截至今天已完成且可确认的证据。这需要仓库维护者进一步核对执行时间与版本来源。<Cite ref="turn587030view1"/>

**我的第一条审稿意见是：项目目前更像一个建设得相当完整的 decision-transfer research program，而不是一篇已经用决定性理论与实验支撑核心结论的方法论文。**

下面是我认为最可能导致 Reject 的实质性问题，以及怎样把它们转化为更高水平的研究贡献。

## 二、八个可能导致 Reject 的重大问题

### Major Concern 1：你还没有证明 ATM 之外存在足够强、足够普遍的新科学问题

这是整篇论文的创新性风险。

ATM 已经证明：真实与预测 latent transition 的动作可恢复性，以及跨域 inverse consistency，与规划结果存在重要联系。它还提供了 model screening 的实证结果。<Cite ref="turn185936view0"/>

你现在希望继续证明：

$$

\text{Action Identifiability}
\not\Rightarrow
\text{Decision Sufficiency}

$$

这个命题是正确的研究方向，但仅仅证明它成立，**已经不足以成为顶会主贡献**。

原因是 DeepMDP、value-aware model learning、DA-LeWM 和 D-JEPA 等工作，早已从不同角度指出表示预测质量与下游决策价值之间的差异。<Cite refs={["","turn185936academia26","turn185936view1"]}/>

真正需要的是：

**在严格控制 ATM、MSE、action support、model capacity 和 planner reachability 之后，仍然发现能够稳定预测实际决策错误的新变量。**

我会要求你进行一个非常严格的 matched experiment。

| 对照条件 | 必须回答的问题 |
|---|---|
| MSE matched | 是否只是普通预测准确率不同？ |
| ATM matched | 是否只是动作语义不一致？ |
| Support matched | 是否只是训练动作覆盖不足？ |
| Geometry matched | 是否只是目标代价函数不同？ |
| Planner matched | 是否只是候选搜索质量不同？ |
| HTSC diagnostic | 是否有真正额外的决策风险信息？ |

主实验必须在未见 roots、未见模型训练 seeds 上成立。

更重要的是，不能只展示两个模型的 regret 不同，而需要证明新诊断能够**在不知道测试真实后果的条件下，预测这种差异**。

否则审稿人会写：

> The work demonstrates another instance of prediction–decision misalignment, but does not establish a substantially new mechanism beyond existing action-identifiability and decision-alignment studies.

**我的要求：C1 不能只是一篇反例论文，必须发展成可外推的决策失效判别机制。**

### Major Concern 2：PointMaze 最好的正结果可能不需要 world-model dynamics

这是当前最值得立即排查的因果归因问题。

我检查了 [`audit_indexed_canonical_action_geometry.py`](https://github.com/BokaiGuo/LeWorldModel/blob/main/le-wm-main/experiments/gdd_wm/audit_indexed_canonical_action_geometry.py)。

其特征包含：

$$

x_i=
\left[
z_{\mathrm{source}},
\hat z_i-z_{\mathrm{source}},
\operatorname{Bank}(\mathrm{candidate\_id}_i)
\right]

$$

其中 `Bank(candidate_id)` 从训练 roots 重构固定的 goal-relative action summary，任务 readout 使用训练阶段的 simulator-labeled real task costs。<Cite ref="turn514229view2"/>

这个做法作为离线任务评分方法没有原则性错误；训练集监督也是允许的。

但它有一个非常严重的替代解释：

**模型可能主要通过固定动作模板、候选 ID 和统计规律预测任务代价，而不是通过 learned rollout 的物理结构作出决策。**

这会直接动摇你想宣称的 *world-model repair*。

我要求四项 kill tests：

1. **Bank-only**：删除所有 predicted latent，只保留训练允许的状态、目标和动作模板特征。
2. **No-dynamics**：保留 source latent 与动作编码，不执行 learned dynamics rollout。
3. **Shuffled successor**：打乱 predicted successor，同时保持候选 ID 和其他输入不变。
4. **Unseen candidate bank**：完全更换训练时未出现的动作序列、候选 ID，甚至使用 CEM 生成连续新动作。

最关键的是第四项。

如果你的 +0.1090 regret reduction 在固定 16-candidate bank 中成立，但换一批真实动作序列就消失，那么这只能证明一个特定离线候选评分器有效。

如果 Bank-only 与 Full model 的差别接近零，论文甚至必须重新考虑是否需要使用 world-model predictor。

这些实验如果失败，不应通过再添加一个 action head 来回避，而应回到方法定义。

### Major Concern 3：HTSC 的有限样本校准代码仍存在一个实质性问题

这个问题需要在继续使用 *certificate* 一词之前修正。

我重新查看了公开源码 [`task_contrast.py`](https://github.com/BokaiGuo/LeWorldModel/blob/main/le-wm-main/experiments/decision_benchmark/task_contrast.py)。

当前 `calibrate_contrast_bound()` 取各 calibration root 上的最大 pairwise contrast error，然后调用 `_quantile_higher()`，其分位数索引相当于：

$$

k_{\mathrm{current}}=
\lceil n(1-\alpha)\rceil

$$

但标准 split-conformal 的有限样本边际覆盖构造应使用：

$$

\boxed{
k_{\mathrm{valid}}=
\lceil(n+1)(1-\alpha)\rceil
}

$$

当所需顺序统计量超过 calibration 样本数量时，需要返回无限阈值，或明确拒绝提供有限的分布无关界，而不是简单截断为样本最大值。

这是可以直接从源码验证的问题。<Cite ref="turn514229view1"/>

例如，在连续、可交换的 root-level scores 下，目标为 90% 边际覆盖：

| Calibration roots | 当前分位数的覆盖率 | 标准构造 |
|---|---:|---|
| 8 | 88.89% | 没有有限的非随机阈值 |
| 20 | 85.71% | 90.48% |
| 40 | 87.80% | 90.24% |

当只有 8 个独立 calibration roots 时，这个问题尤其突出。

还有三个更深的理论缺口：

- **边际覆盖不等于放行动作的条件覆盖。** 即使 root-level 覆盖达到标称值，也不能直接推出 $P(R>\tau\mid\mathrm{permit})$ 被控制。
- **固定 candidate bank 的校准不能自动迁移到自适应 CEM 候选。** 新的 proposal distribution 改变了误差统计对象。
- **单次真实执行的 cost coverage 不等于条件期望代价的 coverage。** 随机系统需要处理执行噪声与期望估计。

此外，目前接口以 `certify_pair()` 为主，需要完成整个候选集合的联合放行，以及相应的 regret bound。

我建议将理论目标严格写为：

$$

\Pr\left(
\max_{i,j\in\mathcal C}
|e_{ij,H}|
\leq q_\alpha
\right)\geq1-\alpha

$$

其中必须明确：模型已冻结、候选生成过程固定、calibration/test roots 满足何种可交换性条件。

如果预测最优候选为 $\hat i$，在该覆盖事件上：

$$

\boxed{
R_H(\hat i)\leq
\max_{j\neq\hat i}
\left[
q_\alpha-
(\widehat J_j-\widehat J_{\hat i})
\right]_+
}

$$

这个 regret 推论本身并不新。

**真正的研究挑战是构造足够紧、适用于实际候选分布的界，而不是从一个普通分位数得到一个漂亮的许可条件。**

2026 年 ICML 已有 <Entity category="other" value="Conformal Risk-Averse Decision Making with Action Conditional Guarantee" disambig="ICML 2026 action-conditional conformal decision research"/>，明确研究比普通边际覆盖更强的动作条件保证。你必须正面区别于这类工作。<Cite ref=""/>

### Major Concern 4：GDD-WM 目前既缺少有效 teacher 证据，也缺少无法被普通 metric learning 替代的创新性

仓库最新 D0/D1 审计的结果非常明确：

- 640 条数据的关联与 schema 校验完成；
- 但每个 root 只有一个 candidate-set context，无法验证偏好跨集合稳定性；
- 当前 `teacher_score` 恰好等于 `-real_task_cost`；
- D1 因此拒绝，D2 正式蒸馏没有启动。<Cite ref="turn587030view4"/>

这里需要区分两件事。

**使用真实执行代价训练模型，本身完全合法。** 它是很有价值的 execution supervision。

问题在于：不能一方面把它声明成独立 teacher，另一方面又用完全相同的真实代价证明 teacher 与执行结果一致。后者是同源标签比较，而不是独立验证。

我甚至建议先不用 D-JEPA teacher，直接做一个明确标记的 *execution-supervised GDD*。只要训练和测试隔离正确，它就可以验证几何学习的可行性。

但更严重的是创新性。

如果 GDD-WM 的核心是：

$$

c_\theta(z,z_g)=\|P(z-z_g)\|_2^2

$$

那么对于固定表示，它等价于：

$$

c_\theta(z,z_g)
=(z-z_g)^\top M(z-z_g),
\qquad M=P^\top P\succeq0

$$

本质是低秩 PSD metric learning。

而 D-JEPA 已经研究 execution-grounded preferences，并把决策关系写入 JEPA-compatible future representations，实现原生 latent-distance planning。AD-WM 已经用动作可辨识训练提升反事实 MPC，并报告仿真和真实机器人结果。<Cite refs={["turn185936view1","turn185936view2"]}/>

因此，我作为审稿人不会接受：

> 我们把执行偏好蒸馏进预测表示，并在推理时移除 reranker。

这是已有工作覆盖过的思想。

我会要求 GDD-WM 提供一个更强的发现：

**在训练之前就能判断哪些错误适合通过几何学习修复，哪些必须修改 dynamics，哪些因为信息不足无法可靠修复。**

而且这个判断必须比不加筛选的 GDD 更好。

如果普通 MLP scorer 或 Mahalanobis metric 在相同反馈预算、相同候选分布下达到相同表现，那么 GDD-WM 不能依靠“内部几何更优雅”来主张顶会级创新。

### Major Concern 5：你可能正在修复 world model，其实真正失败的是 planner

这可能是比 representation learning 更危险的混淆。

2026 年 9 月的 <Entity category="other" value="The Planning Limits of Latent World Models" disambig="Alrasheed et al. 2026 research on latent planning horizon limitations"/> 研究表明，短 horizon 的规划失败甚至可能发生在使用真实模拟器执行精确前瞻的情况下。该论文报告，在一个五步规划设置中，随着目标距离延长，使用真实模拟器预测的成功率也可能从 92% 降到 41%。<Cite ref="turn185936view3"/>

也就是说：

$$

\boxed{
\text{Perfect Dynamics}
\not\Rightarrow
\text{Successful Planning}
}

$$

HTSC 最多能够证明，在声明的条件下模型对当前候选动作的相对代价判断足够准确。

它不能自动证明：

- 当前 candidate bank 包含解决任务的动作；
- 当前 horizon 足以形成有效进展；
- 终点代价能反映绕障碍物或接触操作的长期价值；
- CEM 能找到好候选；
- 重规划后的长期策略能够成功。

因此，你必须建立四种 oracle。

| Oracle / Control | 分离什么失败 |
|---|---|
| Ground-truth candidate-bank oracle | 候选集合是否包含有效选择 |
| Perfect-dynamics MPC | 在同样 horizon 和目标函数下能否规划 |
| Real-successor + learned geometry | 学到的任务几何是否错误 |
| Learned dynamics + oracle task readout | 动力学与任务读出各承担多少误差 |

我会把 **oracle candidate ceiling** 作为下一阶段的第一个实验，而不是等所有 GDD-WM 训练完成后再做。

因为如果 oracle 都无法成功，你继续训练更强的 latent metric，本质上可能只是在优化一个无解或近视的候选选择问题。

### Major Concern 6：PushT 的局部动作辨识改善，还不能作为 GDD-WM 可修复性的证据

你的 PushT 结果揭示了一个很重要的矛盾：

$$

\text{Mode Observability}\uparrow
\quad\text{但}\quad
\text{Within-mode Action Effect}
\text{ 仍不可靠}

$$

10 月 18 日的新记录尤其有信息量。

Orientation gate 在一组三种子审计中获得了很高的 released sign consistency，但实际控制 smoke 中：

- 三种控制方法严格 pose success 均为 `0/8`；
- gate 拦截了约 89.1% 的 contact steps；
- 所有 root 至少使用一次 fallback。

另一项 paired local-effect probe pilot 达到 `2/8` 的 adaptive pose success，但 uncertainty-gated 分支仍仅为 `1/8`，因此不构成稳定规划收益。<Cite ref="turn587030view0"/>

这告诉我们：

**局部 action-effect 估计改善，不等于完整接触动力学被识别；成功拒绝不可信动作，也不等于具有可靠的控制策略。**

尤其需要排除以下替代解释：

1. 状态表示真的缺少接触物理信息。
2. 动作执行坐标、接触接口、候选生成或控制目标仍存在问题。
3. 每次动作效果可局部估计，但接触模式在连续执行中变化，使旧估计迅速失效。
4. Planner 利用预测模型中的系统偏差，反复选择局部看似正确、全局错误的动作。

在正式论文中，PushT 应先作为机制测试，而不是强行承担全部成功证明。

我还会要求上游 LeWM 的正常控制能力在同一协议中复现，以证明新增控制栈没有引入额外失败。

### Major Concern 7：目前的统计显著性还不足以支撑方法的广泛性

你的代码已经很重视 bootstrap、root-disjoint 和 evidence manifests，这是优点。

但存在一个容易被高估的事实：

**960 个 pairwise contrasts 不是 960 个独立统计单位。**

在一个 root 中，16 个候选就能产生 120 个 pairwise comparisons。如果只有 8 个独立 outcome roots，那么绝大部分 pairwise 观测都共享相同初始状态、候选和环境。

同样，10 个 architecture × seed cases 不等于 10 个独立环境。

当前跨环境 matched audit 的 $\Delta R$ 约为 `+0.05336`，但预注册 sign-flip $p=0.0630$，没有通过正式 gate。<Cite ref="turn535549view0"/>

此外，持续围绕同一批 outcome roots 调试 readout、candidate ID、action frame 和 calibration，即使没有梯度泄漏，也可能发生 researcher-induced test-set overfitting。

解决方案不是不断增加 candidate pairs，而是：

**方法完全冻结后，生成全新模型、全新根状态、全新候选动作及全新闭环结果。**

统计上应至少分清 root、episode、model-training seed 和 environment 四个层级，对每种主张采用对应的独立性与层级推断。

### Major Concern 8：ATM、GDD-WM、HTSC 是三个模块，但尚未证明它们必须联合存在

这是统一论文面临的最终拒稿理由。

目前的逻辑是：

$$

\text{ATM}
\rightarrow
\text{GDD-WM}
\rightarrow
\text{HTSC}

$$

但审稿人会问：

> 为什么不直接使用 D-JEPA 的几何修复，再加一个普通 uncertainty 或 conformal gate？

更强的反问是：

> 如果 ATM 只用于诊断，GDD-WM 直接用真实代价学习，HTSC 又是事后校准，那么把这三者串起来，究竟创造了什么不能被简单组合取代的新能力？

所以，除了基本模块消融，你必须增加一个非常难、但非常重要的基线：

**Naive composition baseline：ATM screening + ordinary preference distillation / D-JEPA-style geometry + standard calibrated risk gate。**

这才是 Full method 真正需要超越的对象。

你的新方法必须证明：

**根据错误是否可辨识、是否可表示、是否可修复来选择学习与控制策略，优于简单地把所有模型训练得更好、然后统一施加风险阈值。**

否则三个 Contribution 只是三个合理模块，不是一个新算法原理。

---

## 三、我认为你真正值得冲击 Oral / Best Paper 的科学突破

我不建议继续把核心创新写成：

*Beyond Action Identifiability + Geometry Distillation + Calibrated Planning*

这个表述仍然太接近现有工作的组合。

我更建议提出一个新的中心对象：

<title size="lg">The Decision-Repairability Frontier of Visual World Models</title>

**视觉世界模型的决策可修复性边界。**

其核心问题是：

> **给定有限视觉历史、冻结表示、动作支持、模型预测与少量真实干预数据，能否在部署之前区分：哪些决策错误可以通过任务几何修复，哪些需要修复预测动力学，哪些需要主动获取新信息，哪些在当前观测条件下根本无法被可靠修复？**

这个问题能够让 ATM、GDD-WM 和 HTSC 真正承担不同且不可替代的职责。

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Frozen RGB world model + finite history**
    <text color="secondary" size="xs">Task · Horizon · Action support · Candidate set</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} gap={1} align="center">
    **C1 · Decision-transfer diagnosis**
    <text color="secondary" size="xs">ATM · Real/Predicted Contrasts · Representation Ambiguity</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <text weight="medium" textAlign="center">Geometry error</text>
        <text color="secondary" size="xs" textAlign="center">Execution-grounded GDD repair</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <text weight="medium" textAlign="center">Dynamics error</text>
        <text color="secondary" size="xs" textAlign="center">Constrained predictor adaptation</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <text weight="medium" textAlign="center">Information gap</text>
        <text color="secondary" size="xs" textAlign="center">History / Active probe / Abstain</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <text weight="medium" textAlign="center">Planner limitation</text>
        <text color="secondary" size="xs" textAlign="center">Candidate / Horizon / Subgoal</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **C2 · Selective Repair + Prediction Preservation**
    <text color="secondary" size="xs">Repair only what independent evidence supports</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **C3 · Recalibrated HTSC + RGB Closed-Loop MPC**
    <text color="secondary" size="xs">Permit · Probe · Abstain · Observe · Replan</text>
  </box>
</box>

这张图是我建议研究和验证的目标架构，不代表这些能力已经被实现。

它最大的价值是：**不仅判断 world model 是否可靠，还判断造成不可靠的原因是否能够被当前学习方法消除。**

这才有机会让你的工作从三个现有研究方向的组合，升级成一个新的科学原理。

## 四、我建议补充的真正核心理论

如果想冲 Oral，甚至 Best Paper，我认为至少需要从以下三个方向产生新的、可验证的理论结果。

### Theory 1：把 decision-transfer error 分解为真正不同的可修复机制

给定历史 $h$、目标 $g$ 和候选动作 $U_i,U_j$，定义：

$$

\Delta^*_{ij}=J^*(U_j)-J^*(U_i)

$$

这里 $J^*$ 是真实干预条件下的期望任务代价。

再定义：

- $\Delta^{\mathrm{obs}}_{ij}$：使用真实执行后继 RGB latent、通过当前任务几何得到的代价差异；
- $\widehat\Delta_{ij}$：使用模型预测后继 latent、通过同一任务几何得到的代价差异。

则有一个精确分解：

$$

\boxed{
\Delta^*_{ij}-\widehat\Delta_{ij}
=
\underbrace{
\Delta^*_{ij}-\Delta^{\mathrm{obs}}_{ij}
}_{\text{Realized task-geometry gap}}
+
\underbrace{
\Delta^{\mathrm{obs}}_{ij}-\widehat\Delta_{ij}
}_{\text{Predictive-dynamics gap}}
}

$$

随机环境下，三个量必须统一使用相应的条件期望，否则不能把单次执行结果直接视为期望代价。

这个分解本身只是代数恒等式，不是新定理。

但是，它让 GDD-WM 的研究问题发生了实质性变化。

| 主要观测现象 | 可能的失败源 | 合理研究动作 |
|---|---|---|
| 真实后继 latent 也无法正确排序 | 信息丢失或任务几何错误 | 检验表示混叠；尝试受限几何学习 |
| 真实后继排序正确，预测后继错误 | Predictor transfer 失败 | Dynamics adaptation |
| 同一 latent 对应相反真实动作效果 | Representation aliasing | 增加观测、历史、主动探测或拒绝 |
| 后继预测和代价排序正确，闭环仍失败 | Planner reachability | 修改候选、horizon、subgoal |

需要注意：第一种情况不能仅凭这个分解区分“latent 缺少信息”和“metric 不正确”。这必须由额外的 privileged-state 对照及可表示性实验判断。

**真正的创新应该是把这种误差归因变成一个只利用部署前合法数据、能够预测哪种修复会成功的机制。**

而不是事后观察结果，再给失败贴标签。

### Theory 2：提出固定视觉表示下的 Decision-Repairability Lower Bound

设两个不同的视觉历史满足：

$$

\Phi(h_1)=\Phi(h_2)=\eta

$$

但是某候选动作对的真实代价差异具有相反符号：

$$

\Delta^*(h_1)=+\gamma,\qquad
\Delta^*(h_2)=-\gamma

$$

那么任何只读取当前 $\eta$ 的确定性任务对比预测器，都必须输出相同的 $\widehat\Delta(\eta)$。

于是：

$$

\boxed{
\max_{k\in\{1,2\}}
|\widehat\Delta(\eta)-\Delta^*(h_k)|
\geq\gamma
}

$$

对于需要覆盖这两种状态的系统，固定表示下存在不可消除的 uniform contrast-error 下界。

这可以推广为基于同一表示等价类的 decision ambiguity：

$$

\mathcal A^\Phi_H(\eta)
=
\sup_{\substack{h_1,h_2\\
\Phi(h_1)=\Phi(h_2)=\eta}}
\max_{i,j}
|\Delta^*_{ij,H}(h_1)-\Delta^*_{ij,H}(h_2)|

$$

在声明的状态和候选集合上，任何只依赖 $\eta$ 的确定性预测器，其最坏 pairwise contrast error 至少为：

$$

\boxed{
\epsilon_{\mathrm{unavoidable}}
\geq \frac12\mathcal A^\Phi_H(\eta)
}

$$

这能形式化说明：

**有些决策错误不是 metric 没学好，而是当前输入的信息不允许任何下游 metric 同时做对。**

但我还要提出更严格的要求。

这个下界单独看仍然比较简单。类似的状态混叠、表示充分性与动作激励限制，在控制与 representation-identifiability 文献中已有理论基础。例如，<Entity category="other" value="On the Identifiability of Controlled World Models" disambig="2026 controlled latent dynamics identifiability theory paper"/> 已研究条件动作激励与受控动力学可辨识性之间的关系。<Cite ref="turn535549academia36"/>

要形成真正的新贡献，你还需要：

1. 在有限干预数据下，得到这个 ambiguity 的可检验下界或置信界。
2. 证明某些额外历史或动作探测能够降低与决策有关的 ambiguity。
3. 在给定函数类下，建立可修复情形的正向误差界。
4. 让这个理论判断能预测 GDD-WM 训练或 HTSC 放行的实际效果。

如果 GDD-WM 允许修改 encoder，必须重新定义新的 $\Phi$；旧表示下的不可修复性不能直接扩展为整个模型类的不可修复性。

还必须明确一个无法绕开的限制：**在完全缺少动作支持、又没有额外结构假设时，不可能从现有观测数据分布无关地确定未执行动作的真实后果。**

因此，可修复性边界必须同时声明 representation、action support、model class、task 和 horizon。

### Theory 3：证明哪些执行偏好能被 GDD-WM 的内部几何表达

假设真实执行确认 $U_i$ 优于 $U_j$，冻结模型给出的预测终点相对目标 latent 为：

$$

x_i=\widehat z_H(U_i)-z_g

$$

如果使用 PSD metric：

$$

c_M(U_i)=x_i^\top Mx_i,\qquad M\succeq0

$$

那么几何偏好约束为：

$$

\operatorname{Tr}
\left[
M(x_jx_j^\top-x_ix_i^\top)
\right]\geq m_{ij}

$$

可以要求：

$$

M\succeq0,\qquad\operatorname{Tr}(M)=1

$$

并通过半定可行性问题检验所有可信偏好是否可由一个固定 metric 表达。

这个实验特别值得做，因为它可以在大规模 GDD-WM 训练之前回答：

**当前预测表示是否存在一个能够实现真实动作偏好的简单几何？**

还需要更进一步，检验候选集合之间的偏好一致性。

如果完全相同的历史、目标和两个动作，在仅改变其他无关候选的情况下，被 teacher 反复给出相反的偏好，那么不存在一个候选集合无关的标量代价可以同时重现这些排序。

这对你的 GDD-WM 尤其重要，因为它试图把关系式 teacher 压缩为 candidate-independent geometry。

我会希望最终得到三种明确判断：

<box gap={2}>
  <row align="start" gap={2}>
    <icon name="check-circle-2" color="success"/>
    <box flex="1" gap={1}>
      **Metric-repairable**

      当前真实或预测表示包含足够信息，可靠偏好可以由声明的任务 metric 类表达，允许进行受约束的 geometry distillation。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={2}>
    <icon name="git-branch" color="secondary"/>
    <box flex="1" gap={1}>
      **Dynamics-repairable**

      Realized geometry 能实现正确偏好，但 predicted geometry 无法实现，需要检查 predictor 的动作响应和多步保持。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={2}>
    <icon name="shield-alert" color="secondary"/>
    <box flex="1" gap={1}>
      **Not reliably repairable with current evidence**

      当前视觉历史、动作支持或函数类不足，现有数据无法支持可靠修复，需要补充干预、观测或拒绝。
    </box>
  </row>
</box>

这里的关键不是 SDP 本身，而是**这些预训练判断是否真的能够预测后续修复收益**。

如果可以，你就有机会形成一个新的理论—算法贡献：

> A finite-intervention criterion for deciding when visual world-model decision errors are learnable, repairable, or uncertifiable.

我认为这比再构造一个更复杂的 latent alignment loss 更具有 Oral 潜力。

---

## 五、必须完成的决定性实验

我建议不要继续平均分配研究资源，而是采用以下阶段门。任何阶段不通过，都不应自动进入下一个更复杂的模型训练阶段。

| 阶段 | 必须完成的实验 | 通过标准 | 失败时的动作 |
|---|---|---|---|
| P0：Truth and Oracle | 真实候选执行、oracle ceiling、Bank-only 和 No-dynamics 对照 | 存在明显的 world-model action-selection gap | 先修 benchmark 或 planner |
| P1：Beyond ATM | 匹配 MSE、ATM、support 后检验新诊断 | 留出 learned models 上有增量风险预测力 | 弱化几何诊断主张 |
| P2：Repairability | 预测几何与真实几何误差分解、可表示性实验 | 修复前可预测哪些类型会被 GDD 改善 | 不把普通 distillation 包装为选择性修复 |
| P3：Execution-grounded GDD | 独立真实偏好、student distillation、prediction preservation | 未见候选上真实 regret 降低 | 检查表示损失、teacher 或动力学 |
| P4：HTSC calibration | 冻结 student 后独立校准与风险—覆盖率评估 | 有效且非空洞的上界 | 不部署 certificate |
| P5：RGB closed-loop | Full method 与强基线共享候选、动作和计算预算 | 联合方法有实际闭环收益 | 分析 planner shift / fallback |
| P6：External validation | PushT、独立 backbone、未见环境 | 判别机制具有声明范围内的泛化能力 | 限定论文适用边界 |

### 实验设计最重要的三个细节

**第一，所有主要方法必须共享真正的反事实动作数据。**

同一 root、同一目标、同一物理 horizon 下执行不同动作，而不是仅在已录制轨迹中比较不同预测时域。

模拟器中的 privileged state 可以用于重置、真实成本标注和 oracle 评估，但不能进入 RGB-only planner 的动作选择。

**第二，离线 random-bank 结果和 CEM elite 结果必须分开。**

一个方法可以在全部候选上有很好的 Spearman correlation，却在最终两个 elite candidates 之间选错动作。

这点在 AD-WM 对规划器局部候选后悔值的分析中已经有所体现。<Cite ref="turn185936view2"/>

你的主要排序指标应同时包括：

$$

\text{Global Contrast Error}

$$

$$

\text{Margin-conditioned Ranking Inversion}

$$

$$

\text{CEM-Elite Regret}

$$

以及最终的真实闭环任务收益。

**第三，GDD-WM 训练数据和 HTSC calibration 必须完全独立。**

不能在真实执行偏好上训练 student，然后继续使用相同 roots 对 student 进行风险认证。

我建议至少划分：

- 方法开发及 world-model training 数据；
- execution preference / GDD training 数据；
- student 冻结后的 HTSC calibration roots；
- 完全未接触的 offline outcome roots；
- 最终新生成的 closed-loop episodes。

如果学生模型改变了 encoder、predictor 或任务 metric，必须重新校准风险界。

### 三张主图应该讲清什么？

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="chart-scatter" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Figure 1 — Beyond ATM**

      在 MSE、ATM 与 support 匹配的真实 learned models 中，展示不同 decision-transfer ambiguity 与真实 regret，并验证新诊断在留出模型上的预测增益。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="brain-circuit" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Figure 2 — The Repairability Frontier**

      将候选状态按预先预测的错误机制分组，展示 GDD metric repair、predictor adaptation 和额外历史/动作探测分别在什么条件下有效。重点验证路由选择是否优于无差别训练。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="chart-no-axes-combined" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Figure 3 — Risk–Coverage–Utility**

      在相同放行率、动作预算和规划时间下比较 Frozen LeWM、GDD-only、HTSC-only、D-JEPA + standard calibration 与 Full method，报告实际 regret、success、false release 和 fallback 成本。
    </box>
  </row>
</box>

第三张应当是整篇论文最重要的系统结果。

如果 GDD-WM 确实修复了可修复区域，那么经过重新校准的 HTSC，应该能够在相同风险约束下放行更多有用动作。

如果 HTSC 只把所有动作拒绝掉，它没有证明控制价值。

---

## 六、到底怎样才有 Oral / Best Paper 竞争力？

我会把研究水平分成三个阶段，而不是用一两个显著性数字衡量。

| 层级 | 我作为审稿人需要看到什么 |
|---|---|
| 扎实主会论文 | 独立 RGB-only 闭环收益，严格公平的强基线，有效校准，完整方法消融和失败边界 |
| Oral 竞争力 | 一个新的可辨识/可修复性机制，能够提前预测哪些 world-model 决策错误可以被修复；理论与真实跨模型实验一致，明显优于简单模块组合 |
| Best Paper 潜力 | 建立具有广泛解释力的 decision-repairability 定律或边界，并给出有限数据下可计算、非空洞的判别与干预方法，在多个不同物理系统中经受独立验证 |

我尤其希望看到一个具有“预测性”的科学结果：

> 在不知道最终规划结果的情况下，我们的方法能够判断某个 learned world model 是应该进行几何修复、动力学修复、主动探测，还是拒绝；不同策略的实际实验结果与这个判断一致。

如果这个命题被大规模独立验证，贡献将不再只是某个模型的性能改进，而是有机会改变 world-model 开发流程。

例如，未来研究者训练出一个 world model 后，不再只报告 latent MSE、inverse accuracy、success rate，而可以系统性地回答：

**模型在哪些任务与动作区域具备决策能力？哪些失败可用有限新数据恢复？哪些能力必须增加观察或改变表示才能获得？**

这是我认为最接近 Best Paper 价值的方向。

但要达到这个水平，单个简单下界、一个 SDP、一个 conformal 阈值和若干成功率提升还不够。需要证明这些部件共同产生了此前无法实现的判别能力。

---

## 七、我建议你立刻执行的四项工作

这四项的优先级高于重新训练更大的 world model。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <text weight="medium" tabularNums>01</text>
    </box>
    <box flex="1" gap={1}>
      **修复 HTSC calibration，建立严格独立的风险验证**

      修改 `decision_benchmark/task_contrast.py` 的有限样本分位数，加入样本不足的 fail-closed 逻辑、全候选 regret bound、coverage simulation 和 selection-shift tests。修复后在新 roots 上重新校准，而不是继续复用旧结果。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <text weight="medium" tabularNums>02</text>
    </box>
    <box flex="1" gap={1}>
      **对 PointMaze +0.1090 正结果做 kill tests**

      Bank-only、No-dynamics、Shuffled-successor、New-bank 四组消融必须完成。与此同时计算每个 root 的 oracle candidate ceiling，并确定真实模型选择误差究竟占了多少。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <text weight="medium" tabularNums>03</text>
    </box>
    <box flex="1" gap={1}>
      **完成第一个可信的 GDD repairability pilot**

      先用合法的 execution-only preference 训练固定预测几何，比较 PSD metric、普通 MLP scorer、naive preference learning 与具有可修复性筛选的 GDD。只有真实 held-out regret 改善并保持多步预测时，才继续复杂 student training。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <text weight="medium" tabularNums>04</text>
    </box>
    <box flex="1" gap={1}>
      **在 PointMaze 上跑通最小完整 RGB-only MPC**

      先只比较原始 frozen LeWM、GDD-only、HTSC-only、Full method 和 perfect-dynamics oracle。确保 RGB 观测、动作执行、真实 successor、重规划与失败日志完整闭合，再增加其他环境和大规模模型矩阵。
    </box>
  </row>
</box>

工程上，我不建议另建一套总入口。继续使用已有的研究运行与证据治理结构，把新的研究内容实现成独立 stage，并记录固定协议、代码版本、输入指纹、训练来源、校准来源和测试产物。

另外，GPU smoke、CPU protocol test、真实模型训练与完整闭环结果应继续分开记录。现有仓库在这方面做得相对扎实，不应该为了提高论文完成度而放松。

---

## 八、我会怎样重写你的一篇论文的三个 Contribution？

不再强调三个缩写本身，而强调三个逐层增强、共同回答一个问题的科学发现。

| Contribution | 新的表述 | 最低必须成立的证据 |
|---|---|---|
| **C1 · Characterize** | Task-conditioned Decision Identifiability and Repairability | 新的机制判别能在 ATM/MSE 匹配之后预测真实决策错误及其可修复性 |
| **C2 · Repair** | Execution-Grounded Selective Predictive Geometry | 根据预先判别选择性修复，在未见候选上降低 regret，并保持预测质量 |
| **C3 · Certify** | Horizon-Conditioned Calibrated Selective Planning | 修复后的 student 经独立重新校准，在 RGB-only 闭环中改善风险—放行—效用权衡 |

我推荐的最终论文标题是：

<title size="lg">Know, Repair, or Abstain: Decision-Transfer Sufficiency in Visual World Models</title>

也可以采用理论导向更强的：

*The Decision-Repairability Frontier of Visual World Models*

这两个名称都更明确地表达了你相对于 ATM、D-JEPA 和普通风险校准的潜在差异。

你的论文不应该只是说明“world model 会失败”，也不应该只是“我们训练了一个更好的 world model”。

而应该证明：

**我们能够区分为什么失败、哪些失败能够用当前信息和有限干预修复、哪些不能，以及怎样利用这个判断实现更可靠的规划。**

## 九、最终模拟审稿决定

<box border radius="lg" padding={3} gap={2}>
  <row justify="between" align="center">
    **Recommendation**
    <text weight="medium" color="danger">Reject — Current Version</text>
  </row>
  <divider color="subtle"/>
  **Strengths**

  科学问题重要；准确区分预测、动作辨识与决策充分性；保留理论反例与负结果；工程与实验审计较规范；PointMaze 已有可继续验证的离线正信号；PushT 提供了难得的失败机制证据。

  **Major Weaknesses**

  创新性与 ATM、D-JEPA、DA-LeWM、AD-WM 存在较大重叠；最强 PointMaze 结果仍可能受 action-bank shortcut 解释；HTSC 有待修正有限样本校准；GDD-WM 缺少正式独立修复结果；真实视觉闭环尚未证明三模块互补；跨环境和统计独立性不足。

  **What would change my decision?**

  一套新鲜、可重置的 counterfactual decision benchmark；一个能够提前预测修复成败的新判别准则；一个经独立验证的 GDD 修复方法；一个经过有效校准、在真实 RGB 闭环中产生非空洞收益的 selective planner。
</box>

### 最终研究取舍

我建议你暂时不要继续追求“普适 world model”“通用 commutator certificate”或“所有 alignment metrics 都能预测规划成功”这些过强主张。

把主要精力投入一个更深、更可证伪的问题：

> **给定一个已经训练好的视觉 world model，我们能否在不预先知道真实测试结果的情况下，判断它当前缺少的是可观测信息、可辨识动作效应、正确预测动力学、正确决策几何，还是有效规划范围；并利用这一判断决定修复、探测或拒绝？**

**这是我认为 LeWorldModel 最值得追求的 Oral / Best Paper 方向。**

当前最有希望让论文质量发生质变的，不是增加一个更强的 memory module 或 spectral indicator，而是完成一个实验：**在 PointMaze 上，先证明哪些错误可修复，再让这种预测指导 GDD-WM，最后由重新校准的 HTSC 在全新 RGB 闭环中验证它。**

如果这个结果成立，而且比 ATM + D-JEPA + 普通校准的简单组合更好，你的研究才真正拥有了一个难以被替代的核心贡献。

-----

> 出现一个和我们研究重叠度这么高的研究怎么办？
> 一辆车驶向路口，前方有足够的通行空间，侧前方的车辆却正在靠近。规划器可以给出多条候选轨迹：保持速度、提前减速，或者稍微改变行驶方向。难点不在于生成一条“像人开出来的”轨迹，而在于判断哪条轨迹在接下来的几秒里仍然安全、合理。
>
> 这也是自动驾驶引入世界模型的初衷：不能只看当前一帧，还要利用历史信息判断交通参与者的运动趋势，并推断场景接下来可能怎样变化。
>
> 但“预测未来”本身并不是终点。如果模型花了大量容量重建建筑纹理、路面细节和光照变化，却没有保留会改变轨迹选择的动态线索，这样的未来即使看起来很真实，对规划也未必有用。
>
> PlanWAM真正想解决的，正是这一层问题：**世界模型应该记住怎样的历史、学习怎样的未来，才能真正帮助车辆规划？**
>
> 作者给出的答案可以概括成三步：
>
> 1. **由未来约束历史信息的保留**：历史表示不追求完整保存过去，而要留下能够支持未来推理和轨迹规划的信息。
> 2. **由规划任务定义有用的未来**：未来表示不由图像重建或通用视觉特征单独决定，而由轨迹生成和轨迹评分共同塑造。
> 3. **用预测的未来表征改善规划**：训练时借助真实未来建立目标，部署时只看历史预测这份未来表征，并让它参与轨迹生成与选择。
>
> 这三步不是三个互不相关的模块，而是一条首尾相接的学习链路：**未来告诉模型历史中什么值得记，规划告诉模型未来中什么值得学，模型再把预测到的未来交还给规划。**
>
> 论文在 NAVSIM-v1 上取得93.8 PDMS，在 NAVSIM-v2 上取得90.9 EPDMS，并在零样本闭环 HUGSIM 测试中达到38.7 HD-Score。更值得关注的是，消融实验基本沿着上面这条链路给出了证据。
>
> 本文依据论文 arXiv v1 全文与附录展开。文中原始图片和表格均来自论文，文末附有论文信息与开源状态。
>
> ## 01｜问题不只是“如何预测未来”，而是“什么未来值得预测”
>
> 现有自动驾驶世界动作模型，大体经历了三种思路。
>
> 第一类显式重建未来图像、视频或稠密场景状态。它能提供直观的未来画面，却需要花费大量计算去建模与规划关系不大的视觉细节。
>
> 第二类把预测目标放到潜在特征空间，例如视觉特征或 BEV 表示。这样省去了像素级生成，但未来表示仍然往往由通用预测目标预先定义。
>
> 第三类把预测到的未来特征直接送入规划器，让未来信息参与轨迹决策。这向前走了一步，但仍没有完全回答：**进入规划器的未来表征，是否真的保留了影响决策的关键信息？**
>
> [图片](https://mmbiz.qpic.cn/mmbiz_jpg/pic4YdoDXFiacrzOEib5MjB4T1CKTcxeaFlKiaOIMOk3KicnDXL5fV7yqHn5gYGPkZzgNIxx4Khbt1MPjYt6ZaC2J4fclwPPToP6LcAbGV3hORdU/640?wx_fmt=webp\&from=appmsg)imageimageimage*图1｜从显式未来重建、潜在特征预测，到未来条件规划，再到 PlanWAM。来源：论文 Figure 1，第2页。*
>
> PlanWAM把问题重新表述为“面向规划的预测表征学习”。它不要求历史表示完整复述过去，也不要求未来表示完整重建场景，而是让整条链路围绕轨迹规划筛选信息。
>
> 论文将未来条件规划写成：
>
> 插图
>
> 其中，v 表示多视角图像，c 表示自车状态和导航信息，z 表示未来潜在表征，tau 表示规划轨迹。这个分解说明：模型先根据历史预测未来表征，再让规划器同时利用历史和未来。
>
> PlanWAM的关键不在这个概率分解本身，而在于同时追问三个问题：历史表示 h 应该保留什么，未来表示 z 应该编码什么，以及预测出的 z 应该怎样进入规划。
>
> [图片](https://mmbiz.qpic.cn/mmbiz_jpg/pic4YdoDXFiafFIoJdEn0Xhc5V14vA6nqBmTKOWiaJw3v86SAriaJXAMTgLOMGPeGJ3ytB1UcnZBUMrGC3G9JXmJnvKyHItaCVb7BJEHhPAUonU/640?wx_fmt=webp\&from=appmsg)imageimageimage*图2｜上方蓝色先验分支负责训练和推理，下方橙色后验分支只能在训练时读取真实未来。来源：论文 Figure 2，第4页。*
>
> 沿着图2从左往右读，正好对应本文的三条主线。
>
> ## 02｜第一步：由未来约束历史信息的保留
>
> 多帧、多视角历史图像包含大量冗余。四个时刻、四个摄像头，如果每个摄像头和时刻提取16个场景寄存器，未经压缩的历史流就有：
>
> 插图
>
> 个寄存器。全部保留不仅成本高，也可能让与未来推理无关的细节占据表示容量。
>
> PlanWAM采用 Temporal Register Pyramid，也就是时间寄存器金字塔，对历史信息进行“近重远轻”的压缩。从最早到当前时刻，每个摄像头分别分配4、4、8、16个查询，越接近当前，保留的容量越大。
>
> [图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/pic4YdoDXFiafoVUwASH6hjqOff6gicKFcz8ia6fxDlv91ic32rqGibMF7g5wdewuGEHNRd1k9WPf3SEKWyPVnCbiaibam5ibT0nCbK7DTwEQggAdgwk/640?wx_fmt=webp\&from=appmsg)imageimageimage*图3｜查询融合时间、摄像头、压缩层级、自车状态和当前场景信息，再通过 Q-Former 压缩历史场景寄存器。来源：论文 Figure 3，第5页。*
>
> 第 l 个时间层级的压缩过程可写为：
>
> 插图
>
> 所有层级合并后，历史表示被压缩为：
>
> 插图
>
> 个 token。
>
> 如果只看到这里，很容易把这一部分理解成普通的时间压缩：近期信息更重要，所以多留一些；远期历史更冗余，所以少留一些。
>
> 但这还不是作者所说“由未来约束历史信息保留”的完整含义。
>
> PlanWAM没有给历史表示附加一个“还原历史画面”的重建目标。压缩后的历史 h 会继续进入未来预测和轨迹规划，并同时接受未来潜在表征蒸馏损失与规划任务损失的反向约束。也就是说，历史编码器最终学到的，不是怎样完整复述过去，而是：
>
> - 哪些历史动态有助于推断接下来的场景演化；
> - 哪些历史线索会影响候选轨迹的生成与判断；
> - 在有限 token 预算下，哪些内容值得被保留下来。
>
> 一个直观例子是：连续几帧中，旁车的相对位置变化可能暴露它正在加速并线；路边建筑的纹理变化，对轨迹选择却未必有直接价值。未来预测目标会迫使历史表示保留“运动趋势”，规划目标则进一步检验这种趋势是否会改变驾驶决策。
>
> 因此，第一条主线中的“未来约束历史”并不是让模型在推理时偷看未来。真实未来只在训练阶段提供学习信号，帮助历史编码器形成面向未来推理的选择性记忆。
>
> 论文表7给出了结构层面的证据。在相同的128个历史 token 预算下：
>
> | **时间查询分配**    | **Token数** | **延迟** | **PDMS** |
> | ------------- | ---------- | ------ | -------- |
> | [8, 8, 8, 8]  | 128        | 343 ms | 93.4     |
> | [16, 8, 4, 4] | 128        | 346 ms | 92.9     |
> | [4, 4, 8, 16] | 128        | 342 ms | 93.8     |
>
> *数据整理自论文 Table 7；延迟为单张 NVIDIA A800、batch size 1、FP32 条件下的测量值。*
>
> 平均分配不如近重远轻，反向分配下降更明显。即使把历史容量扩大到256个 token，PDMS 也只有93.6。这说明，在这套模型和数据上，**如何筛选历史信息，比单纯增加历史 token 更重要。**
>
> ## 03｜第二步：由规划任务定义有用的未来
>
> 有了紧凑的历史表示，接下来的问题是：什么样的未来表征才算“有用”？
>
> PlanWAM在训练阶段引入一个具有特权信息的后验分支。它能读取未来1、2、3、4秒的真实多视角图像，并从每个未来时刻、每个摄像头提取场景寄存器。这些未来寄存器带上时间嵌入后，直接作为世界模型的查询；历史表示 h 则提供键和值：
>
> 插图
>
> 这样得到的 z 并不是最终目标。真正决定 z 学成什么样的，是后面的轨迹规划器和轨迹评分器。
>
> 历史表示与后验未来表示被拼接为规划上下文：
>
> 插图
>
> 它同时服务于候选轨迹生成和候选轨迹评分。核心监督为：
>
> 插图
>
> 轨迹生成损失要求未来表征支持模型提出合理的驾驶方案，轨迹评分损失要求它帮助模型识别哪条候选轨迹更安全、更合规、更有效率。两类损失反向传播到 z，有限的未来潜在空间便会优先保留能够改变规划结果的信息。
>
> 这就是“规划塑造未来”的真正含义：不是先定义一份通用未来特征，再让规划器想办法使用；而是让规划任务参与决定未来表征本身。
>
> 需要特别区分两件事。PlanWAM仍然使用 DINOv2 提取视觉特征，但 DINOv2 只是视觉编码器；未来潜在空间并不是简单复刻一份固定 DINO 特征。它通过规划器和评分器接受任务监督，学习的是面向轨迹决策的未来表征。
>
> 论文表5直接比较了不同未来目标：
>
> [图片](https://mmbiz.qpic.cn/mmbiz_jpg/pic4YdoDXFiafNmL10LOSfYiaClwL4p4ZJDO5SaSww67A3wjmo49uaHP8UtUe0XTYYMQ7orumq6lk1N4RNCqKiaPJicf5bRD16J6G5HfeN0oiax4o/640?wx_fmt=webp\&from=appmsg)imageimageimage*表1｜左侧比较未来表征由什么目标塑造，右侧比较预测的未来表征被用于哪里。来源：论文 Tables 5-6，第9页。*
>
> 不使用未来表征时，PDMS 为93.4；用未来场景寄存器重建作为通用特征监督，得到93.5；让轨迹生成和轨迹评分共同塑造未来潜在表征，达到93.8。
>
> 0.3分的差距看起来不大，但它回答了论文最核心的问题：**预测未来本身只有有限收益，预测“对规划有用的未来”才是关键。**
>
> 当然，“规划塑造的未来”仍然是潜在特征，不能直接解释为每个 token 对应一辆车、一次碰撞或一种驾驶意图。论文验证的是它对规划任务的整体价值，而不是逐 token 的语义可解释性。
>
> ## 04｜第三步：用预测的未来表征改善规划
>
> 后验分支虽然能学到高质量的未来目标，却依赖真实未来观测，部署时无法使用。PlanWAM因此再训练一个只依赖历史的先验分支，把训练阶段的“事后信息”转换为推理阶段的“前瞻能力”。
>
> 先验分支使用可学习的未来查询，并添加未来时间、摄像头和寄存器槽位嵌入。它的查询顺序与后验未来 token 一一对应，只从历史表示中预测未来：
>
> 插图
>
> 作者把这个过程称为 Hindsight-to-Foresight Distillation，即“从后见到前瞻”的蒸馏。先验预测与冻结的后验目标通过下式对齐：
>
> 插图
>
> sg 表示停止梯度，保证学生去追随教师，而不是让教师目标跟着学生漂移。论文使用256个未来 token，每个 token 的特征维度为256；蒸馏损失即使在混合精度训练中也以 FP32 计算。
>
> 训练完成后，真实未来图像、后验分支和相关数据加载全部移除。车辆只需读取历史观测，先预测规划导向的未来表征，再把它与历史表示拼接为规划上下文。
>
> [图片](https://mmbiz.qpic.cn/mmbiz_jpg/pic4YdoDXFiaerBG6QSzNdLIHQJBJbhtxSg4qkrzIJ97rMicqB6brlfcqrnpkbDloPibEREBex2pKyEJ8AiaFwtT8yMYkJtPHETpjoRvNtnrpbnM/640?wx_fmt=webp\&from=appmsg)imageimageimage*图4｜推理阶段的三个 Transformer：先验世界模型预测未来表征，规划器生成候选轨迹，评分器完成轨迹选择。来源：论文 Figure 6，第14页。*
>
> PlanWAM的规划器一次生成64条候选轨迹，每条轨迹包含8个点，覆盖未来4秒。评分器再从无责任碰撞、可行驶区域合规、碰撞时间、自车推进程度、行驶方向合规和舒适度等方面评估候选方案。
>
> 预测的未来表征同时进入生成器和评分器，这一点很重要。它既可以帮助模型提出更符合场景演化的候选轨迹，也可以帮助模型判断已有候选中哪一条更值得执行。
>
> 论文表6把这两个作用拆开验证：
>
> | **未来表征的使用方式** | **NAVSIM-v1 PDMS** |
> | ------------- | ------------------ |
> | 仅作为辅助监督       | 93.5               |
> | 只用于轨迹生成       | 93.5               |
> | 只用于轨迹选择       | 93.7               |
> | 同时用于生成与选择     | 93.8               |
>
> 只给轨迹生成器使用，分数与辅助监督相同；只给评分器使用，已经达到93.7；两者共同使用，得到完整模型的93.8。
>
> 这组结果揭示了预测未来的一项具体价值：在当前架构下，它对“选对轨迹”的帮助，比单独改善“生成轨迹”更加明显。
>
> 这并不意味着轨迹生成不重要。如果候选集合中没有可行方案，再准确的评分器也无从选择。它说明的是，PlanWAM的主要增益更多来自对候选轨迹未来后果的判断。
>
> ## 05｜三条主线如何在两阶段训练中闭环
>
> PlanWAM采用两阶段训练。
>
> 第一阶段只训练能看到真实未来的后验分支，以及相应的规划器和评分器。规划任务在这一阶段定义什么样的未来表征有用。
>
> 第二阶段冻结后验教师，用其结构兼容参数初始化先验分支，再让只依赖历史的先验分支学习预测这份未来表征。总损失为：
>
> 插图
>
> 其中，任务损失不仅训练世界模型，也继续训练历史编码器、规划器、评分器和 LoRA 参数。原因是：第一阶段的下游模块接收真实未来塑造的后验特征，第二阶段接收历史预测出来的先验特征。两者即使相似，也并不完全相同，下游模块需要适应部署时真正会看到的输入。
>
> [图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/pic4YdoDXFiafZlIDWlsH4feicgJia7W51ereicsDcxDt9E6t7CCVafubFic2Mx2TZtGJRIa2xQbldXuTxT2fRcCgiaN8oDV2nTxq0gtMMfRyQPqjI/640?wx_fmt=webp\&from=appmsg)imageimageimage*表2｜只追求未来特征相似，并不能得到最好的规划性能；下游模块必须共同适应预测表征。来源：论文 Table 17，第20页。*
>
> 只训练先验世界模型、只使用潜在特征蒸馏时，未来特征相似度达到0.96，PDMS 却只有93.0。加入规划损失并逐步开放评分器、规划器与其余模块训练，PDMS 才依次升到93.4、93.5和93.8。
>
> 附录表16也显示，把蒸馏权重从0.05增加到0.10，特征相似度从0.93升到0.95，PDMS 反而从93.8降到93.7。
>
> 这些实验把三条主线重新连了起来：历史不是为了预测得像而压缩，未来也不是为了与教师更相似而学习；最终标准仍然是，预测表征能否改善规划。
>
> ## 06｜主结果：从开环规划到零样本闭环
>
> ### NAVSIM-v1：93.8 PDMS
>
> PlanWAM在 NAVSIM-v1 navtest 上取得93.8 PDMS。论文对比中，DrivoR 为93.1，DriveFuture 为90.7，Human Agent 为94.8。PlanWAM相比 DrivoR 提高0.7分，相比此前最强的世界模型规划器 DriveFuture 提高3.1分，但仍低于该协议下的人类轨迹基线。
>
> PDMS按场景计算：
>
> 插图
>
> NC 和 DAC 是乘法安全门控，TTC、推进程度 EP 与舒适度 C 构成加权质量项。由于各子项先逐场景计算再平均，不能把表中的子项均值直接代入公式复现最终 PDMS。
>
> ### NAVSIM-v2：90.9 EPDMS
>
> [图片](https://mmbiz.qpic.cn/mmbiz_jpg/pic4YdoDXFiafkKicd61zMwSYibkVASgE7jL89QeKYhKByZTLtyeXPyClvUyicVG5o3G0PP2UiaSKdYr4M7lw9bckNcympoQdYDYgqv7aZCvKeQVo/640?wx_fmt=webp\&from=appmsg)imageimageimage*表3｜PlanWAM在 NAVSIM-v2 的综合分数与各子项。来源：论文 Table 1，第7页。*
>
> PlanWAM在 NAVSIM-v2 上取得90.9 EPDMS，高于 DriveFuture 的89.9和论文表中的 Human Agent 90.3。其 NC 为99.0，TTC 为98.5，EP 为90.8。
>
> 但完整表格也显示，PlanWAM的扩展舒适度 EC 只有70.5，明显低于 Human Agent 的90.1和 Latent-WAM 的87.3。90.9说明它在这套综合协议下得分更高，不代表所有子项占优，更不能直接推导真实道路能力超过人类。
>
> ### HUGSIM：38.7 HD-Score，检验零样本闭环迁移
>
> NAVSIM采用非反应式候选轨迹仿真，其他交通参与者主要沿记录轨迹运动。HUGSIM则进行闭环测试：自车行为会改变下一步观测和仿真状态，更容易暴露持续规划中的误差累积。
>
> 论文将 NAVSIM 训练的 PlanWAM 直接部署到 HUGSIM，没有进行 HUGSIM 专用微调。测试覆盖来自 nuScenes、Waymo、KITTI-360 和 PandaSet 的436个场景。
>
> [image](https://mmbiz.qpic.cn/mmbiz_jpg/pic4YdoDXFiadMzswT4OLlOzL36DNibENnYrKHn6jEE7gIFOrHWV7ZWVwdHkXVuvfhkMhpjickeUhdySQGE4ogV3Z320ega3QOZfS0zGT02pTHc/640?wx_fmt=webp\&from=appmsg)imageimageimage*表4｜HUGSIM零样本闭环结果。E、M、H、X 分别表示 Easy、Medium、Hard、Extreme。来源：论文 Table 3，第8页。*
>
> PlanWAM的平均路线完成度为53.7，HD-Score 为38.7；论文对比方法中的最佳结果分别是45.9和28.9。Hard 和 Extreme 场景的 HD-Score 分别为35.6和22.0。
>
> 这组结果支持模型具有一定的跨域闭环泛化能力，但 Extreme 场景22.0的得分也说明，高强度交互和对抗式交通参与者仍然很难处理。
>
> [图片](https://mmbiz.qpic.cn/mmbiz_jpg/pic4YdoDXFiafy5odaFSubFJ9cGYu7obz99rzwCiaJicGIHDZiaUWePJJB8gichTS1sfNUqe2M7297R0uduWZHAgZJ7GI7XrZ4khbQXKRlibnUzwUE/640?wx_fmt=webp\&from=appmsg)imageimageimage*图5｜左侧为 NAVSIM 中与 DrivoR 的轨迹对比，右侧为不同数据域中的 HUGSIM 闭环案例。来源：论文 Figure 4，第8页。*
>
> ## 07｜失败分析：未来表征主要减少了“选错轨迹”
>
> [图片](https://mmbiz.qpic.cn/sz_mmbiz_jpg/pic4YdoDXFiack8fN4FlbJwbCuAwZAAFI58IBSbEV5GOSMiaIoJWk2gWwbiaibAucniae3VYdSkQfxtReyLOVDiaqhWO89ziaRqibB7LT0oDE9RyIz4A/640?wx_fmt=webp\&from=appmsg)imageimageimage*图6｜完整模型与内部基础模型的失败类型及失败减少归因。来源：论文 Figure 5，第10页。*
>
> 作者统计 NAVSIM-v1 中 PDMS 为零的失败案例。加入完整 PlanWAM 后，失败总数从314降到237，减少77例。
>
> 进一步归因显示：轨迹选择错误减少87例，候选质量不足减少6例，候选塌缩却增加16例。三项相加，最终净减少77例。
>
> 按失败类型看，NC 相关失败下降62.3%，TTC 相关失败下降52.4%，DDC 相关失败下降66.7%；DAC 相关失败反而增加7.3%。
>
> 这与前面的使用位置消融相互印证：PlanWAM最明显的改进，是利用预测的未来表征提升候选轨迹评估和选择，而不是彻底解决候选生成、可行驶区域合规或多样性问题。论文中的百分比来自特定基准的失败统计，不能直接解释为真实道路事故率下降。

我会把 PlanWAM 当作真正的直接竞争工作来评估，而不是简单安慰你“方向不同”。先核对论文原文和发表时间，再逐项比较它与你的 ATM、GDD-WM、HTSC 三条贡献：哪些主张已经被抢先，哪些仍有独立的新意，以及怎样调整实验和论文定位。

我认真看了你提供的 PlanWAM 分析，也核对了论文原文。我的判断是：

**这确实是一篇需要高度重视的竞争论文。它与你的 LeWorldModel 研究在科学动机上高度重叠，但还没有覆盖你真正有机会做成 Oral / Best Paper 的核心问题。**

尤其重要的是，PlanWAM 已经不只是提出一个想法：它在视觉历史建模、任务驱动的未来表征学习、规划监督、候选轨迹选择和闭环测试之间，建立了相当完整的实证链条。

我确认 <Entity category="other" value="PlanWAM: Planning-Shaped Future Representations for End-to-End Autonomous Driving" disambig="Xu et al., arXiv 2610.11382, October 2026"/> 于 **2026 年 10 月 8 日**公开。其论文报告 NAVSIM-v1 的 93.8 PDMS、NAVSIM-v2 的 90.9 EPDMS，以及 HUGSIM 零样本闭环的 38.7 HD-Score。<Cite ref="turn446453view0"/>

因此，如果我们原本想把下面这些作为主要创新：

- 历史表示应该保留与规划相关的信息；
- 未来 latent 不应该只优化预测误差，而应该受规划目标塑造；
- 用任务偏好监督改善世界模型内部表征；
- 让预测的未来表示改善候选动作排序；

那么 PlanWAM 会明显削弱这些主张的新颖性。

但你的研究还有一个非常重要的不同点：

**PlanWAM 主要回答：怎样学习有利于规划的未来表征？**

**LeWorldModel 应该回答：这样的未来表征是否真的保留了正确的反事实动作后果？如果没有，哪些错误可以修复，哪些无法辨识，以及什么时候不应该信任它执行动作？**

这两个问题相关，却不是同一个问题。

我不建议你放弃 ATM + GDD-WM + HTSC 的统一论文。我的建议是：**立即收缩已经被覆盖的创新叙事，把主贡献转向可检验的 Decision Repairability 与可信决策边界，并用 PlanWAM 作为一个重要的反例研究对象和竞争基线。**

下面具体说明应该怎样调整，哪些地方需要做真正的研究突破。

## 一、最关键的发现：PlanWAM 预测的是共享未来信息，而不是显式的候选动作反事实未来

这是我核对论文方法部分后，发现的最值得你抓住的区别。

PlanWAM 的未来表示在部署时由历史预测：

$$

\boxed{\hat z_{t}^{+}=P_\theta(h_t)}

$$

它再使用共享的历史和未来表征生成、评价候选轨迹：

$$

s_i=G_{\mathrm{score}}(U_i,h_t,\hat z_t^+)

$$

这是论文第 3.1、3.4 节以及附录中实际写出的结构。其评分器确实会分别读取各条候选轨迹，不能说它无法区分动作。但其世界模型先验本身，并没有显式地针对每条候选动作生成不同的受干预未来。<Cite refs={["turn166382view1","turn586713view0","turn586713view5"]}/>

而 LeWorldModel 的关键研究对象是：

$$

\boxed{
\hat z_{t+H}(h_t,\operatorname{do}(U_i))
}

$$

进一步比较：

$$

\Delta_{ij}^{*}
=
J_H^*(h_t,U_j,g)-J_H^*(h_t,U_i,g)

$$

这里有两个层次的区别。

<box border radius="lg" padding={3} gap={3}>
  <row align="start" gap={3}>
    <box flex="1" gap={1}>
      <text color="secondary" weight="medium" size="xs">PLANWAM</text>
      <title size="md">Planning-shaped foresight</title>
      让历史预测对规划有用的未来信息，再让候选轨迹评分器利用这份信息进行选择。
    </box>
    <box background="surface-secondary" radius="md" padding={2}>
      <icon name="route" size="lg"/>
    </box>
  </row>
  <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
    **History → Shared future latent → Candidate scoring**
    <text color="secondary" size="xs">优化规划表现，但不直接输出动作反事实误差证书</text>
  </box>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box flex="1" gap={1}>
      <text color="secondary" weight="medium" size="xs">LEWORLDMODEL · 目标方法</text>
      <title size="md">Counterfactual decision sufficiency</title>
      分别检验候选动作真正造成的未来后果，并判断其差异是否可辨识、可修复和可认证。
    </box>
    <box background="surface-secondary" radius="md" padding={2}>
      <icon name="git-compare-arrows" size="lg"/>
    </box>
  </row>
  <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
    **History + Candidate actions → Intervention outcomes → Regret / Certificate**
    <text color="secondary" size="xs">先验证反事实动作排序，再决定修复或放行</text>
  </box>
</box>

这不意味着 PlanWAM 一定会在反事实动作上失败。它的评分器完全可能从历史和轨迹特征中学到有效的动作差异。

真正尚未由它的论文直接解决的是：

**当候选动作进入训练支持不足的区域、改变其他参与者的反应，或者遇到历史不能辨识的隐藏动力学状态时，如何知道评分器的判断仍然可信？**

这正是你能够建立独立科学贡献的位置。

### 把它放到论文开头的路口例子中

假设同一个路口，车辆 A 可以直行或减速。侧方车辆 B 可能主动让行，也可能抢行。

模型从历史推断出一份未来场景信息，并给出两个动作的分数。

但是，真实未来不一定只由当前历史决定：

$$

P(Y_{t+H}\mid h_t,\operatorname{do}(U_{\rm go}))
\neq
P(Y_{t+H}\mid h_t,\operatorname{do}(U_{\rm slow}))

$$

尤其当 B 会对 A 的速度作出反应时，这两个反事实未来可能存在很大差异。

如果训练数据主要来自一种驾驶行为，即使模型能够准确预测这类行为下的未来，也不意味着它可靠了解另一种动作的后果。

因此，我建议你的新核心命题是：

> **A planning-shaped future representation is not necessarily a counterfactually sufficient representation.**

中文：

**一个被规划目标优化过的未来表征，并不必然足以支持可靠的反事实动作选择。**

这句话应当成为研究假设，而不是现在就宣称已经证明的结论。

更关键的是：它不能仅用一个抽象反例证明。你需要在真实 learned world models 上展示，在规划分数、普通预测误差甚至动作可辨识性相近时，反事实排序错误仍然存在，而且你的方法能识别并处理它。

---

## 二、逐项比较：哪些创新已经危险，哪些值得保留？

| 研究维度 | PlanWAM 已提供 | LeWorldModel 应追求的差异 | 威胁 |
|---|---|---|---|
| 任务塑造未来 latent | 有，核心贡献 | 不再作为主要新意 | <text color="danger" weight="medium">高</text> |
| 历史信息筛选 | Temporal Register Pyramid | 历史对动作效应可辨识性的增量，而非简单时间压缩 | <text color="danger" weight="medium">高</text> |
| 规划偏好监督 | 轨迹生成与多指标评分 | 真实执行偏好、可表示性和修复成败预测 | <text color="danger" weight="medium">高</text> |
| 未来表征蒸馏 | Hindsight-to-Foresight | 可辨识性约束下的预测几何修复 | <text color="caution" weight="medium">中高</text> |
| 反事实 action effects | 有候选评分，但未显式提供逐动作动力学认证 | ATM + task-contrast intervention audit | <text color="success" weight="medium">较低</text> |
| 修复前判断可修复性 | 未作为论文主要方法 | Repairability criterion | <text color="success" weight="medium">较低</text> |
| 有限样本 regret 认证 | 未作为论文贡献 | HTSC 校准与联合候选 regret 控制 | <text color="success" weight="medium">较低</text> |
| 不可辨识时拒绝或探测 | 未提出对应的决策认证机制 | Permit / Probe / Abstain | <text color="success" weight="medium">较低</text> |
| 闭环规划实证 | HUGSIM 已有较强结果 | 必须在独立视觉闭环中建立实际互补收益 | <text color="danger" weight="medium">高</text> |

<caption>威胁等级是我的科研新颖性判断，不是对论文内容相似度的量化测量；“较低”也不代表尚无其他竞争工作。</caption>

这里有一个必须面对的事实：**PlanWAM 的实验完成度目前明显领先于你统一论文中 GDD-WM 和 HTSC 的部署级验证。**

你的仓库统一论文计划仍将 GDD-WM 的独立修复结果和 RGB-only MPC 列为尚未完成的证据门槛。因此真正的风险不只是“观点被抢”，而是竞争研究已经拿出了更完整的系统证据，而我们仍在建立自己的机制证据链。<Cite ref="turn683646view1"/>

另一方面，PlanWAM 报告的闭环提升不能自动视为有限样本的决策可靠性保证。它在 NAVSIM 上使用非反应式轨迹评估，在 HUGSIM 上测试闭环；这些是有价值的规划结果，但并不等价于对未执行候选动作的误差界或选择性风险保证。<Cite ref="turn287229view0"/>

## 三、但不能低估 PlanWAM：你的新论文必须再跨过三道关

### 关卡 A：PlanWAM 已经有候选级任务监督

这里要非常公正。

PlanWAM 不是简单地预测一份未来，然后拿欧氏距离随意给轨迹评分。

它对 64 条候选轨迹使用官方 proposal simulator 生成 NC、DAC、TTC、EP 等监督目标，并训练专门的评分器。也就是说，**它已经使用了候选级规划质量监督**，只是这些评分与规划目标是基于其相应仿真协议构造的。<Cite ref="turn287229view0"/>

所以你不能宣称：

“我们是第一次让真实任务质量监督 world model 的候选动作选择。”

这条新意已经不充分了。

你必须更具体地证明：即使存在候选级评分监督，模型仍然可能因为缺少反事实动作支持、历史状态混叠或动力学模式不可辨识而产生可靠性问题。

### 关卡 B：GDD-WM 的竞争者不止 PlanWAM

比 PlanWAM 更直接挑战你 GDD-WM 的，其实还有两篇论文：

<Entity category="other" value="D-JEPA: A Decision-Aligned Latent World Model" disambig="2026 execution-grounded decision preference and geometry research"/> 已经研究利用执行结果学习候选间的决策关系，并把关系信息实现到 JEPA-compatible future representations，使原生 latent-distance planning 成为可能。<Cite ref="turn706018academia49"/>

<Entity category="other" value="AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control" disambig="2026 action-discriminative latent world models for MPC"/> 已经在训练中加入动作辨识约束，并报告反事实 MPC 与机器人控制收益。<Cite ref="turn706018academia48"/>

因此，“把偏好蒸馏进预测几何”和“增强动作信息”也不能独立承担你的核心创新。

你真正要争取的是：

**在学习之前判断哪一种修复是值得做的，并对修复后的决策重新建立独立可靠性证据。**

这比继续设计一个更复杂的几何损失更有价值。

### 关卡 C：HTSC 也不能只是普通 conformal gate

ICML 2026 的 <Entity category="other" value="Conformal Risk-Averse Decision Making with Action Conditional Guarantee" disambig="Zhu et al., ICML 2026"/> 已研究动作条件化的风险保证。<Cite ref=""/>

所以“给决策风险加一个 conformal bound”同样不足以单独形成高水平新意。

HTSC 必须与世界模型的动作可辨识性、候选分布、修复操作及实际 regret 联系起来。尤其要证明：

**经过有依据的 GDD 修复后，HTSC 能够在相同风险水平下放行更多真正有用的动作，而不是简单扩大拒绝范围。**

---

## 四、我建议重新设计统一论文的三个 Contribution

现在的论文可以从：

*Diagnose → Repair → Decide*

进一步升级为：

<title size="lg">Identify → Repair What Is Learnable → Certify What Remains Trustworthy</title>

### Contribution 1：Counterfactual Decision Sufficiency Beyond Planning-Shaped Representations

这一贡献不再仅仅强调低 MSE 或 ATM 不保证决策正确，还要检验：

**即使未来表征经过规划监督，是否仍然存在不可忽略的反事实决策失配？**

目标不是证明所有 planning-shaped representations 都失败，而是刻画哪些条件下它们可能失败。

给定有限候选集合 $\mathcal C$，定义模型的决策传递误差：

$$

E_{ij,H}^{DT}
=
\left|
\Delta_{ij,H}^{*}
-
\widehat\Delta_{ij,H}
\right|

$$

除了 MSE、ATM 和历史信息，还要明确区分动作支持、任务代价的可表示性与真实反事实状态转移。

你需要一个让审稿人信服的结果：即使不同 world models 的常规预测指标和规划监督表现相近，它们对未见候选动作的真实排序仍可能显著不同。

但最关键的是，你的诊断必须能在不知道最终测试结果时预测哪些排序会出错。

### Contribution 2：A Decision-Repairability Frontier for Visual World Models

这是我认为最可能成为整篇论文标志性创新的部分。

不要一发现排序错误就立即训练 GDD-WM。

应该首先判断：当前错误能否由允许修改的函数类修复？

定义某个受限修复函数类 $\mathcal F$ 的最佳可达决策误差：

$$

\mathcal E_{\mathcal F}^{*}
=
\inf_{f\in\mathcal F}
\sup_{i,j\in\mathcal C}
\left|
\Delta_{ij}^{*}
-
\Delta_{ij}^{f}
\right|

$$

这里的关键是：它是一个理论目标，而非可以直接从少量样本精确计算的量。你真正要研究的是如何通过有限独立动作干预和结构约束，对它建立有用的估计或可检验边界。

例如对于固定 latent 的 PSD 几何修复：

$$

c_M(U)=x_U^\top Mx_U,\qquad M\succeq0

$$

真实执行确认的偏好能否由同一个 $M$ 表达，可以转化为带有 margin、归一化和容错项的半定可行性检验。

这个检验可以帮助排除某些**固定表示与固定 metric 类下**不可能同时满足的偏好，却不能证明任意可训练神经网络都无法修复这些偏好。

对 latent aliasing、dynamics mismatch 和 action-support insufficiency，你需要其他独立测试。

真正的新算法应当根据诊断结果决定：

<box border radius="lg" padding={3} gap={2}>
  <box background="surface-secondary" radius="md" padding={3}>
    <text color="secondary" weight="medium" size="xs">PROSPECTIVE REPAIR DECISION</text>
    **Before optimization, predict which intervention can work**
  </box>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <icon name="scan-eye" color="secondary" size="lg"/>
        **Metric repair**
        <text color="secondary" size="xs">信息已存在，几何排序错误</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <icon name="workflow" color="secondary" size="lg"/>
        **Dynamics repair**
        <text color="secondary" size="xs">真实后继有效，预测传播错误</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <icon name="radar" color="secondary" size="lg"/>
        **Active probe**
        <text color="secondary" size="xs">缺少观测或动作激励，可尝试补充信息</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <icon name="shield-alert" color="secondary" size="lg"/>
        **Abstain**
        <text color="secondary" size="xs">现有证据不能支持可靠修复或执行</text>
      </box>
    </grid-item>
  </grid>
</box>

这里必须检验一个很强的科学主张：

**我们能否在 GDD-WM 训练之前，就预测它会在哪些状态—动作区域有效，在哪些区域即使增加训练也不会可靠改善？**

如果能够证明这种前瞻性判断明显优于统一训练或事后风险筛选，那么即使 PlanWAM、D-JEPA 和 AD-WM 在某些 benchmark 上性能更高，你仍然建立了一个不同的科学结果。

### Contribution 3：Post-Repair Calibrated Counterfactual Planning

这一贡献需要给出真正可以落到在线控制的结果。

对已经冻结的修复后模型，使用独立数据校准完整候选集合的误差，而不是复用训练 GDD-WM 的执行偏好数据。

在适当的校准与可交换性条件下，设 $q_\alpha$ 同时覆盖测试 root 的全部相关候选对比误差：

$$

\Pr\left(
\max_{i,j\in\mathcal C}
|\Delta^*_{ij}-\widehat\Delta_{ij}|
\leq q_\alpha
\right)\geq1-\alpha

$$

当模型选择 $\hat i$ 后，可以得到该覆盖事件上的候选集合 regret 上界：

$$

\boxed{
R_H(\hat i)
\leq
\max_{j\ne\hat i}
\left[
q_\alpha-\widehat\Delta_{\hat i j}
\right]_+
}

$$

然后根据预先声明的 regret 容差决定 permit、probe 或 abstain。

需要强调：这属于固定模型与声明候选生成协议下的边际事件保证，不能自动升级为对每个状态或任意在线 CEM 自适应分布的条件安全保证。

你最终要证明的是：

$$

\boxed{
\text{Reliable Repair}
\;\Longrightarrow\;
\text{More Useful Certified Decisions}
}

$$

这里的箭头是待验证的研究目标，不是任何 GDD 修复都会自动满足的定理。

**这三个贡献连在一起，才真正超越 PlanWAM 的论文叙事：不仅学习有用的未来，还判断这种未来能否支持可靠反事实决策，并决定如何有条件地恢复这种能力。**

## 五、最应该立刻做的实验：让 PlanWAM 成为你的新科学问题的试金石

不要马上投入大量 GPU 去复制它的 93.8 PDMS。PlanWAM 原文报告使用 16 张 A800 训练；我检查的公开 [PlanWAM GitHub 仓库](https://github.com/jinchan9/PlanWAM) 目前仍写着代码与模型即将发布，因此还不能把其完整实现当作随时可运行的公平基线。<Cite refs={["turn287229view1","turn446453view2"]}/>

我建议先在你现有 PointMaze / PushT 实验框架内构造一个 *PlanWAM-inspired planning-shaped latent baseline*，明确它是方法思想对照，而不是原始 PlanWAM 的完整复现。

然后按照下面的顺序推进。

| 优先级 | 实验 | 必须回答的问题 |
|---|---|---|
| P0 | Planner oracle + action-bank kill tests | 当前动作候选集真的包含更优解吗？预测动力学是否真的被使用？ |
| P1 | Planning-shaped vs. counterfactual-sufficient representation | 任务监督是否仍留下 ATM/MSE 无法解释的反事实动作排序错误？ |
| P2 | Prospective repairability test | 能否在修复前预测 metric repair、dynamics repair 或 probe 哪种有效？ |
| P3 | GDD-WM 与强基线公平比较 | 相同执行反馈预算下，选择性修复是否优于普通蒸馏与 MLP scorer？ |
| P4 | Post-repair HTSC | 独立校准后能否降低错误放行，并保持有用的覆盖率？ |
| P5 | RGB-only closed-loop MPC | 前述收益能否转化为真实执行表现，并跨未见候选与环境保持？ |

### 我最看重 P1：Planning-shaped 仍然可能缺少反事实信息

构建一个可重置的环境，同一初始物理状态、同一视觉历史、同一规划时域下，分别执行若干不同的候选动作。

可以主动控制接触模式、摩擦参数、动力学方向或其他参与者的反应。

在这些测试中，分别测量真实任务代价、不同模型给出的排序、真实后继几何、预测后继几何，以及动作支持状态。

你希望观察到一个有判别力的现象：

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" weight="medium" size="xs">目标性实验现象 · 尚未证明</text>
  <grid columns={2} gap={3}>
    <grid-item>
      <box gap={1}>
        <text color="secondary" size="xs">Model A</text>
        **Planning-shaped future**
        <text size="sm">Prediction 较好，普通任务评分较好，但在某些反事实干预下排序错误</text>
      </box>
    </grid-item>
    <grid-item>
      <box gap={1}>
        <text color="secondary" size="xs">Model B</text>
        **Identifiability-guided future**
        <text size="sm">预测误差相近，却能提前识别这些不可靠区域，或利用执行反馈修复</text>
      </box>
    </grid-item>
  </grid>
  <divider color="subtle"/>
  **真正决定性的测试：**

  在完全未参与方法开发的动作集合和根状态上，B 是否具有更低的真实 action regret，以及更好的 regret–coverage 权衡？
</box>

即使这个实验中 PlanWAM-style baseline 的总体任务分数仍然很高，也不妨碍你建立不同的贡献。

因为你的目标不是证明 planning-shaped learning 没价值，而是证明其**能力边界可以被预先判别，且这种判别能指导后续修复**。

### P2 则是决定能否冲 Oral 的实验

假设测试集中存在三类错误：任务几何错误、预测动力学错误和当前观测不足导致的不可辨识错误。

在训练任何修复模型之前，运行你的 repairability diagnostic，预先为每个 root 或状态—动作区域给出修复建议。

之后分别尝试普通 metric learning、GDD-WM、predictor adaptation、额外有效观测或探测。

使用全新 held-out outcomes 判断这些建议是否正确。

如果你只能在 GDD-WM 修复成功之后解释为什么成功，这只是事后分析。

但如果你能够在训练之前预测哪个方法会成功，哪个方法不应该尝试，而且这种判断在未见环境上仍然成立，那就是一个实质性的新能力。

注意，所谓“不可修复”必须相对于明确的观测条件、数据预算和允许的函数类定义，不能把一次有限训练失败当成信息论不可能性。

### P4、P5 决定它最终是不是一篇完整论文

最终应该在同一批新测试 episodes 中比较这些方法：

| 方法 | 主要检验 |
|---|---|
| Frozen LeWM | 原有基线 |
| Planning-shaped future + scorer | 任务塑造未来是否已经足够 |
| AD-WM / ATM-guided training | 动作辨识监督是否足够 |
| D-JEPA-style / naive GDD | 普通执行偏好几何学习是否足够 |
| Naive GDD + calibrated gate | 简单模块组合是否已经足够 |
| Full repairability-guided GDD + HTSC | 新判别准则是否带来独立收益 |

最终主要报告真实决策 regret、ranking inversion、错误放行率、regret–coverage 曲线、闭环成功率、fallback 代价和执行反馈预算。

所有比较应尽量共享基础模型、候选集合、动作时域及计算预算，允许不同方法按照其设计接受明确记录的额外训练数据。

**最难战胜的基线不是 Frozen LeWM，而是普通执行偏好学习加普通校准。**

如果它与你的三模块方法表现基本相同，那么就必须重新审视 repairability-guided 方法的复杂度是否值得。

---

## 六、现在应该修改论文定位，而不是推翻研究

我建议你把原论文标题进一步收敛。

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" weight="medium" size="xs">建议主标题</text>
  <title size="xl">Planning-Shaped Is Not Enough: Identifying, Repairing, and Certifying Decision Transfer in Visual World Models</title>
  <text color="secondary" size="sm">这是强调科学问题的暂定标题，不能在结果成立前将其当作已证明结论。</text>
  <divider color="subtle"/>
  **更偏理论的备选：**

  *The Decision-Repairability Frontier of Visual World Models*
</box>

如果最终主证据仍主要来自 frozen LeWM，我会选择更准确的：

*When Can Frozen Visual World Models Be Repaired for Reliable Decision-Making?*

现在的 Introduction 应该承认 PlanWAM、D-JEPA、AD-WM 已经分别解决了规划导向表征、执行偏好学习和动作辨识训练的重要部分，然后指出：

**现有学习方法可以提高平均规划表现，但这本身没有回答一个世界模型的具体决策错误是否可以用当前信息可靠修复，以及修复后能否获得独立认证。**

这就是统一论文仍可能拥有的科学缺口。

从学术优先权角度，也不建议试图通过措辞绕开 PlanWAM。它的 arXiv v1 日期是 2026 年 10 月 8 日，应正常引用。你的早期提交、实验记录和研究演进可以保留为可核验的独立研究过程，但不能代替对公开先行工作的正确归因。

## 七、我会如何重新判断 Oral / Best Paper 潜力？

| 研究最终证明了什么 | 我的判断 |
|---|---|
| 在已有 GDD-WM 上增加规划监督，提高 success rate | 可能成为方法改进论文，但与 PlanWAM 等重叠较高 |
| 发现 planning-shaped 表征仍有真实反事实决策失效，提出新指标 | 有机会成为有价值的诊断研究 |
| 预先预测哪些错误可被几何或动力学修复，并在未见数据验证 | **有明显 Oral 潜力** |
| 在上述基础上，经过独立 HTSC 校准，联合改善闭环 regret、成功率和有用放行率 | 更完整的 Oral 竞争路线 |
| 建立有限数据、动作支持和表示类约束下的可修复性边界，并证明其跨物理系统有效、可以指导实际干预与控制 | **具有 Best Paper 研究潜力** |

这不是接收概率预测。真正的 Best Paper 还取决于理论深度、实验质量、可复现性与整个领域的同期进展。

有一件事我会特别坚持：**不要为了与 PlanWAM 竞争，突然把主要研究资源从 PointMaze 和 PushT 转向大规模自动驾驶。**

自动驾驶是一个很好的后续外部验证领域，但你当前最值得保留的是冻结视觉模型、可重置物理干预、动作支持审计和可辨识性边界。这些条件让你能够做 PlanWAM 原有实验并未直接回答的机制研究。

---

## 八、我现在最建议执行的决定

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="book-open" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **首先，建立正式的 PlanWAM overlap audit**

      在统一论文计划中，逐条标注哪些创新表述已经由 PlanWAM、D-JEPA、AD-WM 覆盖。删除泛泛的 *planning-aware future* 新颖性主张，把反事实充分性与修复可辨识性作为待检验的新主张。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="flask-conical" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **其次，优先做一个 prospective repairability pilot**

      先不要扩大 GDD-WM 网络规模。基于现有 counterfactual 数据，证明能够在训练前区分至少两种修复机制，并且这种区分比普通误差、不确定性或 ATM 筛选有额外价值。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **最后，把全部方法价值压到一项新鲜闭环评估中**

      修复后的 student 冻结，HTSC 使用独立校准，最终在未见 RGB-only episodes 中验证。若不能超过 naive GDD + standard gate，则不要把复杂的三模块组合包装成已经实现的理论突破。
    </box>
  </row>
</box>

## 我的最终判断

PlanWAM 的出现确实提高了你这篇论文的门槛。尤其是，你原来可能想强调的“规划定义有用未来”和“历史记忆为规划服务”，现在都很难再作为主要新颖性。

但它也让最重要的未解决问题更清楚了：

**我们已经有越来越多的方法能够训练出对规划有用的未来表征。下一步真正需要知道的是：这些表征在什么条件下足够可靠？它们的错误哪些可以通过有限反馈修复？哪些必须获得额外信息？修复后又如何避免虚假的模型自信？**

所以我的建议很明确：

**不要去做一个更复杂的 PlanWAM。去做一篇解释 PlanWAM 这一类方法何时有效、何时可能失效，以及如何有根据地修复和认证它们的论文。**

如果最终你能证明：在同样的规划监督下，一些 learned world models 仍然存在可预先识别的反事实决策盲区，而你的 ATM-guided repairability、GDD-WM 和 HTSC 能够将这些盲区转化为真实的修复、探测或拒绝决策——那么 PlanWAM 就不再只是你的竞争工作，它会成为推动你提出更深层科学问题的重要参照。

这才是我认为现在最值得坚持、也最有机会形成真正代表性成果的路线。

-----

> 站在他们研究的基础或者假设上，我们要如何继续往下做？方法换一下或者切换视角？他们有没有什么未解决的问题？

我会沿着 PlanWAM 的实际假设往下推：先找它**没有证明、甚至现有架构可能难以解决的命题**，再比较三条可选路线——改训练目标、改 world model 的反事实结构，或把研究问题提升为“何时值得修复与信任”。最后我会为 LeWorldModel 选一条能复用现有 ATM、GDD-WM、HTSC 积累、又不只是 PlanWAM 增量改进的路线。

# 从 PlanWAM 继续往下做：我建议改变研究问题，而不只是更换模型结构

**我认为有一条比直接改进 PlanWAM 更值得追求的路线：从 *Planning-Shaped Future Representation* 转向 *Counterfactual Decision-Sufficient World Models*。**

PlanWAM 已经证明，利用规划任务塑造未来表征，可以改善轨迹生成与选择。但它没有证明一个更强的命题：

> **被规划目标优化过的未来表征，是否足以判断那些尚未执行、甚至不属于常见行为分布的动作，会在真实环境中造成什么结果？**

这个问题恰好能把你的 ATM、GDD-WM、HTSC 三条研究线重新组织起来。

我重新核对了 PlanWAM 的论文正文、附录和公开代码仓库。它的核心设计确实是：

$$

h_t=E(o_{t-L:t}),\qquad
\hat z_t^+=P_\theta(h_t)

$$

然后利用 $(h_t,\hat z_t^+)$ 生成候选轨迹，并为各条候选轨迹评分。未来 latent 本身由历史产生，并不针对每一条候选动作单独进行受干预预测；但其评分器会显式读取候选轨迹，因此不能说 PlanWAM 完全不具备动作区分能力。<Cite refs={["turn921569view0","turn921569view2"]}/>

真正值得进一步研究的，是这种**共享未来信息 + 候选评分**的架构，与显式的 **action-conditioned counterfactual prediction** 之间，究竟存在怎样的能力边界。

## 一、PlanWAM 留下的六个关键开放问题

| 开放问题 | PlanWAM 的现有做法或证据 | 我们可以继续研究什么 |
|---|---|---|
| **1. 反事实充分性** | 历史预测共享未来 latent，候选由评分器区分 | 对不同 `do(action)`，是否真的预测正确的后果与排序？ |
| **2. 多模态未来** | 使用 token 级平方误差进行未来蒸馏 | 面对多个可能未来，单一预测是否掩盖关键风险？ |
| **3. 历史的真实价值** | 固定的近重远轻时间压缩 | 哪些较早事件包含决定当前动作效果的信息？ |
| **4. 偏好能否内化为几何** | 规划器和评分器联合学习任务信息 | 学到的任务偏好能否成为候选无关、可迁移的内部预测几何？ |
| **5. 预测何时可信** | 通过规划分数和闭环测试证明平均效果 | 能否在执行前判断当前 action ranking 是否可靠？ |
| **6. 候选集与闭环限制** | 64 候选轨迹；选择错误大幅减少，但仍有候选塌缩 | 如何区分预测错误、排序错误与候选搜索本身的限制？ |

这些是**论文尚未系统回答的研究问题**，不等于作者的模型已被证明在这些方面失败。

其中，问题 1、2、3、5 与你的现有研究最有结合价值；问题 4 必须进一步正面区别于 D-JEPA 等决策几何学习工作。

有两个很值得注意的论文内证据：

- PlanWAM 的 privileged posterior 达到 **95.0 PDMS**，而部署使用的先验模型达到 **93.8**，说明后见信息到可预测未来之间仍存在性能差距，但不能仅凭这一差距断言全部由预测不确定性造成。<Cite ref="turn783259view5"/>
- 加入完整模型后，失败数量由 **314 降至 237**，主要收益来自减少轨迹选择错误；作者同时承认候选塌缩仍存在，可行驶区域违规的部分结果也有所恶化。<Cite ref="turn474512view4"/>

这些现象给了我们很好的下一步研究切入点。

## 二、最有价值的视角转换：从“什么未来有用”到“什么未来足以决策”

PlanWAM 的主要假设是：

> 规划任务可以帮助模型学习更有用的未来表示。

我们可以接受这个假设，并进一步追问：

> **一种未来表示对于某个已训练规划器有用，是否意味着它对更广泛的候选动作和反事实决策也是充分的？**

这涉及两种不同的充分性。

### 1. 规划表现充分，不等于反事实决策充分

一个模型可能在训练分布中产生很好的规划轨迹，但如果未曾观察某种动作的真实后果，对该动作的判断仍然可能有很大误差。

例如，自动驾驶模型在历史数据中主要学习到“接近路口时减速”，于是很好地预测减速情况下其他车辆的未来位置。

但如果自车保持速度驶入路口，旁车可能突然刹车，也可能继续抢行。两种反应会显著改变动作排序。

这时，我们真正需要比较的是：

$$

P(Y_{t+H}\mid h_t,\operatorname{do}(U_i))

$$

而不仅是行为策略下的未来：

$$

P(Y_{t+H}\mid h_t,\pi_{\mathrm{behavior}})

$$

PlanWAM 已使用 NAVSIM 官方 proposal simulator，为候选轨迹提供 NC、DAC、TTC、EP 等监督标签，因此它不是只学习演示轨迹。但这种监督并不自动证明对训练候选集之外的动作、或其他交通参与者会响应自车动作的情形，具有可靠的反事实泛化能力。<Cite ref="turn921569view3"/>

**这里就是第一个值得冲击顶会的科学问题：任务监督可以塑造有用表征，但在何种动作支持与观测条件下，才能塑造出反事实决策充分的表征？**

### 2. 我建议用一个新的数学对象来定义未来表示的价值

给定历史 $h$、目标 $g$、动作序列 $U_i,U_j$，定义真实的干预条件下代价差异：

$$

\Delta^*_{ij}(h,g)
=
J^*(h,U_j,g)-J^*(h,U_i,g)

$$

我们可以定义两个历史之间的**决策差异距离**：

$$

d_{\mathrm{dec}}(h,h')
=
\sup_{g,U_i,U_j\in\mathcal C}
\left|
\Delta^*_{ij}(h,g)-
\Delta^*_{ij}(h',g)
\right|

$$

这个定义非常关键。

它不关心两段历史的图像是否相似，也不关心未来 latent 是否接近。

它只问：

**这两段历史对未来候选动作的真实任务偏好是否相同？**

如果两段视觉历史在压缩后映射成同一个表征：

$$

\Phi(h_1)=\Phi(h_2)

$$

但：

$$

d_{\mathrm{dec}}(h_1,h_2)=2\gamma

$$

那么对于任意只依赖当前压缩表征、目标与候选动作的确定性预测器，至少存在一个历史，在某个任务与动作对上的 contrast error 不小于 $\gamma$。

这是由相同预测输出和三角不等式得到的下界。它本身不是足以成为 Best Paper 的新定理，但指出了一个有价值的研究方向：

**未来表示应该尽可能压缩对任务决策无关的变化，同时保留能够改变干预动作偏好的信息。**

这比单纯说“task-relevant features”更精确，也给出了可以被反例、干预和测试数据检验的对象。

它要求明确任务集合、动作支持、历史条件和预测函数类；如果全部视觉历史本身都无法区分隐藏状态，单靠重新压缩历史也无法解决信息缺失。

我认为，这可以成为新的理论主线：

**Counterfactual Decision-Sufficient Representation Learning。**

## 三、但有一个新的竞争风险：不能只把 PlanWAM 改成 Action-Conditioned World Model

我进一步检索了相关论文，发现需要修正一个可能过于简单的研究方案：

**仅仅把 PlanWAM 的共享未来预测改成候选动作条件化预测，并不足以保证新的方法具有顶会级创新性。**

因为相关问题已经被其他研究推进了。

| 相关工作 | 已经覆盖的思想 | 我们不能再单独宣称的创新 |
|---|---|---|
| <Entity category="other" value="How Can Driving World Models Do Counterfactual Prediction?" disambig="Zhang et al., August 2026 driving counterfactual prediction paper"/> | 区分普通动作条件预测与忠实于特定已发生事件的反事实重建，并提出 abduction–action–prediction 视角 <Cite ref="turn769631view0"/> | 仅仅指出 action-conditioned prediction 不保证正确反事实 |
| <Entity category="other" value="CausalDrive" disambig="June 2026 causal reactive driving world renderer"/> | 通过可控世界模型研究自车动作与其他交通参与者的交互反应 <Cite ref="turn769631view1"/> | 只增加响应自车动作的未来预测 |
| <Entity category="other" value="Learning Task-Sufficient World Models by Synergizing Agentic Exploration and Structured Modeling" disambig="Feng et al., ICML 2026"/> | 主动探测任务相关隐藏因素，并学习紧凑、任务充分的表示 <Cite ref="turn769631view2"/> | 只提出 task-sufficient latent 或 active probing |
| <Entity category="other" value="D-JEPA" disambig="September 2026 decision-aligned latent world model"/> | 执行反馈驱动的候选关系学习与原生预测几何适配 <Cite ref="turn769631view3"/> | 只提出 execution-grounded preference distillation |

其中第一篇还有一个需要精确区分的概念：它研究的是在事实结果已经观测到之后，怎样忠实重建同一事件的替代结果。你的在线规划问题则是在动作执行前，基于当前可得信息评估潜在干预的后果。

二者有关联，却不能混为同一种 counterfactual estimand。

因此，真正尚有希望建立独立贡献的交叉点是：

$$

\boxed{
\text{Counterfactual Identifiability}
+
\text{Prospective Repairability}
+
\text{Post-Repair Decision Reliability}
}

$$

即：

**给定有限动作证据，能否提前判断一种 planning-shaped world model 缺少什么决策信息、当前哪种修复可以有效恢复这些信息，以及修复后哪些决策可以可信地执行？**

我建议你不要把整个项目简单改名为 Causal World Model。因果世界建模本身已经有很多强工作。

应该把研究切得更精准：**Decision-Repairability of Pretrained Visual World Models**。

---

## 四、三条可选的新研究路线

下面是我作为共同研究设计者会认真比较的三个方向。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="git-branch" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      <row align="center" wrap="wrap" gap={2}>
        **路线 A｜Counterfactual Contrast Learning**
        <badge color="success">最推荐作为方法核心</badge>
      </row>
      不再将完整 future latent 的逐 token 相似度视为主要目标，而是直接学习候选动作引起的任务后果差异，并把这种差异写回可用于预测的表示。

      优点是能继承 GDD-WM，科学问题具体，实验可控。主要风险是与已有 preference learning 过近，需要证明可修复性筛选的额外价值。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="brain-circuit" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **路线 B｜Decision-Sufficient Belief & Active History**

      改变 PlanWAM 的固定 recency-based 历史压缩，把“哪些历史重要”定义为它们能否减少当前真实动作偏好的不确定性。必要时通过信息获取动作主动补足历史。

      理论潜力很高，尤其适合 PushT 的接触模式与隐藏摩擦，但与已有 POMDP、active system identification、task-sufficient representation 文献重叠更多。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **路线 C｜Calibrated Selective World-Model Planning**

      保留现有 backbone 和大部分 planner，把主要创新放在有限干预数据下的误差校准、选择性放行与风险—效用控制。

      最容易复用 HTSC，能够形成清晰的系统结果；但如果没有新的 decision-specific calibration 理论，容易被审稿人视为常规 conformal planning 的应用。
    </box>
  </row>
</box>

如果只能选一条主路线，我会选 **A 为方法、B 为必要的状态辨识机制、C 为最终部署验证**。

理由是它们能回答同一个问题，而不是各自训练一个网络来提高指标。

### 我会具体怎样改 GDD-WM？

原来的几何蒸馏大体围绕真实执行偏好：

$$

U_i\prec U_j
\quad\Longrightarrow\quad
c_\theta(U_i)<c_\theta(U_j)

$$

新的版本应进一步要求：

**不仅知道谁更好，还要学习两种动作为什么产生不同结果、差多少，以及这种差异能否从当前历史辨识。**

我建议首先做一个受限的 action-conditioned student：

$$

\hat z_H(U)=F_\theta(\Phi(h),U)

$$

并在合法的干预数据上训练其任务对比：

$$

\widehat\Delta_{ij}^{\theta}
=
c_\theta(\hat z_H(U_j),g)
-
c_\theta(\hat z_H(U_i),g)

$$

核心损失可写成：

$$

\boxed{
\mathcal L=
\mathcal L_{\rm pred}
+
\lambda_1\mathcal L_{\rm contrast}
+
\lambda_2\mathcal L_{\rm preserve}
+
\lambda_3\mathcal L_{\rm support}
}

$$

其中：

$$

\mathcal L_{\rm contrast}
=
\sum_{(i,j)\in\mathcal P_{\rm valid}}
w_{ij}
\left(
\widehat\Delta_{ij}^{\theta}
-
\Delta_{ij}^{*}
\right)^2

$$

对于只具有可靠偏好符号、没有可信代价幅值的监督，可以换成 margin ranking loss。对于随机环境中的期望代价差异，则应使用重复执行估计或明确的噪声模型，不能直接将一次 realized cost 当作精确期望。

各项作用应该明确：

- $\mathcal L_{\rm pred}$：保证 student 仍是预测模型，而不是纯任务打分器。
- $\mathcal L_{\rm contrast}$：学习与执行相关的动作代价差异。
- $\mathcal L_{\rm preserve}$：限制错误的 geometry adaptation 破坏原有动作动力学。
- $\mathcal L_{\rm support}$：对缺乏可靠干预证据的样本进行排除、降权或显式不确定性建模；这不是单独就能保证 OOD 正确性的损失项。

**真正新的部分不应该是这四项损失相加，而是决定哪些样本允许进入哪一项修复的机制。**

这也是我希望 ATM 发挥新作用的地方。

ATM 不只是一个离线 inverse accuracy 指标，而是结合真实/预测动作一致性、局部干预覆盖、状态可观测性和可表示性检验，辅助选择修复方式。

对于明确可由现有几何表达的关系，训练 GDD metric；对于 realized geometry 正确但预测传播失败的关系，尝试 predictor adaptation；对于当前视觉历史不能提供充分信息的关系，首先增加有效观测或进行探测，而不是让 student 强行拟合冲突标签。

这样，GDD-WM 的研究定位就从 *Preference Distillation* 升级为：

**Identifiability-Constrained Counterfactual Geometry Repair**。

注意，这个名称代表待研发方法，而不是已经获得验证的新成果。

## 五、还有一个更深的方向：重新定义“值得保留的历史”

PlanWAM 采用近重远轻的 Temporal Register Pyramid，并在相同 128-token 预算下证明这一分配优于均匀或反向分配。<Cite ref="turn783259view5"/>

这个结论支持它在当前数据上的设计，但还没有证明：

**对所有决策任务而言，近期历史总比远期历史包含更多有用的物理信息。**

例如 PushT 中，较早发生的一次接触可能暴露物体摩擦特性。即使最近几帧的外观相同，不同的历史接触事件仍可能意味着不同的后续旋转响应。

这里我不建议你简单换成 Mamba、GRU 或 Transformer，因为你之前的 memory audit 已经表明，模型容量或记忆形式的改变不保证正确恢复物理动作效应。

更值得研究的是：

### Decision-Information-Aware Memory

让历史信息的保留由它对**动作偏好不确定性的减少量**决定，而不只是时间远近。

设 $b_t$ 表示基于合法历史形成的隐状态信念，定义决策不确定性代理：

$$

\mathcal A_{\mathrm{dec}}(b_t)
=
\sup_{U_i,U_j\in\mathcal C}
\operatorname{Width}
\left(
\mathcal I_{ij,H}(b_t)
\right)

$$

其中 $\mathcal I_{ij,H}$ 是由有效估计方法得到的真实动作代价差异置信区间。

历史事件是否值得保留，取决于它是否显著缩小与当前候选决策有关的置信区间。

如果现有历史不足，才考虑一个信息获取动作：

$$

a_{\mathrm{probe}}^*
=
\arg\max_{a\in\mathcal A_{\mathrm{probe}}}
\left[
\mathbb E\bigl[
\mathcal A_{\mathrm{dec}}(b_t)
-
\mathcal A_{\mathrm{dec}}(b_{t+1})
\bigr]
-\lambda C_{\mathrm{probe}}(a)
\right]

$$

这是一个待实现的实验设计目标。实际中无法直接知道真实区间缩小量，必须用已验证的模型集合、置信集或其他可计算代理估计，而且 probe 本身需要满足允许的安全约束。

这个方向的独特价值并非“主动探索”——ICML 2026 已经有相关工作——而是**面向当前候选动作排序歧义进行信息获取，并量化新信息能否把原本无法认证的决策变成可以认证的决策**。

我建议把它作为 Contribution 1 和 Contribution 3 之间的机制实验，而非立即新增第四个大型网络模块。

---

## 六、我会如何真正落地：先做一个能决定研究命运的实验

结合你的仓库现状，我建议首先在 PointMaze 完成一个 **Counterfactual Sufficiency Pilot**，然后再进入 PushT。

你的公开统一论文计划仍明确将正式 GDD-WM 修复、独立 HTSC 校准和 RGB-only 在线 MPC 作为尚待通过的关键阶段。尤其是当前已发布的离线候选数据不能自动替代新鲜 RGB realized successor 与真实在线规划。<Cite ref="turn360038view0"/>

### 实验设计：同样的规划监督，为什么有些未来依然不能可靠选动作？

<box border radius="lg" padding={3} gap={2}>
  <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
    **Shared Counterfactual Dataset**
    <text color="secondary" size="xs" textAlign="center">Same RGB roots · Same goals · Same candidate actions · Real executed outcomes</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <text color="secondary" size="xs">BASELINE</text>
        **Planning-shaped foresight**
        <text color="secondary" size="xs">Shared future context + trajectory scorer</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <text color="secondary" size="xs">BASELINE</text>
        **Action-conditioned model**
        <text color="secondary" size="xs">Candidate rollout + native or learned cost</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <text color="secondary" size="xs">BASELINE</text>
        **Naive preference GDD**
        <text color="secondary" size="xs">All reliable execution labels, no repairability routing</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <text color="secondary" size="xs">PROPOSED</text>
        **Repairability-guided GDD**
        <text color="secondary" size="xs">Identify mechanism → selective repair → preserve prediction</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="md" padding={3} align="center" gap={1}>
    **Independent HTSC Calibration**
    <text color="secondary" size="xs">Frozen model · Root-level contrast bounds · Support checks</text>
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
    **Held-out RGB-only Closed-loop Evaluation**
    <text color="secondary" size="xs">Real regret · Ranking errors · Useful coverage · Task success</text>
  </box>
</box>

其中 Planning-shaped foresight 可以首先使用与 PlanWAM 同思想的轻量基线，不能将其报告为对原始 PlanWAM 的严格复现。

### 最重要的三个实验判据

**判据 1：先证明存在有效的改进空间。**

同一 root 下，ground-truth candidate oracle 应明显好于当前 learned planner。如果 oracle 也无法完成任务，说明候选生成、horizon 或目标函数本身可能是主要限制。

同时完成 Bank-only、No-dynamics、Shuffled-successor 和 New-bank 对照。如果完全不用 world-model rollout 也能得到几乎相同的结果，不能把改进归因于预测模型。

**判据 2：修复前的诊断必须预测修复后的收益。**

将测试状态按照事先规定的方法分为 geometry-repairable、dynamics-repairable、information-limited 等类别。随后对这些类别应用对应修复，并在全新数据上验证。

不要根据修复后的结果反过来修改分类标准。

特别要报告：某个区域被判断不适合当前 metric 修复时，这个判断是否确实优于“所有样本一律使用 GDD”的方法。

**判据 3：认证必须具有非空洞控制价值。**

Student 完全冻结后，使用独立 calibration roots 标定 HTSC，再在新的 root 和候选分布上检验。

建议将主要结果画成：

<text color="secondary" weight="medium" size="xs">预期的论文主图形式，仅说明应比较的对象，不代表已取得结果</text>

<Chart content={{"chartType":"line","meta":{"footer":"Illustrative curves only — not experimental results"},"xKey":"coverage","series":[{"dataKey":"baseline","label":"Naive GDD + gate"},{"dataKey":"full","label":"Repairability-guided + HTSC"}],"yAxisMin":0,"xAxisLabel":"Decision coverage (%)","data":[{"coverage":"20","baseline":0.04,"full":0.03},{"coverage":"40","baseline":0.11,"full":0.055},{"coverage":"60","baseline":0.2,"full":0.095},{"coverage":"80","baseline":0.34,"full":0.19},{"coverage":"100","baseline":0.49,"full":0.41}]}}/>

目标是 Full method 在相同覆盖率下实现更低的真实 regret，或者在相同风险预算下放行更多动作。图中曲线纯属示意，不能被视为任何实验结果。

这里需要格外谨慎：root-level conformal 的边际覆盖，不意味着所有被选择放行的动作都享有同样的条件覆盖。自适应 CEM 改变候选分布后，也需要匹配的独立校准与验证协议。

这些都是 Contribution 3 必须明确解决或限定的理论边界。

---

## 七、我建议的最终三项贡献

如果以 PlanWAM 为新的研究起点，我会这样重写统一论文，而不是按三个独立模块的名字介绍方法。

| Contribution | 核心研究问题 | 新方法与必要证据 |
|---|---|---|
| **C1 · Counterfactual Sufficiency** | 规划导向表征何时不足以支持真实反事实动作选择？ | Task-contrast identifiability、history ambiguity 和 action-support 诊断；证明相较 ATM、MSE、普通任务评分有增量信息 |
| **C2 · Predictive Repairability** | 哪些不足可以通过几何或动力学学习修复？ | 有限干预支持的 repairability criterion、选择性 GDD-WM；在训练前预测修复成败 |
| **C3 · Calibrated Decision Transfer** | 如何判断修复后的模型何时值得信任？ | 独立重新校准的 HTSC，在真实 RGB-only 闭环中改善 regret–coverage–utility |

我更喜欢这三个贡献，因为它们有一个非常严格的内在依赖关系。

C1 必须发现 C2 能够利用的机制信息；C2 必须产生 C3 可以独立验证的真实改进；C3 必须证明联合系统比简单的 preference learning + standard risk gate 有额外价值。

任何一步没有通过，都应该收缩主张，而不是再添加网络模块。

### 对应论文题目

<title size="lg">Beyond Planning-Shaped Futures: Counterfactual Sufficiency and Selective Repair of Visual World Models</title>

如果理论上的可修复性边界最终成为最强结果，则可以考虑：

*The Decision-Repairability Frontier of Visual World Models*

这两个题目都强调：我们研究的不是再创造一种更有用的未来表征，而是确定未来表征支撑决策所需的信息条件，以及如何对其进行有根据的修复。

---

## 八、哪些情况下应该停止、转向或拆分论文？

这也是避免把研究拖成一个无限扩张的工程项目的关键。

| 实验发现 | 应采取的研究决策 |
|---|---|
| Planning-shaped scorer 已经在未见反事实动作上非常可靠 | 不强行主张其不足，转向研究其可靠性条件与可验证范围 |
| ATM / MSE / uncertainty 已经充分预测修复收益 | 不再将复杂 repairability indicator 作为独立创新 |
| Full GDD 与普通 MLP scorer 表现相同 | 重新审查预测几何学习的必要性；转向模型压缩、跨目标迁移或理论边界 |
| 几何修复有效，但修复前不能预测哪里有效 | 将论文定位为条件性几何学习，不宣称 prospective repairability |
| HTSC 只能拒绝绝大多数动作 | 不宣称有效在线认证；首先研究误差界紧致性及 fallback 的真实价值 |
| 诊断和几何修复都成立，但闭环收益无法建立 | 将研究收敛为 decision-sufficiency 诊断论文，不强行包装成完整规划方法 |

这个策略可以避免一个审稿人最容易批评的问题：研究提出大量正确概念，却缺少一个明确、无法由更简单方法替代的结果。

## 最终建议

**我不建议你现在完全换项目，也不建议仅仅把 PlanWAM 改成 Action-Conditioned PlanWAM。**

我建议更换的是主要研究视角：

<box gap={2}>
  <box border radius="md" padding={3} gap={1}>
    <text color="secondary" size="xs" weight="medium">PLANWAM 的问题</text>
    **What future should a world model learn for planning?**

    世界模型应该预测怎样的未来，才能帮助规划？
  </box>
  <row justify="center">
    <icon name="arrow-down" color="secondary"/>
  </row>
  <box border radius="md" padding={3} gap={1}>
    <text color="secondary" size="xs" weight="medium">建议我们继续研究的问题</text>
    **What makes a predicted future sufficient, repairable, and trustworthy for counterfactual decisions?**

    怎样的预测未来才具有反事实决策充分性？哪些不足能修复？哪些决策可以被信任？
  </box>
</box>

最值得保留的是 LeWorldModel 已有的 **ATM 真实/预测动作辨识、GDD-WM 执行偏好几何、HTSC 有条件规划** 三条资产。

最需要改变的是 GDD-WM 的方法目标：从“把正确偏好蒸馏进未来 latent”，改成“**先判断偏好是否可辨识、是否能由当前函数类表达，再有选择地修复，并验证修复是否真正改变决策**”。

如果最终能拿到一个结果——**在相同 planning-shaped 训练监督下，模型对某些动作的可靠性差异能够被预先判断，而且这种判断指导的修复确实优于统一训练，在闭环中还获得独立的风险—效用收益**——那么这篇研究就不仅是 PlanWAM 的后续工作。

它会提出一个更基础、可在世界模型研究中反复使用的科学问题：**世界模型所学习到的“有用未来”，究竟在什么信息条件下可以成为“可信未来”？**

我认为这比现在直接追逐更高的 NAVSIM 分数，更有机会形成属于你的、难以被现有方法组合取代的核心贡献。

-----

> A 为方法、B 为必要的状态辨识机制、C 为最终部署验证  展开成一个具体的研究方案，先看看你提到的这些研究有哪些存在的未解决的问题，然后看看我们这个研究方案需要怎么解决

我会先把相关论文按 A（反事实决策学习）、B（历史与状态辨识）、C（校准规划）重新审一遍，区分“论文明确承认的限制”和“我们推断但尚未验证的空白”。然后把这些空白收敛成一个统一问题，给出模型结构、损失函数、理论命题、数据采集协议、强基线以及每个阶段的停止条件，尽量做到可以直接转成 LeWorldModel 的实验计划。

# LeWorldModel 下一阶段研究方案：从 Planning-Shaped Futures 到可辨识、可修复、可认证的反事实世界模型

**研究定位：A 为方法核心，B 为状态辨识机制，C 为部署与可靠性验证。**

我核对了 PlanWAM、D-JEPA、AD-WM、Controlled World Models Identifiability，以及 ICML 2026 关于 Task-Sufficient World Models 和 Action-Conditional Conformal Decision Making 的论文。

我的结论是：**这项研究仍有值得投入的科学缺口，但不能再把“规划导向表征”“主动探测”“执行偏好蒸馏”或“动作条件校准”单独当成主要创新。**

它们都已经有很强的先行工作。

我们真正应该建立的是一种新的能力：

> 给定一个预训练视觉世界模型、有限视觉历史与少量合法动作干预，系统能否在修改模型之前，判断当前决策错误来自哪里、哪种修复能够有效改善它，以及修复后的哪些动作值得执行？

我建议暂时把整个方法命名为：

<title size="lg">Counterfactual Decision Repairability for Visual World Models</title>

<text color="secondary" size="sm">工作名称：CDR-WM。ATM、GDD-WM 和 HTSC 保留为三个有明确职责的组成部分，而不是三种独立算法的简单拼接。</text>

## 一、文献审查：六类工作究竟还缺少什么？

这里先区分两件事：

**已确认的证据边界**：论文实际解决了什么、在哪些条件下验证了什么，或作者明确承认的局限。

**我们提出的开放假设**：论文没有系统验证的性质，但不能因此断言它的模型必然失败。

| 研究 | 已完成的主要贡献 | 仍值得研究的问题 | 对我们 A/B/C 的影响 |
|---|---|---|---|
| <Entity category="other" value="PlanWAM" disambig="Xu et al. 2026 planning-shaped autonomous driving world action model"/> | 规划目标塑造未来表示、历史压缩、后见到前瞻蒸馏、候选生成与评分 | 历史预测的共享未来上下文，能否支持未见干预动作的可靠对比？ | A：反事实对比；B：历史辨识 |
| <Entity category="other" value="D-JEPA" disambig="Liu et al. 2026 decision-aligned latent world model"/> | 从执行结果学习候选关系，并把决策顺序实现为可由 latent distance 读取的未来表示 | 哪些偏好能够以不依赖候选集合的函数稳定表达？修复前能否预测成败？ | A：可表示性与选择性修复 |
| <Entity category="other" value="AD-WM" disambig="Qiu et al. 2026 action-discriminative world models"/> | 用动作恢复训练增强预测器，在多项操控任务中改善 MPC | 动作可区分是否保证真实任务对比正确？怎样事先判断仍会选错的区域？ | A：超越动作辨识 |
| <Entity category="other" value="On the Identifiability of Controlled World Models" disambig="Zhang et al. 2026 identifiability theory"/> | 在特定高斯潜在状态和行为策略假设下，给出表示与受控转移的可辨识条件 | 对冻结非高斯视觉表征、部分可观测历史和接触切换，如何得到可检验的局部结论？ | B：状态与动作效应辨识 |
| <Entity category="other" value="Learning Task-Sufficient World Models by Synergizing Agentic Exploration and Structured Modeling" disambig="Feng et al. ICML 2026"/> | 主动探测任务相关因素、学习紧凑任务充分状态，并验证组合泛化 | 如何针对当前候选动作的排序歧义设计最小信息获取，而非一般性环境探索？ | B：决策定向观测与探测 |
| <Entity category="other" value="Conformal Risk-Averse Decision Making with Action Conditional Guarantee" disambig="Zhu et al. ICML 2026"/> | 提供动作条件风险保证及相应有限样本学习方法 | 自适应 MPC 候选集合、修复后的模型和有限真实执行预算下，如何构造有用的 regret 证书？ | C：选择性决策认证 |

上述文献覆盖范围可从各论文的方法、定理、实验及结论核对。<Cite refs={["turn269661view1","turn269661view2","turn269661view3","turn269661view4","turn269661view5","turn269661view6"]}/>

### 最值得注意的三个具体缺口

**第一，PlanWAM 并没有直接研究当前动作集上的因果反事实充分性。**

它部署时的未来潜在表征由历史产生，随后作为轨迹生成和候选评分的上下文。评分器确实读取候选轨迹，但世界模型的前瞻分支没有显式为每条候选动作分别模拟受干预后果。

作者还发现，未来表征对于轨迹选择的帮助强于仅用于轨迹生成；同时，候选塌缩问题没有完全解决。<Cite refs={["turn753606view1","turn826532view6","turn753606view2"]}/>

因此，我们不应该再问“未来表示是否有用”，而应该问：

$$

\boxed{
\text{Planning relevance}
\;\not\Rightarrow\;
\text{Counterfactual sufficiency}
}

$$

这只是待严格验证的非蕴含关系，不是对 PlanWAM 的既成失败判定。

**第二，D-JEPA 已经把决策关系写进 future latent，GDD-WM 必须再往前走一步。**

D-JEPA 不只是 reranker。其论文还研究通过修改未来表示，让原生 latent distance 反映关系模型产生的排序；相关实现可以包含候选集合的相对顺序信息。<Cite ref="turn826532view2"/>

所以我们不能再把“推理时不依赖额外关系评分器”作为足够的新颖性。

更有价值的问题是：

**对于新的历史、目标和未见候选动作，能否提前判断这种内生几何修复会不会成功，而不是先训练、再观察它是否改善？**

**第三，AD-WM 的原文限制说明，动作恢复和闭环成功之间仍有值得研究的机制空间。**

AD-WM 在仿真和真实机器人上取得了明显收益；但作者也明确指出，详细的 planning diagnostics 主要局限于 Cube，Scene 的收益在三个种子上仍未解决，真实机器人试验还依赖人工子目标，并使用单一站点、相机和 backbone。<Cite ref="turn826532view0"/>

这意味着我们应当争取的不是“比 AD-WM 多提升一点成功率”，而是：

**为什么一部分 action-identifiable futures 可以被有效规划，另一部分即使动作恢复良好，也仍然无法支持正确的任务选择？**

---

## 二、统一科学问题：Decision-Repairability，而不只是 Decision Alignment

我建议将整项研究定义为一个有限信息下的决策问题。

给定：

$$

\mathcal P=
(E_0,F_0,\mathcal H,\mathcal A,\mathcal G,\mathcal D_{\rm int})

$$

其中：

- $E_0,F_0$：预训练视觉 encoder 与动作条件 predictor；
- $\mathcal H$：部署时允许使用的视觉和动作历史；
- $\mathcal A,\mathcal G$：候选动作与任务目标；
- $\mathcal D_{\rm int}$：有限、独立且具有可追溯来源的动作干预结果。

研究目标不再是训练一个可以拟合所有执行偏好的模型，而是学习一种决策规则：

$$

\boxed{
\pi_{\rm repair}:
(h,g,\mathcal C,\mathcal D_{\rm int})
\longrightarrow
\{\text{Metric},\text{Dynamics},\text{Observe/Probe},\text{Abstain}\}
}

$$

这里最关键的要求是：**修复策略必须在相应测试结果尚未观察到时就确定。**

然后选择性地训练 GDD-WM，并使用独立的 HTSC 校准决定是否放行规划。

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Frozen visual world model**
    <text color="secondary" size="xs">RGB history · Action candidates · Task and goal</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **B · Decision-state identification**
        <text color="secondary" size="xs" textAlign="center">History · Action support · Hidden-mode ambiguity</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **ATM + Transfer audit**
        <text color="secondary" size="xs" textAlign="center">Real/predicted action effects · Task contrasts</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **Prospective repairability assessment**
    <text color="secondary" size="xs">Which repair can work under the allowed information and function class?</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **A · Selective counterfactual GDD-WM**
    <text color="secondary" size="xs">Geometry repair · Restricted dynamics adaptation · Prediction preservation</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **C · Independent HTSC recalibration**
    <text color="secondary" size="xs">Regret bounds · Risk/coverage validation · Support gates</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **RGB-only closed-loop MPC**
    <text color="secondary" size="xs">Permit · Observe/Probe · Fallback · Execute · Replan</text>
  </box>
</box>

这是我认为可以真正把 A、B、C 组成一篇论文的方法架构。下面进一步落实为模型、理论和实验协议。

## 三、B：必要的状态辨识机制应该怎样设计？

我建议先实现 B，再正式训练新的 GDD-WM。

因为如果模型缺少正确状态信息，直接训练偏好损失可能只会使它更加自信地输出错误排序。

### B1. 保留冻结基础模型，增加最小化的 Decision-State Belief

从历史：

$$

h_t=(o_{t-L:t},a_{t-L:t-1})

$$

构造：

$$

\eta_t=B_\psi(E_0(o_{t-L:t}),a_{t-L:t-1})

$$

进一步得到一个关于决策相关隐藏状态的信念：

$$

b_\psi(\xi_t\mid h_t)

$$

这里 $\xi_t$ 不一定是完整物理状态，可以只包含影响当前任务动作效果的隐藏因素，例如运动方向、摩擦类别、接触模式或速度。

我不建议第一版直接引入大型 stochastic Transformer。先使用三种严格匹配参数量与历史长度的对照：

| 历史建模 | 作用 |
|---|---|
| Recency-based compression | 对照 PlanWAM 的近重远轻思想 |
| Fixed delay / Fourier history | 对照现有 LeWorldModel 的历史提升方法 |
| Decision-event belief | 新方法：优先保存能够改变候选动作后果判断的历史事件 |

第三种方法的关键不是 attention 架构，而是学习目标。

定义当前候选集合的任务对比向量：

$$

\boldsymbol\Delta^*(h_t)
=
\left(
\Delta^*_{ij,H}(h_t,g)
\right)_{(i,j)\in\mathcal P}

$$

让历史表征尽可能保留对这个向量有预测价值的信息，而不是无差别保留所有视觉变化。

可以通过一个轻量的 task-contrast prediction head 建立辅助监督，但在训练时只使用训练划分中的真实干预结果。

这与 ICML 2026 的 task-sufficient exploration 工作不同的潜在新意，在于我们关注的是**固定候选决策的排序歧义及修复可能性**，而不是学习一个尽量充分的通用任务状态。该区别仍须通过实验验证。<Cite ref="turn269661view5"/>

### B2. 定义 Decision Aliasing，而不是普通 latent 相似性

对于冻结或当前声明的历史表征 $\Phi$，定义：

$$

\mathcal A_H^\Phi(\eta)
=
\sup_{\substack{h,h':\\
\Phi(h)=\Phi(h')=\eta}}
\max_{i,j\in\mathcal C}
\left|
\Delta^*_{ij,H}(h)-\Delta^*_{ij,H}(h')
\right|

$$

若存在两个历史被映射成相同的 $\eta$，但真实动作偏好相差 $\delta$，则任何仅根据 $\eta$ 输出单一 contrast prediction 的方法，至少在其中一个历史上产生 $\delta/2$ 的误差。

因此：

$$

\boxed{
\inf_{\widehat\Delta(\eta)}
\sup_{h:\Phi(h)=\eta}
\|\widehat\Delta(\eta)-\Delta^*(h)\|_\infty
\geq
\frac12\mathcal A_H^\Phi(\eta)
}

$$

这个不等式属于固定信息接口下的基本不可辨识性下界，并非新的高难度定理。

**真正困难的是有限样本估计。**

对于真实高维 latent，几乎不可能指望在数据中找到两个完全相等的 embedding。因此我建议采用两套互补实验：

第一套是**受控构造**。在合成与可重置物理系统中，有意让表示忽略速度、接触模式或其他任务相关变量，建立具有已知标签的 decision-aliasing benchmark，用于检验理论。

第二套是**自然 learned models**。对真实训练的 LeWM / DINO / JEPA 表示，搜索表征接近而真实动作对比显著不同的样本，并在独立 roots 上验证。

第二套得到的是经验近似混叠证据，不能直接解释为全局不可辨识性定理。若要把它变成有置信保证的下界，需要对局部平滑性、覆盖及测量噪声提出可验证的附加假设。

这恰恰可以成为 Theory 1 的核心研究内容。

### B3. 将动作支持和状态辨识分开

Controlled World Models Identifiability 已说明，在其受控高斯设定下，动作条件变异不足会妨碍动力学可辨识性。其理论中使用了：

$$

\rho_{\rm tr}(\pi)
=
\lambda_{\min}
\left(
\mathbb E_z[\operatorname{Cov}(a\mid z)]
\right)

$$

作为条件动作激励的刻画。<Cite ref="turn826532view5"/>

但我们不能直接把这个高斯模型下的结论套用于 PushT 接触动力学。

对于 LeWorldModel，我建议分别估计：

- **State ambiguity**：历史是否包含区分不同真实动作后果的信息；
- **Action excitation**：现有干预是否覆盖了需要辨识的动作方向；
- **Real–predicted transfer consistency**：预测与真实后继中的动作响应是否一致；
- **Task contrast consistency**：相同动作对是否具有正确的真实任务代价排序。

其中，ATM 可以承担第三项的一部分，但不能代替第四项。

而且，不能将不可恢复动作标签等同于决策失败：如果两个动作具有相同的任务后果，它们无法被 inverse probe 区分，也不一定损害当前决策。

### B4. 只在可减少决策歧义时进行 Active Probe

对 PushT，可以设计接触点附近的小幅正反向探测；对导航任务，可以利用短时动作观察隐藏动力学参数或运动趋势。

但 probe 的目标不应是“最大化任意隐藏状态信息”。

我建议定义：

$$

a_{\mathrm{probe}}^*
=
\arg\max_{a\in\mathcal A_{\rm safe}}
\left[
\widehat{\operatorname{EIG}}_{\rm decision}(a)
-\lambda c(a)
\right]

$$

其中 $\widehat{\operatorname{EIG}}_{\rm decision}$ 估计执行探测后，当前候选动作偏好不确定性的减少量；$\mathcal A_{\rm safe}$ 是由独立验证的约束限定的可探测动作集。

第一版只在可重置仿真环境中测试这一机制。若不存在经过验证的安全探测动作，就应保留 abstain 或明确的 fallback，而不是把 probe 当作必然安全的替代动作。

**B 的验收标准**不是历史越长越好，而是：

在相同历史 token 预算下，新方法能否降低 held-out counterfactual contrast error、减少决策状态混叠，并提高后续 GDD 修复的可预测性。

---

## 四、A：具体的新方法——Repairability-Guided Counterfactual GDD

这是整篇论文最应该投入方法研究资源的部分。

我建议不要重新发明一个庞大的 world model，而是让 GDD-WM 从普通偏好蒸馏变成**受可辨识性与函数类约束的选择性修复算法**。

### A1. 首先对真实决策错误做可操作的分解

对于同一历史 $h$、目标 $g$ 和候选对，构造：

$$

\Delta^*_{ij}
=
J^*(U_j)-J^*(U_i)

$$

定义真实执行后继的几何代价差异：

$$

\Delta^{\rm obs}_{ij}
=
\mathbb E[
c_\theta(z^{\rm real}_{j,H},g)
-
c_\theta(z^{\rm real}_{i,H},g)
\mid h,\operatorname{do}(U_i),\operatorname{do}(U_j)]

$$

这里的表达表示分别在对应动作干预分布下对两项代价取期望，不是要求在同一次物理实验中同时执行两个互斥动作。

预测代价差异为：

$$

\widehat\Delta^\theta_{ij}
=
c_\theta(\hat z_{j,H},g)
-
c_\theta(\hat z_{i,H},g)

$$

于是：

$$

\boxed{
\Delta^*_{ij}-\widehat\Delta^\theta_{ij}
=
\underbrace{\Delta^*_{ij}-\Delta^{\rm obs}_{ij}}_{\text{Task-geometry gap}}
+
\underbrace{\Delta^{\rm obs}_{ij}-\widehat\Delta^\theta_{ij}}_{\text{Predictive-transfer gap}}
}

$$

这是精确的代数分解，前提是代价、时间范围和期望定义一致。

但它只能分解误差，不能自动识别误差原因。还需要 B 的历史/动作支持审计，以及下面的可表示性测试。

### A2. 在训练之前，判断现有预测几何是否能够表达执行偏好

以固定表示上的 PSD metric 为最小可分析函数类：

$$

c_M(U_i)=x_i^\top Mx_i,\quad
M\succeq0,\quad
\operatorname{Tr}(M)=1

$$

其中：

$$

x_i=\hat z_H(U_i)-z_g

$$

对真实执行确认的 $U_i$ 优于 $U_j$，要求：

$$

\operatorname{Tr}
\left[
M(x_jx_j^\top-x_ix_i^\top)
\right]
\geq m_{ij}-\xi_{ij}

$$

其中 $\xi_{ij}\geq0$ 是容错松弛量。

然后解决一个半定优化问题，检验在声明的函数类和 margin 下需要多少不可避免的偏好违背或松弛。

需要在两条路径上分别做：

| 测试 | 使用的特征 | 所能支持的结论 |
|---|---|---|
| Realized geometry feasibility | 真实执行后继的编码 | 当前 encoder + metric 类是否可以表达执行偏好 |
| Predicted geometry feasibility | learned rollout 的未来 latent | 当前 predictor + metric 类是否可以表达相同偏好 |
| Flexible scorer control | 相同输入下的 MLP 或其他非二次模型 | 排除“只是 PSD 函数类太弱”的解释 |
| History-lifted feasibility | 增加合法的历史状态信息 | 检验改变信息接口后是否改善可表达性 |

例如，若真实后继 PSD 可行、预测后继 PSD 不可行，这支持检查 dynamics 的必要性，但尚不能仅凭可行性差异证明 dynamics 是唯一原因。

如果当前表示下两个候选具有完全相同的目标相对向量 $x_i=x_j$，但真实代价不同，那么任何只根据这个向量打分的函数都无法区分它们；这属于输入信息或预测特征的可表达性问题。

这类实验可以让我们**在大规模 student 训练前，就排除一些错误的修复方案**。

### A3. 建立一个 Prospective Repair Router

我建议先使用可解释的结构化路由规则，而不是立即训练黑箱 router。

<box border radius="lg" padding={3} gap={2}>
  <table>
    <table-row>
      <table-cell width="42%">
        **训练前可观察证据**
      </table-cell>
      <table-cell>
        **候选修复**
      </table-cell>
    </table-row>
    <table-row>
      <table-cell>历史信息充分、动作支持可靠、真实几何可表达，但原 metric 排序错误</table-cell>
      <table-cell>Geometry GDD</table-cell>
    </table-row>
    <table-row>
      <table-cell>真实几何可表达，预测 rollout 出现显著动作效应错误</table-cell>
      <table-cell>Restricted Dynamics GDD</table-cell>
    </table-row>
    <table-row>
      <table-cell>历史提升可以明显消除动作后果混叠</table-cell>
      <table-cell>History / Belief adaptation</table-cell>
    </table-row>
    <table-row>
      <table-cell>缺少条件动作支持，但存在经过验证的安全信息获取方式</table-cell>
      <table-cell>Active probe</table-cell>
    </table-row>
    <table-row>
      <table-cell>证据不足、可表示性不成立或风险无法可靠估计</table-cell>
      <table-cell>Uncertified / Abstain</table-cell>
    </table-row>
  </table>
</box>

这些规则在第一版都应该是可证伪的判断，而不是预先认定它们百分之百正确。

尤其应允许 `unknown`。有限样本检验无法区分修复失败、缺少数据和优化困难时，不应该强行给出“不可修复”的结论。

之后可以利用**独立的开发环境与修复试验**，学习每种方法在不同证据特征下的成本—收益估计：

$$

\hat r(h,g,\mathcal C)
=
\arg\min_{r\in\mathcal R}
\widehat{\mathcal E}_{\rm post}(r)
+\lambda\,\operatorname{Cost}(r)

$$

其中 $\widehat{\mathcal E}_{\rm post}(r)$ 是仅基于开发数据预测的修复后误差，不得包含最终测试 outcomes。

这一步才是整项研究中最有潜力的新算法对象：**在不知道某次修复的最终测试结果前，决定应当采用哪种修复。**

### A4. GDD-WM 的训练目标

在路由确定以后，使用可靠、受支持的执行候选对训练对应 student。

我建议使用：

$$

\begin{aligned}
\mathcal L_{\rm CDR}
={}&
\mathcal L_{\rm pred}
+\lambda_c\mathcal L_{\rm CF}\\
&+\lambda_g\mathcal L_{\rm realized}
+\lambda_p\mathcal L_{\rm preserve}
+\lambda_s\mathcal L_{\rm stability}
\end{aligned}

$$

其中最核心的是反事实对比损失：

$$

\boxed{
\mathcal L_{\rm CF}
=
\frac1{|\mathcal P_{\rm valid}|}
\sum_{(i,j)\in\mathcal P_{\rm valid}}
w_{ij}\,
\rho\left(
\widehat\Delta^\theta_{ij}
-\widetilde\Delta^{\rm exec}_{ij}
\right)
}

$$

$\widetilde\Delta^{\rm exec}_{ij}$ 是由独立执行反馈构造的训练标签，$\rho$ 可以采用 Huber loss；如果只有可信偏好符号，则改用排序损失。

权重 $w_{ij}$ 体现测量精度、动作支持和训练样本的可靠性，不能由最终测试结果反向确定。

对于经过确认的三类修复，训练参数应有所区别：

| 修复类型 | 更新参数 | 保留约束 |
|---|---|---|
| Geometry-only | 投影/metric 或受限表示 head | 冻结 predictor、检查新候选泛化 |
| Dynamics-only | Predictor tail / 低秩 residual adapter | 保留真实 successor prediction、动作符号及多步一致性 |
| History-guided | 小型 belief/history module，再训练必要的预测适配 | 保留基础视觉 encoder；验证新增历史确实提高动作辨识 |

我特别建议保留两套部署方法，避免一个隐藏的函数类问题：

**Strict native-geometry GDD**：仅在目标相对预测几何足以表达任务偏好时使用欧氏或 PSD 代价。

**Expressive decision-cost control**：当任务代价必须依赖路径、接触事件或方向性时，允许使用显式 path-aware / directed cost，并将其作为单独方法或强基线。

例如，两个动作可能产生相同终点，但中途一个碰撞、一个没有碰撞。单纯终点距离无法区分，不应假装增加 metric 维度就能自动修复；必须改变包含的信息或代价接口。

这也与 TD-JEPA 的有向任务进展工作存在直接关联。<Cite ref="turn920801academia70"/>

这里的核心贡献不是提出更多 loss，而是**根据可辨识性与可表达性证据，决定何时应使用哪一种学习接口**。

## 五、C：把 HTSC 升级成真正的 Post-Repair Decision Certification

HTSC 是整个系统的最后一环，但我不建议让它继续承担“证明 world model 总体安全”这种过大的任务。

它应该解决一个更明确的问题：

**修复后的模型，在当前候选动作集合和声明的规划时域中，其动作选择是否具有足够小的真实 regret？**

### C1. 认证对象必须是候选决策，而不是 latent prediction MSE

修复后的模型完全冻结，给定候选集合：

$$

\mathcal C(h_t,g)=\{U_1,\ldots,U_K\}

$$

预测代价为：

$$

\widehat J_i^\theta=\widehat J_\theta(h_t,U_i,g)

$$

设 $J_i^*$ 是相同条件下的真实期望任务代价。

对一个独立 calibration root $r$，定义误差统计量：

$$

S_r=
\max_{i,j\in\mathcal C_r}
\left|
(J_{r,j}^*-J_{r,i}^*)
-
(\widehat J_{r,j}-\widehat J_{r,i})
\right|

$$

对于确定性可重置环境，可以计算真实候选代价；对于随机环境，必须进一步处理有限重复执行估计 $J^*$ 的误差。

令 $n$ 个独立 calibration roots 的统计量排序为：

$$

S_{(1)}\leq\cdots\leq S_{(n)}

$$

标准 split-conformal 的 root-level 有限样本构造为：

$$

k=\left\lceil(n+1)(1-\alpha)\right\rceil

$$

$$

q_\alpha=
\begin{cases}
S_{(k)},&k\leq n\\
+\infty,&k>n
\end{cases}

$$

在模型、候选生成算法预先冻结，以及 calibration/test root-level 数据可交换等条件下：

$$

\Pr(S_{\rm test}\leq q_\alpha)\geq1-\alpha

$$

这给出的是**跨测试 roots 的边际覆盖保证**，并不是任意状态的条件保证。

### C2. 从 contrast bound 得到 regret-aware permit

设：

$$

\hat i=\arg\min_i\widehat J_i

$$

在上述有效覆盖事件上：

$$

\boxed{
R_H(\hat i)
\leq
\max_{j\ne\hat i}
\left[
q_\alpha-
(\widehat J_j-\widehat J_{\hat i})
\right]_+
}

$$

记右侧为 $\widehat B_H$。

那么可以设计：

$$

\pi_{\rm HTSC}=
\begin{cases}
\text{permit},&
\widehat B_H\leq\tau,\ \text{support valid}\\
\text{probe},&
\text{information insufficient and probe admissible}\\
\text{fallback/abstain},&
\text{otherwise}
\end{cases}

$$

其中 $\tau$ 是事先声明的候选 regret 容差。

但需要明确：普通边际覆盖不自动保证

$$

\Pr(R_H>\tau\mid\mathrm{permit})\leq\alpha

$$

也不自动保证自适应在线多步 MPC 的整条轨迹可靠。

因此要让 C 真正产生研究新意，还需要处理两个问题。

**问题一：放行选择本身的统计偏差。**

除了 root-level 边际覆盖，需要对冻结的许可规则使用独立测试数据，报告被放行动作的实际错误率及其置信区间。

如果想声明明确的 selective risk guarantee，则应针对放行事件设计相应风险控制方法，而不是直接借用普通 conformal 的结论。

**问题二：CEM 自适应候选的分布变化。**

若候选生成器、模型和 CEM 协议在校准前已冻结，并将整个自适应候选生成过程视为 root 到 candidate bank 的固定映射，则可在相同协议下对完整候选集合进行 root-level 校准。

但如果在线更新模型、调整搜索分布，或采用新的探测策略，旧校准结果不能自动延用。

ICML 2026 的 Action-Conditional Conformal 工作已经证明普通边际保证并非研究终点，所以我们的贡献必须落在**world-model-specific regret、候选集合选择与修复后重新认证**上，而不是单纯引入 conformal prediction。<Cite ref="turn269661view6"/>

### C3. HTSC 还必须区分 Model Failure 和 Planner Failure

这里有一篇不能忽略的竞争工作：

<Entity category="other" value="The Planning Limits of Latent World Models" disambig="Alrasheed et al. September 2026 latent planning range study"/> 证明，短 horizon 规划在某些目标设置下，即使使用真实模拟器作为完美动力学预测器也可能失败。<Cite ref="turn553222view4"/>

因此，HTSC 可以证明当前候选集合内的排序具有有限 regret，却不能证明候选动作最终足以完成任务。

我建议在每个主要环境中同时构造：

| Oracle | 排查的问题 |
|---|---|
| True candidate-bank oracle | 候选中是否存在明显更好的动作 |
| Perfect-dynamics MPC | 同样 horizon 与目标函数是否可规划 |
| True successor + learned metric | 决策几何是否限制排序 |
| Learned successor + privileged cost | Predictor 是否限制排序 |
| Expert/subgoal planner | 失败是否来自远距离目标或近视规划 |

这四层区分将帮助我们避免错误归因，也会提高论文机制分析的可信度。

---

## 六、理论部分：应该争取哪些真正新的结论？

我特别建议把理论研究集中在 **Repairability Frontier**，而不是重新证明 prediction MSE 不保证控制。

2026 年的 <Entity category="other" value="A Control Theory of Predictability in Latent World Models" disambig="You et al. 2026 plan-cost fidelity and off-manifold control theory"/> 已经证明 plan-cost discrepancy 与控制 regret 的关系，并研究了 planner-reachable distribution 与训练数据分布不一致的问题。<Cite ref="turn532221view0"/>

因此，如果我们仅证明：

$$

R_H\leq 2\sup_U|\widehat J(U)-J^*(U)|

$$

审稿人不会认为这是新的理论贡献。

我会把目标定为三个相互连接的结果。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="sigma" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Theory 1 — Decision Information Lower Bound**

      在固定表示、任务、动作支持和 horizon 下，刻画 latent 混叠造成的不可消除决策误差；研究有限干预数据能够支持何种下界估计。

      **真正新增的难点**：从已知的抽象不可辨识反例，走向有限数据下可检验的决策歧义条件。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="layers" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Theory 2 — Function-Class Repairability**

      对受限 metric / dynamics / history-adaptation 函数类，建立何种执行偏好可以被表达的条件，并刻画几何表达误差、动力学传播误差与数据不足的不同作用。

      **真正新增的难点**：在修复尚未进行时，就得到对实际修复收益有预测意义的量，而不是仅仅求解一个 SDP。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="shield-check" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Theory 3 — Repair-then-Certify Decision Control**

      在明确的 support、校准和候选生成条件下，给出修复后有限时域 regret 的有效界，并刻画何时应放行、探测或拒绝。

      **真正新增的难点**：把修复选择和后验可靠性联系起来，获得相对于统一训练加标准风险门控的可量化优势。
    </box>
  </row>
</box>

这三个结果可以构成一个新的理论对象：

$$

\boxed{
\mathfrak F_H
\left(
\Phi,\mathcal F,\mathcal D_{\rm int},
\mathcal U,g
\right)
}

$$

我建议将其称为 **Decision-Repairability Frontier**。

它不应只是一个综合得分，而应明确表示：在给定表征、函数类、数据与预算下，哪些动作区域有证据支持修复，哪些仅能得到部分保证，哪些无法从现有信息可靠判断。

如果能够进一步得到一个关于修复选择成本、误差上界和动作覆盖率的非空洞结果，并在真实 learned models 中验证，这才更接近理论型 Oral 的创新深度。

---

## 七、具体实验方案：怎样一步一步证明 A+B+C？

我建议论文优先覆盖三种不同的动力学机制，而不是立即转向大规模自动驾驶。

<box gap={3}>
  <row align="start" gap={3}>
    <AsyncImage query="2D point maze robot navigation benchmark top down simulated continuous control environment" aspectRatio="5:3" maxWidth="160px"/>
    <box flex="1" gap={1}>
      **PointMaze / TwoRoom — 反事实决策与几何修复**

      用于建立最小可控的 action-contrast benchmark、可表示性实验、预测几何修复和完整候选 regret 认证。首先完成这里的主方法验证。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <AsyncImage query="DeepMind Control Suite reacher robotic arm simulation reaching colored target" aspectRatio="5:3" maxWidth="160px"/>
    <box flex="1" gap={1}>
      **Reacher — 连续动力学与未见动作**

      检验动作激励、历史信息、连续动作插值与多步预测对决策充分性的影响，减少只依赖固定 action bank 的风险。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <AsyncImage query="Push T reinforcement learning benchmark robot pushing T shaped blue block 2D environment" aspectRatio="5:3" maxWidth="160px"/>
    <box flex="1" gap={1}>
      **PushT — 部分可观测与接触模式切换**

      检验同一视觉表示下接触前、接触中和模式切换的动作响应是否可辨识；是否需要保留更早的接触历史或补充探测才能实现稳定姿态控制。
    </box>
  </row>
</box>

这些场景应共享研究问题与指标，但不要求使用完全相同的网络结构和物理动作接口。

### 数据协议：真正的同根反事实执行

对每个初始 root $r$，固定目标 $g_r$，构造：

$$

\mathcal C_r=\{U_{r1},\ldots,U_{rK}\}

$$

然后分别重置到相同物理初始条件，执行每个候选动作。

记录：

$$

(o_t^{RGB},h_t,g,U_i,
o_{t+1:t+H}^{RGB},J_i^{real})

$$

物理状态、接触标签和真实动力学参数可以用来重置及分析，但不能在测试时进入 RGB-only planner。

对非确定性环境，要有重复执行或明确的随机性估计协议。

我建议的第一阶段数据规模如下，它们是**启动预算建议，不是统计充分性的固定标准**：

| 用途 | 建议初始规模 | 目的 |
|---|---:|---|
| P0 pilot | 约 100–200 个独立 roots，每 root 8–16 个候选 | 测试数据采集、oracle ceiling、动作效果可测性 |
| B/A 方法开发 | 约 300–500 个独立 roots，按物理模式分层 | 估计可辨识性，训练 repair router 与 GDD student |
| HTSC calibration | 单独约 150–300 roots | 校准完整候选对比统计量 |
| Locked final offline test | 另取约 200–300 roots | 评估 held-out regret、route accuracy 和覆盖率 |
| Closed-loop test | 独立 episodes，多环境及多个训练种子 | 判断实际控制收益 |

对于最终统计功效，应根据 pilot 的 root-level 方差和希望检测的效应大小重新计算样本量。不能通过增加同一 root 的 pair 数冒充独立样本。

另外，完整 64 候选 CEM 的真实后果标注可能昂贵，可以先使用较小的完整候选集合证明基本机制，再进入 CEM elite audit。**如果只标注少量 elite，就不能把它宣称为全部未标注候选上的同时覆盖。**

### 严格的数据隔离

至少保证下面五类数据互不泄漏：

1. Base world-model pretraining / development。
2. 状态辨识与 repairability-router training。
3. GDD-WM execution preference training。
4. Student 冻结后的 HTSC calibration。
5. 全新 offline outcomes 与 closed-loop final evaluation。

同一物理 root 的所有候选动作及重复执行必须整体划分。

模型级泛化还需要独立训练的 world-model seeds；环境级泛化需要从未参与阈值、模型结构或方法选择的任务。

---

## 八、必须包含的强基线和消融

这是决定是否能说服 ICML / ICLR / NeurIPS 审稿人的核心。

| 方法 | 状态辨识 B | Counterfactual GDD A | Repair router | HTSC C |
|---|:---:|:---:|:---:|:---:|
| Frozen LeWM | — | — | — | — |
| Planning-shaped future + scorer | 可使用历史压缩 | 任务评分监督 | — | — |
| ATM / AD-WM-style | 动作辨识 | 动作恢复训练 | — | — |
| D-JEPA-style geometry alignment | — | 执行偏好监督 | — | — |
| Naive GDD | — | ✓ | — | — |
| Naive GDD + standard risk gate | — | ✓ | — | ✓ |
| Belief + naive GDD | ✓ | ✓ | — | — |
| Repairability-guided GDD | 可选 | ✓ | ✓ | — |
| **Full A+B+C** | ✓ | ✓ | ✓ | ✓ |

尤其必须补以下四类负控制：

**Bank-only、No-dynamics、Shuffled-successor、Unseen-candidate-bank。**

如果预测后继被删除或打乱之后，GDD 仍获得同样效果，那么性能提升很可能不需要真实 world-model rollout。

这会直接削弱“我们修复了预测几何”这一主张。

同时报告：

- 在相同执行监督预算下，普通 MLP scorer 和 PSD metric 能达到什么水平；
- Router 是否在训练前正确预测不同修复的收益；
- Student 的 realized 与 predicted geometry 是否同时改善；
- 多步预测误差是否被保持；
- CEM elite 的真实 ranking inversion 是否下降；
- 同一风险约束下，HTSC 是否释放更多有用动作；
- 新方法是否在 RGB-only 闭环中超越简单模块组合。

其中应把**修复策略选择的准确性和经济价值**作为新指标。例如比较 routing regret：

$$

R_{\rm route}
=
\mathcal E_{\rm post}(r_{\rm selected})
-
\min_{r\in\mathcal R}
\mathcal E_{\rm post}(r)

$$

这里 oracle 最佳修复方法仅用于事后评价，不能进入 router 的实际选择。

如果 Full 方法只是提升最终成功率，却没有证明 router 优于随机修复或统一 GDD，那么本研究仍然缺少核心新意。

---

## 九、按阶段推进：具体做到哪一步才能进入下一步？

| 阶段 | 要完成的任务 | 必须通过的门槛 | 失败时怎么办 |
|---|---|---|---|
| **P0：Truth & Oracle** | 新鲜 counterfactual roots、候选 oracle、原始 LeWM MPC、shortcut kill tests | 存在真实可改善的 action-selection gap | 先修 candidate generator、控制协议或任务代价 |
| **P1：B State Identification** | 历史混叠、条件动作支持、ATM、正反动作干预 | 状态辨识诊断在留出数据上提供真实增量信息 | 降低 belief 复杂度或重新定义可观测对象 |
| **P2：Repairability** | PSD / MLP feasibility、realized/predicted geometry 分解、预先路由 | 能在训练之前预测哪些修复实际有效 | 不宣称 prospective repairability |
| **P3：A GDD Repair** | Geometry-only、Dynamics-only、Selective GDD | 未见候选 regret 改善，且 prediction preservation 合格 | 检查函数类、动作支持或改为显式决策模型 |
| **P4：C Calibration** | 修复后冻结、独立 root-level 校准、CEM shift audit | 可靠、非空洞的 regret–coverage 表现 | 不称为证书，不进入正式许可规划 |
| **P5：RGB Closed Loop** | PointMaze / Reacher 的完整闭环 MPC | Full 方法优于 Naive GDD + gate，收益具有实际控制意义 | 排查 candidate ceiling、fallback 与规划时域 |
| **P6：External Validation** | PushT、未见模型训练 seed、独立动力学条件 | 判别准则的有效范围可复现、失败边界清楚 | 缩小理论与实验适用范围 |

我尤其建议把 **P0–P3 视为决定是否继续统一三贡献论文的关键阶段**。

如果 P2 的 repairability router 没有比普通的 MSE/ATM/uncertainty 分层明显更强，就不应该继续投入大量资源把它包装成新的理论对象。

---

## 十、怎样接入现有 LeWorldModel 仓库？

从当前可核验的 [统一论文计划](https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/UNIFIED_DECISION_TRANSFER_PAPER_PLAN_20261014.md) 和 [GDD 执行方案](https://github.com/BokaiGuo/LeWorldModel/blob/main/docs/GDD_WM_EXECUTION_GROUNDED_GEOMETRY_PLAN_20261014.md) 看，仓库已有 task contrast、ATM 研究基础、风险门控与 GDD-WM 的设计协议；正式 GDD 修复及后续在线收益仍需要独立证据。公开 README 与后续执行记录可能存在状态更新差异，因此我不会把计划中的阶段当作已运行完成。<Cite refs={["turn553222view0","turn151573view3"]}/>

我建议沿用现有统一执行框架，新增下列功能，而不是另建第二套总入口。

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" weight="medium" size="xs">建议新增的代码模块与职责（不是声称这些文件已经存在）</text>
  <table>
    <table-row>
      <table-cell>
        `decision_benchmark/counterfactual_data.py`
      </table-cell>
      <table-cell>
        同根干预数据、候选 provenance、RGB successors 与 oracle ceilings
      </table-cell>
    </table-row>
    <table-row>
      <table-cell>
        `decision_benchmark/decision_aliasing.py`
      </table-cell>
      <table-cell>
        固定表示的历史混叠、任务对比差异与可信证据
      </table-cell>
    </table-row>
    <table-row>
      <table-cell>
        `decision_benchmark/action_support.py`
      </table-cell>
      <table-cell>
        局部动作激励、支持度、ATM 与 task contrast 分层
      </table-cell>
    </table-row>
    <table-row>
      <table-cell>
        `gdd_wm/geometry_feasibility.py`
      </table-cell>
      <table-cell>
        Realized / predicted PSD 可行性、函数类对照
      </table-cell>
    </table-row>
    <table-row>
      <table-cell>
        `gdd_wm/repair_router.py`
      </table-cell>
      <table-cell>
        训练前修复选择、unknown 判定、routing cost
      </table-cell>
    </table-row>
    <table-row>
      <table-cell>
        `gdd_wm/selective_gdd.py`
      </table-cell>
      <table-cell>
        条件化几何、动力学或历史适配；预测保持
      </table-cell>
    </table-row>
    <table-row>
      <table-cell>
        `decision_benchmark/regret_calibration.py`
      </table-cell>
      <table-cell>
        Root-level split conformal、有限样本检查、许可风险审计
      </table-cell>
    </table-row>
    <table-row>
      <table-cell>
        `decision_benchmark/closed_loop_eval.py`
      </table-cell>
      <table-cell>
        RGB-only MPC、fallback、风险覆盖与任务成功
      </table-cell>
    </table-row>
  </table>
</box>

正式运行需要统一记录：Git commit、模型权重 checksum、root split、真实执行来源、动作坐标系、candidate-generation seed、teacher 来源、calibration 来源以及最终测试结果。

HTSC 相关旧校准还应进行有限样本分位数与 fail-closed 逻辑的专项复核。任何对 encoder、predictor、metric、router 或在线候选生成协议的修改，都应触发适用性检查，必要时重新校准。

---

## 十一、三张主图与一个理论主结果

最终论文不应该被组织成“ATM 一张图、Memory 一张图、GDD 一张图、HTSC 一张图”。

应该围绕三个连续的科学发现展开。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="scan-search" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Figure 1 — Planning-Shaped Is Not Necessarily Counterfactually Sufficient**

      在相同任务监督、MSE 和动作支持匹配条件下，展示真实反事实动作排序误差，以及 B 的状态辨识信息能否预测这些错误。

      这一图回答：为什么现有任务塑造表征仍可能不足？
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="git-branch" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Figure 2 — The Decision-Repairability Frontier**

      按训练前诊断的不同机制分层，展示 Geometry GDD、Dynamics GDD 和 History/Probe 的真实修复结果；比较预先路由与普通统一训练。

      这一图回答：我们是否能提前知道什么值得修？
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="chart-no-axes-combined" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Figure 3 — Post-Repair Risk–Coverage–Utility**

      在独立 RGB-only 闭环中，比较完整方法与普通几何学习、标准不确定性门控和简单模块组合，报告真实 regret、success、false release、coverage 与执行成本。

      这一图回答：这些诊断和修复是否真正改变了实际决策？
    </box>
  </row>
</box>

理论主结果则围绕同一个问题：

**在声明的表示、动作支持与修复函数类下，我们能否从有限干预数据推断某些决策信息的不可辨识性、某些修复的可表达性，并在修复后建立有效的选择性决策风险界？**

这应是一个有统一假设与可检验结论的研究框架，而不是把三个经典不等式合并后声称发现了新定理。

---

## 十二、研究取舍：我会怎样判断这条路线的价值？

| 最终结果 | 论文定位 |
|---|---|
| 只证明 task-shaped latent 在部分动作上排序错误 | 机制诊断研究，创新性取决于新诊断是否有额外价值 |
| Geometry GDD 明显降低 regret，但没有可靠的修复前预测 | 有条件的 decision-aligned world-model 方法论文 |
| 修复前能预测修复策略，且跨未见模型/环境有效 | **开始具备 Oral 竞争力** |
| 上述机制再与独立有效的 HTSC 形成真实闭环互补收益 | 较完整的 Oral 竞争方案 |
| 得到可计算、非空洞、具有广泛解释力的 repairability frontier，并在多种动力学下独立验证 | 具有进一步冲击 Best Paper 的研究潜力 |

## 最终建议：先攻克一个足以改变研究方向的实验

我最建议你**先不增加新的大型 backbone，也不急着复现整套 PlanWAM**。

先使用 LeWorldModel 现有的 PointMaze 反事实研究基础，完成一次预先注册的实验：

**在训练 GDD-WM 前，利用真实后继、预测后继、动作支持与历史辨识信息，预测当前哪一种几何或动力学修复能够降低 held-out decision regret；然后真正执行这些修复，在完全独立数据上检验预测是否正确。**

这项实验包含 A 与 B 的核心机制，也能直接为 C 提供修复后模型。

如果它成功，就不仅证明模型“可以被修得更好”，还证明我们**知道为什么能够修、应当修哪里、什么时候不应当修**。

而一旦再由 HTSC 在全新 RGB-only 闭环中证明：这种有根据的修复带来了比普通 GDD + standard calibration 更好的真实 regret–coverage–utility 权衡，三个 Contribution 就真正形成了一条不可替代的科学证据链。

**这就是我认为最值得把 LeWorldModel 下一阶段研究资源集中投入的方向：从学习对规划有用的未来，走向识别、修复和认证对反事实决策真正充分的未来。**

-----

> 请你作为ICLR/ICML/NIPS审稿人锐评一下这个研究的缺点[BokaiGuo/LeWorldModel: Transfer-sufficient diagnostics and reproducible evidence for frozen visual world models](https://github.com/BokaiGuo/LeWorldModel)  并且补充需要做什么才能达到oral/best paper级别研究？建立在PlanWAM已经发出来的基础上我们该如何继续往下做？做出我们独特的insight，基于谱分析视觉表征完善促进下游规划plan任务的效果

我会按当前公开仓库做一次重新审稿，重点不再停留在“ATM＋GDD-WM＋HTSC 是否能拼成一篇”，而是检验你最想保留的**谱分析视觉表征 → 改善真实规划**这条因果链。随后对照 PlanWAM 和近期直接竞争工作，找出哪些谱结论已经不够新、哪些能发展成真正的训练方法与理论，并给出冲 Oral / Best Paper 的决定性实验和止损标准。

核查后有两个比“PlanWAM 抢了任务导向表征”更严峻的事实：一是已有研究直接做过**谱模态选择与 PushT 规划失效**的实验；二是还有工作分别研究了**真实干预效应错误、动作对比训练和规划器访问分布偏移**。因此，LeWorldModel 的新意不能停在“发现某个谱方向很重要”，必须升级为：**用有限真实动作证据找出决策关键模态，选择性修复它们，并在未见候选动作与闭环中证明因果收益。**

# LeWorldModel：ICLR / ICML / NeurIPS 强审稿，以及后 PlanWAM 时代的研究升级方案

<text color="secondary" size="sm">评审基准：2026 年 10 月 23 日｜公开 LeWorldModel 仓库、实验记录、PlanWAM 原文、近期谱表征与决策导向世界模型研究</text>

<badge color="danger">当前模拟审稿决定：Reject</badge>

我的核心判断是：

**LeWorldModel 有机会发展成一篇很有辨识度的研究，但目前最缺少的不是一个新诊断指标，而是证明“谱结构发现了规划需要却被表征学习忽略的信息，并且能够指导模型修复，最终提升真实规划表现”。**

你希望保留三个核心方向：

- **谱分析视觉表征**：covariance、spectrum、Schur、commutator、transient dynamics、action-to-task transfer。
- **决策导向表示学习**：ATM、GDD-WM，以及让预测几何符合真实动作效果。
- **可靠规划**：HTSC、RGB-only MPC、有限时域 regret 和有条件的动作放行。

我认为可以保留三者，但必须彻底改变它们之间的逻辑。

不是：

$$

\text{Spectral metrics}
+\text{GDD loss}
+\text{Risk gate}

$$

而应该是：

$$

\boxed{
\text{Identify missing decision modes}
\rightarrow
\text{Restore learnable modes}
\rightarrow
\text{Verify their planning value}
}

$$

这就是我建议下一阶段真正追求的独特 insight：

> **A visual world model may accurately preserve dominant or predictable latent modes while suppressing weak but action-controllable, task-critical modes. Decision-reliable planning requires identifying, preserving, and calibrating the transfer of these modes—not merely making future representations planning-shaped.**

中文：

**视觉世界模型可能很好地保留高方差、容易预测的主导表征方向，却丢失低能量但对动作控制和任务结果至关重要的方向。真正提升规划能力，需要辨识、恢复并验证这些决策关键动力学模态，而不只是用规划损失塑造未来表示。**

这个命题比“预测误差不等于规划性能”更有具体机制，也能让你的谱研究真正成为方法核心。

但有一项重要的新竞争工作必须正视。

## 一、我发现了另一项与你的谱方向高度相关的研究

2026 年 10 月 2 日，Hugues Van Assel 发表了技术文章 <Link url="https://huguesva.github.io/blog/2026/world-models/" title="Feature Selection in World Models: A Spectral View"/>。

它在共同的线性高斯设定下，分析了不同 world-model 目标偏好保留的谱模态，包括 VAE、JEPA、Dreamer 等，并给出了一个很有冲击力的 PushT 实验：添加一个容易预测、但与任务无关的视觉颜色信号，可能大幅损害规划表现。文章报告，在其设定下，灰色对照的平均成功率为 79.0%，加入稳定颜色后降至 1.1%，改为不可预测颜色后回升至 67.7%。<Cite ref="turn388563view4"/>

这是技术博客中的作者报告，并非我已经核验的同行评审论文或独立复现结论。

但它非常重要，因为它已经直接研究了：

**预测目标选择了哪些谱模态，而这些模态是否真正对控制有用。**

这意味着，你不能仅仅证明“JEPA 偏好可预测模态，有时会忽略任务信息”，然后将其称为首次发现。

你必须进一步研究它没有直接解决的部分：

**如何在已经训练好的真实视觉世界模型上，利用动作干预与任务传递结构，识别缺失的决策模态，并主动修复这些模态，使下游在线规划真正受益。**

这条路线比单纯比较谱分布、主成分或特征值更加有前景。

---

## 二、目前仓库的真实成熟度：最大的距离仍在实验闭环

公开仓库的统一论文计划把当前研究分为 Diagnose、Repair、Decide，但明确指出：GDD-WM 尚未形成正式验证的 student 结果，HTSC 仍需要合法的动作支持和独立校准，RGB-only 在线规划的主要证据尚未完成。<Cite ref="turn504293view2"/>

我核对了公开执行记录中截至 10 月 23 日可使用的部分，结果非常清楚。

| 研究证据 | 已有进展 | 我作为审稿人的判断 |
|---|---|---|
| PointMaze task contrast | 一项离线审计报告约 40.8% 的候选对排序反转 | 决策失配存在，但还不能说明谱指标能够预先预测它 |
| Indexed canonical-action readout | 2 类架构 × 5 seeds，报告 regret reduction +0.1090 | 有正结果，但可能依赖固定 candidate bank 的动作特征 |
| Cross-environment matched audit | $\Delta R=+0.05336$，sign-flip $p=0.0630$ | 没通过预先声明的统计检验 |
| GDD-WM | 数据与证据审计已有进展，正式蒸馏结果仍缺 | 未证明 predictive geometry repair |
| HTSC | 风险门控与 task contrast 接口已存在 | 有限样本认证及在线有效性需要重新核验 |
| PushT visual MPC | 10 月 15 日三种子、24 个测试 episodes 中，基础和修复版均为 0/24 | 尚未实现稳定姿态闭合 |
| PushT paired probe | 10 月 18 日 pilot 中 adaptive-online 达到 2/8 | 值得继续验证的局部干预信号，不是稳定规划结果 |

数据来自公开执行记录与严格审稿准则。<Cite refs={["turn504293view1","turn504293view3"]}/>

特别注意：仓库执行报告还有标为 **2026 年 10 月 24 日**的后续条目，晚于本次评审日期 10 月 23 日。我不将这些条目计入当前已完成的实验成绩，也不会据此推断数据真实性；需要单独核对原始运行时间和产物来源。

现在的问题是：你已经有很多合理的物理、数学与实验解释，但它们还没有形成审稿人最看重的链条：

$$

\boxed{
\text{Spectral finding}
\;\Rightarrow\;
\text{Specific representation intervention}
\;\Rightarrow\;
\text{Better real decisions}
}

$$

**这条链是下一阶段的第一优先级。**

## 三、最可能导致 Reject 的七个重大问题

### Major Concern 1：谱分析目前主要解释模型，尚未真正改变模型

这是我最严厉的评价。

你已经研究了 covariance–predictor alignment、Schur、commutator、resolvent、transient amplification，以及：

$$

G_h^\Phi=D_\Phi A_\Phi^{h-1}B_\Phi

$$

这些对象能够帮助分析有限时域传递，但现有论文主张中还缺少一个明确的**谱驱动干预**：

假设诊断发现某个特征方向对任务很重要，却没有被 predictor 正确传递。

接下来究竟应该改变什么？

是重新加权这个方向？增加历史信息？修正 action Jacobian？还是只修改规划代价？

目前的 GDD-WM 主要依赖执行偏好训练几何，而不是由谱结构明确决定哪些模态需要修复。

**因此，谱理论和 GDD-WM 之间仍然缺少算法层面的必要联系。**

审稿人可能写：

> The spectral analysis remains largely descriptive; the proposed representation repair does not demonstrably depend on the identified spectral mechanism.

解决这个问题，不能只是把一个 spectral score 加到现有 GDD loss 里。必须让谱分析决定具体的修复子空间，并证明这种选择优于随机、PCA、普通可预测性排序和统一偏好学习。

### Major Concern 2：你研究的谱对象可能不是 planner 真正使用的对象

当前项目已经认识到：

$$

\operatorname{spec}(A)
\not\Rightarrow
DA^{h-1}B

$$

但即使知道 $DA^{h-1}B$，也不能自动判断哪个动作更好。

因为规划还依赖目标、代价函数、候选集合以及不同动作的实际执行时间。

例如，同一个 action-to-task operator，在两个不同目标下可以对应相反的动作偏好。

对于平方任务代价：

$$

J(U)=\|y_H(U)-g\|_Q^2

$$

在局部线性条件下：

$$

y_H(U)=y_H^0+\mathcal G_HU

$$

于是：

$$

J(U)=
\|y_H^0-g+\mathcal G_HU\|_Q^2

$$

它不仅由 $\mathcal G_H$ 的奇异值决定，还由任务残差 $y_H^0-g$ 与动作方向之间的关系决定。

因此，单纯报告 transfer-operator singular values、condition number 或 principal angles，仍然不足以证明真实 action ranking 改善。

你需要明确地把谱变化映射到**某个真实候选对的代价差异变化**。

这也符合现有 DA-LeWM 的重要教训：全局排序、CEM elite 排序和真实闭环成功不是同一个对象。<Cite ref="turn147534academia69"/>

### Major Concern 3：理论中存在一个需要立即修正的数学表述

我检查了仓库公开的 [`FINITE_HORIZON_TRANSFER_RISK_THEOREM_20261013.md`](https://github.com/BokaiGuo/LeWorldModel/blob/main/FINITE_HORIZON_TRANSFER_RISK_THEOREM_20261013.md)。

它在构造有限时域误差分解时定义：

$$

g_j=\|DA^j\|,\qquad s_j=\|D\|\rho^j

$$

并写道，对于 normal $A$，当 $\rho$ tight 时有 $g_j=s_j$。<Cite ref="turn432663view1"/>

这个等式一般不成立。

例如：

$$

A=\begin{bmatrix}0.9&0\\0&0.1\end{bmatrix},
\qquad D=[0,1]

$$

这是 normal 矩阵，但当 $j=1$ 时：

$$

g_1=0.1,\qquad s_1=0.9

$$

因此正确的是：

$$

g_j\leq s_j

$$

对于 normal $A$，这足以保证所定义的 $q_j=(g_j-s_j)_+$ 为零，但不能声称两者相等。

这是一个局部表述错误，不会自动推翻该误差上界；不过，作为以谱理论为核心的投稿，这种数学细节必须非常严格。

更大的理论问题是：当前 transfer-risk theorem 依赖 externally valid residual bounds 和局部适用条件。它没有自动解决这些误差界怎样从有限真实动作干预中可靠估计的问题。

所以不能仅凭定理形式就宣称获得了实际 learned world model 的 planning certificate。

### Major Concern 4：HTSC 的有限样本校准代码仍有需要修复的地方

我核查了公开源码 [`task_contrast.py`](https://github.com/BokaiGuo/LeWorldModel/blob/main/le-wm-main/experiments/decision_benchmark/task_contrast.py)。

当前 `calibrate_contrast_bound()` 计算每个 calibration root 的最大 pairwise contrast error，再使用：

$$

k_{\rm current}=\lceil n(1-\alpha)\rceil

$$

取顺序统计量。<Cite ref="turn432663view3"/>

如果目标是标准 split-conformal 的有限样本边际覆盖，应该使用：

$$

\boxed{k_{\rm valid}=\lceil(n+1)(1-\alpha)\rceil}

$$

当 $k_{\rm valid}>n$ 时，标准的分布无关构造需要使用无限阈值等处理，而不是把最大观测误差当成具有指定覆盖率的有限上界。

例如，20 个 calibration roots、目标覆盖率 90% 时：

- 当前实现取第 18 个顺序统计量；
- 标准构造取第 19 个顺序统计量。

在连续、可交换、无并列值的理想条件下，对应覆盖率分别为：

$$

18/21\approx85.7\%,\qquad
19/21\approx90.5\%

$$

这是实质性差别。

此外，当前主要是 pairwise `certify_pair()` 接口，并没有完成整个 CEM candidate set、修复后重新校准和在线许可后实际风险的全部验证。

**这一项应当作为立即修复的 P0 阻断问题。**

### Major Concern 5：PointMaze 的正结果仍可能来自 action-bank shortcut

公开执行报告说明，当前 indexed canonical-action readout 通过训练 roots 建立固定候选动作摘要，并按照 `candidate_id` 查表。它在相同固定候选协议下取得了 +0.1090 的离线 regret reduction。<Cite ref="turn969976view5"/>

问题是：

**即使 world-model rollout 没有正确预测动作后果，固定 action bank 的标识与目标方向也可能提供有用的任务统计信息。**

所以必须加入四种负控制：Bank-only、No-dynamics、Shuffled-successor、Unseen-bank。

尤其重要的是完全未见的连续动作，以及真正 CEM 产生的新候选。

如果这些测试中，谱修复方案没有明显优于只使用动作编码和简单 task scorer 的方法，就不能将其称为 world-model predictive dynamics improvement。

### Major Concern 6：谱结构不具备天然的表示坐标不变性

对于任意可逆 latent 变换：

$$

z'=Tz

$$

相应的线性局部实现变化为：

$$

A'=TAT^{-1},\quad
B'=TB,\quad
D'=DT^{-1}

$$

那么：

$$

D'(A')^{h-1}B'=DA^{h-1}B

$$

任务动作传递保持不变。

但是普通欧氏距离、某些 covariance alignment、非正规性范数与未指定 metric 的谱诊断，可能随这种坐标变化而改变。

你此前已经意识到 gauge-aware geometry，但这必须进入**实际方法与实验**，而不只是附录中的数学说明。

我会要求一个很直接的测试：对同一个 world model 做多组可逆 latent reparameterization，保持实际预测的物理输出不变，检查新诊断与修复决策是否保持一致。

如果只需改变 latent 坐标标尺，某个模式就从“值得修复”变为“不值得修复”，它很难被称为可靠的决策物理机制。

### Major Concern 7：PlanWAM 之外，还有三篇工作进一步挤压创新空间

我认为尤其需要正视下面三篇：

| 工作 | 已经研究到哪里 | 我们不能再单独宣称什么 |
|---|---|---|
| <Entity category="other" value="A Control Theory of Predictability in Latent World Models" disambig="You et al. 2026 control-theoretic world-model analysis"/> | 从 planner-reachable 分布、预测代价差异、off-manifold divergence 和 non-normal spectral amplification 解释控制失败 | “我们首次用谱和非正规性解释 world model 规划错误” |
| <Entity category="other" value="The Intervention Gap in Latent World Models" disambig="Vakalis 2026 matched-intervention latent world model audit"/> | 直接审计真实与模型的动作效应，并在 LeWM 中观察到任务方向旋转和幅值放大 | “我们首次发现真实动作效应与 imagined dynamics 可以不一致” |
| <Entity category="other" value="Do-JEPA: From Masking to Intervention in Latent World Models" disambig="Resani and Shi 2026 intervention-supervised JEPA"/> | 使用同根动作干预的 latent difference 训练，研究动作作用位置与传播，并改善视觉世界模型的动作效果预测 | “我们首次通过真实动作对比监督改善 world-model dynamics” |

这些均有公开原文。<Cite refs={["turn301435view2","turn301435view0","turn301435view1"]}/>

其中 *The Intervention Gap* 尤其危险：它已经在 LeWorldModel 上研究 task-direction rotation 和 gain mismatch，与你原来的 action-to-task transfer 诊断非常接近。

所以现在真正值得争取的新颖性必须进一步收缩为：

**能否利用具有明确任务含义的谱子空间，选择性修复具体的反事实动作误差，并证明这种谱选择比普通动作对比监督或普通决策偏好学习更加有效？**

这是我认为最有机会形成实质性独特贡献的问题。

---

## 四、PlanWAM 和 Do-JEPA 的实验，实际上揭示了一个更深的研究机会

我认为这两篇论文放在一起，比单独阅读 PlanWAM 更有启发。

### 现象 A：未来 latent 越接近，不意味着规划越好

PlanWAM 的附录 Table 17 显示，当仅训练 prior world model、冻结下游规划器与评分器时，即使未来 latent 的 cosine similarity 达到 0.96，PDMS 也只有 93.0。

完整系统需要进一步适配 scorer、planner 和其他模块，才能达到 93.8。Table 16 也显示，把未来特征相似度从 0.93 提高到 0.95，规划分数反而由 93.8 略降到 93.7。<Cite ref="turn969976view3"/>

这说明，即使 teacher future 是 planning-shaped 的，也不能认为每个 token、每个表征方向都同等值得精确蒸馏。

### 现象 B：真实动作效应预测更准确，也不意味着规划更好

Do-JEPA 的附录 Planning Study 更直接。

作者报告，在一个 LeWM PushT 预训练规划器上，直接用配对动作效应损失微调，虽然 effect RMSE 相对对照降低约 29.1%，规划成功率却从 94% 降到 54%。

随后使用带有中心化结构和多步 rollout-preservation 约束的 adapter，能够保留原有约 94% 的规划成功率，并获得部分动作效应误差改善，但没有证明规划成功率超过原始基线。<Cite ref="turn457347view0"/>

这一对结果尤其值得研究。

它说明：

$$

\text{Better effect prediction}
\not\Rightarrow
\text{Better planning}

$$

不仅是因为某些动作效应没有被正确学到，也可能因为修正错误的方向、幅度或方式，破坏了 planner 已经能够利用的预测结构。

### 这启发我们提出一个更锋利的问题

> **Which errors in the action-conditioned predictive spectrum actually change the planner's preferred action, and how can these errors be corrected without destroying the useful predictive structure?**

中文：

**哪些动作条件预测谱中的误差真正改变了规划器的动作选择？怎样只修复这些误差，同时保留原本正确的预测动力学？**

这就是我希望 LeWorldModel 研究的新 insight。

不再追求将整个 latent 预测得更像，也不再追求让所有动作 effect 都更准确，而是：

**针对 planner 当前可能混淆的动作，在真正影响决策的有限时域模态上修正预测。**

这与 PlanWAM 的任务导向未来表示形成一个有意思的递进关系：

PlanWAM 证明规划目标能够塑造表示；我们的目标是进一步找出这种塑造应该发生在**哪些可辨识的动力学子空间**，以及什么情况下改变这些子空间确实能够改变真实规划后果。

---

## 五、我建议的新方法：Intervention-Calibrated Decision Spectral Repair

我建议暂时将方法称为 **IC-DSR（Intervention-Calibrated Decision Spectral Repair）**。

它可以发展为 LeWorldModel 统一论文的真正技术核心。

其思想是：

**用有限真实干预估计哪些动作—任务模态被当前视觉世界模型错误传播，再利用受约束的低秩更新修复这些模态，而不是无差别地优化整个 latent 空间。**

### 5.1 从普通 latent spectrum 转向 Decision-Transfer Spectrum

保留你已有的有限时域局部模型：

$$

\eta_{t+1}=A_\Phi\eta_t+B_\Phi a_t

$$

$$

y_t=D_\Phi\eta_t

$$

但对完整动作序列 $U=(a_t,\ldots,a_{t+H-1})$，使用完整的 Markov parameter block：

$$

\mathcal G_H^\Phi=
\begin{bmatrix}
D_\Phi A_\Phi^{H-1}B_\Phi&
D_\Phi A_\Phi^{H-2}B_\Phi&
\cdots&
D_\Phi B_\Phi
\end{bmatrix}

$$

再引入两个必须预先声明的对象：

- $Q_g$：目标条件化的任务输出度量；
- $\Sigma_{\mathcal C}$：候选动作变化的协方差或合法支持尺度。

定义任务加权动作响应算子：

$$

\boxed{
\mathcal K_H(h,g,\mathcal C)
=
Q_g^{1/2}
\mathcal G_H^\Phi
\Sigma_{\mathcal C}^{1/2}
}

$$

然后进行 SVD：

$$

\mathcal K_H=U\operatorname{diag}(\sigma_1,\ldots,\sigma_r)V^\top

$$

这个算子的意义与原来的 covariance 或 predictor spectrum 不同。

它刻画的是：

**哪些候选动作方向，经过有限时域传播，能够影响当前任务关心的输出方向。**

例如在 PushT 中：

| Latent / physical mode | 普通预测目标可能关心什么 | 任务传递需要判断什么 |
|---|---|---|
| 背景纹理或稳定颜色 | 高可预测性 | 是否与真实动作效果无关 |
| 推杆位移 | 易观测、易预测 | 是否对物体姿态有实际作用 |
| 物体角度变化 | 可能低幅度、受接触影响 | 是否决定最终姿态闭合 |
| Contact-mode transition | 不连续、依赖历史 | 是否改变后续动作的方向和效力 |
| 未观测摩擦参数 | 单帧可能无法读取 | 是否需要历史或探测才能辨识 |

这里特别注意：

**我们不是最大化 $\sigma_i$，也不是简单让高奇异值方向得到更多权重。**

大的 predicted gain 可能恰恰是错误放大的结果。

我们真正需要的是模型预测与真实干预的**任务响应谱一致性**。

### 5.2 引入真实与预测之间的 Decision-Spectral Mismatch

使用同根动作干预估计真实任务响应：

$$

\mathcal G_H^*
\approx
\frac{\partial\mathbb E[y_H\mid h,\operatorname{do}(U)]}
{\partial U}

$$

以及模型的预测响应：

$$

\widehat{\mathcal G}_H^\theta
=
\frac{\partial\widehat y_H^\theta(h,U)}
{\partial U}

$$

实际实验不必形成庞大的完整 Jacobian。可以通过选定方向的有限差分、Jacobian-vector products 和配对动作测试估计所需子空间。

然后构造：

$$

\boxed{
\mathcal E_H
=
Q_g^{1/2}
(\widehat{\mathcal G}_H^\theta-\mathcal G_H^*)
\Sigma_{\mathcal C}^{1/2}
}

$$

$\mathcal E_H$ 是一个比单独 eigenvalue 更直接与物理决策相关的谱误差对象。

其 SVD 能揭示：

- 哪些任务输出方向发生了错误传递；
- 哪些候选动作方向激发这些错误；
- 错误是增益放大、衰减，还是方向旋转；
- 哪些错误方向在 planner 的实际候选分布中会被访问。

这也与 *The Intervention Gap* 报告的 rotation / excess gain 现象相接，但我们必须进一步证明：**识别这种误差后，确实能够有选择地修复它。**

### 5.3 进一步引入 Decision Leverage：不是所有谱误差都影响动作选择

这是整个方法最值得研究的地方。

设当前任务代价为：

$$

J(y,g)=\|y-g\|_Q^2

$$

给定两条候选动作 $U_i,U_j$，在局部一阶近似下，算子失配对两者相对任务代价的影响，与以下量有关：

$$

\left|
\nabla_yJ^\top
(\widehat{\mathcal G}_H-\mathcal G_H^*)
(U_i-U_j)
\right|

$$

对谱误差做分解：

$$

\mathcal E_H=P\operatorname{diag}(s_k)V^\top

$$

可以进一步分析每个失配模态的决策影响，其绝对贡献具有如下形式：

$$

\boxed{
L_k^{ij}
=
s_k
\left|\left\langle \bar q,p_k\right\rangle\right|
\left|\left\langle
v_k,\Sigma_{\mathcal C}^{-1/2}(U_i-U_j)
\right\rangle\right|
}

$$

其中 $\bar q$ 是任务代价梯度在 $Q$ 度量下的相应加权表示。

这个量同时涉及三件事：

**谱误差大小 × 任务敏感性 × 当前候选动作是否激发它。**

这比仅比较 covariance principal angles 或整体 $\|\mathcal E_H\|_F$ 更接近 planner 的实际选择机制。

它是一个待验证的局部近似对象，而不是精确 regret 恒等式。尤其在任务梯度接近零时，平方代价的二阶项与

$$

\mathcal G_H^\top Q\mathcal G_H

$$

仍可能决定动作排序，不能忽略。

对于存在接触切换的系统，还应将不同模式下的有限动作差分和模式转移分别估计，不能将接触不连续性强行套入同一个光滑 Jacobian。

### 5.4 这个新谱对象为什么比旧指标更合理？

最简单的比较如下。

| 指标 | 能回答什么 | 不能独立回答什么 |
|---|---|---|
| Latent covariance eigenvalues | 哪些表示方向变化大 | 是否可控、与目标有关 |
| Predictor eigenvalues | 哪些动力学模态衰减或增长 | 动作是否能激发这些模态 |
| Commutator / Schur / pseudospectrum | 动力学几何耦合与瞬态放大 | 实际任务动作排序是否错误 |
| Controllability / observability Gramians | 哪些状态模式可由动作影响并被任务读出 | Learned model 的对应响应是否与真实世界一致 |
| **Intervention-calibrated Decision Spectrum** | 哪些有限时域任务响应模态被真实模型错误传递 | 仍需要校准才能声明统计可靠性 |

需要承认：controllability Gramian、observability Gramian、Hankel singular values 和 balanced truncation 都是经典控制理论。

**因此，单纯构造一个任务加权 Gramian 并不新。**

可能成立的新贡献在于：针对冻结视觉世界模型、有限真实干预、部分可观测状态以及 planner-reachable 候选，建立可估计的任务响应谱，再用它指导训练前修复选择和实际有限时域规划。

另一个很重要的优点是：如果 $Q_g$ 定义在真实任务输出空间，并且 $A,B,D$ 随 latent 坐标变换一致变化，那么上述输入—输出响应算子保持不变。

这样可以让主要决策谱指标具备比任意 latent 欧氏几何更明确的坐标意义。

---

## 六、怎样让谱分析真正改进表示学习？

这是 Contribution 2，也应该是论文最核心的算法创新。

我建议引入：

### Spectrally Targeted GDD-WM

不是像普通 GDD 那样，对所有可信动作偏好统一蒸馏，而是先识别**高决策影响、可修复、受动作支持的误差子空间**。

训练流程应如下：

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="md" padding={3} align="center">
    **Frozen visual encoder + pretrained dynamics**
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center">
    **Action-balanced interventions + Real/Predicted spectral audit**
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center">
    **Identify decision-critical error modes**
    <text size="xs" color="secondary">Task sensitivity · Spectral mismatch · Action support</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **Information present**
        <text color="secondary" size="xs" textAlign="center">Low-rank metric or dynamics repair</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **Information missing**
        <text color="secondary" size="xs" textAlign="center">History, probe, or abstain</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box background="surface-secondary" radius="md" padding={3} align="center">
    **Spectrally constrained GDD student**
    <text color="secondary" size="xs">Match important action effects · Preserve useful dynamics</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center">
    **Independent HTSC calibration + RGB-only MPC**
  </box>
</box>

### 6.1 方法上的具体更新

在 frozen backbone 上加入受限 residual adapter：

$$

\widehat z_{t+1}
=
F_0(z_t,a_t)
+
R_\psi(\eta_t,a_t)

$$

其中 $\eta_t$ 是经过状态辨识的历史表示。

关键是：并非全部 residual 都可以自由修改，而应利用已确定的决策关键子空间限制其作用。

设 $P_{\rm crit}$ 表示经过独立干预估计、通过稳定性检验的任务关键子空间投影，则可先测试：

$$

R_\psi=P_{\rm crit}R_\psi

$$

以及更灵活的低秩更新。

但这个限制并不总是正确。如果任务相关信息完全不存在于当前 latent 或该投影无法捕捉需要的 dynamics，强制投影会损害模型。因此 B 的可观测性与可修复性审计必须先通过。

### 6.2 损失函数

我建议：

$$

\boxed{
\mathcal L_{\rm SD\text{-}GDD}
=
\mathcal L_{\rm pred}
+
\lambda_1\mathcal L_{\rm decision}
+
\lambda_2\mathcal L_{\rm spectral}
+
\lambda_3\mathcal L_{\rm preserve}
}

$$

其中：

$$

\mathcal L_{\rm spectral}
=
\sum_{r,H}
\left\|
W_{r,H}^{1/2}
\left(
\widehat{\mathcal K}_{r,H}
-
\widetilde{\mathcal K}_{r,H}^{\rm real}
\right)
\right\|_F^2

$$

$W_{r,H}$ 依据训练阶段估计的决策模态重要性确定，并且在最终测试之前冻结。

$\widetilde{\mathcal K}^{\rm real}$ 由合法训练干预估计，不能由模型自己预测的结果构造自我监督标签。

$\mathcal L_{\rm decision}$ 监督真实动作对的任务代价差异；$\mathcal L_{\rm preserve}$ 约束模型保留原来的事实预测、多步动力学和未被选中子空间的稳定性。

这里我会非常强调公平性：

如果这种损失没有胜过用相同干预数据训练的普通 action-effect loss、Do-JEPA-style pairwise loss 和普通 GDD，那么“谱选择”就没有证明独立价值。

### 6.3 一个必须成功的机制实验

你需要做一个 **Spectral Mode Intervention Study**。

固定同一 pretrained world model，使用相同训练数据和相同训练预算，比较：

| 方法 | 修改方式 | 要证明什么 |
|---|---|---|
| PCA-top-k | 保留高方差 latent 模态 | 普通表示能量是否足够 |
| Predictability-top-k | 保留易预测模态 | 预测性是否足够 |
| Gramian / balanced modes | 保留经典可控、可观测模态 | 经典控制表征是否足够 |
| Random rank-k | 随机低秩更新 | 是否仅因参数增加而受益 |
| Do-JEPA-style effect learning | 统一学习配对动作效果 | 普通干预监督能做到什么 |
| D-JEPA / Naive GDD | 统一学习任务偏好 | 普通决策几何能做到什么 |
| **Decision-spectral GDD** | 按真实任务响应误差与决策影响选择模态 | 谱识别是否真正带来额外收益 |

尤其要加一个因果对照：

**随机旋转或置换被选中的错误模态，保持相同参数量、训练预算和监督数量，观察规划收益是否消失。**

还要做一致的 latent gauge transformation，检验方法是否只对某种坐标表示有效。

只有这一组实验成立，谱分析才真正从论文的背景理论变成独立方法贡献。

---

## 七、B 和 C 应该怎样融入，而不是成为两个额外工程模块？

### B：状态辨识用于回答“当前谱误差是否可以修复”

我建议沿用前面讨论过的历史、动作支持和状态混叠框架，但让所有判断服务于 Decision Spectrum。

例如 PushT 中，如果两个接触状态具有相同 frozen latent，却对同一动作产生相反的角度变化，那么单独调整 predictor 中某个谱方向无法同时恢复两种响应。

应当先检查：增加合法历史是否能够让两种状态在决策相关表征中被区分。

若可以，使用 History-conditioned Spectral Repair；若不能，但允许安全探测，则测试是否能通过短动作获得辨识信息；否则应保留不确定判断或拒绝，而不是强行训练更复杂的 residual。

B 的主要验收指标不应是 reconstruction accuracy，而应是：

$$

\text{History}
\rightarrow
\text{Reduced action-effect ambiguity}
\rightarrow
\text{More reliable spectral repair}

$$

如果历史变长只改善 MSE，却没有降低真实 action contrast error，就不应称之为 decision-state recovery。

### C：HTSC 用于回答“谱修复后的结果是否真的可用”

使用修复后的 frozen student，在独立 calibration roots 上计算完整候选集合的 cost contrast residual。

修复前后需要分别重新校准 HTSC，并比较：

$$

\text{Real decision regret}

$$

$$

\text{Useful decision coverage}

$$

$$

\text{Closed-loop task success}

$$

三个量。

在适当的有限样本条件下，HTSC 的 pairwise bound 可以支持候选 regret 上界，但不能直接宣称每个状态都有确定性可靠性，也不能假定 CEM 自适应候选与旧校准分布天然一致。

最终我希望看到：

**谱选择性修复降低真实决策误差，HTSC 因此能在相同错误风险预算下，允许执行更多有效动作。**

这是 A、B、C 真正互补的证明。

---

## 八、建议把研究收敛成三项正式 Contribution

| 贡献 | 科学问题 | 关键新意 | 决定性证据 |
|---|---|---|---|
| **C1：Decision-Critical Spectral Modes** | 哪些视觉 latent 动力学模态真正控制任务动作排序？ | 任务条件化、动作支持感知、真实干预校准的有限时域响应谱 | 匹配 MSE、ATM、普通谱指标后，仍能预测 held-out ranking error |
| **C2：Spectrally Targeted Predictive Repair** | 能否只修复错误的决策关键模态，保留正确动力学？ | 谱选择指导 GDD-WM、低秩几何与动力学适配、历史可辨识性路由 | 同等监督预算下，明显优于普通 GDD、Do-JEPA-style 监督和非谱低秩更新 |
| **C3：Calibrated Spectral Decision Transfer** | 修复后何时可以信任模型规划？ | 修复后独立校准、候选 regret 控制与 RGB-only 选择性 MPC | 独立闭环中的 regret–coverage–success 联合收益 |

这是我更推荐的三个 Contribution。

其中 ATM 应作为 C1 的动作辨识参照和支持审计，GDD-WM 作为 C2 的核心学习机制，HTSC 作为 C3 的可靠性机制。

谱分析则不是第四个模块，而是**贯穿三个贡献的统一理论对象**。

---

## 九、最值得投入资源的三个决定性实验

我不建议一开始就投入资源复现 PlanWAM 在 NAVSIM 上的全部训练。

更合理的是先用 LeWorldModel 已有环境证明新的物理机制，再去自动驾驶场景展示迁移价值。

### Experiment 1：低能量但高决策价值的模态是否被学错？

先从 synthetic system + PointMaze 开始。

设计两类视觉或动力学因素：

- 一类高方差或高可预测，但与任务动作效果无关；
- 一类表示能量较低，但能显著改变目标推进、碰撞或动作排序。

这里要严格控制动作支持，不能让任务模态低能量只是因为训练中根本没有对应动作激励。

训练多个世界模型，比较普通表示指标、谱诊断、ATM 和新的 Decision-Spectral Mismatch。

真正需要证明：

**新的谱对象比 latent PCA、prediction MSE 和 ATM 更能提前识别哪些模型会在真实候选动作上选错。**

之后做谱定向干预：只修复错误的任务响应方向，检验是否降低新 roots 上的 regret。

不要只报告某个新的谱指标与 success 的相关性。

### Experiment 2：谱选择性修复能否解决 Do-JEPA 暴露的矛盾？

这是我认为最有 Oral 潜力的实验。

Do-JEPA 已经表明，改善 latent action effects 并不保证规划成功。

你的实验应该直接针对这个难题：

**在相同的执行干预训练数据下，我们的方法是否能够比普通 effect learning 更可靠地将动作效果改进转化为规划收益？**

测试矩阵：

| 结果维度 | 必须检验的变化 |
|---|---|
| Realized action effect | 是否正确解释真实动作变化 |
| Imagined action effect | 预测方向、幅值、重要谱子空间是否改善 |
| Multi-step prediction | 是否保持原有预测能力 |
| Elite ranking | CEM 最终竞争候选是否减少排序反转 |
| Unseen actions | 未见连续候选上是否仍然成立 |
| Closed-loop MPC | 实际目标达成和 regret 是否改善 |

如果你的方法只把 effect RMSE 降得更低，规划成功率却没有提高，那么核心研究仍然失败。

相反，如果普通 action-effect learning 对全部模态优化后损害控制，而你的方法能够识别并只修复影响决策的模态，同时保持预测结构并提高真实成功率，这将是非常有说服力的新结果。

### Experiment 3：跨环境的谱机制是否具有预测性？

我建议采用三类环境：

| 环境 | 验证机制 | 预期作用 |
|---|---|---|
| PointMaze / TwoRoom | 有限时域动作响应、障碍与目标几何 | 最干净的决策谱验证 |
| Reacher | 连续动作、局部控制雅可比、不同目标 | 验证谱选择不是固定 action bank shortcut |
| PushT | 接触模式、历史混叠、非线性角度响应 | 验证 B 的状态辨识和修复边界 |

PushT 不要在前期强行承担全套主结果。如果没有先复现可靠的原始视觉控制基线，且 privileged oracle 在相同接口下也无法稳定完成任务，就应该先修正控制协议。

整个研究真正强的实验设计，是**同一模型在不同任务、动作支持与物理模式下，谱方法能够提前判断哪些模态有用、哪些模态错误、哪些无法可靠恢复。**

最终再迁移到自动驾驶的候选轨迹选择，例如 NAVSIM / HUGSIM，比较 PlanWAM-style 历史未来表示与 action-conditional spectral transfer 的效果。

在自动驾驶中，还要区分记录轨迹式评估与其他交通参与者会响应自车动作的闭环仿真，不能以普通闭环成功代替反事实机制证据。

---

## 十、建议的研究执行顺序及停止条件

| 优先级 | 具体工作 | 通过标准 | 不通过时的处理 |
|---|---|---|---|
| **P0** | 修正 HTSC quantile、理论中 normal operator 等式；复现原始 planner 与 candidate oracle | 理论和代码正确，存在真实可改善的 planner gap | 先修协议 |
| **P1** | 构造任务加权真实/预测有限时域响应谱 | 谱失配对 held-out regret 有独立解释力 | 不继续堆谱指标 |
| **P2** | 完成 synthetic mode-selection 与 nuisance intervention | 能预先识别任务关键低能量模态 | 重新检查任务度量和动作支持 |
| **P3** | 训练 rank-matched spectral GDD adapter | 优于 random/PCA/Do-JEPA-style/naive GDD，保持预测能力 | 不能宣称谱驱动修复 |
| **P4** | Student 冻结后重新校准 HTSC | 有效且非空洞的 regret–coverage 表现 | 不称为 certificate |
| **P5** | RGB-only PointMaze/Reacher MPC | 新 episodes 上真实 regret 和 success 改善 | 检查候选可达性及规划目标 |
| **P6** | PushT 接触与跨模型验证 | 可辨识区域得到改善，不可辨识区域有正确边界 | 缩小论文主张 |
| **P7** | PlanWAM-style / 自动驾驶外部验证 | 新方法在新任务上有机制或实证增量 | 保留为独立的后续研究 |

我会优先投入 P0–P3，而不是一次性推进所有七个阶段。

还有一个必须遵守的原则：**修复模态的选择不能使用最终评估 roots 的真实 outcomes。**

否则在训练后发现哪个方向有效，再把它称为谱诊断提前发现，会形成严重的因果归因与测试集泄漏问题。

统计上也需要按独立 root、world-model training seed 和 environment 区分推断单位。同一 root 的数百个候选对并不是数百个独立实验。

---

## 十一、什么程度才可能达到 Oral / Best Paper？

### 达到扎实主会论文

至少需要证明：谱选择性方法在独立的 learned visual world models 上，确实改善真实 action ranking，并能够进入有效的 RGB-only MPC。

它需要超过同等数据、参数与计算预算下的普通对比训练、普通几何学习和强控制基线。

### 达到 Oral 竞争力

我会要求三件事同时成立。

**第一，新的谱机制具有预测能力。** 它不是修复成功之后的事后解释，而能在修复前判断哪些动作模态将导致决策错误，以及哪些模态值得修复。

**第二，谱选择具有真正的因果作用。** 在相同训练资源下，修复正确的 task-sensitive modes 明显优于随机修复、高方差修复或统一干预监督，并且这种收益跨模型和环境成立。

**第三，真实控制收益不是拒绝动作换来的。** 修复后的 world model 经过重新校准，在相同风险约束下取得更高有用覆盖率、更低 regret 和更好的闭环表现。

如果能再构造一个被多个预训练模型共享的真实失败模式，并通过任务响应谱准确预测它，我认为论文就开始具有鲜明辨识度。

### 达到 Best Paper 潜力

需要比以上更进一步。

真正有分量的科学结果应该是一个**可检验的 Decision-Spectral Sufficiency / Repairability Boundary**：

给定视觉表征、动作支持、任务指标与有限数据预算，我们不仅知道哪些任务模态需要被保留，还能说明哪些模态能够通过受限制的表示或动力学适配恢复，哪些因观测混叠或动作支持不足而无法可靠修复。

理论上，你可以从局部有限时域系统出发，研究：

$$

\mathcal G_H^*
-
\widehat{\mathcal G}_H

$$

在任务敏感、动作可激发子空间上的误差怎样影响真实 candidate regret。

例如，若在声明的动作支持范围中有：

$$

\left\|
Q^{1/2}
(\mathcal G_H^*-\widehat{\mathcal G}_H)
\Sigma_{\mathcal C}^{1/2}
\right\|_2
\leq\epsilon

$$

并且候选动作满足：

$$

\left\|
\Sigma_{\mathcal C}^{-1/2}U
\right\|\leq R

$$

则局部线性响应造成的任务输出误差至多为 $\epsilon R$，还需要加上基准轨迹误差和非线性 remainder。

对平方任务代价，可以进一步得到：

$$

|J_H(U)-\widehat J_H(U)|
\leq
2\|\widehat y_H(U)-g\|_Q e_H(U)
+
e_H(U)^2

$$

这里 $e_H(U)$ 是包含基准预测、算子失配和适用区域误差的有效上界。

如果对所有候选同时有效，就能进一步得到候选集合 regret 控制。

这类代数推论本身属于经典控制和决策误差分析，并不能直接成为新理论贡献。

**真正新的定理必须回答：怎样利用有限真实干预，估计这个任务关键响应误差；怎样证明特定修复类可以降低它；何时因状态信息不足而无法降低；以及这些结论怎样转化为有限数据下非空洞的规划许可。**

如果理论和跨环境结果能够共同建立这样的边界，这项工作才可能改变视觉世界模型的评价与训练方式。

---

## 十二、我会怎样重写最终论文标题和故事？

我不建议继续使用太宽泛的 *Transfer-Sufficient Diagnostics* 作为最终方法论文的唯一定位。

现在最推荐的是：

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" size="xs" weight="medium">首选研究标题</text>
  <title size="xl">Which Latent Modes Matter for Planning? Identifying and Repairing Decision-Critical Dynamics in Visual World Models</title>
  <divider color="subtle"/>
  <text color="secondary" size="xs" weight="medium">更偏理论的版本</text>
  **From Predictive Spectra to Decision-Sufficient World Models**
  <divider color="subtle"/>
  <text color="secondary" size="xs" weight="medium">更偏方法的版本</text>
  **Intervention-Calibrated Spectral Repair for Reliable Visual World-Model Planning**
</box>

这三个题目都比单独罗列 ATM、GDD-WM、HTSC 更能够表达同一个研究核心：

**世界模型中哪些动力学模态真正决定任务成功，以及我们怎样利用它们改善规划？**

---

## 最终模拟审稿结论

| 维度 | 当前判断 | 提升为 Oral 所需的关键变化 |
|---|---|---|
| 研究问题 | 很重要，但概念覆盖范围过广 | 聚焦决策关键谱模态 |
| 原创性 | 多个组成部分已被先行工作覆盖 | 谱识别必须产生不可替代的修复机制 |
| 理论 | 有价值的反例和条件性误差界 | 有限干预下可检验的谱充分性/可修复性边界 |
| 方法 | GDD-WM 与谱诊断仍未真正闭合 | Spectrally targeted predictive repair |
| 实验 | 离线正结果与明确负结果并存 | 新鲜候选、真实干预和 RGB-only 闭环 |
| 可信度 | 证据治理较好，校准尚需修正 | 修复后独立认证和非空洞的风险—效用收益 |

**我目前仍会给 Reject。** 但这次的研究建议和之前有一个重要区别：

我不再建议仅把整个项目定位为“判断 world model 哪些错误可以修复”。

我更建议把你最初的谱分析优势重新放回研究中心，但让它直接服务于**模型学习与实际决策**。

最终应该争取证明：

> **The modes most strongly preserved by predictive learning are not necessarily the modes most consequential for action selection. By identifying decision-critical input–output dynamics from limited interventions, we can selectively repair predictive representations and improve planning without sacrificing their reliable dynamics.**

如果只做出第一句，那只是又一篇谱诊断论文，而且已有研究十分接近。

如果能证明第二句——**谱分析不仅发现失效，而且直接指导模型训练，在真实 learned world models 上稳定改善规划**——才真正形成了 LeWorldModel 相对于 PlanWAM、D-JEPA、Do-JEPA 和普通 task-aware representation learning 的独特价值。

**我建议下一步就把全部主力集中在一个决定性实验：PointMaze 上，同样的数据与训练预算，基于真实干预估计的 Decision-Spectral Repair，能否比普通 GDD 和普通 action-effect learning 更准确地修复未见动作排序，并在真正 RGB-only MPC 中获得可重复的收益。**

如果这个实验成立，理论、方法和整篇论文的研究主线就有了值得冲击 Oral 的核心支点。

-----

> 参考SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model的insight继续改进我们的研究但是不要做后训练

我会先核对 SuperMimic 的核心方法，特别是它怎样利用**可微世界模型**做控制，以及哪些机制不依赖额外训练。然后把这些 insight 映射到 LeWorldModel 的谱分析、动作传递和 MPC 上，设计一个**冻结 encoder、predictor 和全部学习参数，不做 GDD-WM 后训练**的新方案，并指出它与 PlanWAM 的实质区别。

# 借鉴 SuperMimic，重构 LeWorldModel：不做后训练，让冻结世界模型的梯度真正改善规划

**我建议把研究方向从“修复世界模型参数”转为“判断冻结世界模型的哪些梯度值得用于规划”。**

这里有一个比普通谱分析更有潜力的科学问题：

> **A world model can predict future states accurately while providing misleading action gradients. Which gradients can be trusted to improve real decisions?**

中文：**世界模型可能预测得很准确，却给规划器提供错误的动作优化方向。我们能否通过有限时域谱分析，判断哪些梯度可信，并在不更新任何模型参数的情况下改善真实规划？**

我建议将这个方向暂命名为：

<title size="lg">Spectral Gradient Trust for Frozen Visual World-Model Planning</title>

简称 **SGT-MPC**。

## 一、SuperMimic 真正值得借鉴的三个 Insight

<Entity category="other" value="SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model" disambig="Liao et al., CoRL 2026 Oral, differentiable world-model humanoid control"/> 于 2026 年 10 月 20 日发布预印本，已被 CoRL 2026 接收为 Oral。<Cite ref="turn128459view0"/>

<row align="start" gap={3}>
  <AsyncImage query="SuperMimic Scaling Humanoid Whole-Body Control differentiable world model Berkeley training diagram human robot G1" aspectRatio="4:3" maxWidth="168px"/>
  <box flex="1" gap={1}>
    论文的关键不是开发了另一种复杂的机器人表征，而是利用一个共享的可微动力学模型，在大量不同参考动作之间复用动力学知识，并通过短时域 imagined rollouts 提供解析梯度。

    原论文报告，在特定步行、跑步与冲刺实验中，相比同样使用世界模型 rollout 的 PPO-WM，解析梯度训练达到相同目标回报所需的 imagined steps 约少 64 倍。<Cite ref="turn685775view1"/>
  </box>
</row>

我提炼出三个可以迁移的思想：

| SuperMimic 的 Insight | 对 LeWorldModel 的启发 |
|---|---|
| **同一个动力学模型服务大量目标** | 冻结一个 RGB encoder + predictor，面向不同目标和任务规划，不再为每个任务训练 adapter |
| **可微预测不仅提供未来状态，还提供动作梯度** | 从分析预测误差转向分析 $\nabla_U\widehat J$ 是否忠实于真实环境 |
| **短时域可微 rollout 更实用** | 利用短时梯度优化、重规划及任务条件化 horizon，而不是盲目加长预测时域 |

但必须纠正一个容易产生的误解：

**SuperMimic 本身并不是无需训练的方法。** 它每次迭代都会更新世界模型、policy 和 critic；只在一次 policy update 内冻结世界模型参数。它使用的还是仿真器状态和动力学上下文，而非 LeWM 的纯 RGB latent 输入。<Cite ref="turn128459view3"/>

所以我们不照搬它的训练循环。

我们的研究约束更严格：

$$

\boxed{
\theta_E,\theta_F\ \text{frozen},
\qquad
\text{No fine-tuning, no adapters, no policy training}
}

$$

**唯一在线优化的变量是当前动作序列 $U$，不是神经网络参数。**

这意味着原来需要后训练的 GDD-WM，要从这篇论文的主方法中移除；ATM 与 HTSC 可以保留为不修改模型权重的诊断和决策机制。

## 二、必须正视：可微世界模型规划已经有直接竞争工作

仅仅将 LeWM 的 CEM 改成反向传播优化，已经不够新。

2026 年 ICML 至少有两篇直接相关论文：

- <Entity category="other" value="GRASP: Parallel Stochastic Gradient-Based Planning for World Models" disambig="Psenka et al., ICML 2026 differentiable planning with virtual states"/> 已经提出对视觉世界模型使用并行随机梯度优化，通过 virtual states 和 soft dynamics constraints 改善长时域搜索；而且已有基于 LeWM 的公开实现。<Cite refs={["turn520594view0","turn520594view3"]}/>
- <Entity category="other" value="Dream-MPC: Gradient-Based Model Predictive Control with Latent Imagination" disambig="Spieler and Behnke, ICML 2026"/> 已经结合候选轨迹、世界模型梯度优化、风险或不确定性调节以及跨步复用优化结果。<Cite ref="turn520594view1"/>

另外，<Entity category="other" value="World Modeling through Spectral Alignment" disambig="Molin et al., October 2026 SpecWM spectral representation learning paper"/> 已经把谱对齐发展成世界模型的训练目标，改善了多个机器人任务的规划。<Cite ref=""/>

因此，我们不能以“第一次用梯度规划”或“第一次通过谱结构改善视觉世界模型”为创新点。

**真正还值得攻克的是：在冻结模型上，如何识别并限制错误的动作梯度，让可微规划利用可靠方向，而不放大模型的虚假物理响应。**

这是与 SuperMimic 相呼应、但又属于 LeWorldModel 自己的研究问题。

## 三、方法核心：从预测谱转向“可信动作梯度谱”

LeWM 原始规划流程是：

$$

z_t=E(o_t),\qquad
\hat z_{t+1}=F_\theta(z_t,a_t)

$$

通过 CEM 搜索动作序列，让预测终点的 latent 尽可能接近目标图像的 embedding。<Cite ref="turn520594view4"/>

SuperMimic 给我们的启发是：既然 $F_\theta$ 可微，除了采样候选，我们还可以获得：

$$

\boxed{
\widehat g_H(U)
=
\nabla_U\widehat J_H(U)
}

$$

其中：

$$

\widehat J_H(U)
=
\|\widehat z_{t+H}(U)-z_g\|_2^2

$$

这表示：按照冻结模型的预测，当前整段动作序列应该向什么方向调整，才能更接近目标。

但有一个重要区别：

**预测代价的梯度，并不必然是真实环境任务代价的下降方向。**

如果对真实任务代价定义：

$$

g_H^*(U)=\nabla_UJ_H^*(U)

$$

则可能出现：

$$

\left\langle
\widehat g_H(U),g_H^*(U)
\right\rangle<0

$$

也就是世界模型自信地建议一个更新方向，而真实任务沿该方向反而变得更差。

我建议将它定义为 **Decision-Gradient Transfer Failure**。

### 3.1 用有限时域 Jacobian 建立谱对象

对冻结的非线性 predictor，在某条候选 rollout 周围线性化：

$$

A_k=
\frac{\partial F_\theta}{\partial z_k},
\qquad
B_k=
\frac{\partial F_\theta}{\partial a_k}

$$

完整动作序列到预测终点 latent 的导数为：

$$

\mathcal G_H^\theta
=
\frac{\partial\widehat z_{t+H}}{\partial U}

$$

它的块结构为：

$$

\mathcal G_H^\theta=
\begin{bmatrix}
A_{H-1}\cdots A_1B_0&
A_{H-1}\cdots A_2B_1&
\cdots&
B_{H-1}
\end{bmatrix}

$$

然后定义：

$$

\boxed{
K_H
=
Q_g^{1/2}
\mathcal G_H^\theta
\Sigma_U^{1/2}
}

$$

这里 $Q_g$ 是预先指定的目标代价度量；如果沿用 LeWM 的 latent Euclidean goal cost，可先取 $Q_g=I$。$\Sigma_U$ 用来表示候选动作各维度的预先指定尺度。

进行奇异值分解：

$$

K_H=P\operatorname{diag}(\sigma_k)V^\top

$$

这时 $V$ 的列向量就描述了：

**哪些动作序列变化方向，在冻结世界模型中能够强烈影响目标相关的未来表示。**

但我特别反对直接选择最大的奇异值。

因为最大的响应方向也可能恰好是模型最严重的虚假放大方向。

### 3.2 从“高响应模态”进一步筛选“真实可信模态”

这是研究的新方法核心。

对于每个候选动作方向 $v_k$，估计：

$$

\widehat d_k
=
v_k^\top\nabla_U\widehat J_H

$$

并通过可重置的真实环境进行对称有限差分：

$$

d_k^*
\approx
\frac{
J_H^*(U+\delta v_k)
-
J_H^*(U-\delta v_k)
}{2\delta}

$$

然后检查两者是否方向一致。

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium">我建议明确区分三种动作模态</text>
  <grid columns={3} gap={2}>
    <grid-item>
      <box background="rgba(22,163,74,0.07)" radius="md" padding={2} gap={1}>
        <icon name="check-circle-2" color="success"/>
        <text weight="medium" size="sm">Reliable</text>
        <text size="xs" color="secondary">预测与真实方向一致，且干预证据充分</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="rgba(234,179,8,0.07)" radius="md" padding={2} gap={1}>
        <icon name="alert-triangle" color="secondary"/>
        <text weight="medium" size="sm">Uncertain</text>
        <text size="xs" color="secondary">动作支持不足、梯度幅值太小或误差界过宽</text>
      </box>
    </grid-item>
    <grid-item>
      <box background="rgba(239,68,68,0.07)" radius="md" padding={2} gap={1}>
        <icon name="x-circle" color="danger"/>
        <text weight="medium" size="sm">Misleading</text>
        <text size="xs" color="secondary">模型预测与真实任务效果经常反向</text>
      </box>
    </grid-item>
  </grid>
</box>

这些判断应基于独立的干预开发数据形成，而不是在测试结果出来之后才给模态分类。

对高维动作空间，不必显式计算完整 Jacobian，可以使用自动微分的 Jacobian-vector product、vector-Jacobian product 和随机化 SVD。

**不训练新模型，只分析已冻结模型对动作输入的导数。**

需要强调一个限制：若模型使用的是 latent distance，而真实任务代价使用物理距离，它们的梯度尺度未必一致。这时应首先比较方向、符号和实际下降效果，不能未经统一代价定义便将梯度数值差当成有效误差界。

---

## 四、提出一个真正无需后训练的规划算法

我建议采用：

<title size="md">Spectrally Trusted Differentiable MPC</title>

方法结构：

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
    **Frozen RGB encoder + Frozen action-conditioned predictor**
    <text color="secondary" size="xs">No network updates</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center" gap={1}>
    **CEM / previous MPC solution generates initial action sequences**
    <text color="secondary" size="xs">Keep global candidate exploration</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center" gap={1}>
    **Differentiate short imagined rollouts**
    <text color="secondary" size="xs">BPTT with respect to actions U only</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center" gap={1}>
    **Spectral trust filter**
    <text color="secondary" size="xs">Reliable action modes · Support · Horizon · Step size</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
    **Optimize the action sequence, not model parameters**
    <text color="secondary" size="xs">Gradient refinement with a bounded trust region</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center" gap={1}>
    **Candidate decision gate**
    <text color="secondary" size="xs">Accept · Return to CEM · Use fallback</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box background="surface-secondary" radius="md" padding={3} align="center" gap={1}>
    **Execute first action · Observe RGB · Replan**
  </box>
</box>

### 4.1 为什么保留 CEM，而不是全部换成梯度下降？

因为 SuperMimic 的主要优化对象是 policy 参数，且其学习过程会反复更新世界模型，使模型逐渐适应当前策略访问的状态。

你的约束更严格：模型始终冻结。

在冻结 world model 下，直接沿着错误梯度反复优化，可能比普通 CEM 更严重地利用模型缺陷。

所以我更建议：

**CEM 负责探索多个候选区域，可靠梯度只负责局部精修。**

这类似 GRASP 和 Dream-MPC 中的混合规划思想，因此必须通过谱可信度筛选体现新意，不能仅把 CEM + GD 作为主要贡献。<Cite refs={["turn520594view0","turn520594view1"]}/>

### 4.2 谱过滤后的动作更新

令：

$$

\widehat g=\nabla_U\widehat J_H(U)

$$

取通过可信度检验的动作方向集合 $\mathcal T$。

定义：

$$

P_{\mathcal T}
=
\sum_{k\in\mathcal T}v_kv_k^\top

$$

最简单的更新是：

$$

\boxed{
U^{(m+1)}
=
U^{(m)}
-
\eta P_{\mathcal T}\widehat g
}

$$

再施加动作幅值、速度和合法支持区域约束。

不过第一版我甚至更推荐对每个模态分别缩小梯度，而不是二值保留：

$$

s_k=
\bigl(|v_k^\top\widehat g|-\epsilon_k\bigr)_+

$$

$$

\boxed{
\Delta U=
-\eta
\sum_{k\in\mathcal T}
s_k\,
\operatorname{sign}(v_k^\top\widehat g)v_k
}

$$

$\epsilon_k$ 表示该方向经过独立验证的梯度误差容限。

它的直观含义是：**对真实效果没有足够证据的梯度分量，不让它推动动作优化。**

这里的 $\epsilon_k$ 不能来自模型自报的 confidence，也不能凭空设定为一个具有统计保证的值。它需要独立的方向性干预数据；如果没有有效误差界，就只能把这个算法作为经验可信度筛选方法，而不能称其为认证梯度下降。

---

## 五、这可以产生一个非常清晰的理论结果

相比之前研究预测误差上界，我认为这次应该围绕**何时可微世界模型的梯度能够真正改善真实任务代价**建立理论。

假设真实任务代价 $J^*(U)$ 在当前可信区域内具有 $L$-Lipschitz gradient，并且对于选中的正交方向 $v_k$，满足：

$$

\left|
v_k^\top
(\nabla J^*(U)-\nabla\widehat J(U))
\right|
\leq\epsilon_k

$$

记：

$$

s_k=
\left(
|v_k^\top\nabla\widehat J(U)|-\epsilon_k
\right)_+

$$

使用上面的谱梯度更新。

如果整段更新留在该光滑区域和合法动作集合内，并取：

$$

0<\eta<\frac2L

$$

则由 smoothness inequality 可以得到：

$$

\boxed{
J^*(U+\Delta U)
\leq
J^*(U)
-
\eta
\left(1-\frac{L\eta}{2}\right)
\sum_{k\in\mathcal T}s_k^2
}

$$

这个结果说明：

**只要所选梯度方向的误差界有效，且存在足够大的可信梯度分量，模型梯度就能产生真实代价下降。**

证明的关键是，每个保留的方向满足：

$$

\operatorname{sign}
(v_k^\top\nabla\widehat J)
(v_k^\top\nabla J^*)
\geq s_k

$$

而 $\{v_k\}$ 正交，因而一阶下降项能够压过受控的二阶项。

但要精确限定：

这是一条基于正确梯度误差界、共同代价定义和局部光滑性的**充分条件**，不是无条件的 learned world-model 保证。

特别是在 PushT 接触切换处，梯度可能不存在、出现跳变，或者无法由局部线性化正确描述。

这种情形应切换回采样式优化、降低规划时域，或使用允许的 fallback，而不是继续沿错误导数下降。

已有 ICLR 2026 关于可微仿真器梯度估计器的研究，也明确指出非连续动力学会造成一阶梯度偏差。<Cite ref=""/>

因此，你的理论创新不能仅是这个 smoothness 推论；真正困难且可能有新意的是：**在冻结 RGB 世界模型中，以有限、可重置的真实动作干预构造有效且不空洞的方向性误差界，并让它指导实际规划。**

---

## 六、如何保留原先 A、B、C 的研究积累？

我建议改成下面三项。

| 旧研究资产 | 不做后训练后的新职责 | 是否更新模型参数 |
|---|---|---|
| **A：ATM + Spectral Transfer** | 衡量真实和预测动作梯度的方向、增益、奇异子空间及任务敏感性 | 否 |
| **B：History / Identifiability** | 利用历史残差、动作支持和模式切换证据，决定当前能否信任梯度 | 否 |
| **C：HTSC** | 对优化后的动作进行候选后悔值与许可审计，必要时返回 CEM 或 fallback | 否 |
| **原 GDD-WM** | 不再进行偏好蒸馏；改为 test-time action-space geometric preconditioning | 否 |

其中 B 特别需要注意：

如果原始 predictor 只接受当前 $z_t$，那么额外收集历史并不能自动让它变成 history-conditioned dynamics model。

在不修改任何模型权重的条件下，历史首先只能用于决定**哪些状态属于已验证的可信区域**。

例如，可以使用：

$$

r_t=
E(o_t)-
F_\theta(E(o_{t-1}),a_{t-1})

$$

作为局部模型失配的一个信号，再结合最近的动作、接触事件或其他合法视觉变化。

但小的 $r_t$ 不保证动作 Jacobian 正确，因此它应是风险特征，而不是独立的梯度证书。

如果单帧表示真的丢失了控制所需的隐状态信息，纯粹的后处理无法恢复不存在的信息。这个边界应当保留为研究结果。

### HTSC 是否还算后训练？

严格区分两件事：

- **不允许**：GDD student training、LoRA、adapter 微调、policy training、critic training、学习新的 task scorer。
- **可以作为无参数统计步骤**：在独立数据上计算固定的误差阈值与覆盖区间，不更新 encoder、predictor 或额外神经网络。

后者可以用于评估和校准，但必须如实称为 *nonparametric calibration*，而不是声称系统完全没有使用后续数据。

如果你连任何形式的独立校准也不希望使用，那么仍然可以实现谱梯度 MPC，不过就不能在缺少额外理论假设时宣称具有有限样本的真实决策保证。

---

## 七、必须做的决定性实验

我建议用一个实验直接回答整篇论文的主问题：

**为什么某些冻结世界模型使用梯度规划比 CEM 更差，而我们能否提前识别并避免这些错误？**

### Experiment 1：Gradient Fidelity 与真正的任务效果

在 PointMaze 和 Reacher 上，使用相同根状态和候选动作，通过环境重置执行：

$$

U+\delta v_k,\qquad U-\delta v_k

$$

分别比较：

$$

\widehat d_k
\quad\text{与}\quad
d_k^*

$$

报告：

- 梯度方向一致率；
- 任务条件化梯度余弦相似度；
- 奇异模态上的虚假放大与方向旋转；
- 真实任务代价变化；
- 这些量是否比 MSE、ATM、普通 spectrum 更能预测优化失败。

还要把**预测动力学错误**与**latent goal metric 本身错误**分开。

即使使用真实 successor，若 latent goal distance 仍然给出错误动作偏好，那么冻结规划器单靠修正 predictor 梯度也无法解决全部问题。

### Experiment 2：不训练任何参数，仅替换规划器

在同一冻结 checkpoint 和相同计算预算下进行对照。

| Planner | 作用 |
|---|---|
| LeWM CEM | 原始基线 |
| Vanilla BPTT / Gradient MPC | 检验直接使用动作梯度的效果 |
| CEM + GD | 排除普通混合优化已经足够的解释 |
| GRASP | 已有的强可微规划基线 |
| Gradient clipping / damping | 排除普通梯度稳定化的收益 |
| PCA / Largest-singular-mode filtering | 检验只根据谱能量筛选是否有效 |
| Random rank-matched filtering | 检验是否只是减少更新维度 |
| **SGT-MPC** | 使用真实干预验证的可信模态优化 |

尤其应在 CEM elites 上检验。

因为普通梯度方法可能显著降低模型预测代价，却把真实世界的动作排序变得更差。

### Experiment 3：PushT 的接触分层

我建议分为：

- Free motion；
- Contact onset；
- Stable contact；
- Sliding / rotation；
- Contact-mode transition。

具体比较每类中的模型梯度是否真实可靠。

最有价值的结果不一定是所有状态都可以梯度优化。

如果你能证明：

**光滑运动区域里的梯度相当可靠，但接触切换后的某些谱方向严重失真；SGT-MPC 能正确限制这些方向，并在闭环中优于无差别 BPTT。**

这本身就比再增加一个 contact head 更有科学解释力。

### 最重要的三个结果指标

<box gap={2}>
  <row align="center" justify="between">
    <box flex="1" gap={1}>
      **Real action-selection regret**
      <text color="secondary" size="xs">真正执行后的动作选择错误是否减少</text>
    </box>
    <icon name="trending-down" color="success" size="lg"/>
  </row>
  <divider color="subtle"/>
  <row align="center" justify="between">
    <box flex="1" gap={1}>
      **Closed-loop success**
      <text color="secondary" size="xs">RGB-only MPC 是否真正完成更多任务</text>
    </box>
    <icon name="trending-up" color="success" size="lg"/>
  </row>
  <divider color="subtle"/>
  <row align="center" justify="between">
    <box flex="1" gap={1}>
      **Planning compute**
      <text color="secondary" size="xs">相同 GPU 时间与模型调用预算下是否更有效</text>
    </box>
    <icon name="timer" size="lg"/>
  </row>
</box>

SuperMimic 的重要优势是计算效率，因此我们也不能只报告成功率。

但其 64 倍 imagined-step 效率来自政策训练对照，不能直接转化为 SGT-MPC 的预期加速倍数。我们的实际延迟必须单独测量。

---

## 八、三项 Contribution 应这样重新组织

我建议整篇论文以 **Frozen Differentiable World Models: Which Gradients Can Be Trusted to Plan?** 为主问题。

| Contribution | 科学内容 | 最低成立证据 |
|---|---|---|
| **C1：Decision-Gradient Transfer Sufficiency** | 证明预测准确与真实动作梯度可靠是不同性质；提出任务与时域条件化的谱诊断 | 在 MSE / ATM 匹配后，仍能预测梯度优化的真实失败 |
| **C2：Training-Free Spectral Gradient Planning** | 不修改任何世界模型参数，只选择可信动作谱方向进行有限时域优化 | 优于 vanilla GD、CEM+GD、GRASP 及随机谱过滤，且计算成本可接受 |
| **C3：Selective RGB-Only Control** | 根据状态辨识、动作支持和决策风险，决定何时使用梯度、何时退回采样规划 | PointMaze / Reacher / PushT 的真实闭环收益和有效失败边界 |

这与 PlanWAM、SpecWM、SuperMimic 的区别也更加清楚：

| 方法 | 主要研究对象 | 是否需要训练或更新学习参数 |
|---|---|---|
| PlanWAM | 什么样的未来表示对规划最有用 | 需要训练规划导向表征与规划组件 |
| SpecWM | 如何通过谱相似结构学习更适合控制的表征 | 需要训练世界模型 |
| SuperMimic | 如何借助可微世界模型高效训练控制策略 | 世界模型与策略在训练循环中更新 |
| GRASP | 如何有效地对世界模型执行梯度规划 | 可使用预训练模型，无须为每个新任务更新世界模型 |
| **我们的 SGT-MPC** | 冻结模型中哪些动作梯度忠实于真实环境，如何只利用可信梯度规划 | **不进行任何模型或策略后训练** |

因此，我们相对于 GRASP 的新颖性尤其必须用实验建立：**不是优化器更复杂，而是有物理证据支持的梯度方向筛选能够抑制 world-model gradient exploitation。**

---

## 九、最短落地路线

| 阶段 | 工作 | 通过标准 |
|---|---|---|
| P0 | 冻结 LeWM，确保 autograd 能贯通动作序列；复现 CEM 和 GRASP 基线 | 纯动作梯度计算正确，模型权重 hash 始终不变 |
| P1 | PointMaze / Reacher 真实动作方向有限差分 | 找到可复现的 predicted-vs-real gradient mismatch |
| P2 | 构造任务条件化 Jacobian spectrum 与可信模态筛选 | 能在独立 roots 上提前预测 vanilla GD 的真实失败 |
| P3 | 实现 CEM + spectrally filtered BPTT | 同等预算下真实 regret 低于 CEM+GD |
| P4 | 加入独立无参数许可校准与 fallback | 风险与放行率具有实用权衡 |
| P5 | 推进 RGB-only MPC 与 PushT 接触模式实验 | 在未见 episodes 上获得可重复控制收益 |

针对现有仓库，最值得新增的不是训练脚本，而是三个独立组件：

`action_gradient_audit.py`：比较冻结模型的梯度与可重置真实环境的方向性动作效果。

`spectral_trust_mpc.py`：执行有限时域 Jacobian 谱分析、可信梯度过滤和动作序列更新。

`gradient_transfer_eval.py`：完成独立 roots、真实 regret、CEM/GRASP 对照及闭环结果审计。

这些是建议的新组件名称，不代表仓库目前已经存在。

原有 `GDD-WM D0–D2` 蒸馏工作应从该方案的主实验路径中暂时移出，避免它违背不做后训练的研究约束。现有 HTSC、task contrast 和谱分析代码则可以作为测试基础。仓库公开的统一计划也明确承认，当前离线 candidate bank 尚不足以直接支持完整的视觉在线规划验证。<Cite ref="turn955295view1"/>

---

## 十、最关键的研究取舍

我认为现在有两个非常不同的目标，不能混在一起。

**目标一：让冻结 LeWM 的规划性能更好。**

这可以通过 GRASP、CEM + GD、短时域优化、fallback 等方法实现，但即使成功，也未必是一个新的重要科学发现。

**目标二：发现为什么冻结视觉世界模型的动作梯度有时可信、有时会误导规划，并用有限时域谱结构预测与控制这种失效。**

这是更适合 ICLR / ICML / NeurIPS 研究的方向。

如果最终实验能够出现下面的现象，就很有说服力：

- 两个冻结模型 prediction MSE 和 ATM 相近，却具有不同的真实动作梯度方向一致性；
- 这些差异能够被任务条件化的梯度谱提前识别；
- 无差别 BPTT 可能利用模型错误，而可信模态筛选减少了真实决策后悔值；
- 这种改善不需要任何 fine-tuning、adapter、policy 或 critic training；
- 同样的机制在连续控制和接触切换区域都具有可解释的有效范围。

**我的最终建议是：借鉴 SuperMimic 的可微预测能力，但不要复制它的训练方法。**

真正属于 LeWorldModel 的新 Insight 应该是：

> **Differentiability gives a planner gradients. It does not make those gradients physically trustworthy. We study the spectral structure of decision-relevant gradient transfer and use it to plan reliably with frozen visual world models.**

换句话说：

**SuperMimic 告诉我们“世界模型的梯度能够帮助控制”；LeWorldModel 下一步应该回答“什么情况下，这些梯度真的值得相信”。**

这既保留了你最初的谱分析、ATM 和 HTSC 研究优势，也不再依赖 GDD-WM 后训练，而且比直接把 CEM 替换成梯度优化，更有希望形成独立的理论与方法贡献。

-----

> 不要把研究往可信的什么什么东西去推进，应该基于我提到的这些研究，结合他们的insight和我们的insight去做研究，提供一种新的方法论出来才能达到oral/best paper级别研究

你这个纠正是关键：之前我把研究重心放在“可信梯度、认证和拒绝规划”上，实际上偏离了你想做的**新规划方法论**。这次我会保留“不做后训练、冻结世界模型”的约束，以 PlanWAM 的任务导向未来、SuperMimic 的可微动力学，以及 LeWorldModel 的谱与动作传递分析为起点，重点研究：能否直接改变**规划器理解和搜索未来状态的方式**，而不只是给现有优化器加一个过滤器。

# 我建议重新定义 LeWorldModel：不是让 World Model 更适合规划，而是让规划适应 World Model 的动力学结构

你说得对。之前我不断把研究推向“可信梯度、风险认证、选择性拒绝”，这实际上把 **planning methodology** 变成了可靠性评估研究，偏离了你真正想做的事情。

结合 PlanWAM、SuperMimic，以及你在 LeWorldModel 中已经积累的谱分析、有限时域动作传递、Schur/commutator 和视觉表征几何，我现在更推荐一个新的核心命题：

> **World models should not only predict where the future goes; their action-induced geometry should determine how a planner searches, composes, and reaches future states.**

中文：

**世界模型不应该只是给规划器提供未来状态和代价。世界模型内部由动作产生的可达几何，应当直接决定规划器如何构造子目标、组织动作序列，以及搜索长时域未来。**

这个方向的区别很大。

PlanWAM 让规划任务塑造 future representation；SuperMimic 利用可微世界模型中的梯度改进控制。我们则进一步提出：

**不修改任何网络权重，直接从冻结视觉世界模型的有限时域动力学中，构造一种用于规划的动态坐标系。**

这不是普通的 gradient descent，也不是在原有 CEM 上增加谱过滤。

我建议把新方法暂称为：

<title size="lg">Reachability-Shaped Imagination: Spectral Action Geometry for Frozen Visual World Models</title>

简称 **RSI Planning**（暂定研究名称）。

## 一、三项研究真正可以组合的 Insight

| 研究基础 | 应继承什么 | 我们进一步改变什么 |
|---|---|---|
| <Entity category="other" value="PlanWAM" disambig="2026 planning-shaped future representations autonomous driving"/> | 未来信息必须服务于规划，而非仅仅提高预测相似度 | 不再通过训练塑造 future latent，而是在规划时从动作可达结构动态生成规划几何 |
| <Entity category="other" value="SuperMimic" disambig="Liao et al. 2026 differentiable humanoid world model"/> | 一个共享动力学模型可以服务大量任务，可微 rollout 提供高效优化信息 | 不训练 policy/critic；用梯度构造动作空间中的可达模态和新动作程序 |
| **LeWorldModel** | 谱分解、输入—输出传递、非正规性、有限时域动作效应与 latent geometry | 把这些谱对象转化为规划坐标、子目标与动作搜索，而不是只做事后诊断 |

PlanWAM 的原文强调 planning-shaped future representation；SuperMimic 的主要效率收益来自共享可微动力学和通过 imagined rollout 传播的梯度。不过 SuperMimic 本身仍会训练世界模型、policy 和 critic，我们不会复制这部分。<Cite refs={["turn101246academia60",""]}/>

同时，有三项竞争工作必须避开：

**GRASP** 已经研究对冻结或预训练世界模型进行梯度式、collocation 风格的规划；**Hierarchical Planning with Latent World Models** 已经研究多时间尺度子目标；**SAGE** 已经研究可达 latent subgoal 引导的候选动作生成。<Cite refs={["","","turn265773academia70"]}/>

因此，单独提出“梯度 MPC”“可达子目标”“多层规划”都不够新。

我们需要的是一个将它们连接起来、但现有方法并未直接提供的机制：

$$

\boxed{
\text{Spectral action geometry}
\;\longrightarrow\;
\text{Reachable subgoals and action programs}
\;\longrightarrow\;
\text{Differentiable planning}
}

$$

**谱结构不是用来判断规划器是否应该行动，而是直接生成规划器的搜索空间。**

## 二、真正的新 Insight：任务目标不应该直接等同于下一步要追逐的 latent 目标

现有 LeWM-style MPC 经常从视觉目标出发：

$$

z_t=E(o_t),\qquad z_g=E(o_g)

$$

然后搜索动作序列 $U$，最小化：

$$

\widehat J(U)=\|\widehat z_{t+H}(U)-z_g\|^2

$$

但它忽略了一个几何事实：

**目标虽然存在于 latent space，不代表当前动作能够朝这个目标的所有方向移动。**

例如，在 PushT 中，推杆当前接触位置可能只能使物体平移，却无法有效旋转。此时继续优化目标角度对应的 latent distance，可能只会让 CEM 不断重复无效推动。

真正正确的动作可能是先移动推杆、改变接触位置，然后再推动物体旋转。

这引出一个新的方法论：

> **The useful future is not just the future closest to the goal. It is the future that creates the best action-reachable geometry for subsequent planning.**

也就是说：

**一个好的中间状态，不一定立即接近目标，但应该让后续动作更容易实现目标。**

这不是一般的子目标学习。我们的子目标不由另一个网络预测，而是由冻结世界模型的**有限时域可控结构**计算出来。

## 三、方法核心一：把冻结世界模型变成局部可达几何

### 3.1 从单个动作梯度升级为整段动作的可达算子

设冻结视觉世界模型为：

$$

\hat z_{k+1}=F_\theta(\hat z_k,a_k)

$$

对长度为 $H$ 的完整动作序列：

$$

U=[a_t^\top,\ldots,a_{t+H-1}^\top]^\top

$$

定义：

$$

\boxed{
\mathcal G_H(z_t,U)
=
\frac{\partial \widehat z_{t+H}(U)}{\partial U}
}

$$

对于非线性世界模型，在一条名义动作轨迹附近线性化：

$$

A_k=\frac{\partial F_\theta}{\partial z_k},
\qquad
B_k=\frac{\partial F_\theta}{\partial a_k}

$$

则：

$$

\mathcal G_H=
\begin{bmatrix}
A_{H-1}\cdots A_1B_0&
A_{H-1}\cdots A_2B_1&
\cdots&
B_{H-1}
\end{bmatrix}

$$

这一步继承你已有的有限时域 input–output transfer 研究。

但我们不再把 $\mathcal G_H$ 主要用作离线指标，而是把它当成当前动作空间的局部坐标生成器。

引入任务输出度量 $Q$ 和动作尺度矩阵 $R$：

$$

\widetilde{\mathcal G}_H
=
Q^{1/2}\mathcal G_H R^{-1/2}

$$

进行奇异值分解：

$$

\boxed{
\widetilde{\mathcal G}_H
=P\Sigma V^\top
}

$$

此时：

- $V$：当前哪些**动作序列组合**能产生不同的未来变化；
- $P$：这些动作在**目标相关的未来表示**中能到达哪些方向；
- $\Sigma$：在声明的局部尺度下，动作变化与未来变化之间的敏感程度。

重要的是，这个分解随状态、目标、动作序列和 horizon 改变。

因此，我们不再认为 visual latent space 具有一套对所有规划任务都适用的静态几何。

而是提出：

$$

\boxed{
\text{Planning geometry}
=
\text{Task geometry}
+
\text{Finite-horizon action reachability}
}

$$

如果暂时没有合法的 task-output readout，第一版可以直接使用预训练模型原本的 goal-latent distance，即 $Q=I$。这不需要额外训练，但也意味着方法仍受原始 latent metric 的表达能力限制。

### 3.2 计算 Reachable Goal Projection

设当前名义 rollout 终点与目标之间的偏差为：

$$

r=Q^{1/2}
(z_g-\widehat z_{t+H}(\bar U))

$$

将它分解成：

$$

r=r_\parallel+r_\perp

$$

其中：

$$

r_\parallel=P_rP_r^\top r

$$

代表在当前有限时域线性化下可以通过动作扰动影响的部分。

而：

$$

r_\perp=(I-P_rP_r^\top)r

$$

代表目前的一阶动作响应无法直接覆盖的部分。

这是第一个算法变化。

**原始 LeWM 让 MPC 直接追逐整个 $r$。我们的规划器先构造局部可达子目标，再决定应该怎样改变当前可达方向。**

需要注意：$r_\perp$ 只是当前轨迹附近的**一阶不可达部分**，不是全局不可达结论。

这恰好引出下一步方法。

---

## 四、方法核心二：无需训练 Critic，直接从谱结构计算 Spectral Cost-to-Go

这是我认为整个研究最值得尝试的新技术对象。

SuperMimic 通过短时域可微 rollout 和 critic 相结合，解决较长时间尺度上的控制问题。其 critic 需要训练。<Cite ref=""/>

你的研究既然不允许后训练，我们可以考虑另一种形式：

**通过当前冻结世界模型的局部可达几何，直接构造一个解析的、任务条件化的短时域 cost-to-go。**

给定状态 $z$ 与目标 $z_g$，定义：

$$

\boxed{
V_{\mathrm{reach}}(z,g)
=
\min_{\delta U}
\left[
\left\|r-\widetilde{\mathcal G}_H\delta U\right\|^2
+\lambda\|\delta U\|^2
\right]
}

$$

这里 $\delta U$ 已经用 $R$ 进行归一化。

这个优化问题在局部线性、无额外动作约束的情况下有解析解：

$$

\delta U^*
=
(\widetilde{\mathcal G}_H^\top
\widetilde{\mathcal G}_H+\lambda I)^{-1}
\widetilde{\mathcal G}_H^\top r

$$

代入 SVD，可得到：

$$

\boxed{
V_{\mathrm{reach}}
=
\|r_\perp\|^2
+
\sum_{i=1}^{r}
\frac{\lambda}{\sigma_i^2+\lambda}
(p_i^\top r)^2
}

$$

这个公式的含义非常直观：

一个目标状态的价值，不再仅取决于它距离当前状态有多远，还取决于**需要怎样的动作努力才能朝目标移动**。

<box border radius="lg" padding={3} gap={2}>
  <text weight="medium">比较两种规划评价方式</text>
  <grid columns={2} gap={3}>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <text color="secondary" weight="medium" size="xs">EUCLIDEAN GOAL COST</text>
        <svg viewBox="0 0 170 120" width="100%" xmlns="http://www.w3.org/2000/svg">
          <rect x="1" y="1" width="168" height="118" rx="8" fill="none" stroke="currentColor" strokeOpacity=".12"/>
          <path d="M20 78 L145 28" stroke="#94a3b8" strokeWidth="1.4" strokeDasharray="4 4"/>
          <path d="M20 78 L118 78" stroke="#0d9488" strokeWidth="3" strokeLinecap="round"/>
          <path d="M20 78 L70 58 L100 44 L145 28" fill="none" stroke="#d97706" strokeWidth="2.5" strokeDasharray="5 3"/>
          <circle cx="20" cy="78" r="5" fill="#0d9488"/>
          <circle cx="145" cy="28" r="6" fill="#d97706"/>
          <text x="17" y="97" fill="currentColor" fontSize="10">Start</text>
          <text x="122" y="18" fill="currentColor" fontSize="10">Goal</text>
          <text x="47" y="72" fill="#0d9488" fontSize="9">Reachable</text>
        </svg>
        <text color="secondary" size="xs">把目标方向当作可直接优化的方向，即使当前动作只能影响其中一部分。</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} gap={1}>
        <text color="secondary" weight="medium" size="xs">REACHABILITY-SHAPED VALUE</text>
        <svg viewBox="0 0 170 120" width="100%" xmlns="http://www.w3.org/2000/svg">
          <rect x="1" y="1" width="168" height="118" rx="8" fill="none" stroke="currentColor" strokeOpacity=".12"/>
          <path d="M20 78 L118 78 L145 28" fill="none" stroke="#0d9488" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
          <path d="M118 78 L145 28" fill="none" stroke="#d97706" strokeWidth="1.5" strokeDasharray="4 3"/>
          <circle cx="20" cy="78" r="5" fill="#0d9488"/>
          <circle cx="118" cy="78" r="5" fill="#0d9488"/>
          <circle cx="145" cy="28" r="6" fill="#d97706"/>
          <text x="17" y="97" fill="currentColor" fontSize="10">Start</text>
          <text x="89" y="97" fill="currentColor" fontSize="10">Subgoal</text>
          <text x="122" y="18" fill="currentColor" fontSize="10">Goal</text>
        </svg>
        <text color="secondary" size="xs">优先到达能够改变后续动作可达性的中间状态，再朝最终目标推进。</text>
      </box>
    </grid-item>
  </grid>
  <caption>概念示意：图中的两段轨迹不是实验结果；真实可达性还取决于非线性动力学和动作约束。</caption>
</box>

例如在 PushT 中，某个中间推杆位置可能距离最终目标的 latent 很远，但从那里推动物体旋转更容易。

那么它的 $V_{\mathrm{reach}}$ 可能比视觉上更接近目标、却无法有效施加转矩的位置更低。

**这就提供了一个不训练新 Critic，也不依赖任务演示的 planning value surrogate。**

但必须正视已有理论：这个二次最小化本身是经典阻尼最小二乘、逆运动学和局部最优控制的基本形式，不能单独构成论文创新。

我们真正要创造的是：把这个解析 value 用于**选择下一处可达几何，以及生成能够改变可达几何的动作程序**。

---

## 五、方法核心三：不是只利用当前可达模态，而是主动改变未来的可达模态

我认为这是整项方法最有希望形成独特 insight 的部分。

### 5.1 当前一阶动作空间不够，不代表任务做不到

设当前某种动作控制只能使 latent 沿两个方向变化：

$$

\operatorname{span}\{p_1,p_2\}

$$

但目标方向 $r$ 与这两个方向几乎正交。

传统 gradient planner 会发现：

$$

\mathcal G_H^\top r\approx0

$$

因此可能停滞。

但真正的解决方案未必是更好的梯度。

**可以先执行一段动作，改变下一时刻的动力学结构，让新的可达方向出现。**

例如在机器人操控中，移动接触点可以改变推动物体时的旋转响应。

在导航中，暂时改变朝向可以创造此前无法直接执行的运动方向。

这对应一种新的规划原则：

> **Actions can be valuable because they reshape future controllability, not only because they immediately reduce goal error.**

### 5.2 利用动作顺序产生高阶可达方向

这部分可以结合你之前研究的 commutator 思想，但必须使用正确的控制理论对象。

对于两个短动作 $a_i,a_j$，计算冻结世界模型的两种顺序：

$$

z_{ij}
=
F_\theta(F_\theta(z,a_i),a_j)

$$

$$

z_{ji}
=
F_\theta(F_\theta(z,a_j),a_i)

$$

定义：

$$

\boxed{
c_{ij}(z)=z_{ij}-z_{ji}
}

$$

它描述交换动作顺序对预测终点产生的差异。

如果两个动作交换顺序后产生几乎相同的结果，那么这种组合可能没有提供新的动态效应。

但如果交换顺序能产生明显不同的方向，就可以将这组动作视为一个候选的 **Action-Order Primitive**。

在光滑连续控制系统中，经过适当的小幅正反向动作组合，这类高阶效应与向量场的 Lie bracket 有关。

**不过，你此前研究的矩阵 commutator $[A,C]$ 与这里的控制向量场 Lie bracket 不是同一个数学对象。** 需要区分使用条件。对于 PushT 的接触切换，也不能直接假设光滑 Lie bracket 近似成立；在那里，应使用实际的离散动作顺序 rollout 进行比较。

### 5.3 为什么这种机制值得研究？

考虑经典非完整运动系统：

$$

\dot x=v\cos\theta,\qquad
\dot y=v\sin\theta,\qquad
\dot\theta=\omega

$$

当 $\theta=0$ 时，瞬时控制方向为：

$$

f_v=(1,0,0),\qquad f_\omega=(0,0,1)

$$

所以在这个姿态下，一阶控制方向不包含 $y$ 方向。

但它们的 Lie bracket 为：

$$

[f_\omega,f_v]=(0,1,0)

$$

即：通过适当的转向与前后运动组合，可以产生单次瞬时控制无法直接实现的横向位移。

这正说明：

$$

\boxed{
\text{Instantaneous reachable directions}
\neq
\text{Directions reachable through composed actions}
}

$$

Lie bracket controllability 是经典控制理论，不是我们的新发现。

**我们的新研究问题，是如何从一个已经训练好的、高维视觉世界模型中，自动发现这些由动作组合产生的有效方向，再利用它们生成规划动作，而不是预先由人设计 motion primitives。**

这比单纯对世界模型梯度做 SVD 后滤除某些方向，更像一种全新的 world-model planning methodology。

---

## 六、整合成一个算法：Spectral Reachability Atlas Planning

我建议最终算法叫 **Spectral Reachability Atlas（SRA）**。

核心不是一直在同一个 action space 中搜索。

而是将冻结 world model 在不同状态、不同 nominal rollouts 下形成的局部可达子空间，看作一系列局部图册（charts）。

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" padding={3} radius="lg" align="center" gap={1}>
    **Current RGB + Goal RGB**
    <text color="secondary" size="xs">Frozen encoder and action-conditioned predictor</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border padding={3} radius="lg" align="center" gap={1}>
    **Compute finite-horizon action Jacobian**
    <text color="secondary" size="xs">Task-weighted SVD · Locally reachable directions · Analytic spectral cost-to-go</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <grid columns={2} gap={2}>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **Progress actions**
        <text size="xs" color="secondary" textAlign="center">Optimize in reachable spectral action coordinates</text>
      </box>
    </grid-item>
    <grid-item>
      <box border radius="md" padding={3} align="center" gap={1}>
        **Reconfiguration actions**
        <text size="xs" color="secondary" textAlign="center">Change contact, orientation, or future reachable directions</text>
      </box>
    </grid-item>
  </grid>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border padding={3} radius="lg" align="center" gap={1}>
    **Connect local reachability charts**
    <text color="secondary" size="xs">Multi-chart search · Action-order primitives · Adaptive horizon</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box background="surface-secondary" padding={3} radius="lg" align="center" gap={1}>
    **Differentiable trajectory refinement**
    <text color="secondary" size="xs">Optimize only action coefficients · No model or policy updates</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="lg" padding={3} align="center">
    **Execute action · Observe RGB · Replan**
  </box>
</box>

### 6.1 第一类动作：Progress Actions

对于已有的可达奇异方向：

$$

V_r=[v_1,\ldots,v_r]

$$

将动作优化从原始高维空间：

$$

U\in\mathbb R^{mH}

$$

转换成：

$$

\boxed{
U=\bar U+R^{-1/2}V_r\alpha
}

$$

其中：

$$

\alpha\in\mathbb R^r

$$

这样优化器搜索的不是所有原始动作，而是当前真正能够影响预测任务结果的独立动作组合。

可以使用解析阻尼更新、Adam、L-BFGS 或短时域 BPTT 优化 $\alpha$。

关键是实际执行前必须通过冻结世界模型重新 rollout 完整动作，不能只依赖线性化预测。

### 6.2 第二类动作：Reconfiguration Actions

如果当前 goal residual 的大部分处于 $r_\perp$，系统不应该仅仅增加梯度下降次数。

而应主动生成能改变下一时刻可达谱的动作序列。

例如候选：

$$

U_{\mathrm{bridge}}\in
\left\{
[a_i,a_j],\
[a_j,a_i],\
U_{\mathrm{null}},\
U_{\mathrm{sample}}
\right\}

$$

其中 $U_{\mathrm{null}}$ 来自当前一阶任务响应较弱的动作方向，而 $U_{\mathrm{sample}}$ 来自保留的 CEM 或随机探索候选。

这些动作可能短期没有显著任务进展，但可能改变接触关系或动力学姿态。

对每个 bridge candidate，利用模型 rollout 得到中间状态 $z'$，重新计算该状态的 Spectral Cost-to-Go。

选择：

$$

\boxed{
U_{\mathrm{bridge}}^*
=
\arg\min_{U_{\mathrm{bridge}}}
\left[
J_{\mathrm{stage}}(U_{\mathrm{bridge}})
+
\beta V_{\mathrm{reach}}(z',g)
\right]
}

$$

其中 $J_{\mathrm{stage}}$ 包含已有任务协议允许的动作、路径、碰撞或能量代价，不需要新网络。

这一步非常重要。

**我们的规划器不仅选择离目标更近的候选，还主动选择能够改善后续可达性的中间状态。**

### 6.3 第三类机制：Spectral Horizon Expansion

有些目标不是在当前 horizon 不可达，而是动作的有效影响需要更长时间传播。

因此，可以计算不同 horizon 下的：

$$

\mathcal G_2,\mathcal G_4,\mathcal G_8,\ldots

$$

比较其任务加权可达子空间。

如果增加 horizon 能引入对目标有用的新动作方向，就扩展规划时域。

如果只是放大原有方向，而没有提升目标相关可达性，就不必盲目增长 horizon。

你以前研究的 Schur、non-normal transient amplification 和有限时域 Markov parameters，可以在这里发挥真正作用：

**它们解释为什么某些动作影响具有延迟、瞬态放大或复杂的模态耦合，并帮助选择有效的规划时间尺度。**

这样，谱分析同时控制三件事：

1. 搜索哪些动作组合；
2. 是否应该先改变动力学配置；
3. 需要在什么时间尺度上规划。

这才是一个统一的方法，而非三个并列的诊断模块。

---

## 七、相对于现有方法，我们独特在哪里？

| 方法 | 核心做法 | 我们要超越的边界 |
|---|---|---|
| PlanWAM | 训练 planning-shaped future latent | 在推理时直接构造由动作可达性决定的未来规划几何 |
| SuperMimic | 可微 world model 为 policy learning 提供梯度 | 不训练 policy，利用动力学导数生成规划坐标与动作程序 |
| GRASP | 优化虚拟状态与动作，以改善长时域 world-model planning | 搜索空间由状态相关的谱模态与可达 chart 主动构造 |
| Hierarchical WM | 学习多时间尺度模型，通过 latent subgoal 分层规划 | 不学习新的时间尺度模型，在冻结 predictor 上计算动态子目标 |
| SAGE | 学习可达子目标与动作 proposal | 不训练 proposal generator，直接计算可达子空间、action-order primitives 和解析局部价值 |
| **SRA** | **有限时域谱坐标 + 解析 cost-to-go + 高阶动作组合 + 多 chart 规划** | **同一个冻结视觉世界模型完成任务相关的动态规划几何构造** |

前面三个独立部件都能在经典控制与优化中找到对应思想。因此，**SRA 是否具有顶会级新颖性，取决于它们组合后是否解决了现有 frozen-world-model planner 不能高效解决的一类任务**，特别是必须先改变接触、姿态或可达方向才能完成目标的 non-greedy tasks。

还有一个值得强调的数学性质：如果使用一致变换的任务度量 $Q$，则 latent 可逆坐标变换 $z'=Tz$ 下：

$$

\mathcal G'_H=T\mathcal G_H,\qquad
Q'=T^{-\top}QT^{-1}

$$

因此：

$$

\boxed{
(\mathcal G'_H)^\top Q'\mathcal G'_H
=
\mathcal G_H^\top Q\mathcal G_H
}

$$

这意味着我们的主要动作空间几何能够在正确的度量变换下保持不变。

这个性质很适合与你已有的 representation gauge 分析形成联系。不过，如果只是固定使用原始 latent 欧氏距离，不能直接声称具有完整坐标不变性。

---

## 八、论文最重要的实验，不应该是“新优化器平均多几个点”

我认为你必须专门设计能够证明新 insight 的实验。

### Figure 1：为什么需要改变可达几何，而不只是优化动作？

首先在有明确非完整约束的系统中验证，比如 unicycle、带姿态的导航或受限制操控。

构造相同的初始状态、目标与冻结模型。

比较 vanilla GD、CEM、GRASP、阻尼 Gauss–Newton 与 SRA。

关键指标不是只有 success，而是：

**当当前目标方向无法通过一阶动作响应直接实现时，谁能有效生成改变未来可达方向的动作序列？**

这个实验必须展示 SRA 的 action-order / chart-transition 机制，而不是仅仅说明 SVD 降维提高了优化速度。

### Figure 2：在 RGB 世界模型上，谱构造是否真正带来规划收益？

首先用 PointMaze、Reacher，再迁移到 PushT。

做完整消融：

| 变体 | 去掉什么 | 验证什么 |
|---|---|---|
| Raw GD | 不进行谱重参数化 | 普通可微规划能力 |
| Damped Gauss–Newton | 不引入多 chart 和高阶动作 | 经典局部逆控制能达到什么程度 |
| Spectral-only | 只有 SVD 动作坐标 | 降维是否足够 |
| Reachability-value-only | 只有解析 $V_{\rm reach}$ | 新规划代价是否有效 |
| Action-order-only | 只有交换顺序的动作生成 | 高阶动作组合本身的价值 |
| No chart expansion | 不主动探索改变可达结构的中间状态 | 多 chart 是否真正必要 |
| **Full SRA** | 全部机制 | 是否产生不可替代的联合收益 |

特别建议加入两个具有真实物理意义的测试。

在 PointMaze 中，目标的直接 latent 方向被障碍阻断时，是否能先选择有利于后续路径的中间状态？

在 PushT 中，目标角度不能从当前接触点有效实现时，规划器是否能先调整接触位置，再完成旋转？

这比展示 SVD 图或者额外几个 PCA 指标有价值得多。

### Figure 3：在冻结模型与相同计算预算下，能否大幅提高长时域规划效率？

所有主要 training-free planner 使用完全相同的 world-model checkpoint，不训练新的 policy、critic、proposal network 或 adapter。

报告真实 success、任务完成步数、wall-clock planning latency、模型 forward/backward 次数，以及不同任务长度与动作维度下的性能。

如果 SRA 需要额外大量 Jacobian 和多 chart 计算，但最终只比 CEM 提升一两个百分点，它不足以支撑 Oral 级贡献。

真正有说服力的结果应该是：**在需要非贪心中间动作的任务上，SRA 用更小的搜索预算找到传统动作空间优化难以找到的动作程序。**

这也避免与 SuperMimic 的训练效率指标混淆——我们研究的是在线动作规划计算，而非 policy-training GPU-hours。

---

## 九、我建议现在真正执行的研究路径

| 阶段 | 实现内容 | 决定性问题 |
|---|---|---|
| P0 | 冻结 LeWM checkpoint，确认 action-to-latent Jacobian；复现 CEM、vanilla GD、GRASP | 是否确实存在可被规划器利用的动作响应结构？ |
| P1 | 实现局部 Spectral Cost-to-Go | 不训练 Critic，能否更好地比较中间状态？ |
| P2 | 实现 Action-Order Primitives 和多 chart 搜索 | 能否解决普通一阶优化陷入停滞的任务？ |
| P3 | 增加 horizon-dependent spectral expansion | 是否更高效地完成长时域目标？ |
| P4 | 在 PointMaze、Reacher、PushT 的 RGB-only 闭环中检验 | 方法是否真正改善视觉规划？ |
| P5 | 扩展到其他冻结世界模型和任务类别 | 这种动态可达几何是否具有跨架构价值？ |

**我会优先完成 P1 和 P2 的小型严格实验，而不是立即搭建大型统一系统。**

一个足够好的初始实验是：

在同一个冻结 PointMaze 或受限二维操控模型上，对每个状态计算局部可达谱，再比较两种中间动作选择策略——最小化当前 latent goal distance，以及最小化下一状态的 Spectral Cost-to-Go。

如果后者能够更早地选择那些短期不接近目标、但能显著改善后续可达性的动作，并且真实闭环收益成立，就有了推进完整 SRA 的科学依据。

同时我会保留一个明确的研究边界：**冻结世界模型如果完全没有编码某种关键动作的物理后果，推理时的谱方法不能凭空恢复它。** 所以你必须在多个已具备一定动作响应能力的 checkpoint 上测试，而不能把不可观测的信息缺失包装为搜索算法能够解决的问题。

---

## 十、怎样成为真正有 Oral / Best Paper 潜力的方法论？

我认为最值得证明的不是“我们用 SVD 加速了 MPC”。

而是一个更普遍的结果：

> **The geometry of a latent future is not fixed. It depends on the actions available, their temporal composition, and the horizon over which they are applied. Planning should operate over this dynamically generated reachable geometry rather than a static latent goal metric.**

它有三个层次的科学价值。

**第一层：表示视角的改变。** 从静态 task-shaped representation，转向由动作诱导的、任务相关的动态可达几何。

**第二层：规划算法的改变。** 从直接搜索动作序列，转向搜索可达谱模态、动作顺序组合和不同局部可达 chart 之间的转换。

**第三层：计算方法的改变。** 利用一个冻结可微世界模型，计算局部解析 cost-to-go，并将其用于多步非贪心规划，不需要训练专门的 policy、critic 或 subgoal generator。

其中局部最小二乘、SVD、Lie bracket 和 graph search 都有经典先例。要达到 Best Paper 竞争力，必须证明的是这些机制结合后产生的**新规划能力和可解释的计算优势**，而不是仅仅重新命名经典控制工具。

### 最终建议

我建议将主论文的三个 Contribution 改写为：

| Contribution | 核心内容 |
|---|---|
| **C1 — Action-Induced Future Geometry** | 从冻结视觉世界模型的有限时域响应谱，构造动态、任务相关的局部可达几何 |
| **C2 — Spectral Reachability Atlas Planning** | 通过解析 cost-to-go、动作顺序与 chart transition，规划能够改变未来可达性的动作程序 |
| **C3 — Training-Free Differentiable Visual Planning** | 在同一冻结模型上，以更高搜索效率完成长时域、非贪心与接触操作任务 |

这样，原先的 ATM 和谱分析可以保留为理论与机制分析基础，HTSC 不再是论文重点，GDD-WM 的后训练整条路线则从本篇方法中移出。

我现在最推荐的正式研究题目是：

<title size="lg">Beyond Planning-Shaped Futures: Learning-Free Spectral Reachability Planning with Visual World Models</title>

这里的 *Learning-Free* 仅表示不进行额外模型学习；基础视觉世界模型仍然是预训练得到的。

**如果这条路线成立，LeWorldModel 的独特贡献就不再是告诉别人世界模型什么地方不可靠，而是提出一种新的原则：利用世界模型本身的动作动力学谱，动态创造规划坐标、可达子目标和非贪心动作程序。**

这才更接近你想做的——**以谱分析和世界模型为科学基础，真正发明一种新的规划方法论。**

-----

> 有点像在是做世界模型的harness，参考一些相关的最新最接近研究的insight，结合他们的insight和我们的insight去推进研究

我觉得“世界模型的 harness”这个定位比单纯做谱诊断或再发明一个 MPC 优化器更有潜力：**模型保持冻结，研究如何把它的想象能力组织成有效的规划过程**。我会先核对最近最接近的 test-time planning、可微世界模型、分层搜索和 world-model orchestration 工作，再判断它们留下的真正方法空白，并把 LeWorldModel 的谱分析变成一个不可替代的规划机制。

# LeWorldModel 下一步：从世界模型的 Harness，走向一种新的 Planning Runtime

我认为你抓到了一个重要的方向：**一个预训练世界模型的实际规划能力，不仅取决于它预测了什么，还取决于推理时如何组织想象、搜索、动作组合和真实执行反馈。**

但我核对了最新相关研究后，发现一个必须面对的事实：

**“为世界模型设计 harness”本身已经不是一个足够新的论文贡献。**

2026 年出现了直接使用这一概念的工作，包括 <Entity category="other" value="HarnessPAI" disambig="2026 physical AI executable harness research"/> 和 <Entity category="software" value="WorldForge" disambig="world-model planning harness framework"/>。前者通过可执行程序组织机器人动作，并跨 rollout 改进程序；后者已经提供 world-model provider、动作提议、预测、评分和 CEM 的可组合运行框架。<Cite refs={["turn924988view0","turn924988view4"]}/>

因此，我建议区分两个层次：

**工程层的 World Model Harness**：连接模型、工具、规划器、执行器和实验系统。

**科学层的 World Model Planning Runtime**：研究在模型完全冻结的条件下，如何从预测动力学中**自动构造规划所需的计算结构、动作程序与搜索策略**。

前者可以成为项目的开源基础设施；后者才应该是冲 ICLR / ICML / NeurIPS Oral 的论文核心。

## 一、最新且最接近的工作，真正告诉了我们什么？

| 工作 | 已经验证的 Insight | 对我们形成的约束 |
|---|---|---|
| **PlanWAM**，2026-10 | 为规划塑造的未来表示比单纯学习未来相似性更有用 | 不能只提出 task-aware future |
| **SuperMimic**，2026-10 | 同一个可微动力学模型能够服务大量控制目标，短时想象与解析梯度具有很高计算价值 | 不能只提出 BPTT 或可微 MPC |
| **GRASP**，ICML 2026 | 虚拟状态、随机化与软动力学约束改善长时域视觉规划 | 不能只提出更好的梯度优化器 |
| **Hi-LeWM**，2026-07 | 冻结低层 LeWM 后做分层规划可以有效，但高层子目标分布与低层执行能力不匹配会失败 | 不能只提出 latent subgoal hierarchy |
| **HarnessPAI**，2026-09 | 不重训底层动作模型，通过程序组织与跨次执行改进可以显著增强物理任务完成能力 | 不能把程序执行框架当作主要创新 |
| **WorldForge** | 已有现成的 world-model planning harness 与 provider 抽象 | 不能靠搭建一个通用框架证明算法新意 |
| **AVIC**，NeurIPS 2026 | 测试时想象并非越多越好，按任务需要动态分配想象有价值 | 不能只提出 adaptive imagination budget |

这些结论来自各论文或项目的一手材料。<Cite refs={["turn924988view5","","turn804209view5","turn804209view1","turn924988view0","turn924988view4","turn804209view4"]}/>

其中我认为对你最重要的新发现，不是 PlanWAM，也不是 SuperMimic，而是 **Hi-LeWM**。

它几乎直接触及我们上一版 Spectral Reachability Atlas 的研究目标：

在冻结 LeWM 的前提下，通过高层 latent subgoals 改善长时域规划。

论文报告：只要高层搜索脱离低层控制器实际能够执行的动作分布，分层规划就可能失败；限制到经验支持的 macro-actions 后，在其 PushT 长时域设置中取得了相对于 flat LeWM 的明显改善。<Cite ref="turn804209view1"/>

这意味着：

**我们不能再把“冻结模型 + 可达子目标 + 分层搜索”当作核心新意。**

更强、也更有研究价值的切入点是：

> **Can the planner construct its own executable action abstractions from the local dynamics of a frozen world model, instead of relying on pretrained skills, learned macro-action models, or unconstrained latent subgoals?**

中文：

**能不能不训练额外技能模型、不预设动作原语，而是直接从冻结世界模型的局部动力学中，生成可组合、可执行、能主动改变未来控制能力的动作程序？**

我认为这个问题值得把 LeWorldModel 下一阶段的工作集中起来。

## 二、还有一篇刚发布的论文，会直接改变我们的定位

10 月 21 日，UC Berkeley、NVIDIA 等团队发布了 <Entity category="other" value="Agentic World Action Models (AW-0)" disambig="Liu et al. October 2026 agentic code video action model"/>。

它将 **code、future video、robot action** 作为三个耦合的生成流，让模型生成可复用的参数化技能程序，然后执行这些程序。<Cite ref="turn152478view0"/>

这意味着，连“让 world model 使用可执行动作程序，而不只是输出逐步动作”也已被直接研究。

不过，AW-0 的论文明确指出一个仍然存在的问题：程序可以选择正确的技能，却给出错误的物理参数；程序语法正确并不保证执行成功。其失败案例涉及接触丢失、抓取位置偏差、物体拖动和精细对齐。<Cite ref="turn353874view0"/>

这对我们尤其有启发。

AW-0 的程序主要从训练中获得语义技能与控制结构。我们可以研究另一个互补问题：

**能否让一个从未训练过程序生成器的冻结世界模型，直接通过微分动力学和有限时域动作谱，合成带有具体动作参数与预测状态转移的控制程序？**

我认为这才是与 AW-0 的真正区别。

| 方向 | 程序从哪里来 | 是否训练额外模块 |
|---|---|---|
| HarnessPAI | 外部代码推理、执行反馈与程序改进 | 底层动作模型可冻结，程序跨 rollout 演化 |
| AW-0 | 联合训练的 code expert 和技能库 | 需要 |
| Hi-LeWM / SAGE | 学习高层子目标、macro-action 或 proposal 模型 | 需要额外学习组件 |
| **我们要探索的方向** | 冻结世界模型的动作响应、可达子空间和组合动力学 | **不需要任何模型后训练** |

这里要小心区分：**不需要额外训练**是一个重要约束，但不能单独成为主要科学创新。

我们必须进一步证明：这种从动力学结构中合成程序的方法，确实产生了已有采样、梯度搜索、分层学习和代码程序方法难以得到的规划能力。

---

## 三、建议提出的新方法论：World Model as a Planning Compiler

我建议把前面讨论的 Spectral Reachability Atlas 再提升一层。

**不要只把谱分析当作更好的 MPC 搜索空间，而是把它当作一个“将任务目标编译成可执行动作程序”的中间表示。**

暂定方法名称：

<title size="lg">Dynamics-to-Programs: Test-Time Action Program Synthesis from Frozen Visual World Models</title>

简称 **D2P**。

其中 *program* 指由可执行动作片段、参数、状态转换和控制流程组成的数值控制程序，不是必须由 LLM 生成的 Python 代码。

它的核心命题是：

> **A frozen world model should serve not only as a future predictor or cost oracle, but as a compiler that derives executable action abstractions from its own dynamics.**

这比“给世界模型套一个 harness”更接近一项新方法论。

我们可以把规划分为三个相互依赖的计算过程：

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Task specification + Current RGB**
    <text color="secondary" size="xs">Frozen encoder · Frozen differentiable world model</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **1 · Compile local dynamics**
    <text color="secondary" size="xs">Finite-horizon action Jacobian · Task-weighted spectral directions</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **2 · Synthesize executable programs**
    <text color="secondary" size="xs">Goal-progress actions · Configuration-changing actions · Composed action chunks</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **3 · Search and refine programs**
    <text color="secondary" size="xs">Reachability-shaped lookahead · Hierarchical action optimization</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Execute action chunk → Observe RGB → Recompile**
    <text color="secondary" size="xs">Model parameters remain unchanged</text>
  </box>
</box>

这套结构的关键不是把 CEM、梯度规划和 graph search 全部装进去，而是：**三个步骤使用同一个有限时域动力学对象，并且后面的动作程序必须由前面的可达计算构造出来。**

## 四、D2P 的算法核心：把谱分析变成“程序生成规则”

我希望这个方法与上一版 SRA 最不一样的地方是：

**SRA 主要改变搜索坐标；D2P 则改变搜索对象——从原始动作序列变成由世界模型推导出的、可组合的动作程序。**

### 4.1 编译第一类程序：Goal-Progress Program

给定冻结的视觉世界模型：

$$

z_t=E(o_t),\qquad
\hat z_{t+H}=F_\theta^{(H)}(z_t,U)

$$

在候选动作序列 $\bar U$ 附近计算：

$$

G_H=
\frac{\partial F_\theta^{(H)}(z_t,U)}
{\partial U}\Bigg|_{\bar U}

$$

考虑目标度量 $Q$ 和动作尺度 $R$，定义：

$$

K_H=Q^{1/2}G_HR^{-1/2}

$$

对其做 SVD：

$$

K_H=P\Sigma V^\top

$$

目标残差为：

$$

r=Q^{1/2}(z_g-\hat z_{t+H}(\bar U))

$$

由此可以直接生成一个低维动作程序：

$$

\boxed{
U_{\rm progress}
=
\bar U+
R^{-1/2}
V
\operatorname{diag}
\left(
\frac{\sigma_i}{\sigma_i^2+\lambda}
\right)
P^\top r
}

$$

这是局部阻尼最小二乘形式，数学本身是经典方法。

我们的用途却不同：把它当作**程序生成器**，而不只是优化器。

一个程序不仅保存动作 $U$，还保存其模型预测的后继 $\hat z'$、动作响应子空间 $G_H$、所需时域和局部执行代价。

这使得 planner 能够组合程序，而不是重新从整个原始动作空间盲目采样。

### 4.2 编译第二类程序：Reconfiguration Program

这是我认为最有机会产生新发现的部分。

某些动作暂时不能让机器人接近目标，却能改变机器人下一步的控制能力。

例如 PushT 中，移动推杆到物体另一侧，可能暂时增加 goal distance，但使之后的旋转动作变得有效。

我们不依赖人工定义“先移动接触点”这种技能，而是从预测动力学中自动发现它。

定义当前局部目标残差中的一阶不可达部分：

$$

r_\perp=(I-PP^\top)r

$$

然后不继续对当前 $U$ 做无效梯度下降，而是生成短动作候选 $B$，预测其到达的新状态：

$$

z_B=F_\theta^{(k)}(z_t,B)

$$

在新状态重新计算 $G_H(z_B)$ 和局部可达目标代价：

$$

V_{\rm reach}(z_B,g)
=
\min_{\delta U}
\left[
\|r_B-K_B\delta U\|^2+
\lambda\|\delta U\|^2
\right]

$$

其中 $r_B$ 和 $K_B$ 都在状态 $z_B$ 重新计算。

一个 reconfiguration 程序可通过以下目标进行比较：

$$

\boxed{
B^*=
\arg\min_B
\left[
c_{\rm stage}(B)+
\beta V_{\rm reach}(z_B,g)
\right]
}

$$

它选择的不是立即最接近目标的动作，而是**能够让未来目标更容易通过动作实现的状态转换**。

这里有一个很好的研究判据：

如果某个动作短期不降低任务误差，但显著改善下一步的任务相关可达子空间，并让真实闭环更快完成目标，那么它证明了一种单纯 goal-distance MPC 不容易表达的程序级规划收益。

对于光滑系统，可以研究正反动作组合与高阶可达方向；对于 PushT 这种接触模式可能不连续的任务，应使用实际短动作 rollout 和动作顺序差异，而不能机械地应用连续 Lie bracket 近似。

### 4.3 编译第三类程序：Composition Program

现在我们不再规划：

$$

U=(a_0,a_1,\ldots,a_{H-1})

$$

而是规划：

$$

\boxed{
\Pi=
P_1\circ P_2\circ\cdots\circ P_M
}

$$

其中每个 $P_i$ 是有具体动作参数的 progress 或 reconfiguration 程序。

<box border radius="lg" padding={3} gap={2}>
  <text color="secondary" weight="medium" size="xs">程序记录结构（概念示例）</text>
  <box background="surface-secondary" radius="md" padding={3} gap={1}>
    **Program A — Reconfigure contact**
    <text size="xs" color="secondary">Input: Current RGB-derived latent</text>
    <text size="xs" color="secondary">Action chunk: Move to a new predicted contact configuration</text>
    <text size="xs" color="secondary">Predicted postcondition: Improved rotation-related action response</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box background="surface-secondary" radius="md" padding={3} gap={1}>
    **Program B — Goal progress**
    <text size="xs" color="secondary">Input: Postcondition from Program A</text>
    <text size="xs" color="secondary">Action chunk: Optimize controllable rotation modes</text>
    <text size="xs" color="secondary">Predicted postcondition: Reduced object pose error</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center" gap={1}>
    **Compose → Roll out → Refine → Execute**
    <text size="xs" color="secondary">Every intermediate target has an explicit generating action sequence</text>
  </box>
</box>

相较纯 latent subgoal search，一个重要区别是：

**每个中间状态都携带能够在冻结模型内生成它的动作见证（action witness）。**

这不保证它在真实物理环境中一定可达，但能避免规划器直接提出完全没有对应动作解释的任意 latent waypoint。

这正面回应了 Hi-LeWM 发现的高层子目标与低层动作接口失配问题。<Cite ref="turn353874view3"/>

同时，D2P 不训练新的 high-level model 或 subgoal generator，这又与 <Entity category="other" value="SAGE" disambig="Cheng et al. 2026 subgoal-conditioned action generation"/> 形成区别。SAGE 已经证明，训练多时域子目标和条件动作 proposal 能显著改善冻结 LeWM 的长时域规划表现；因此我们必须把它当作强基线，而不能假装自己首次发现子目标的重要性。<Cite ref="turn152478view2"/>

---

## 五、Harness 真正应该研究的不是“工具更多”，而是“计算流程如何由动力学决定”

这是我建议再向前推进的一步。

普通 harness 可能具有固定执行链：

$$

\text{Generate}\rightarrow
\text{Predict}\rightarrow
\text{Score}\rightarrow
\text{Execute}

$$

我们的 harness 则应该让模型当前的可达几何决定**下一步计算什么**。

例如，定义局部无法覆盖的目标残差比例：

$$

\chi_H=
\frac{\|r_\perp\|^2}
{\|r\|^2+\epsilon}

$$

当目标的大部分方向已经处于局部动作响应空间时，优先生成 progress program。

当目标主要位于局部一阶不可达方向时，扩展 reconfiguration candidates，或探索能形成新动作响应方向的动作组合。

当扩大 horizon 能改变任务相关可达子空间时，增加对应的 imagined rollout 长度。

这不需要一个新的 LLM router 或可训练 classifier。它可以通过**当前模型的有限时域谱与实际规划目标**决定计算资源的分配。

但 $\chi_H$ 不应被包装为新的安全保证，或者仅靠一个手工阈值就声称最优。它是计算调度信号；真正评价标准是相同 world-model 调用预算下，能否找到更好的动作程序。

这里可以吸收 NeurIPS 2026 <Entity category="other" value="When and How Much to Imagine" disambig="Yu et al. 2026 adaptive test-time visual imagination"/> 的 insight：增加想象量未必单调提高效果，因此应根据任务需要分配预测计算。但它主要研究视觉空间推理，我们这里将计算分配与可微动力学和物理动作搜索直接耦合。<Cite ref="turn804209view4"/>

**这才是科学意义上的 harness：模型不只提供预测结果，还决定该如何使用预测计算来组织控制。**

## 六、我建议这篇论文真正主张的三项贡献

| Contribution | 不是做什么 | 真正要做什么 |
|---|---|---|
| **C1 — Dynamics-Native Action Abstractions** | 不是训练技能 token 或统一动作编码器 | 从冻结视觉世界模型的有限时域动作响应谱，生成任务条件化、具有动作见证的控制程序 |
| **C2 — Reconfiguration-Aware Program Synthesis** | 不是普通 latent subgoal、CEM+GD 或简单 SVD | 搜索能够改变后续可达空间的动作片段，并与直接进展动作组成非贪心计划 |
| **C3 — Model-Native Test-Time Planning** | 不是后训练 policy、critic 或 GDD student | 由动力学结构决定程序生成、搜索深度和计算预算，在同一冻结模型上完成不同任务 |

这里我会把 LeWorldModel 的旧研究资产重新分配职责。

**谱分析**负责生成程序，而不是只生成诊断报告；有限时域 transfer operator 负责发现时序动作方向；Schur 和 transient 分析帮助研究延迟作用与 horizon；ATM 保留为物理动作语义分析和机制对照。HTSC 与 GDD-WM 可以留在独立研究线上，不再成为这篇方法论文的主要贡献。

这也保持了你已经明确提出的约束：**不做后训练**。不更新 encoder、predictor、policy、critic 或额外 adapter。只在规划时优化动作参数、构造临时程序和缓存本次任务的模型预测。

---

## 七、最关键的实验：怎样证明我们不是简单的 Harness Engineering？

我建议采用三组互相依赖的实验。

### Experiment 1：Executable Subgoal Gap

直接对齐 Hi-LeWM、SAGE 的科学问题。

构造三种中间目标生成方式：

| 方法 | 中间状态如何产生 |
|---|---|
| Arbitrary latent waypoint | 直接在 latent 中优化目标 |
| Learned subgoal prior | 采用已有分层方法训练得到的 subgoal/proposal |
| **Dynamics-compiled program** | 从冻结模型的动作响应推导动作序列，再得到对应中间状态 |

关键不是单纯比较 latent distance，而是比较中间状态能否由低层 controller 实现、执行过程中是否出现偏差，以及最终闭环成功率。

如果 D2P 明显改善的是**高层子目标与低层动作之间的衔接**，那么它就有了超越普通框架工程的实质机制。

### Experiment 2：Controllability-Changing Actions

这是我认为最值得做成论文 Figure 2 的实验。

在带朝向约束的导航、受限推动或 PushT 中，专门设计目标无法靠当前局部动作直接完成、但通过重新配置姿态或接触能够完成的状态。

比较：

| 基线 | 必须排除的替代解释 |
|---|---|
| CEM / MPPI | 增加采样是否就可以解决 |
| Vanilla gradient MPC | 普通动作梯度优化是否已足够 |
| GRASP | 更先进的可微规划是否已解决 |
| Local Gauss–Newton | SVD 和阻尼最小二乘本身是否足够 |
| SRA without program composition | 仅改变动作坐标是否已足够 |
| Kinodynamic search | 经典动力学采样搜索能否达到相同结果 |
| **Full D2P** | 动态编译和组合动作程序是否具有额外价值 |

必须预先确定真实的物理任务成功指标，不能以预测出的 rank 增加或 latent distance 下降代替任务完成。

如果 Full 方法不能显著优于 SRA without program composition，说明新 harness 并没有比上一版谱规划方法多创造实质性能力。

### Experiment 3：One Frozen Model, Many Tasks

这直接继承 SuperMimic 关于共享动力学可以跨大量目标复用的 insight。

在相同冻结 checkpoint 上切换任务目标、规划 horizon 和动作限制，比较执行成功率、规划耗时、模型调用次数、backward 次数，以及长时域任务完成情况。

建议用 PointMaze 验证可达子空间和绕行，用 Reacher 验证连续动作及谱优化，用 PushT 验证接触配置切换。

如果基础模型在某类任务上连正确的动作响应都不具备，那么 compiler 无法凭空弥补这些物理知识。那应当作为明确的模型能力边界，而不是通过增加 harness 复杂度掩盖。

---

## 八、必须先完成的两个研究决策

我建议目前不要同时实现所有程序类型。

先做两个最小实验，它们能直接决定是否值得推进完整 D2P。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="flask-conical" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P0：谱编译是否比普通动作搜索更有用？**

      冻结 PointMaze / Reacher checkpoint，使用有限时域 Jacobian SVD 生成 goal-progress actions。与相同计算预算的随机、CEM、Gauss–Newton 和 GRASP 比较。记录完整动作、真实执行结果以及模型调用成本。

      如果仅有低秩参数化的收益，就把工作定位为规划优化，而不要声称已经发明了程序合成方法。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="git-branch" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **P1：Reconfiguration Program 能否产生非贪心规划能力？**

      在模型已有相应动作动力学的前提下，选择当前直接梯度容易停滞、但两阶段动作可达的状态。先通过短动作改变可达几何，再生成 progress action，检验真实闭环中是否优于单层搜索及普通 latent subgoal。

      如果这项实验明显成立，才值得实现多程序组合、动态 horizon 和跨场景 harness。
    </box>
  </row>
</box>

我会把第二项作为整篇论文的 **kill experiment**。

因为它能真正检验这条研究路线是不是超越了“给世界模型加一个更复杂的优化器”。

---

## 九、最终论文定位：Harness 是实现形式，Planning Compiler 是研究贡献

我最推荐的题目是：

<title size="lg">World Models as Planning Compilers: Synthesizing Action Programs from Frozen Visual Dynamics</title>

备选：

*Beyond Imagined Futures: Dynamics-Native Program Synthesis for Visual World-Model Planning*

这篇论文要回答的主问题不是“如何让 world model 更可靠”，也不是“如何设计最通用的 harness”。

而是：

> **Can a pretrained world model generate the action abstractions needed to solve a task, without learning a separate skill model, value function, or hierarchical planner?**

如果研究成功，最大的 Insight 将是：

**规划能力不只存在于世界模型预测的未来状态中，还隐藏在不同动作如何改变未来可达空间的动力学结构里。把这种结构编译成可组合的动作程序，可以从同一个冻结模型中提取新的规划能力。**

不过，经典可控性分析、局部逆动力学、motion primitives 和图搜索都已有长期研究传统。D2P 目前仍是一个明确的研究假设，不是已被证明的新范式。

**要有 Oral 竞争力，需要证明动态编译生成的程序相对已有强规划器具有不可替代的闭环收益；要进一步具有 Best Paper 潜力，则需要证明这种方法揭示了跨模型、跨任务可复现的“动力学结构如何转化为程序级规划能力”的新规律。**

我的最终取舍是：**把 LeWorldModel 做成一个真正以冻结世界模型为计算核心的 Planning Compiler，而不仅是一个能换模型、换 planner 的 Harness。** 前者才可能形成你希望的独特方法论。

-----

> GDD teacher 分支违反“无后训练”约束，请你把这个部分的研究模块独立出来变成一个另外的独立研究。基于我提到的这些研究，结合他们的insight和我们的insight去做研究，基于GDD teacher提供一种新的方法论出来，可以参考蒸馏和OPD这种方法论的最新最相关论文，参考一些相关的最新最接近研究的insight，结合他们的insight和我们的insight去推进研究

我会把 **GDD teacher → student 蒸馏**彻底拆成第二条独立论文线，与“不做后训练的 Planning Compiler”并行，但不再混入它的方法贡献。先核对你说的 OPD 在最新文献中的具体含义，以及与 D-JEPA、在线策略蒸馏、执行偏好蒸馏最接近的研究；然后设计一个真正有新方法论的 GDD 研究，而不是简单地把 teacher 排序拟合进 latent。

文献核对后，新的研究分界已经很清楚了：**WAM-OPD 已做 student-induced 训练，ΔWAM 已做动作切向场蒸馏，D-JEPA 已做候选关系与原生 latent 几何对齐。** 所以第二篇不能只把这三种损失相加。我更建议把 GDD 的核心提升为：**把 teacher 的候选关系先转成可跨候选集合表达的“决策势场”，再在 student 自己不断变化的 MPC 候选分布上蒸馏这个势场及其动作方向导数**。下面我会把这个方法的可实现性、训练循环和决定性对照写具体。

# 将 GDD Teacher 独立成第二条研究线：从偏好蒸馏走向 On-Policy Decision Geometry Learning

我赞成把 GDD teacher 分支从 LeWorldModel 的 **Training-Free Planning Compiler** 中彻底拆出来，而且建议将它发展成一篇具有独立理论问题、训练方法和实验结果的论文。

但在核对 2026 年最新蒸馏文献后，我认为必须进一步升级原始 GDD-WM：**简单把执行偏好从 teacher 蒸馏到 student 的 latent geometry，已经不足以成为 Oral 级创新。**

这里的 OPD 我按当前研究领域中最常用的 **On-Policy Distillation（同策略分布蒸馏）**理解。它的关键不是使用某种 KL loss，而是让 student 在自己产生的状态、动作或预测轨迹分布上接受 teacher 指导。

## 一、先把两篇研究的边界明确分开

<box border radius="xl" padding={3} gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="route" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      <text size="xs" color="secondary" weight="medium">RESEARCH I · 不做后训练</text>
      <title size="md">World Models as Planning Compilers</title>
      使用冻结的视觉 encoder 和 world model，通过有限时域谱几何、可达动作程序与 test-time 搜索改善规划。不训练 student、policy、critic 或 adapter。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="brain-circuit" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      <text size="xs" color="secondary" weight="medium">RESEARCH II · 允许后训练</text>
      <title size="md">GDD: On-Policy Decision Geometry Distillation</title>
      使用独立执行反馈、teacher 的决策关系和 student 自己产生的规划候选，改变 student 的未来表示与动作条件动力学，使真正影响动作选择的关系成为模型本身的能力。
    </box>
  </row>
</box>

两篇可以共享原始 pretrained checkpoints、物理环境、数据格式和评估协议，但**不应互相依赖最终方法组件**。Research I 不能偷偷调用经过 GDD 训练的 student 来证明自己的 training-free 性质；Research II 也不能把 Planning Compiler 的收益当成蒸馏成功的证据。

## 二、最新直接竞争文献已经把创新门槛推得很高

| 论文 | 已经解决的问题 | GDD 不能再单独宣称什么 |
|---|---|---|
| <Entity category="other" value="D-JEPA" disambig="2026 decision aligned latent world model"/> | 从执行结果学习候选间关系，并将排序实现为可用于 latent-distance planning 的未来表示 | 首次把执行偏好写进 future latent |
| <Entity category="other" value="WAM-OPD: Joint Video–Action Supervision" disambig="Yang et al. October 2026 world action model on policy distillation"/> | 在 student 实际访问的历史上，利用 teacher 的视频与动作目标训练 WAM | 首次让 WAM 接受 student-induced OPD |
| <Entity category="other" value="WAM-OPD: Sharpening World Action Models" disambig="Liu et al. September 2026 prefix weighted on policy distillation"/> | 通过固定执行轨迹池及前缀加权，减少重复环境交互的 OPD 成本 | 首次用轨迹重放解决 WAM 蒸馏的分布变化 |
| <Entity category="other" value="OPRD" disambig="Yang et al. June 2026 on policy representation distillation"/>、<Entity category="other" value="FocusOPD" disambig="Lin et al. October 2026 structure guided on policy diffusion distillation"/> | 从仅匹配输出走向中间表示、特定层与特定时间位置的选择性蒸馏 | 首次提出 representation-aware OPD |
| <Entity category="other" value="ΔWAM" disambig="Wu et al. October 2026 distilling action tangent fields"/> | 将动作诱导的局部未来变化场蒸馏到世界动作模型 | 首次蒸馏 action Jacobian 或 tangent field |

这些都已经有公开论文，其中两篇 WAM-OPD 的研究对象与你的新分支尤其接近。<Cite refs={["turn882577view6","turn882577view3","turn882577view4","turn576705academia62","turn576705academia60","turn882577view5"]}/>

因此我建议新的 GDD 研究聚焦一个更尖锐的问题：

> **Can a world-model student learn the decision structure of a stronger teacher on its own planning distribution, while preserving action-conditioned predictive dynamics and learning relationships that transfer beyond the candidate set used for supervision?**

即：

**能不能不只是模仿 teacher 在某一批候选上的排序，而是让 student 在自己不断变化的规划分布上，学到跨候选集合、跨任务目标、跨规划时域仍然有效的决策结构？**

这才可能把 OPD、GDD、谱分析和世界模型规划真正结合起来。

## 三、GDD 原方案真正需要推翻的部分

仓库中的 GDD-WM 方案其实已经有比较清楚的研究目标：把 D-JEPA 风格的执行偏好转化为**候选集合无关的预测几何**，并保持多步动作条件预测。原方案还包含 realized geometry、predicted geometry、CEM elite mining 和 prediction preservation。<Cite ref="turn175289view1"/>

因此，新论文不应该把这些重新命名成创新。

我认为应重点解决三个尚未闭合的问题。

### 3.1 Teacher 的关系知识，并不天然等于可蒸馏的状态价值

D-JEPA 的 relational teacher 可以同时读取整个候选集合。它给某个候选的评分，可能依赖其他候选的存在。

但你想部署的 student 是：

$$

c_\theta(h,U,g)
=
\frac1d
\left\|
F_\theta^{(H)}(E_\theta(h),U)-E_\theta(g)
\right\|^2

$$

它不应该知道当前候选集合里还有哪些动作。

这两种函数类之间存在本质区别。

对于相同历史、目标和动作对 $U_i,U_j$，如果 teacher 在候选集合 $\mathcal C_1$ 中认为 $U_i$ 更好，在 $\mathcal C_2$ 中却认为 $U_j$ 更好，那么任何候选集合无关的标量代价，都不可能同时复现这两次严格相反的判断。

**因此，原始 teacher 的全部输出不是一个合理的蒸馏目标。**

我们需要先确定：哪些关系可以表示成一个固定的 decision potential，哪些只是集合相关的比较效应。

### 3.2 Offline teacher 数据不等于 Student Planning Distribution

原 GDD 可以先在固定 action bank 上训练，再在新候选上测试。

但 OPD 的启发是：student 一旦改善，就会提出不同的动作。

设 student 当前的 MPC 候选分布为：

$$

q_{\theta_k}(U\mid h,g)

$$

更新以后：

$$

q_{\theta_{k+1}}(U\mid h,g)
\ne
q_{\theta_k}(U\mid h,g)

$$

尤其 CEM 会不断集中到新的 elite regions。

原来用于蒸馏的候选偏好可能越来越不代表 student 现在真正需要比较的动作。

所以 GDD 的训练对象不应是永远固定的一批离线 action pairs。

**Teacher 必须持续回答 student 现在实际提出的规划问题。**

这与两篇 WAM-OPD 的 student-induced supervision 一致，但我们监督的是完整动作候选之间的**决策关系与其对规划几何的影响**，不是仅仅模仿动作流或预测视频。<Cite refs={["turn882577view3","turn882577view4"]}/>

### 3.3 Teacher 并不一定在所有动作上优于 Student

10 月 18 日的 <Entity category="other" value="Beyond the Teacher: Understanding and Preserving Distillation Overshoot in Agentic On-Policy Distillation" disambig="Zhong et al. October 2026 on policy distillation overshoot"/> 提出了一个很重要的发现：

Student 可以在蒸馏过程中一度超过 teacher，之后因为继续模仿 teacher 而退步。作者将其部分归因于：蒸馏修复了错误选择，同时也可能侵蚀 student 原本成功的选择，并提出 Success-Preserving OPD。<Cite ref="turn392872view2"/>

这个 insight 对 GDD 非常重要。

假设当前 student 的某个动作在真实环境里已经成功，但 teacher 偏好另一个实际更差的动作。

我们不应该继续强迫 student 模仿 teacher。

所以新的 GDD 不是纯 Teacher Imitation，而应当是：

$$

\boxed{
\text{Teacher's useful relations}
+
\text{Student's successful behaviors}
+
\text{Actual intervention outcomes}
}

$$

我们希望 student 最终能够**选择性吸收 teacher 的知识，同时保留甚至超过 teacher 的任务能力**。

---

## 四、我建议的新方法：On-Policy Decision-Field Distillation

暂定名称：

<title size="lg">GDD-OPD: Learning Intrinsic Decision Fields from Relational Teachers on Student-Generated Plans</title>

核心贡献不是一种新的 preference loss，而是一个三阶段学习机制：

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **Independent execution-grounded relational teacher**
    <text color="secondary" size="xs">Frozen teacher · No outcome leakage</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **1 · On-policy candidate graph**
    <text color="secondary" size="xs">Student MPC proposals · Overlapping candidate sets · Hard comparisons</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **2 · Integrable teacher decision field**
    <text color="secondary" size="xs">Preference graph · Potential projection · Task-dependent action directions</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **3 · Student predictive geometry distillation**
    <text color="secondary" size="xs">Realized and imagined future geometry · Directional matching · Prediction preservation</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="lg" padding={3} align="center" gap={1}>
    **Update student → Regenerate MPC candidates → Repeat**
    <text color="secondary" size="xs">Teacher only during training · No teacher at deployment</text>
  </box>
</box>

我最看重的是第二步。

如果它成立，我们就不仅是在蒸馏 teacher 的答案，而是在**蒸馏 teacher 对哪些动作改变能够改善任务的局部决策结构**。

### 4.1 建立 Student-Induced Candidate Graph

对一个历史 $h$、目标 $g$、horizon $H$，当前 student 运行自己的 MPC：

$$

\mathcal C_k(h,g)
\sim q_{\theta_k}(U\mid h,g)

$$

候选不只取随机动作，而应同时覆盖当前 elite、elite 附近扰动，以及少量有较大动作差异的连接候选。

构造图：

$$

\mathcal G_k=(\mathcal V_k,\mathcal E_k)

$$

其中每个节点对应一条**有明确动作序列内容**的候选，而边对应 teacher 对两条动作的相对偏好。

同一个 root 的多组候选集合应保留共同的 anchor actions，使不同 candidate sets 之间可以建立比较连接。

这里可以吸收 ICML 2026 <Entity category="other" value="Coverage Improvement and Fast Convergence of On-policy Preference Learning" disambig="Kim et al. ICML 2026"/> 的 insight：on-policy preference learning 的优势，与训练分布逐渐改善目标区域的覆盖密切相关。该论文的正式收敛结论属于其 contextual-bandit/Bradley–Terry 等设定，不能直接套用于 world-model MPC，但它支持把**候选覆盖质量**作为一项独立研究对象。<Cite ref="turn392872view5"/>

这会导致 GDD 与普通 OPD 的一个区别：

**Student 的候选集合不只是收集训练样本的工具，还是我们学习决策几何所需的比较图。**

### 4.2 将 Teacher 的关系偏好转化为一个 Decision Potential

设 teacher 给出候选之间的偏好强度：

$$

d^T_{ij}\approx
\text{teacher-implied cost}(U_j)
-
\text{teacher-implied cost}(U_i)

$$

我们希望学习一个候选集合无关的势函数：

$$

\phi_T(h,U,g)

$$

使得：

$$

\phi_T(U_j)-\phi_T(U_i)
\approx d^T_{ij}

$$

但 teacher 可能给出不一致的关系。

因此，对候选图的有向边构造 incidence matrix $B$，每条边表示两个候选的代价差。

将 teacher 的边信号记为 $d_T$，则可以计算：

$$

\boxed{
u_T^*=
\arg\min_{\mathbf1^\top u=0}
\left\|
W^{1/2}(Bu-d_T)
\right\|^2
}

$$

其中 $u_T^*$ 是节点上的最小二乘 decision potential。

残差为：

$$

e_{\mathrm{nonpot}}
=
d_T-Bu_T^*

$$

这个残差揭示：teacher 的哪些比较无法由同一个候选集合无关的标量代价同时解释。

注意两个数学边界。

第一，**HodgeRank 和 pairwise graph potential projection 本身是经典研究**，包括 Jiang、Lim、Yao、Ye 的统计排序与组合 Hodge 理论。我们不能把图分解本身包装成新发明。<Cite ref=""/>

第二，如果 teacher 只有偏好符号而没有经过统一标定的差值，不能直接把所有胜负强行设为 $\pm1$，然后把最小二乘残差解释为真实偏好循环。对于仅有符号的情况，应使用可行的排序不等式或经独立校准的 pairwise preference model。

真正值得研究的，是这个经典工具与当前问题的结合：

**在 student 自己产生且持续变化的动作候选图上，哪些 teacher 关系可转化成可由 predictive latent cost 表示的潜在价值，哪些不能？**

进一步还可以研究候选图 Laplacian：

$$

L_{\mathcal G}=B^\top WB

$$

的第二小特征值 $\lambda_2(L_{\mathcal G})$。

如果比较图不连通或连接很弱，节点相对价值的估计会缺乏足够约束。因而可以主动选择重叠候选与 bridge comparisons，使有限 teacher 查询预算能够连接关键候选区域。

这让**图谱结构真正决定如何蒸馏和采样数据**，而不只是作为统计报告。

### 4.3 从 Teacher Potential 构造局部 Decision Field

这是与 D-JEPA 和普通 GDD 拉开距离的关键。

普通蒸馏拟合：

$$

c_S(U_i)-c_S(U_j)
\approx
u_T^*(U_i)-u_T^*(U_j)

$$

但我们还可以进一步关注：

**如果规划器微小修改一条候选动作，teacher 希望任务代价怎样变化？**

在每个 candidate neighborhood 中，使用相邻动作与其 potential 差值估计局部方向场：

$$

\boxed{
g_T(U_i)
=
\arg\min_v
\sum_{j\in\mathcal N(i)}
w_{ij}
\left[
v^\top(U_j-U_i)
-
(u^*_{T,j}-u^*_{T,i})
\right]^2
+
\lambda\|v\|^2
}

$$

$g_T$ 是从 teacher 的局部偏好中恢复出来的动作空间方向性信息。

它与 ΔWAM 中的 Action Tangent Field 并不相同：

- ΔWAM 主要监督动作改变引起的 **future-world variation**；
- 我们要监督动作改变引起的 **task-decision variation**。

同一种动作可能引起很大的视觉变化，但对当前目标毫无帮助；也可能只引起很小的物体角度变化，却决定任务是否成功。

**GDD-OPD 应蒸馏第二种信息。**

不过这种局部场只能在动作邻域得到足够激励、且偏好强度有意义时可靠估计。它不是从几个排序标签就能无条件恢复的真实连续代价梯度。

## 五、真正改变 Student 的方法：蒸馏 Decision Field，而不是仅蒸馏排序

对于 student，仍然保留原始 GDD 的关键目标：让训练后的预测 future latent 自己能够体现正确的任务偏好。

$$

c_\theta(h,U,g)
=
\frac1d
\|\widehat z_H^\theta(h,U)-z_g^\theta\|^2

$$

但相比原来的 pairwise ranking loss，我建议增加**动作方向层面的蒸馏**。

### 5.1 Decision-Field Matching

student 对动作序列的预测代价梯度为：

$$

\boxed{
g_S(U)=
\nabla_U c_\theta(h,U,g)
}

$$

利用 teacher 所估计的 $g_T(U)$，定义：

$$

\mathcal L_{\rm field}
=
\mathbb E_{U\sim q_{\theta_k}}
\left[
\left\|
P_{\mathcal S(U)}
(g_S(U)-g_T(U))
\right\|^2
\right]

$$

其中 $\mathcal S(U)$ 是当前数据支持的局部动作扰动子空间。

之所以使用投影，是因为高维动作空间中，我们只通过有限候选差分观察到了某些方向，不能假设所有方向都被 teacher 数据辨识。

**这项监督会直接改变 student 对动作变化的敏感性，而不只是改变候选的最终排序。**

从 world-model 结构看：

$$

\boxed{
\nabla_Uc_\theta
=
\left(
\frac{\partial\widehat z_H^\theta}{\partial U}
\right)^\top
\nabla_{\widehat z_H^\theta}c_\theta
}

$$

这个表达式恰好连接你已有的两项研究积累：

- 谱分析与 ATM：动作扰动如何通过有限时域 predictor 传播；
- GDD-WM：预测后继中的目标几何如何反映任务价值。

它表明，student 的规划梯度由**动作传递结构**和**任务几何**共同产生。

因此，新 GDD 不是简单让 student 的 feature 更接近 teacher feature，而是让它们在任务相关的动作效应方向上表现一致。

### 5.2 避免昂贵的二阶训练

直接优化 $\nabla_Uc_\theta$ 与 teacher field 的差异，可能需要计算对模型参数的混合二阶导数。

第一版我建议采用方向有限差分：

$$

\mathcal L_{\rm field}^{FD}
=
\sum_{(U,v)}
w_{U,v}
\left[
\frac{
c_\theta(U+\epsilon v)
-
c_\theta(U-\epsilon v)
}{2\epsilon}
-
g_T(U)^\top v
\right]^2

$$

这样 student 仍只需要普通的参数反向传播，额外代价是多次模型 rollout。

这些方向 $v$ 不是任意采样，而来自当前 student MPC 的 elite candidate differences、teacher 高决策影响方向以及少量用于连接候选区域的扰动。

**这正是 OPD 对我们最重要的启发：监督应该跟随 student 目前真正需要优化的决策局部区域。**

### 5.3 不要对所有预测模态无差别蒸馏

这里可以直接借鉴 OPRD 与 FocusOPD。

OPRD 证明了在语言模型中，中间表示监督能够传递单纯输出模仿没有显式约束的结构；FocusOPD 则发现，针对特定层和扩散时间位置的方向性监督，可以优于全层、全时间监督。<Cite refs={["turn392872view1","turn392872view0"]}/>

但我们的选择维度不应是机械照搬“深层”和“早期”。

对于视觉 world model，更自然的选择单位是：

$$

\boxed{
\text{Task}
\times
\text{Prediction horizon}
\times
\text{Action-response mode}
}

$$

可以在当前 student 的有限时域动作响应算子上分析：

$$

G_H^\theta=
\frac{\partial\widehat z_H^\theta}{\partial U}

$$

再根据 teacher 的决策场、当前任务梯度和动作支持，选择真正需要修改的响应方向。

我建议把这叫 **Decision-Response Spectral Distillation**。

它与 ΔWAM 必须有明确区别：ΔWAM 已经使用动作扰动提取世界状态变化方向，再增强这些方向上的未来预测监督；我们必须证明，**只选择有显著决策影响的动作—任务响应方向**，比只选择动作敏感的方向更有用。<Cite ref="turn882577view5"/>

这可以形成一项非常强的消融：

同样训练参数量、teacher 查询次数与执行反馈预算，比较 action-tangent distillation 和 decision-field distillation。

如果后者没有额外收益，新的谱机制就不能作为主贡献。

---

## 六、完整 GDD-OPD 训练目标

我建议将原来大量并列的 loss 收敛成三组核心约束。

$$

\boxed{
\begin{aligned}
\mathcal L_{\rm GDD\text{-}OPD}
={}&
\lambda_r\mathcal L_{\rm relation}
+
\lambda_f\mathcal L_{\rm field}\\
&+
\lambda_p\mathcal L_{\rm predictive}
+
\lambda_k\mathcal L_{\rm keep}
\end{aligned}
}

$$

| 损失 | 核心作用 | 它解决什么问题 |
|---|---|---|
| $\mathcal L_{\rm relation}$ | 匹配经过势函数投影的 teacher 相对代价，同时使用合法真实执行偏好 | 候选选择的任务几何 |
| $\mathcal L_{\rm field}$ | 匹配 student 规划分布上的局部动作决策方向 | 从有限候选排序推广到邻近未见动作 |
| $\mathcal L_{\rm predictive}$ | 保留多步真实 successor、动作条件传播和必要的原始模型能力 | 防止为了排序扭曲世界模型 |
| $\mathcal L_{\rm keep}$ | 在独立执行证据表明 student 原有动作有效时保护该行为 | 防止盲目模仿 teacher 造成性能侵蚀 |

原始 GDD 的 realized geometry 与 predicted geometry 两条路径仍然保留。

对于相同合法动作，要求：

$$

c_\theta^{\rm obs}(U_j)
-
c_\theta^{\rm obs}(U_i)

$$

以及：

$$

c_\theta^{\rm pred}(U_j)
-
c_\theta^{\rm pred}(U_i)

$$

都尽可能反映训练数据中的真实任务对比，而不是只让 imagined futures 的距离被重新排列。

如果学生模型通过扭曲预测终点得到更好的 ranking，却显著破坏真实多步预测，那不是我们希望的成果。

另外，对任务存在路径依赖、不可逆运动或接触事件成本的情况，单一终点欧氏距离可能在函数类上不足。应将严格 Euclidean GDD 作为主设定，另设 goal-conditioned/path-aware cost 扩展实验，不能暗中改变代价函数后仍宣称原生欧氏几何已经充分。

---

## 七、我更推荐采用“两层 On-Policy”训练协议

这点需要严格区分。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="git-branch" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Level 1 — Planner-On-Policy Candidates**

      在一批固定、独立的环境初始状态上，让更新中的 student 生成新的 MPC 候选。每轮都重新生成候选图、查询冻结 teacher、选择监督边并更新 student。

      这解决的是动作候选分布变化，但不意味着环境状态也服从当前 student 的真实闭环分布。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <icon name="refresh-cw" size="lg"/>
    </box>
    <box flex="1" gap={1}>
      **Level 2 — Environment-On-Policy Histories**

      定期用当前 student 在环境中执行 MPC，收集它真正访问的 RGB 历史。在这些历史上构造候选、查询 teacher，并对选定候选执行真实反事实评估。

      这才能进一步解决 student 逐渐改变的环境状态分布问题。
    </box>
  </row>
</box>

我建议先完成 Level 1 pilot，再做 Level 2 的正式方法。

因为如果 Level 1 中新的 decision-field loss 都不能在未见候选上降低 regret，那么投入昂贵的真实环境 OPD 只会增加训练复杂度。

同时必须比较 WAM-OPD 提出的固定轨迹池与前缀加权重放。它可以在不持续新增环境交互的条件下，近似处理部分 student 分布偏移，但不能把重放状态称为严格来自当前 student 的新闭环状态。<Cite ref="turn882577view4"/>

### 训练循环

<CodeBlock language="python" editable={false}># GDD-OPD: conceptual training loop

teacher.freeze()
student = initialize_from_pretrained_world_model()

for iteration in range(num_iterations):
    # 1. Student generates its own planning candidates
    roots = sample_training_roots()
    candidate_sets = student_mpc_propose(student, roots)

    # 2. Add overlapping anchors and local action perturbations
    graph = construct_candidate_graph(candidate_sets)

    # 3. Query an independent frozen relational teacher
    teacher_edges = teacher.compare(graph)

    # 4. Anchor selected comparisons to real executions
    graph = attach_training_outcomes(graph, teacher_edges)

    # 5. Separate potential-representable preference evidence
    potential, residual, eligible = potential_projection(graph)

    # 6. Recover local decision-direction targets
    direction_targets = estimate_decision_field(
        graph, potential, eligible
    )

    # 7. Train both realized and imagined decision geometry
    loss = (
        relation_loss(student, potential, eligible)
        + field_loss(student, direction_targets)
        + prediction_preservation_loss(student)
        + successful_behavior_preservation_loss(student)
    )

    update_student(student, loss)

    # 8. Refresh planner distribution after model changes
    if should_collect_new_environment_rollouts(iteration):
        add_student_executed_training_histories(student)

# Teacher is absent during deployment
deploy_with_native_student_mpc(student)</CodeBlock>

这段是拟议流程，并非当前仓库已经完成的算法实现。

---

## 八、这篇研究有两个值得追求的理论问题

### Theory 1：哪些 Teacher Relations 可以真正写成 Student Potential？

对于固定的候选图、具有可比较尺度的 teacher 边信号 $d_T$，任何单一节点势函数 $u$ 都必须满足：

$$

\|W^{1/2}(Bu-d_T)\|^2
\geq
\|W^{1/2}e_{\rm nonpot}\|^2

$$

这是图最小二乘投影直接给出的结果，并不是新定理。

但它明确揭示了一个问题：

**关系 teacher 能够表达的东西，不一定能够全部压缩进一个不依赖候选集合的 scalar future cost。**

真正的新理论应进一步连接：

- candidate-graph 连通性；
- teacher preference 的跨集合一致性；
- student latent cost 的函数类表达能力；
- 未见候选上的几何蒸馏误差。

如果能够证明哪些图结构和样本覆盖条件下，student 能可靠学习这些关系，并解释为什么某些 teacher 信息无法被严格欧氏几何表达，才具备更高的理论价值。

### Theory 2：为什么 Field Distillation 有机会改善未见候选？

假设某个局部动作邻域内，teacher 对应一个可微的目标势函数 $V_T(U)$，student 代价为 $c_S(U)$。

对于两条动作 $U_i,U_j$，沿连接路径有：

$$

\begin{aligned}
&
[c_S(U_j)-c_S(U_i)]\\
&\quad-
[V_T(U_j)-V_T(U_i)]\\
&=
\int_0^1
\left\langle
\nabla c_S(U(t))-\nabla V_T(U(t)),
U_j-U_i
\right\rangle dt
\end{aligned}

$$

其中：

$$

U(t)=U_i+t(U_j-U_i)

$$

这个恒等式说明：

如果沿合法动作路径的决策梯度误差较小，那么两端候选之间的代价差异误差也会较小。

所以相比只拟合有限个训练 pair 的 ranking loss，**方向场监督有可能帮助 student 学习邻近动作之间更连续、可泛化的代价几何**。

但注意：只在几个动作端点匹配梯度，还不能自动保证整条连接路径上的积分误差小。需要额外的覆盖、平滑性或轨迹内采样条件。

真正值得做的理论工作，是刻画有限学生候选采样与 student-induced distribution shift 下，方向场误差如何影响未见动作 ranking 与 finite-horizon regret。

这会使理论直接服务于 GDD-OPD，而不是停留在又一个 margin inequality。

---

## 九、当前仓库的首个阻断问题：先获得真正的 Teacher

这是实现新论文之前必须解决的事项。

我查看的公开 GDD 执行方案记录表明，D0 已完成部分真实 RGB successor 数据关联；D1 曾发现：

$$

\texttt{teacher\_score}
=
-\texttt{real\_task\_cost}

$$

即所谓 teacher 分数直接来自真实代价，因此不能再用同一份代价证明 teacher 的独立正确性。

后续又构造了跨 candidate sets 的重复候选比较，但记录中的 teacher 仍是 `offline_environment_cost_v1`，不是独立验证的 D-JEPA teacher；D2 正式蒸馏因此继续被阻断。<Cite ref="turn263290view1"/>

这项结论针对所核对的公开审计记录，不代表之后所有版本都没有更新。

我建议采取双轨设计：

| Teacher 路径 | 用途 | 可以宣称什么 |
|---|---|---|
| **Execution-supervised oracle pilot** | 直接使用训练划分的真实代价关系，先验证 potential/field 学习是否可行 | 真实执行监督的决策几何学习 |
| **Independent relational teacher** | 使用独立执行反馈训练 D-JEPA-style teacher，冻结后在新 training roots 上提供偏好与相对分数 | 真正的 teacher–student decision-structure distillation |

这两条路径都合理，但不能混淆。

特别建议先运行 execution-only pilot，因为它能最快回答一个更根本的问题：

**即使给 student 正确的决策偏好，我们提出的 decision-field distillation，是否真的比普通 pairwise ranking distillation 更有效？**

如果答案是否定的，就没有必要首先投入大量资源训练大型 relational teacher。

---

## 十、怎样设计能够冲击 Oral 的决定性实验？

我建议将实验收敛成三个主结果。

### Figure 1：Output Imitation 与 Decision-Field Distillation 的差异

研究问题：

**匹配 teacher 在固定候选上的排序，是否足以把关系结构推广到未见动作？**

在相同 pretrained student、相同执行预算下比较：

| 方法 | 监督对象 |
|---|---|
| Pointwise teacher-score KD | teacher 的候选分数 |
| Pairwise preference KD | 候选对排序 |
| D-JEPA-style representation lifting | 将完整集合的排序实现为 future latent |
| OPRD-style representation alignment | Teacher / student 共同接口下的中间特征 |
| ΔWAM-style tangent distillation | 动作导致的预测未来变化 |
| **GDD-OPD decision-field distillation** | Candidate potential + 局部动作决策方向 |

重点测试固定候选、全新候选、CEM elite 和不同目标之间的迁移。

如果 Full 方法只在训练候选上改善，未见动作上没有增益，那么方向场蒸馏的核心主张不成立。

### Figure 2：为什么 Student-On-Policy Candidate Learning 是必要的？

在固定反馈预算下比较：

| 采样方式 | 它检验什么 |
|---|---|
| Fixed offline candidate bank | 普通离线蒸馏 |
| Teacher-generated candidate bank | Teacher 分布上的训练 |
| Initial-student bank + replay | 固定初始学生分布 |
| Student-refreshed candidate bank | Student 每轮更新后的候选分布 |
| Student-refreshed + overlapping candidate graph | 是否有必要显式维护关系图的覆盖与连通性 |
| Full environment-on-policy OPD | 新状态历史和新候选都由当前 student 产生 |

最终重点不是简单看平均 pairwise accuracy。

我希望看到：**当 student 的 MPC 优化逐渐集中到新的候选区域时，普通离线 GDD 的收益停滞，而新的 candidate-graph OPD 仍能改善真实 elite ranking。**

### Figure 3：蒸馏后能否形成真正更强的世界模型？

这张图应该成为主论文的最终成果。

需要报告：

| 指标 | 我们要看到的现象 |
|---|---|
| Realized geometry alignment | 真实 RGB successor 编码后的任务关系改善 |
| Imagined geometry alignment | Student 自己预测的未来关系改善 |
| Multi-step dynamics | 预测能力得到保持，不依靠扭曲未来表示作弊 |
| Unseen-action regret | 对训练时未出现的候选动作仍然有效 |
| Closed-loop success | Teacher-free student MPC 真正取得控制收益 |
| Teacher–student comparison | 检验 student 是否保留原有优势，有没有可能超过 teacher |
| Latency / FLOPs | 部署时不运行 relational teacher 或额外 candidate-set reranker |

完整实验应从 PointMaze 的重置式候选对开始，扩展到 Reacher 的连续动作，再用 PushT 检验接触条件下的方向场和候选几何。

PushT 必须先有合理的原始闭环 baseline 与 oracle ceiling。否则成功率为零时，无法判断究竟是蒸馏失败，还是规划器和动作执行接口本身不具备完成任务的能力。

---

## 十一、完整消融中，最重要的不是多加一个 Loss

我作为审稿人，最关心以下四组消融。

**第一：有没有必要做 On-Policy？**

固定候选分布与 student-refreshed 候选分布的比较，必须在相同真实执行预算和 teacher 查询预算下完成。

**第二：有没有必要蒸馏 Decision Field？**

普通 ranking KD、pairwise KD、Action Tangent Field 与 Full method，需要共享 teacher 和 student 初始化。

**第三：有没有必要做 Potential Projection？**

直接蒸馏全部 teacher 关系，与只蒸馏可表达的 potential 部分，必须比较未见 candidate-set 的 ranking、student 几何和实际控制。

**第四：是否在损害 Student 原有能力？**

对已经真实成功的 student 动作，检验普通 OPD 是否会强行改变其排序；再与 outcome-preserving 更新比较。

这最后一项直接呼应 SPOD 的发现，但我们的新意不能只是重用 SPOD 的遮罩策略，而是需要证明：**保留 student 已有的真实有效决策，同时利用 teacher 学习新的可迁移决策场，可以提高世界模型的总体规划能力。**

---

## 十二、我建议的新论文三个 Contribution

| Contribution | 学术主张 | 必须成立的证据 |
|---|---|---|
| **C1 — On-Policy Decision-Graph Learning** | Student-induced MPC 候选形成不断变化的比较图；有效蒸馏需要兼顾局部 elite 信息、跨集合一致性与图覆盖 | 比固定候选蒸馏更能改善未见动作和 elite regret |
| **C2 — Integrable Decision-Field Distillation** | 关系 teacher 中可表示的决策信息不仅包括候选排序，还包括任务条件化的动作方向场；将其写入内生预测几何 | Field distillation 优于普通 score/rank KD 和 action-tangent supervision |
| **C3 — Prediction-Preserving Beyond-Teacher World Models** | Student 在保持多步预测及已有成功行为的同时吸收 teacher 的有益决策结构，形成无需 teacher 的闭环规划能力 | 在独立 RGB-only MPC 中改善 regret、success，并经受 teacher 与强基线对照 |

这是一个真正独立的研究故事。

它的中心不是“这个世界模型可不可信”，也不是“我们加了 OPD”。

而是：

> **Relational teachers make decisions over sets. World models must learn reusable dynamics and geometry. We develop an on-policy distillation mechanism that converts student-relevant teacher relations into transferable, candidate-independent decision fields.**

---

## 十三、建议采用的阶段门

| 阶段 | 工作 | 失败时的结论 |
|---|---|---|
| **D0：Independent teacher evidence** | 真实 teacher、跨集合重复比较、执行反馈互不泄漏 | 不允许正式 teacher distillation |
| **D1：Potential feasibility** | 比较图连通性、排序一致性、PSD/Euclidean 可表示性 | 如果函数类根本不支持，不应继续增加训练复杂度 |
| **D2：Decision-field pilot** | Execution-only、Pairwise KD vs Field KD | Field 无增量则暂不推进复杂 OPD |
| **D3：Planner-on-policy refresh** | 当前 student 的 CEM proposals、graph coverage、teacher 查询预算 | 只在固定 bank 有效则限制论文主张 |
| **D4：Prediction-preserving co-training** | Encoder/predictor 受限适配、双路径几何、成功行为保护 | Prediction 退化则回退到更受限适配 |
| **D5：Independent deployment** | RGB-only closed-loop，teacher-free，未见动作和目标 | 没有闭环收益，不称为新的 world-model 方法论 |

代码层面，可以在现有 `experiments/gdd_wm/` 基础上形成独立研究入口，例如 `teacher_evidence.py`、`decision_graph.py`、`potential_projection.py`、`decision_field_targets.py`、`train_gdd_opd.py` 和 `eval_gdd_opd.py`。

同时，**Planning Compiler 不应导入这些训练模块，也不应将 GDD student 作为默认主结果模型**。两条线可以共享 `decision_benchmark` 中的数据加载与评估工具，但应分别维护模型指纹、训练协议和结果声明。

---

## 十四、怎样进一步达到 Oral / Best Paper 级别？

我认为这个方向比原始 GDD-WM 更值得研究，但还不能仅凭构想就宣称达到 Oral。

| 层级 | 必须实现的科学结果 |
|---|---|
| **扎实主会论文** | Student-generated candidate OPD 明显优于离线 GDD，在未见候选和真实闭环中提高表现 |
| **Oral 竞争力** | 证明 relational teacher 的可蒸馏部分具有明确的结构；decision-field supervision 在相同反馈预算下产生普通 KD、OPRD-style 和 ΔWAM-style 方法无法复现的增量 |
| **Best Paper 潜力** | 建立一个具有广泛解释力的原则：teacher 的集合关系如何在有限数据、student 分布演化与动作条件动力学约束下，转化为可复用的内生规划能力，并在多个不同类型的 world models 中经受验证 |

尤其是 Best Paper，不能只靠一项图投影公式、一个新的 preference loss 和若干成功率提升。

必须出现一个改变领域理解的结果。

例如，若实验证明：

**Teacher 的完整关系输出往往不能直接被内生 latent cost 学习，但经过学生分布驱动的候选比较、可表达势场提取和任务方向蒸馏，student 能获得比简单模仿 teacher 更好的新动作泛化与长期规划能力。**

那么这项工作才拥有真正鲜明的方法论价值。

## 最终研究取舍

我建议把 LeWorldModel 明确拆成两篇具有不同创新核心的论文：

<box border radius="lg" padding={3} gap={2}>
  <text size="xs" color="secondary" weight="medium">PAPER I · INFERENCE / PLANNING</text>
  **World Models as Planning Compilers: Synthesizing Action Programs from Frozen Visual Dynamics**

  <text color="secondary" size="sm">冻结模型。利用有限时域动力学谱生成可达动作程序，研究 test-time planning computation。</text>
  <divider color="subtle"/>
  <text size="xs" color="secondary" weight="medium">PAPER II · LEARNING / DISTILLATION</text>
  **From Relational Teachers to Predictive Decision Fields: On-Policy Geometry Distillation for World Models**

  <text color="secondary" size="sm">允许后训练。利用学生诱导的候选图、执行反馈和局部方向场蒸馏，研究如何将 teacher 的决策能力内化为未来预测几何。</text>
</box>

**我最建议第二篇立即推进的第一个决定性实验是：**

在完全相同的 PointMaze execution-supervised training split 上，比较原始 Pairwise GDD 与新的 **Potential + Decision-Field GDD**。冻结测试候选生成协议，评估全新 roots、全新动作扰动和真实 MPC elite 的 regret。

如果第二种方法在同样反馈预算下能够降低未见动作 regret，同时保持多步预测，那么就说明我们学习到了**比有限候选排序更丰富、可以指导动作优化的决策结构**。

之后再用真正独立的 D-JEPA teacher 和 student-on-policy 数据刷新验证这种结构能否高效地从强 teacher 迁移。

这样，两篇论文既没有相互污染，又围绕同一个大的研究主题相互呼应：

**第一篇研究如何从世界模型中提取规划能力；第二篇研究如何把新的规划能力真正写回世界模型。**

-----

> 对比实验也帮我看看要对比哪些研究？如果我们对比的论文有报告数值直接用人家的报告值

我会以刚拆分出来的 **GDD-OPD / Decision-Field Distillation** 为主，整理一份可以直接用于论文 Experimental Setup 的对比矩阵：哪些是必须复现的同协议强基线，哪些适合引用原论文数值，哪些只能做方法论对照。引用数值时我会核对原论文的具体表格、环境、输入条件和指标，避免把不同评估协议的成功率混在一起。最后也会补一小节独立的 Planning Compiler 论文基线，防止两条研究线混淆。

-----

> 第一篇研究的对比实验也帮我看看要对比哪些研究？如果我们对比的论文有报告数值直接用人家的报告值

我会按第一篇 **D2P / World Models as Planning Compilers（冻结模型、零后训练）** 的设定，核对最接近的论文和官方结果表。重点区分两类数值：**可以原样引用的论文报告值**，以及**必须在同一 checkpoint、环境、动作预算和评估协议下重跑的主实验基线**。我会优先查 LeWM、GRASP、Hi-LeWM、SAGE 和 Dream-MPC，并把 PlanWAM、SuperMimic 等放在合适的跨领域对照位置。

# 第一篇论文的对比实验：基线选择、可直接引用的数值与主结果设计

针对第一篇 **World Models as Planning Compilers（D2P）**，我建议把实验重点放在三个问题上：是否比普通 CEM/梯度搜索更好、是否比已有分层子目标方法更擅长非贪心规划，以及是否能在**同一个完全冻结的世界模型**上，用更少的规划计算完成更长时域的任务。

先明确一条论文规范：**可以直接引用其他论文报告的结果，但不能把不同 checkpoint、goal offset、执行预算和成功判据下的数字拼成一张公平排行榜。** 我建议最终论文同时提供：

1. **Published Results**：忠实引用各论文原表中的数字，标记为 *reported*。
2. **Controlled Comparison**：在相同冻结 checkpoint 和统一协议上重跑最关键的规划器。这部分才用于证明 D2P 的实际优势。

## 一、最应该首先对比的是 SAGE

<Entity category="other" value="SAGE: Subgoal-Conditioned Action Generation for Latent World Model Planning" disambig="Cheng et al. 2026 latent world model planning paper"/> 是第一篇论文在长时域 PushT 和 OGBench Cube 上最直接的竞争方法之一。它使用预训练且冻结的 LeWM 进行候选评估，但另外训练了子目标生成器和动作 proposal，因此不符合 D2P 的“无后训练”约束，却是必须挑战的性能基线。<Cite ref="turn210778view0"/>

下面直接列出其原论文 Table 1 中的成功率。

<text weight="medium">PushT：不同 goal offset 下的 Success Rate (%)</text>

| 方法 | H=50 | H=100 | H=150 |
|---|---:|---:|---:|
| Base LeWM | 56.0 ± 8.0 | 8.7 ± 3.1 | 4.7 ± 4.6 |
| Subgoal Scoring | 75.3 ± 9.5 | 72.7 ± 7.6 | 58.7 ± 7.6 |
| **SAGE** | **81.3 ± 3.1** | **72.7 ± 9.0** | **64.7 ± 9.5** |

<text weight="medium">OGBench Cube：不同 goal offset 下的 Success Rate (%)</text>

| 方法 | H=50 | H=100 | H=150 |
|---|---:|---:|---:|
| Base LeWM | 56.0 ± 8.7 | 58.7 ± 9.5 | 20.7 ± 5.8 |
| Subgoal Scoring | 63.3 ± 5.0 | 70.7 ± 13.3 | 51.3 ± 4.2 |
| **SAGE** | **76.0 ± 5.3** | **85.3 ± 1.2** | **67.3 ± 3.1** |

<caption>直接转录 SAGE 原文 Table 1。± 表示三个固定 evaluation manifests 的样本标准差；H 为 goal offset，非内部 world-model rollout 长度。原实验在该表内共享冻结 LeWM、300 个初始候选和 30 轮搜索。<Cite ref="turn516474view2"/></caption>

这组结果给 D2P 提出了一个很具体的挑战：

**SAGE 已经证明，长时域规划的瓶颈不只是搜索动作不够充分，还在于短期预测结果究竟应该与哪个子目标比较。**

因此 D2P 不能只在同样的 final-goal distance 上做更快的 CEM。它必须证明：**由动力学谱生成的动作程序及中间可达目标，能够替代部分需要额外训练才能获得的 subgoal prior。**

这也应该是第一篇论文最重要的竞争实验之一。

---

## 二、另有一篇必须加入的直接竞争论文：Anchored Planning

最新核查中，<Entity category="other" value="Aim Short to Reach Far: Your Frozen World Model Can Plan Better Than You Think" disambig="Liu et al. September 2026 training-free anchored planning"/> 比此前讨论的很多分层方法更直接威胁 D2P 的创新性。

它提出 **Anchored Planning（AP）**：不训练额外网络，从已有轨迹中检索中间观测，作为冻结 LeWM 的局部目标。它已经证明“无需后训练，仅改变规划目标”就能显著改善长时域任务。<Cite ref="turn274589view0"/>

原论文 Table 1 的部分报告值如下：

| 任务 | 原 LeWM | AP-CEM | AP-rank |
|---|---:|---:|---:|
| Cube | 15.6 | 46.9 | 73.4 |
| PushT | 7.0 | 69.5 | 66.4 |
| Reacher | 66.4 | 75.8 | 96.9 |
| TwoRoom | 17.2 | 46.9 | 98.4 |
| 四任务宏平均 | 26.6 | 59.8 | 83.8 |

<caption>原论文 Table 1，standard starts，每任务 128 个 paired queries，Success Rate (%)。AP-CEM 从观测记忆选目标并由 CEM 合成动作；AP-rank 额外利用记录的动作片段做候选选择。这些数值只适用于论文自己的长程任务协议，不能与上一节 SAGE 的数字直接比较。<Cite ref="turn302709view0"/></caption>

这篇论文会迫使 D2P 的实验回答一个特别重要的问题：

**如果直接检索一个真实经历过的中间状态，已经能够解决大部分非贪心规划问题，我们为什么还需要从世界模型的谱结构“编译”动作程序？**

我认为这是第一篇最不能回避的对照。

必须在同样的冻结模型、相同观察记忆可用性与计算预算下，比较 *AP-CEM / AP-rank* 和 D2P。还应测试没有接近目标的可复用记忆片段、必须组合多个局部动作才可到达的目标。

这决定了 D2P 究竟有新的方法价值，还是只是更复杂的中间目标生成方式。

---

## 三、GRASP：必须对比，而且原论文已有完整数值

<Entity category="other" value="GRASP: Parallel Stochastic Gradient-Based Planning for World Models" disambig="Psenka et al. ICML 2026 gradient relaxed stochastic planning"/> 是 D2P 在**可微规划与动作搜索效率**方面最重要的直接竞争方法。

原论文使用 DINO-WM 系列学到的视觉动力学，比较 CEM、GD、LatCo 和 GRASP。它与 D2P 的核心差异是：GRASP 优化动作和虚拟中间状态；D2P 计划通过动作响应谱生成可组合的程序。<Cite ref=""/>

<text weight="medium">原文 Table 1：长时域 PushT 开环规划成功率 (%)</text>

| 方法 | H=40 | H=50 | H=60 | H=70 | H=80 |
|---|---:|---:|---:|---:|---:|
| CEM | **61.4** | 30.2 | 7.2 | 7.8 | 2.8 |
| Gradient Descent | 51.0 | 37.6 | 16.4 | 12.0 | 6.4 |
| LatCo | 15.0 | 4.2 | 2.0 | 0.0 | 0.0 |
| **GRASP** | 59.0 | **43.4** | **26.2** | **16.0** | **10.4** |

<caption>GRASP 原论文 Table 1，每个设置 500 次试验；这里是优化后的开环规划成功率，并非 RGB-only 真实闭环 MPC 成功率。H 是其内部规划时域，与 SAGE 的 goal offset H 不同。<Cite ref="turn953273view0"/></caption>

最值得关注的是 H=50：GRASP 的 43.4% 对比 CEM 的 30.2%；H=80 时分别是 10.4% 和 2.8%。但 H=40 时 CEM 反而略高，因此不能声称 GRASP 在所有设置中都最强。

对于 D2P，我建议做两层比较：先在 GRASP 原论文的开环优化任务上比较成功率—计算时间曲线，再将双方实现放在相同冻结 LeWM 的 RGB-only 闭环中比较。前者可以引用论文数字，后者如果要证明 D2P 优势，就必须形成新的统一协议结果。

## 四、LeWM 原论文已有很重要的求解器消融，不需要重新发明

原始 <Entity category="other" value="LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels" disambig="Maes et al. 2026 LeWM original paper"/> 的 Appendix Table 10 对比了四种规划优化器：

| 求解器 | LeWM PushT Success Rate (%) |
|---|---:|
| **CEM** | **96.0 ± 2.83** |
| SGD | 26.0 ± 4.32 |
| RMSProp | 67.33 ± 2.49 |
| Adam | 84.0 ± 7.12 |

<caption>原始 LeWM Appendix Table 10；为原论文的 PushT 协议，goal offset 25、执行预算 50。表中 ± 保留作者原始记法。<Cite ref="turn280087view1"/></caption>

这组结果说明：**在原生 LeWM 上，普通梯度优化已经输给 CEM。** 因此 D2P 不能靠“使用自动微分”作为主要创新。

更好的比较是：在相同 LeWM checkpoint 上，测试 CEM、CEM+GD、阻尼 Gauss–Newton、GRASP 和 D2P 的动作程序合成机制，检验后者是否能跳出普通局部优化的局限。

---

## 五、Hi-LeWM：必须比较“动作程序”是否优于学习得到的 Macro-Actions

<Entity category="other" value="Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel" disambig="Caselli et al. 2026 Hi-LeWM hierarchical planning analysis"/> 在冻结低层 LeWM 的基础上训练高层 macro-action encoder 和 predictor，是 D2P 在**分层动作抽象**方面的直接竞争工作。

原文报告了如下 PushT 结果：

| 方法 | Goal offset 25 | 50 | 75 |
|---|---:|---:|---:|
| Flat LeWM | **94.0** | 52.7 | 18.0 |
| Naive Hi-LeWM | 89.3 | 38.7 | 15.3 |
| Hi-LeWM + empirical macro CEM | — | 48.7 | **32.7** |
| Hi-LeWM + staged execution | — | **64.0** | 22.0 |

<caption>成功率 (%)，源自原论文正文及 Figure 3 的报告值。每个方法与 offset 采用超参数扫描中的最佳配置，并在三个随机种子上汇总。不能将其视作预先锁定超参数后的统一评估。<Cite refs={["turn820228view1","turn666380view2"]}/></caption>

这个实验对 D2P 的意义很明确：

**学到高层动作编码，并不自动获得好的高层搜索空间；直接对潜在 macro-actions 搜索，还可能制造低层控制器无法执行的子目标。**

D2P 应直接证明：由冻结动力学计算出的 action-witness programs，在相同控制接口下，比 unconstrained latent macro-actions 更容易执行，并能完成需要重新配置姿态、接触或运动方向的任务。

此外，我建议用 <Entity category="other" value="Hierarchical Planning with Latent World Models" disambig="Zhang et al. 2026 multi-timescale world model planning"/> 作为另一项跨模型强基线。其原文 Table 2 在 PushT goal offset 75 上报告，Flat DINO-WM 为 **17%**，hierarchical DINO-WM 为 **61%**；在较难的 Diverse Maze 最短路径距离 13–16 上，Flat PLDM 为 **44%**，hierarchical PLDM 为 **83%**。这些是不同的预训练模型与分层训练体系，适合放在外部结果表，而不是与 D2P 在 LeWM 上的结果直接计算提升幅度。<Cite refs={["turn280087view3","turn280087view4"]}/>

---

## 六、另外四篇值得直接引用结果的研究

这四篇不一定需要全部重跑，但我建议将它们作为性能背景与训练预算对照。

### 6.1 PRISM：学习 Action Prior 的效果有多强？

<Entity category="other" value="PRISM: PRior-guided Imagination Sampling in World Models" disambig="Wang et al. 2026 prior-guided latent planning"/>

<text weight="medium">原论文 Table 1，K=128 的成功率 (%)</text>

| 方法 | PushT | Cube |
|---|---:|---:|
| Vanilla MPPI (LeWM) | 57 ± 6 | 44 ± 4 |
| PRISM-MPPI | **89 ± 4** | **79 ± 6** |

PRISM 额外训练约 1M 参数的动作先验，主表采用每种方法 3 个 seeds，每 seed 50 episodes，30 轮 MPPI。它可以证明额外学习 proposal 的价值，但不属于零后训练方法。<Cite ref="turn265724view1"/>

D2P 要回答的是：能否从冻结世界模型的有限时域动力学里**计算出**有效的动作候选，而不需要学习这个先验？

### 6.2 IMWM：演示检索与直觉模型有多大价值？

<Entity category="other" value="IMWM: Intuition Models Complement World Models for Latent Planning" disambig="Gao et al. 2026 intuition-guided world model planning"/>

| 任务 | World-model-only | IMWM |
|---|---:|---:|
| TwoRoom | 87.7 ± 4.7 | **99.2 ± 1.0** |
| Reacher | 83.2 ± 4.1 | 83.8 ± 5.7 |
| PushT | 89.8 ± 4.8 | **92.7 ± 3.4** |
| OGBench Cube | 66.2 ± 5.7 | **94.7 ± 1.9** |

<caption>原论文 Table 2，成功率 (%)；12 个 evaluation cells/任务的均值和 SD。IMWM 需要额外训练 intuition model，并利用演示检索。Reacher 的差别不具有明确统计优势。<Cite ref="turn302709view5"/></caption>

它尤其适合用来说明：**规划器如何产生候选，比简单提高世界模型预测质量更重要。**

### 6.3 最新 Frozen HWM Subgoal 学习

2026 年 10 月 19 日发布的 <Entity category="other" value="Learning Goal-Conditioned Subgoals through Frozen Hierarchical World Models" disambig="Xiang and Liu 2026 frozen HWM subgoal head"/>，冻结两层世界模型并额外训练一个 goal-conditioned condition head。

| 环境 | HWM Search | Learned Condition Head |
|---|---:|---:|
| Cube | 51.3 ± 5.5 | **70.4 ± 2.5** |
| PushT | 48.7 ± 2.4 | 48.7 ± 2.9 |
| FourRoom | 72.0 ± 7.0 | **100.0 ± 0.0** |
| AntMaze | 35.3 ± 0.7 | **71.1 ± 1.9** |

<caption>原文 Table 2，固定 50-task sets、三个 HWM backbone seeds；± 为跨 backbone 均值的标准误。这套 HWM backbone 与普通 LeWM 单层 checkpoint 不同。<Cite ref="turn503892view1"/></caption>

这篇适合作为“训练高层规划模块可以获得多大收益”的参照，不是 D2P 的同条件主对手。

### 6.4 谱分析方向也有一个很接近的新对照

<Entity category="other" value="Control-Geometry Straightening for Sampling-Based Latent Planning" disambig="2026 control geometry representation learning paper"/> 已经直接研究将动作效应几何塑造得更适合 MPPI、CEM 与 GD。

其原文 Table 1 在 K=128、30 轮 MPPI 下报告：

| World Model | PushT | Cube |
|---|---:|---:|
| LeWM + MPPI | 64.0 ± 5.3 | 59.3 ± 4.6 |
| LeWM + CGS + MPPI | **77.3 ± 6.4** | **79.3 ± 4.6** |

CGS 需要改变世界模型训练目标，所以不属于 D2P 的零后训练基线。<Cite ref="turn503892view2"/>

但它很适合用于跨 checkpoint 验证：**D2P 能否在原生 LeWM、具有更好控制几何的 CGS checkpoint，以及 DINO-WM 上，利用同一种程序编译算法进一步改善规划？**

---

## 七、SAGE 主表有一个必须注意的公平性问题

前面列出的 SAGE 主结果使用了 *state-assisted* 的子目标生成器和动作 prior，其低维输入来自环境状态，不只是机器人 proprioception。

**如果你的 D2P 坚持 RGB-only，最直接的外部对照应该是 SAGE 附录中的 Visual-only 版本。**

SAGE 原论文 Table 18 的 Visual-only 成功率：

| 任务 | H=25 | H=50 | H=75 | H=100 | H=125 | H=150 |
|---|---:|---:|---:|---:|---:|---:|
| PushT | 93.3 | 83.3 | 78.0 | 72.0 | 68.0 | 61.3 |
| Cube | 96.7 | 76.7 | 84.0 | 80.0 | 76.0 | 56.0 |

<caption>原文 Table 18，150 个 queries 的合并成功率，非多训练种子的均值。Visual-only 版本不向生成器与先验输入额外环境状态，但仍使用额外训练的两个模块。<Cite ref="turn503892view0"/></caption>

因此最终论文建议分别标记：

**SAGE (reported, state-assisted)** 和 **SAGE (reported, visual-only)**。

不能只拿主表 64.7% 与我们 RGB-only 的结果比较，也不能省略其使用额外训练模块的事实。

---

## 八、我建议最终真正进入 Main Table 的基线

我会把实验分成三个层级。第一层是最重要的，也是唯一适合用来证明 D2P 在**相同冻结模型下**优于现有规划算法的证据。

| 对比方法 | 需要回答的问题 | 建议处理 |
|---|---|---|
| Frozen LeWM + CEM | 相对原始规划器有多少收益 | **统一协议重跑** |
| Frozen LeWM + MPPI | 是否只是换采样优化器 | **统一协议重跑** |
| Frozen LeWM + Adam / CEM+GD | 是否只是利用可微梯度 | **统一协议重跑** |
| Frozen LeWM + Gauss–Newton / SVD | 谱重参数化本身是否已足够 | **实现经典方法对照** |
| **GRASP** | 是否优于强可微与长时域优化 | 至少在一个共同协议重跑；原文数值另列 |
| **Anchored Planning (AP-CEM)** | 真实观测子目标是否已经解决问题 | 建议作为同模型主对照 |
| **AP-rank** | 直接复用动作片段是否更有效 | 在允许访问记录动作的实验中主对照 |
| Spectral-only D2P | 仅动作谱是否带来改善 | 必需消融 |
| D2P without reconfiguration | 不改变未来可达几何会怎样 | 必需消融 |
| **Full D2P** | 程序合成的独立增益 | 主要实验方法 |

第二层是**额外学习组件的性能参照**，建议使用公开报告值并明确标注：SAGE、PRISM、IMWM、Hi-LeWM、Hierarchical WM 和 Frozen HWM Condition Head。它们可以展示训练更多模块带来的性能上限，但不应与零后训练 D2P 混称同资源比较。

第三层是**不同训练得到的世界模型**，包括 DINO-WM、PLDM、CGS、Dual-WM 和 Slot-JEPA。它们更适合跨模型泛化实验：在能够加载合法 checkpoint 的情况下，把 D2P 应用到这些冻结模型上，再与对应模型自己的默认规划器比较，而不是直接拿不同论文的最终成功率互相排序。

## 九、主论文应该设计哪四张结果表？

我会将论文实验证据组织成下面四张主表。

| Table | 比较内容 | 主要指标 | 证明什么 |
|---|---|---|---|
| **Table 1 — Frozen WM Planning** | LeWM CEM、MPPI、GD、GRASP、AP、D2P | Success、regret、time、模型调用次数 | 相同模型下的规划收益 |
| **Table 2 — Long-Horizon Generalization** | 多 goal offsets、多个任务、不同任务长度 | Success vs goal distance、预算 | 能否解决非贪心长时域任务 |
| **Table 3 — Action Program Ablation** | Spectral-only、reachability value、reconfiguration、composition | Success、程序执行率、有效动作数量 | 新方法中的哪个机制不可替代 |
| **Table 4 — Reported Prior Results** | SAGE、Hi-LeWM、PRISM、IMWM、HWM、GRASP 原报告数值 | 原论文指标和原始协议 | 文献中的最新性能背景 |

特别重要的是 Table 3。

你需要检验：

$$

\text{Progress-only}
\quad\text{vs.}\quad
\text{Progress + Reconfiguration}
\quad\text{vs.}\quad
\text{Full Program Composition}

$$

并展示程序合成不只是增加了 CEM 搜索次数。

如果 Full D2P 的改善主要来自更大的计算量，或经典 Gauss–Newton 已能实现相同效果，那么核心新方法论就没有得到证明。

---

## 十、最小 GPU/实验预算方案

如果希望最大限度使用作者公开数值，而不重复大量已完成研究，我建议按如下顺序实施。

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <text tabularNums weight="medium">P0</text>
    </box>
    <box flex="1" gap={1}>
      **直接整理已有 Published Results**

      SAGE、GRASP、Hi-LeWM、Anchored Planning、PRISM、IMWM、HWM 都已经有可引用的原文数字，不需要为了文献回顾重跑。记录原论文版本、表号、指标、样本数、任务定义、模型是否冻结以及额外训练参数。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <text tabularNums weight="medium">P1</text>
    </box>
    <box flex="1" gap={1}>
      **仅重跑最必要的同模型控制组**

      先在 PointMaze / Reacher 复现 LeWM-CEM、CEM+GD、Gauss–Newton 和 Full D2P，使用固定 checkpoint 与 paired roots。先确认 action program composition 相比简单局部优化存在独立收益。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <text tabularNums weight="medium">P2</text>
    </box>
    <box flex="1" gap={1}>
      **用两个公开协议完成最具决定性的竞争**

      在 SAGE 的长时域 PushT/Cube 协议上评估 D2P，并在 Anchored Planning 的长程、扰动起点协议上检验 D2P。优先复用官方 manifests、checkpoint 和 success predicate；对没有执行过的竞争方法，保留其 *reported* 数值，但不进行未经验证的显著性比较。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}>
      <text tabularNums weight="medium">P3</text>
    </box>
    <box flex="1" gap={1}>
      **再投入 PushT 接触重配置及跨模型验证**

      如果前两项通过，就增加真正体现 D2P 创新的接触位置改变、姿态重配置、绕行与动作顺序任务。最后在第二个模型架构上检查方法是否仍然成立。
    </box>
  </row>
</box>

还建议你从开始就记录三个计算量：world-model forward calls、backward calls、实际 wall-clock latency。因为 CEM、GRASP、SVD 和程序编译的计算结构不同，仅匹配“候选数量”并不等价于匹配规划成本。

此外，原 LeWM 的 96% PushT、Hi-LeWM 的 94%、SAGE 的 89.3% Base H25，以及 Anchored Planning 的 7% 长程 PushT，并不互相矛盾：**它们在不同 goal offset、采样 queries、重规划频率和执行预算下测得。** 必须保留这些协议差异。<Cite refs={["turn280087view0","turn820228view1","turn820228view3","turn265724view3"]}/>

## 我的最终取舍

**必须正面挑战的四个研究是：GRASP、Anchored Planning、SAGE、Hi-LeWM。**

它们分别挑战 D2P 的可微搜索效率、无需后训练的中间目标、长时域动作提议，以及高层动作抽象。PRISM、IMWM 和 Hierarchical WM 则构成第二层强性能参照。

如果目标仅是扎实主会论文，我认为同冻结 LeWM 上的 CEM、GRASP、AP 与 D2P，再加一套长时域 RGB-only MPC，就可以形成最基本的可审稿证据。

**但如果目标是 Oral / Best Paper，仅超过这些方法的平均成功率仍不够。** 我更希望看到一种此前方法难以完成的结果：D2P 能从冻结模型中自动合成“先改变未来动作可达性、再完成目标”的动作程序，而且这种收益相对 AP 的记录轨迹检索、SAGE 的训练式 subgoal proposal，以及 GRASP 的直接轨迹优化，都具有明确、可重复的增量。

这才是第一篇论文值得集中证明的核心方法论。

-----

> 需要做什么才能达到研究方案的闭环？为什么我P1 gate一直无法通过 [BokaiGuo/LeWorldModel: Transfer-sufficient diagnostics and reproducible evidence for frozen visual world models](https://github.com/BokaiGuo/LeWorldModel)

我会直接对照仓库最新的 **P1 gate 定义、执行报告、失败日志和对应代码**，先查清它究竟卡在数据、模型、规划器还是统计验收条件，而不是继续泛泛地建议“多跑实验”。然后我会给出一条从 P1 到第一篇 **零后训练 Planning Compiler** 完整闭环的最短路径，并明确哪些 gate 应保留、哪些可能需要重定义。

# LeWorldModel：为什么 P1 Gate 一直过不了，以及怎样完成研究闭环

我核对了公开仓库的统一论文计划、研究进度和执行报告。先给出最关键的诊断：

**你目前的 P1 不是单纯卡在实现或 GPU 上，而是卡在“一个诊断指标是否具有独立的真实决策价值”这个科学验收条件上。**

仓库中正式定义的 P1 是：

> **ATM/MSE decision-transfer baseline：matched diagnostic has independent planning consequence.**

也就是：在控制预测误差、动作辨识等因素后，你提出的额外诊断，必须能够解释或预测真实动作选择的后果。它不是“代码跑通”“发现排序错误”或者“某个 readout 的离线 regret 改善”就能通过的 gate。<Cite ref="turn662685view0"/>

## 一、目前公开证据说明了什么？

<box gap={2}>
  <row justify="between" align="center">
    **P0 · Counterfactual task contrast**
    <badge>部分完成</badge>
  </row>
  PointMaze 已记录 960 个 root 内候选对的约 **40.78% 排序反转率**，说明预测候选排序和真实任务后果之间存在明显差距。但这些候选对共享 root，不能视为 960 次独立实验。<Cite ref="turn551417view1"/>
  <divider color="subtle"/>
  <row justify="between" align="center">
    **P1 · Matched decision-transfer evidence**
    <badge color="warning">未通过</badge>
  </row>
  跨环境匹配实验的报告值为 $\Delta R=+0.05336$，预注册 sign-flip 检验 $p=0.0630$，没有通过原定统计门槛。它是一个正向信号，但不足以支持确认性结论。<Cite ref="turn551417view2"/>
  <divider color="subtle"/>
  <row justify="between" align="center">
    **P2–P4 · Learning / calibration / online planning**
    <badge>尚未闭环</badge>
  </row>
  旧统一计划仍把 P2 设为 GDD-WM、P3 设为 HTSC、P4 设为 RGB-only MPC。这与我们后来明确拆分的两篇论文路线已经不一致，需要正式调整研究阶段定义。<Cite ref="turn662685view0"/>
</box>

因此，**P1 的失败不能直接解释为“谱分析没用”或“整个研究方向失败”。它目前只能说明：按照现有数据、匹配方式、诊断定义和统计验收规则，你还没有证明该诊断具有足够稳定的独立 planning consequence。**

这里还存在两个不同的 P1，必须避免混淆：

- **仓库旧 P1**：ATM/MSE 匹配后的 decision-transfer 诊断效用检验。
- **我们后来为第一篇 Planning Compiler 提议的 P1**：谱动作几何能否直接改进动作搜索与程序合成。

这不是同一个科学假设。**旧 P1 不通过，不应自动阻止新 Planning Compiler 的算法实验；但新方法也不能拿规划收益反过来宣称旧 P1 已经通过。**

## 二、P1 反复失败的五个具体原因

### 1. 你的匹配条件很严格，但满足条件的独立比较太少

我进一步检查了实际实现：

[`run_matched_learned_transfer_benchmark.py`](https://github.com/BokaiGuo/LeWorldModel/blob/main/le-wm-main/experiments/empirical_spectral_audit/run_matched_learned_transfer_benchmark.py)

代码不是简单比较所有 checkpoint，而是先按照环境、训练阶段、latent dimension 和动作分布分组，再要求预测 MSE、完整 predictor eigenvalue spectrum，以及严格模式下的 inverse-action risk 足够接近。

默认相对容差为 5%，并要求匹配集合包含至少两种架构、两个训练种子。只有具备完整 Schur 指标和独立外部证据的分组，才进入主要统计检验。<Cite refs={["turn814154view0","turn336901view1"]}/>

这在控制混杂因素方面合理，但造成一个设计矛盾：

**你希望比较真实 learned models 的结构差异，却又要求不同模型具有非常接近的完整特征值谱。**

跨架构模型即使具有相近的任务性能和局部动作响应，也未必能通过这一严格匹配。

最终只有 **7 个可识别 matched strata**。这解释了为什么增加不少 checkpoint、候选动作和结果记录，依然没有带来足够强的 P1 证据。

更重要的是：增加同一个 root 内的 candidate pairs，并不会增加模型级或环境级的独立重复。

### 2. Bootstrap CI 看起来为正，仍不满足预先固定的显著性规则

你当前结果是：

$$

\Delta R=+0.05336

$$

$$

95\%\ \mathrm{bootstrap\ CI}
=[0.01150,\ 0.11661]

$$

$$

p_{\mathrm{signflip}}=0.0630

$$

表面看置信区间不跨零，但代码明确要求**bootstrap 下界为正，同时 sign-flip 检验 $p<0.05$**。后者没有满足。<Cite refs={["turn357343view1","turn814154view0"]}/>

这不是需要修改 `alpha`、更换 bootstrap seed 或增加抽样次数来解决的问题。

两种统计方法的小样本假设不同，而目前有效的独立 matched units 太少。你需要增加新的、真正独立的检验单元，同时确认各个 strata 是否共享模型训练种子或 outcome 数据，避免把相关的 strata 当作独立观测。

### 3. 数据曾经存在一个很严重的 Action Bank 语义问题

执行报告发现：**相同的 `candidate_id` 在不同 root 下，并不一定代表相同的原始动作序列。**

历史记录中，一些 outcome roots 的候选相对排序甚至大量翻转。随后通过 goal-relative canonicalization，离线 action-effect readout 得到了更稳定的正结果。<Cite ref="turn357343view2"/>

但这个修复带来了新的科学边界：

原先效果较好的某些 readout 使用了 privileged source-state / goal-frame 信息。它证明了正确动作坐标有用，却还不能证明**只有 RGB 输入的冻结模型能够获得相同的规划能力**。

这意味着你可能在解决两个不同的问题：

- 当前模型是否缺乏决策所需的信息；
- 当前候选动作的坐标系是否正确表达了任务方向。

如果不把两者分开，P1 的机制归因会不稳定。

### 4. Prediction Improvement 没有稳定传递到 Decision Improvement

仓库已经留下了一个非常有价值的反例：

在某个 standardized checkpoint 上，RGB successor prediction 改善约 **14.39%**，但 estimated goal-frame candidate regret reduction 反而为 **−0.0249**。

进一步对这个失败样本做 successor calibration，虽然训练预测误差降低，独立 outcome regret 却从 **0.0872** 增加到 **0.1503**。<Cite ref="turn357343view3"/>

这是你最初研究假设的有力动机，却同时解释了 gate 为什么难通过。

**你的某些谱、预测、动作效应指标确实会改善，但它们未必在同一个模型上与实际任务排序的改善同向变化。**

若 P1 依赖把多项诊断的排序平均后选模型，就必须先确认这些指标究竟有没有一致的任务含义。

实际源码采用 MSE、action inverse risk 和四项几何指标的**等权平均排名**，并预先将四项几何指标全部声明为 *lower is better*。<Cite ref="turn526506view1"/>

这里需要进行一次语义审计：每项指标是否经过同向定义、标准化和独立验证？不能因为变量名叫 alignment 就假设越大越好，也不能在结果出来之后反转符号。对于没有明确单调关系的指标，不应继续把它们加入等权综合评分。

### 5. 旧 P1 同时承担了“科学机制”和“跨环境泛化”两种过重任务

当前 cross-environment gate 要求：总体检验通过、至少两个可识别环境、每个环境中有至少两个架构和两个种子，还必须具备独立 planning consequence。<Cite ref="turn336901view1"/>

但以下三个命题强度完全不同：

$$

\begin{aligned}
H_1 &: \text{存在真实 action-to-task 失配}\\
H_2 &: \text{某个诊断能解释这种失配}\\
H_3 &: \text{该诊断可跨模型、跨环境预测规划收益}
\end{aligned}

$$

现在你已拥有支持 $H_1$ 的现象证据，以及对 $H_2$ 的一些条件性正结果；旧 P1 却在要求近似 $H_3$ 的严格确认。

**我建议保留旧 P1 的失败结论，但将机制发现、同环境复现、跨环境确认拆成三个不同的验收层次。**

这不是放宽旧 gate，而是让未来的研究不再被一个混合了多种科学主张的门槛永久阻塞。

## 三、先用代码确定：P1 究竟是哪一层失败

公开仓库里至少有两套相关 gate。

第一套是 `run_cross_environment_transfer_diagnostic.py`，用于检验**匹配后的跨环境诊断增益**。

第二套是 `independent_planning_gain_gate.py`，用于检验**冻结方法在全新规划 episodes 上的独立收益**。

后者的代码要求：

$$

\Delta_{\rm plan}
=
\mathbb E[
L_{\rm baseline}-L_{\rm method}
]

$$

并同时满足最少有效配对数、最低增益、bootstrap CI 下界和单侧 sign-flip $p<\alpha$。它还检查是否使用全新规划 episodes、模型和 planner 是否冻结、测试结果是否参与方法选择，以及运行时是否访问 privileged state。<Cite ref="turn336901view0"/>

这两个 gate 不能互相替代。

建议先在本地重新生成一次只读审计：

<CodeBlock language="bash">cd le-wm-main

# 1. Audit cross-environment matched diagnostics
python experiments/empirical_spectral_audit/run_cross_environment_transfer_diagnostic.py \
  --record-root "<checkpoint_record_directory>" \
  --output "<new_audit_directory>"

# 2. Separately audit fresh, paired planning outcomes
python experiments/empirical_spectral_audit/independent_planning_gain_gate.py \
  --input "<fresh_paired_planning_episodes.json>" \
  --output "<new_planning_gate.json>"</CodeBlock>

第一份审计重点检查 `strata[*].reason_if_gated`、`missing_identifiable_environments`、`primary_delta_risk_mse_minus_full` 和 `p_one_sided`。

第二份审计重点检查 `invalid_records`、`protocol_gate`、`statistical_gate` 和 `paired_inference`。

可以按照下面的诊断表处理：

| 实际失败状态 | 代表什么 | 应做的事情 |
|---|---|---|
| `fewer_than_two_quality_and_spectrum_matched_candidates` | 没有足够严格匹配的模型 | 设计新的 matched model panel |
| `only_one_architecture_in_matched_pool` / `only_one_seed_in_matched_pool` | 模型重复不足 | 补独立架构和训练种子 |
| `task_metric_schur_missing_from_at_least_one_candidate` | 诊断字段不完整 | 重新导出诊断字段，不改变结果 |
| `independent_action_risk_or_planning_consequence_missing` | 缺少独立物理动作或规划证据 | 补真实动作干预与独立 outcome |
| `gated_positive_delta_not_established` | 数据可识别，但增量效应未通过检验 | 分析模型和机制；不能通过放宽阈值解决 |
| `gated_independent_planning_gain` | 独立规划的协议或效果不足 | 根据 `invalid_records` 与统计结果分别排查 |

这些状态和校验条件来自两套实际源码。<Cite refs={["turn814154view0","turn336901view1","turn336901view0"]}/>

**如果目前主要是 `gated_positive_delta_not_established`，那说明 P1 的问题已经不是写更多接口，而是现有指标组合没有产生足够稳定的独立收益。**

---

## 四、怎样才能真正完成第一篇 Planning Compiler 的研究闭环？

结合我们最近明确的新方向，我建议正式把第一篇论文改成以下阶段，而不是继续以旧 ATM→GDD→HTSC 的顺序作为所有实验的前置条件。

<box border radius="xl" padding={3} gap={2}>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **A0 · Establish a solvable control benchmark**
    <text size="xs" color="secondary">Oracle candidate ceiling · Valid action interface · Time-aligned objective</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center" gap={1}>
    **A1 · Extract action-induced planning structure**
    <text size="xs" color="secondary">Finite-horizon Jacobian · Spectral action coordinates · Reconfiguration directions</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center" gap={1}>
    **A2 · Synthesize executable action programs**
    <text size="xs" color="secondary">Progress program · Reconfiguration program · Composition</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center" gap={1}>
    **A3 · Demonstrate independent planning advantage**
    <text size="xs" color="secondary">Same frozen model · Paired roots · Same planning budget</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box background="surface-secondary" radius="lg" padding={3} align="center" gap={1}>
    **A4 · RGB-only closed-loop MPC**
    <text size="xs" color="secondary">Execute · Re-observe · Recompile · Replan</text>
  </box>
  <row justify="center"><icon name="arrow-down" color="secondary"/></row>
  <box border radius="md" padding={3} align="center" gap={1}>
    **A5 · Cross-model and cross-task replication**
    <text size="xs" color="secondary">New checkpoints · New roots · Non-greedy tasks · Contact transitions</text>
  </box>
</box>

### A0：这是目前最值得先做的实验

对相同的 root、goal 和候选动作，分别使用：

| 对照 | 能确定什么 |
|---|---|
| Real-environment candidate oracle | 候选动作中是否存在真正有用的动作 |
| Perfect-dynamics MPC | 当前 horizon、目标函数和搜索预算本身是否有能力完成任务 |
| Real successor + current goal metric | 即使未来预测正确，原本的 latent metric 是否能选对动作 |
| Learned successor + privileged task cost | 动力学预测本身造成多少动作选择误差 |
| Frozen LeWM + CEM | 真正需要超越的原始性能 |

这项分解比继续在同一批 checkpoint 上调谱指标更重要。

如果 perfect-dynamics MPC 也失败，那么首先需要修复规划目标、候选生成、动作接口或 horizon，而不是继续归因于 world-model representation。

这里还有一个非常值得加入的最新对照：LeWMRO 的独立复现实验发现，在 receding-horizon MPC 中，如果优化终点是 $H$，但实际只执行前 $K$ 步，评估目标和执行时刻可能严重错配。该研究在自己的 PushT 设置中报告，`terminal@H` 为 17.0%，`prefix@K` 达到 94.5%。这不是你仓库的结果，但说明**在做任何新的谱规划器之前，必须先检查规划代价的时间索引**。<Cite ref="turn654611view0"/>

### A1：将谱分析从诊断改造成算法输入

当前应该围绕：

$$

G_H(h,U)=
\frac{\partial\widehat z_{t+H}(h,U)}{\partial U}

$$

而不是单独的 predictor eigenvalue spectrum。

再计算：

$$

Q_g^{1/2}G_HR^{-1/2}
=P\Sigma V^\top

$$

用 $V$ 生成低维 progress actions，用局部可达残差以及后续响应结构生成 reconfiguration actions。

真正的 A1 通过标准应是：

**冻结同一个 checkpoint，不改变训练参数，使用这些动作结构构造出的计划，能够比直接使用原始动作梯度或随机同维搜索更有效地完成真实任务。**

如果仅证明 SVD 数值合理，却没有产生可执行动作收益，那么 A1 仍只是理论工具。

### A2–A3：证明 Program Composition 不是增加计算量

在一个需要“先改变姿态或接触，再完成目标”的任务上，比较：

- CEM、CEM+GD、Gauss–Newton、GRASP；
- 只有谱坐标的 D2P；
- 只有 reachability value 的 D2P；
- 只有 reconfiguration programs 的 D2P；
- 完整 progress + reconfiguration + composition。

这里最重要的消融是：

$$

\boxed{
\text{Spectral-only}
\quad\text{vs.}\quad
\text{Full Program Composition}
}

$$

如果 Full D2P 只是因为调用更多次 world model 才获胜，不能作为新方法论的核心结果。

如果 Full D2P 能在相同预算下完成原本需要非贪心动作的任务，那么它才有机会区别于普通 CEM 和可微 MPC。

### A4：完成真正的 RGB-only 闭环

正式执行协议应该是：

$$

\boxed{
RGB_t
\rightarrow E
\rightarrow F^{(H)}
\rightarrow \text{Action Program Synthesis}
\rightarrow a_t
\rightarrow RGB_{t+1}
}

$$

然后重新生成程序。

部署阶段只使用 RGB、目标、动作历史和冻结世界模型，不使用 simulator state、真实后继标签、GDD teacher 或训练得到的额外 scorer。

因此，最终论文至少需要同时证明：

**模型生成的动作程序具有真实执行收益，并且这些收益在不断重新观察和重新规划以后仍然成立。**

A5 再检验这种收益是否能迁移到其他任务和独立 world-model checkpoint。

---

## 五、旧 P1 应如何继续推进，而不是反复碰同一堵墙？

旧 P1 仍有科研价值，但建议保留它作为**独立机制研究**，不要用它阻止 Training-Free Planning Compiler。

我建议这样修订执行策略：

| 现在的做法 | 建议的新做法 |
|---|---|
| 在已有小型 matched panel 上继续叠加指标 | 首先冻结现有负结果，生成新的独立模型和 outcome panel |
| 主要关注平均谱指标与跨模型排名 | 优先测量真正的 finite-horizon action-to-task response |
| 将多项指标等权混合 | 对每项指标单独检验任务含义和增量，再预先指定组合规则 |
| 从固定 action bank 推断规划能力 | 使用 random、CEM initial、CEM elite 和全新动作候选分别评估 |
| 希望一次 gate 同时证明机制与泛化 | 分为同环境机制确认、跨模型重复、跨环境外部确认 |
| 反复调整当前 test panel | 冻结当前数据为开发集，锁定方法后使用全新 confirmatory outcomes |

其中最重要的是增加**预测输入—输出响应与真实动作效果的成对比较**。

可以定义真实与预测的任务输出差异：

$$

d_H^{*}(h,U_i,U_j)
=
y_H^{*}(h,U_i)-y_H^{*}(h,U_j)

$$

$$

\widehat d_H(h,U_i,U_j)
=
\widehat y_H(h,U_i)-\widehat y_H(h,U_j)

$$

针对任务关键方向和真实候选差异，研究二者是否存在增益失真、方向旋转或有限时域效应缺失。

这与原先只比较静态 covariance、alignment、Schur 和 commutator 的做法不同：**它把谱分析对象直接放在实际动作选择所需要的物理变化上。**

不过，已有 P1 的 $\Delta R=0.05336,p=0.0630$ 必须原样保留为未通过结果。新定义的 response experiment 应作为新的预注册实验，不能回写旧 gate 或重新解释旧数据。

---

## 六、现在该停止什么、保留什么？

根据当前研究目标，我建议做一次正式的项目分流。

| 模块 | 第一篇 Planning Compiler | 第二篇 GDD-OPD |
|---|---|---|
| Frozen RGB encoder / predictor | 核心 | Student 初始化或基线 |
| Spectral transfer / finite-horizon Jacobian | 用于生成动作程序 | 用于任务方向蒸馏 |
| ATM / counterfactual task contrast | 机制分析与评估 | Teacher/student action-effect 比较 |
| HTSC | 非主方法；不作为算法依赖 | 可作为独立评估，不作为 teacher 监督 |
| GDD relational teacher | **排除** | 核心 |
| Encoder / predictor adaptation | **不允许** | 允许 |
| Teacher-generated preferences / OPD | **排除** | 核心 |
| RGB-only MPC | 最终主结果 | 最终 teacher-free student 验证 |

**第一篇的闭环是：动力学谱 → 动作程序 → 实际规划 → 闭环收益。**

**第二篇的闭环是：执行反馈 → teacher decision field → student 蒸馏 → 未见动作泛化 → teacher-free 闭环收益。**

这两条链应共享数据治理与 benchmark，但不共享需要后训练的最终方法组件。

---

## 七、最值得立即执行的四项工作

<box gap={3}>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}><icon name="file-search" size="lg"/></box>
    <box flex="1" gap={1}>
      **01 · P1 Failure Attribution Report**

      在旧 P1 的每个 matched stratum 输出可识别状态、被选中的 checkpoint、MSE-only 与 full selector 的差异、真实 regret 和失败原因。把失败分成：匹配不足、外部证据缺失、统计不足、真实效应不存在。不能只看最终一个 `gate=false`。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}><icon name="crosshair" size="lg"/></box>
    <box flex="1" gap={1}>
      **02 · PointMaze Oracle + RGB-only Baseline**

      先确认真实 candidate oracle、perfect-dynamics MPC 和原始 LeWM-CEM 能在相同动作执行协议下形成有效进展。时间索引、goal-frame、动作单位和终止条件都需要固定。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}><icon name="git-branch" size="lg"/></box>
    <box flex="1" gap={1}>
      **03 · D2P 最小 Reconfiguration 实验**

      只实现 progress program 与一层 reconfiguration program，在全新 roots 上测试经典 Gauss–Newton、CEM、GRASP 和 Full D2P。优先选择具有真实非贪心最优动作的状态，而不是普通直线可达目标。
    </box>
  </row>
  <divider color="subtle"/>
  <row align="start" gap={3}>
    <box background="surface-secondary" radius="lg" padding={3}><icon name="flask-conical" size="lg"/></box>
    <box flex="1" gap={1}>
      **04 · 冻结独立确认性实验**

      预先锁定模型、任务、起点、计算预算和主成功指标。使用全新 outcome episodes 检验真实闭环收益；如果结果失败，依据 oracle、metric、dynamics 与 search 的分解定位原因，而不是继续增加模块。
    </box>
  </row>
</box>

## 最终判断

**你的 P1 gate 持续无法通过，主要不是研究工作做得不够多，而是当前研究积累中“能被诊断解释的结构”和“能实际改善任务的结构”还没有成为同一个对象。**

你已经证明过后继预测可以改善、动作 readout 可以改善、局部物理效果可以改善，但这些不一定产生一致的动作选择收益。PushT 后续的接触指标修复也提供了类似证据：局部动作效应误差降低，但稳定姿态闭环收益仍未建立。<Cite ref="turn266230view2"/>

因此我不建议再继续用一个越来越复杂的谱诊断综合评分反复冲击旧 P1。

**更值得做的是把有限时域 action-to-task transfer 变成一个实际参与动作程序生成的算法对象。**

然后用独立真实规划结果回答：它是否让冻结模型获得了原来没有的非贪心搜索能力？

如果答案成立，第一篇就能拥有真正的研究闭环。旧 P1 可以作为机制分析中的未完成外推检验，而不是被强行包装成已经成功的理论证书。

还有一个核验边界需要说明：我检查的是当前能够读取的公开 GitHub `main` 文档及源码，执行报告包含截至 2026 年 10 月 24 日的记录；我无法看到你尚未推送的本地最新 P1 日志。因此，上面的判断针对**公开版本可确认的失败机制**，不能替代对本地最新一次 gate 输出的逐字段审计。