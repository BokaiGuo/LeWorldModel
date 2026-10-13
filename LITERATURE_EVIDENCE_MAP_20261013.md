# Learned latent 短期记忆、接触动力学与视觉规划：证据地图

更新时间：2026-10-14
检索方式：逐条核对 arXiv 元数据与摘要；经典论文优先，2025--2026 预印本不作为已解决问题的证据。链接对应原始论文页面。

## 综述问题与边界

本综述回答：在部分可观测视觉 world model 中，已有工作如何处理短期历史、非平稳动力学、接触模式、价值保持和规划闭环；这些机制为什么不能直接解决当前 LeWM 的 frozen-latent、RGB-only、PushT 接触问题；哪些设计应进入下一轮协议。

不把“latent prediction RMSE 下降”视为规划成功；不把有 privileged contact state、结构化几何或额外 action training 的结果直接外推到 frozen RGB latent。

## 证据地图

| 工作 | 原文实际支持的机制 | 关键边界 | 对 LeWM 的直接启示 |
|---|---|---|---|
| [PlaNet](https://arxiv.org/abs/1811.04551) | 用 deterministic + stochastic latent transition 和 latent overshooting 处理部分可观测、多步 reward prediction，并在像素输入的接触任务上做 latent planning。 | 它训练的是完整 world-model/planner 闭环，不是冻结视觉 latent 后只接一个局部 adapter；成功也不等于接触 mode 和 torque sign 可由任意 latent 恢复。 | 多步 transfer 必须直接训练/评估，不能用一步 latent MSE 代替；需要 stochastic uncertainty 或闭环再观测。 |
| [Dreamer](https://arxiv.org/abs/1912.01603) | 在 learned compact state 中进行 latent imagination，并把 value gradient 反传到长时行为。 | 依赖可用于 imagination 的 learned recurrent state 和下游 policy/value learning；不证明 frozen encoder 的现有 latent 已经是充分状态。 | 把“记忆可解码”与“记忆可支持 action-conditioned imagination”分开；当前 LeWM 只能声称前者的局部证据。 |
| [Deep Visual Foresight](https://arxiv.org/abs/1610.00696) | action-conditioned video prediction + MPC，靠短视滚动规划处理 pushing 和未见物体。 | 依赖模型预测、candidate action proposals 和反复观测；一次错误 action ranking 仍会被执行放大。 | 使用 short-pulse receding horizon、real/predicted successor gate，而不是一次性长 pulse。 |
| [Experience-Embedded Visual Foresight](https://arxiv.org/abs/1911.05071) | 将新物体的 observed trajectories 编成 adaptation embedding，条件化 visual prediction。 | 它显式使用 test-time observed trajectories 做 fast adaptation；与 frozen adapter 的信息预算不同。 | 把 action-effect adaptation 与 frozen latent memory 分开计量；不能把适应收益归因给原 latent。 |
| [ContactNets](https://arxiv.org/abs/2009.11193) | 用 signed distance、contact-frame Jacobian、互补约束和 maximum-dissipation 结构表示 impact、non-penetration、stiction 等不连续接触动力学。 | 需要结构化接触几何/动力学对象；不是 RGB-only frozen latent 的无监督恢复。 | mode/geometry 要成为显式诊断和 gate；普通连续回归器不能独自承担 contact transition。 |
| [Contact Mode Guided Motion Planning](https://arxiv.org/abs/2105.14431) | 将离散 contact mode switch 与连续运动规划组合，自动枚举 mode 并指导搜索。 | 使用显式环境—物体接触结构，且部分计划以 open-loop 执行；不证明视觉 latent 能识别 successor mode。 | 将 mode observability 与 within-mode action identifiability 分开；两者未同时通过时禁止 pose planner。 |
| [HiP-RSSM](https://arxiv.org/abs/2206.14697) | 为变化动力学引入低维 hidden dynamics parameter，并在 RSSM 中做推断。 | 目标是变化系统的参数识别，不是接触摩擦/torque effect 的完整视觉可观测性；仍需足够多的 informative transitions。 | 记忆状态应包含 mode/hidden-dynamics uncertainty，而不是只有 latent history；action coverage 是前置 gate。 |
| [S4](https://arxiv.org/abs/2111.00396) | 用结构化 state-space kernel 建模长程依赖，提供稳定的线性记忆参数化。 | 长序列建模能力不等于决策充分性；平滑 SSM 仍可能把稀疏冲击和 mode switch 平均掉。 | Fourier/SSM 作为低方差 carrier；事件/phase 作为条件 residual，而不是强行统一所有 horizon。 |
| [Mamba](https://arxiv.org/abs/2312.00752) | 让 SSM 参数依赖输入，以选择性保留/忘记内容，解决固定记忆对离散事件不敏感的问题。 | 选择性状态更新本身不提供 contact label、action identifiability 或 planning-risk calibration。 | 可作为后续候选 memory backbone，但必须用 action-ranking 和 successor consistency 验收，而非 sequence loss。 |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 在大规模视频预训练后，用少量 robot video 后训练 action-conditioned latent world model，并展示 image-goal planning。 | 关键贡献包含 action-conditioned post-training、robot data 和完整 planner；不能等价于 frozen generic latent + local adapter。 | LeWM 若要声称规划，必须把 action conditioning、candidate support、receding-horizon verification 写入协议。 |
| [DeepMDP](https://proceedings.mlr.press/v97/gelada19a.html) | 研究 latent transition、reward approximation 与 representation quality 之间的 value-preserving 条件。 | 其抽象通常围绕给定任务/价值函数的 Bellman 保持，不直接处理 frozen RGB、有限动作激励、接触切换或真实 successor 校准。 | LeWM 不能把“只保留决策相关信息”作为新意；增量应是可操作的、候选动作级的 transfer 认证。 |
| *The Value Equivalence Principle for Model-Based Reinforcement Learning* | 说明模型不必恢复全部环境动力学，只要在声明的策略/价值函数集合上保持决策相关更新即可。 | value equivalence 是任务/函数类条件性的，不自动给出有限数据下的 permit/abstain 规则，也不等于视觉动作后果可辨识。 | 将 HTSC 明确定义为对有限候选动作和有限 horizon 的可校准决策证书，而非普适动力学恢复定理。 |

## 文献共识与当前负结果的统一解释

### 1. 记忆不是单一对象

已有方法实际上分别处理四类东西：

1. 场景记忆：物体/遮挡/几何位置；
2. 任务进度记忆：phase、subgoal、episode recap；
3. 动力学记忆：摩擦、接触持续时间、隐藏参数；
4. 决策记忆：上一个 action 造成了什么真实 successor effect。

当前 Fourier 主要承载平滑趋势和相位；event lift 主要承载变化；recurrent 分支试图同时承担全部四类，因而在小样本、非平稳接触数据中方差很高。当前结果“h=1 fixed Fourier、h=4 Fourier+phase、selector 不稳定”与这一分解一致。

### 2. mode observability 与 within-mode action identifiability 是正交瓶颈

ContactNets/CMGMP 通过显式 mode/几何解决 mode transition；HiP-RSSM 通过 hidden parameter 解决变化 dynamics；但这些都没有保证在 RGB-only latent 中，同一接触模式内的两个 torque 候选仍然有可辨识的角度 effect。

因此当前 `mode accuracy` 或 `contact retention` 的局部改善不能推出 pose closure；必须分别报告：

\[
\text{mode observability}
\quad\text{and}\quad
\text{within-mode action identifiability}.
\]

### 3. 规划收益依赖闭环，而不是单步 prediction

PlaNet、Dreamer 和 Visual Foresight 的共同点不是某个特定 memory basis，而是多步训练/imagined rollout 或短视 MPC + re-observation。它们不支持“单步 effect RMSE 改善就足以稳定规划”的推论。

### 4. 选择性 memory 也必须有支持集和不确定性门

Mamba 式 selective update、HiP-RSSM 式 hidden parameter inference 都要求输入带有足够信息。若 action distribution 没覆盖关键 torque、接触转移样本稀疏，任何 memory selector 都可能在 fit split 看似稳定、在新 episode 发生 inversion。

### 5. Value equivalence 不是本项目的空白

Value equivalence、state abstraction、bisimulation 和 DeepMDP 已经说明：world model
不需要恢复所有状态细节，只需要保留给定任务、策略或价值函数集合所需的决策信息。因此，
“决策充分性”不能单独作为创新声明。

LeWorldModel 的可争取增量必须更具体：在 frozen RGB latent、有限动作激励、部分可观测和
固定有限 horizon 下，利用独立真实 successor 数据估计候选动作级的 transfer uncertainty，
并输出是否允许规划的可执行判断。这里的关键不是抽象模型是否 value-equivalent，而是：

```text
有限视觉历史 + 动作支持 + 校准数据
  -> 候选动作的 transfer-risk bound
  -> action-ranking confidence
  -> permit / abstain
```

这也是 HTSC 与传统 value-equivalent abstraction 的区别：前者应当是部署时、候选动作级、
支持集条件化的判断协议；后者主要刻画抽象模型在给定价值/策略类上的保持性质。

### 6. 两条科学主线不能混为一谈

HTSC 与 GDD-WM 共享 candidate data、\(G_h^\Phi\)、action support、real/predicted
successor 和独立规划评估，但成功标准不同：

- **HTSC — Can we trust the world model?** 给定冻结模型，判断任务相关信息是否存在、动作
  后果是否可辨识、预测是否可靠到足以选动作；结果可以是 `permit` 或 `abstain`。
- **GDD-WM — Can reliable decisions improve the world model?** 利用执行验证过的动作偏好
  学习预测几何，同时保持真实动力学一致性，并在新候选分布上取得独立收益。

GDD-WM 的训练收益不能反向证明原 frozen latent 已经具备充分性；反之，HTSC 的拒绝或放行
也不能被写成 GDD-WM 的蒸馏结果。

## 对 LeWM 方法论的调整

### 新研究对象

不再把目标定义成“恢复 latent 的全部历史”，而定义 horizon-specific transfer sufficiency：

\[
P(y_{t+h}\mid \mathcal H_t,a_{t:t+h-1})
\approx
P(y_{t+h}\mid \Phi_h(\mathcal H_t),a_{t:t+h-1}).
\]

候选 memory 只有在独立 held-out 上同时改善 action-to-task transfer、real/predicted successor consistency 和任务风险时才具有 planning relevance。

### 新结构

\[
m_t^{(h)}=F_h(z_{t-L:t})+g_h(t)E_h(z_{t-L:t},a_{t-L:t-1}),
\]

其中 `F_h` 是固定 Fourier carrier，`E_h` 是 phase/event residual，`g_h` 是只在 calibration split 声明的因果 gate。selector 不部署，只报告 selection instability；support 不足或 instability 过高时 fail-closed。

当前代码中的正式 policy profile：

```text
h=1: fixed Fourier
h>=2 candidate profile: Fourier + phase
selector: diagnostic only, never deployed
```

该 h>=2 profile 已在新的 split-seed confirmation 中测试；它仍是同一 JEPA-PushT export 内的 episode-holdout 证据，不能外推为跨架构/跨环境结论。

### 新验收链

```text
memory lift
  -> successor/mode observability
  -> within-mode action identifiability
  -> real/predicted successor consistency
  -> short-pulse closed-loop transfer
  -> pose/planning outcome
```

任一箭头断裂，都报告为机制边界，而不是通过 composite weight 包装成成功。

## Gap verdict

**Partly holds.** 文献已经覆盖了 latent overshooting、recurrent hidden state、SSM memory、视觉 MPC、显式 contact mode、结构化接触动力学以及 value-preserving abstraction；因此“提出一种 memory/adapter”或“只保留决策相关信息”都不是空白。

仍成立的窄 gap 是：在 frozen RGB latent、无 privileged contact state、有限 action support 的条件下，能否以 horizon-conditioned、support-aware 的 transfer sufficiency protocol，校准候选动作级风险并预测何时历史信息真正转化为 action/planning gain。这个 gap 需要跨 architecture、environment、真实 successor 和独立 planning consequence 才能成立。

## 目前可以写进论文的句子

> Prior work resolves partial observability, long-horizon latent prediction, or contact discontinuities by adding recurrent uncertainty, action-conditioned training, closed-loop MPC, or explicit contact structure. These mechanisms are complementary rather than interchangeable. Our controlled audit therefore treats memory as horizon-conditioned transfer sufficiency, keeps fixed Fourier as a low-variance carrier, and reports selector instability as a risk signal instead of deploying an outcome-tuned memory selector.

中文：

> 现有工作分别通过随机/递归隐状态、action-conditioned 训练、闭环 MPC 或显式接触结构处理部分可观测性、长时预测和接触不连续性；这些机制不能互相替代。因此本研究将 memory 定义为有限时域 transfer sufficiency，以固定 Fourier 作为低方差载体，以 phase/event 作为条件 residual，并把 selector instability 作为风险诊断，而不是部署一个根据结果调出来的 memory selector。

进一步需要补充：

> Value-equivalent abstraction 已经说明决策不需要保留全部动力学；本研究的窄问题是，有限 RGB 动作干预能否把 frozen latent 的候选动作 transfer uncertainty 校准为可执行的 `permit/abstain` 证书，并在独立闭环结果上得到验证。

## 证据等级与未完成项

- **已核对的设计先例**：PlaNet、Dreamer、Deep Visual Foresight、Experience-Embedded Visual Foresight、ContactNets、CMGMP、HiP-RSSM、S4、Mamba、V-JEPA 2 的 arXiv 元数据/摘要。
- **当前 LeWM 正向证据**：同一 JEPA-PushT enriched export、fit-only POD rank 32、episode-disjoint split；candidate profile 在 h=2 的 5 个新 split 中相对 fixed Fourier 胜 4/5，在 h=4 胜 5/5。
- **尚未证明**：action ranking、contact-mode successor prediction、pose closure、frozen online planning、跨 architecture/environment transfer、universal world model、commutator certificate。
- **不应升级的原因**：当前正向结果仍是 transfer-sufficiency proxy；还没有独立的 planning consequence，也没有跨环境/架构确认。
