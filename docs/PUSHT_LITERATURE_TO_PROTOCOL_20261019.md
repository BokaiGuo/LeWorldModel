# PushT 文献机制到实验协议的落地说明

更新时间：2026-10-19

## 研究问题

当前 PushT 的负结果不是单纯的预测误差问题：接触模式、动作支持、短期观测验证、姿态任务效用和长程规划语义被混在同一个全局 adapter 中。本文回答一个可检验问题：已有工作如何处理类似的 action-effect 不可辨识，并把这些机制转成 LeWorldModel 的可执行约束。

## 最接近的先例与实际启发

| 先例 | 文献中真正做的事 | 对 PushT 的直接改动 | 不能据此声称 |
|---|---|---|---|
| Deep Visual Foresight / Visual Foresight with a Local Dynamics Model | action-conditioned visual prediction、短视 MPC、频繁重观测；局部模型只在当前邻域使用 | 每个 pulse 后重观测；将预测-successor 与 RGB observed-successor 做残差 gate；禁止长 open-loop macro | 低 prediction error 不等于 pose closure |
| Experience-Embedded Visual Foresight | 用刚执行的轨迹在测试时快速适应局部视觉动力学 | 将全局 adapter 与 episode-local update 分开；单独记录 probe budget、更新收益和 fallback | local adaptation 不等于 frozen latent 的因果充分性 |
| ReOI | 通过真实观察干预不可信的 imagined visual consequence，再决定是否继续使用模型 | 将 observed-effect verification 作为 permit/reject，而不是只做离线 diagnostic | 视觉一致性不能补足未覆盖 action support |
| Balancing Exploration for Online Receding Horizon Learning Control | 将 informative excitation 纳入 receding-horizon 控制，并衡量探索代价 | probe 选择最大化任务相关 action-effect 信息，而不是随机补数据；记录 probe cost 和 regret | 线性系统保证不能直接转给 nonlinear RGB latent |
| Exciting Contact Modes in Differentiable Simulations | 主动激发可区分接触模式，而不是均匀采样 | calibration bank 覆盖 normal/tangent/moment 方向；显式记录 mode coverage | privileged differentiable contact labels 在 RGB 部署时不可直接使用 |
| Contact Mode Guided Motion Planning | 按 free/contact/stick/slide 等 mode 使用局部 chart，再在 mode 内规划 | mode-conditioned local chart；跨 mode 不共享无条件线性外推；mode 不确定时 fail-closed | mode accuracy 高不代表 within-mode torque sign 可辨识 |
| TAMPURA | 将 information-gathering action 纳入规划目标，而非视为额外诊断 | 把 short informative pulse 作为有成本的规划动作；报告 re-probe、abstain 和任务收益 | 探针成功不能替代长程 waypoint/planner 语义 |
| On multi-step prediction models for receding horizon control | 为不同 horizon 单独建模/校准多步误差 | h=1/2/4/8 分开校准 residual 和 ranking regret；不从 one-step RMSE 外推长 horizon | horizon-specific calibration 不是 universal transfer certificate |

来源与核验状态：以上条目沿用 `VERIFIED_LOCAL_TRANSFER_LITERATURE_20261019.md` 中已核对的 arXiv 页面和摘要；预印本只作为机制先例，不作为 LeWorldModel 的独立实验支持。

## 对现有 PushT 负结果的诊断

现有结果应按三个可分离原因解释：

1. **支持缺口。** 训练/探针覆盖轴向短脉冲，测试候选包含未覆盖的 diagonal 或不同 contact edge。任何低误差拟合都不能认证外推动作。
2. **模式混合。** 相同的 RGB 局部变化可能来自 free、sticking、sliding 或 separation；单个全局 adapter 学到的是平均 effect。模式 head 改善只能说明 successor mode 更可预测，不能说明 mode 内 torque sign 已可辨识。
3. **验证缺口。** effect RMSE 下降但 action-ranking regret 不下降，说明误差没有沿任务相关方向下降；必须把 observed RGB successor 放回控制回路。

因此当前 PushT 的正式控制协议冻结为：

```text
RGB observation
 -> current support/mode test
 -> short informative pulse
 -> RGB successor verification
 -> local update only if residual/support gate passes
 -> task-conditioned ranking
 -> translation/orientation phase gate
 -> fail-closed carrier or re-probe
```

## 代码层面的验收约束

以下约束已对应到现有 runner 和测试，后续实验不得绕过：

- `physical_contact_geometry` / `boundary_action_frame`：动作必须在真实 T-shape 边界 chart 中定义，不能把 center-to-center 方向当作接触法向；
- `online_effect_key`：缓存必须区分 edge、fraction、normal/tangent action 分量和 contact distance，避免把不同 friction/mode 的样本混用；
- paired opposite pulses：单个零角度 probe 不足以证明 orientation effect；没有 matched opposite support 时拒绝局部角度更新；
- support-preserving two-pulse：未覆盖 diagonal 时改成 axis pulse → observe → axis pulse，不允许一次性 diagonal extrapolation；
- orientation/translation shield：姿态动作不能增加已闭合的 XY 误差；否则回退 translation carrier；
- held-out consequence：至少报告 ranking regret、observed/predicted residual、contact retention、angle error、strict pose success 和 fallback/probe cost；只改善 RMSE 的结果标为 `prediction_only`。

## 贡献边界与下一轮判据

文献支持的是“局部模型 + 信息性探针 + 重新观测 + 风险门控”的控制设计，不支持“frozen RGB latent 已具有可靠物理因果动力学”。PushT 下一轮只有在同一 root-disjoint manifest 上同时满足以下链条，才可称为正向任务结果：

```text
support/mode evidence
 -> observed-predicted residual decreases
 -> action-ranking regret decreases
 -> held-out angle error and strict pose success improve
```

如果第一、二步成立但第三步断裂，结论应是 within-mode action identifiability 或 pose readout/planner boundary；如果 ranking 改善而闭环不改善，则应把问题归因于 phase/long-horizon planner，而不是继续扩大 memory 或 predictor。

## Gap verdict

“用局部视觉动力学、短脉冲、重观测和在线更新解决 action-effect transfer”并非空白；这些机制分别已有先例。当前仍可成立的窄贡献是：在 frozen RGB/LeWM 条件下，把 support-aware observed-effect verification、mode-conditioned chart、任务排序和 fail-closed permission-to-act 放进同一个 root-disjoint 闭环协议，并显式证明每个环节的失效边界。

## 新增 PushT 机制边界

CUDA privileged line-search 进一步隔离了一个此前混合在一起的失败：纯切向脉冲可以保持 XY，但不能克服 stiction 产生足够角度变化；加入法向分量可以产生角度变化，却会把物体推离已经闭合的 XY 目标。三组参数 sweep 的结果是：纯切向 `(normal,tangent)=(0,0.08)` 得到 `0/16` 姿态成功、平均 XY 误差 `3.57 px`；小幅 `(0.02,0.04)` 得到 `2/16`、`21.17 px`；较大 `(0.10,0.16)` 虽然接触保持 `16/16`，但平均 XY 误差 `181.61 px`；加入短 XY restoration 后仍为 `2/16`、`28.15 px`。

这说明当前负结果不是单纯的 wrong-sign 问题。可辨识的控制对象是一个带约束的 hybrid action-effect：

\[
\text{angle progress} \quad\leftrightarrow\quad \text{normal-force-induced XY drift}.
\]

因此后续正式方法必须把 orientation pulse 的 translation cost 纳入 action utility，并拥有可验证的 phase-level XY restoration；仅增加 orientation ranker、mode classifier 或 contact gate 不足以解决姿态闭合。该 line-search artifact 是 privileged upper-bound/negative-boundary evidence，不是 frozen-LeWM 规划结果。

同一问题的 16-root matched orientation ceiling 也显示：equal-norm opposite-moment pairs 可以在 `16/16` roots 中构造，但 signed response consistency 只有 `9/16`，平均 paired angular separation 仅 `0.00995 rad`。因此 paired probe 能提供局部信号，却不能跨 edge/fraction 直接迁移；正式 gate 必须以 contact stratum 为条件，并将低信号或 sign 不一致状态标为 abstain。

## 组合主张的独立验收门槛

不能把接触保持、observed local effect 可测、以及 prediction error 改善三个局部结果相加来声称组合方法成立。组合主张只有在同一 frozen backbone、同一 action bank、同一 root-disjoint test roots 上同时满足以下三臂比较时才释放：

| 对照 | 必须回答的问题 |
|---|---|
| frozen predictor | 是否提供超出 RGB/reactive carrier 的独立决策信息？ |
| RGB/reactive verified fallback | 只靠观测后验证和安全回退能达到什么上限？ |
| support-aware local repair | 局部更新是否进一步降低 held-out ranking regret，并改善 angle closure？ |

固定估计量为：

\[
\Delta R_{\mathrm{pred}}=R(\mathrm{RGB\ verified})-R(\mathrm{frozen\ predictor+gate}),
\qquad
\Delta R_{\mathrm{repair}}=R(\mathrm{RGB\ verified})-R(\mathrm{local\ repair}).
\]

每个差异必须同时报告 action-ranking regret、observed/predicted residual、contact retention、angle error、strict pose success 和 probe/fallback cost。若 `ΔR_pred` 跨零而 `ΔR_repair` 为正，只能宣称 selective local repair 有效，不能宣称 frozen predictor 有独立增益；若只有 contact retention 改善，则只能宣称 permission-to-act 修复。

同理，任一未见 architecture、environment 或 action-support 条件下 predictor 增量消失或 sign consistency 下降，就构成对 universal transfer claim 的否证，不能用 spectrum、alignment、commutator 或单一 latent geometry 补回这一缺口。

## 2026-10-19 dense-bank learned follow-up：relocation 不是唯一原因

在同一高角度 8-root manifest、同一 dense action bank 和同一 CUDA runner 上，先运行允许接触 relocation 的 focused 三臂，再运行预先声明的 no-relocation 对照。结果如下：

| 协议 | predictor pose | fallback pose | local-repair pose | predictor final XY | local-repair final XY | local-repair angle |
|---|---:|---:|---:|---:|---:|---:|
| relocation enabled | 0/8 | 0/8 | 0/8 | 61.01 px | 61.50 px | 1.959 rad |
| relocation disabled | 0/8 | 0/8 | 0/8 | 30.69 px | 25.53 px | 1.641 rad |

关闭 relocation 将 predictor 的平均 XY 误差降低约一半、local repair 降低约 58%，并消除了 relocation-induced drift；但所有姿态成功仍为 `0/8`。local repair 在 no-relocation 条件下的 observed-effect residual 为 `0.174`，低于 predictor 的 `0.210`，而其 orientation wrong-sign events 仍约为 `4.25/root`。因此此前的负结果由两部分组成：

1. relocation 是显著的 translation-confound，会把“接触恢复”误写成姿态控制；
2. 即使去掉该 confound，within-contact action-effect/sign identifiability 仍未解决，不能通过继续增加 relocation、训练样本或 dense bank 自动得到 pose closure。

本结果把下一步方法冻结为 **contact lease + verified local orientation control**：姿态闭合阶段禁止未经 XY budget 验证的 relocation；每个 normal/tangent pulse 必须通过 observed successor、局部 support 和 task-conditioned translation cost gate。若 gate 不通过，执行 information-gathering pulse 或回退 carrier，而不是强行推进姿态。

对应 artifacts：

- `le-wm-main/results/pusht_highangle_densebank_focused8_cuda_20261019/`
- `le-wm-main/results/pusht_highangle_densebank_norelocation_focused8_cuda_20261019/`

该结果仍是 privileged mechanism audit，不是 frozen RGB/LeWM 独立规划增益；它支持的窄结论是：文献中的“短脉冲—重观测—局部更新”只有在 translation budget 和 contact lease 同时进入控制器时才是可解释的实验机制。

## 2026-10-19 contact lease 与 active excitation follow-up

将 contact lease 显式加入 runner：当状态进入近目标姿态阶段时禁止 relocation，并对 terminal orientation ranker 强制执行 predicted XY shield。相同 8-root panel 的结果为：

| controller | pose | final XY | final angle | effect residual | wrong-sign/root |
|---|---:|---:|---:|---:|---:|
| contact lease | 0/8 | 22.96 px | 1.840 rad | 0.154 | 5.00 |
| contact lease + max-moment escalation | 0/8 | 44.60 px | 1.834 rad | 0.185 | 1.625 |

因此最大 moment excitation 确实减少了 wrong-sign 事件，但会把 translation cost 放大；contact lease 的近目标 shield 也不能把高角度姿态变成可闭合任务。该结果支持把 sign confidence 与 task utility 分开报告，而不是用 wrong-sign 降低推断 pose closure 已解决。

新增 privileged horizon-2 beam ceiling（同一高角度 roots、dense bank、18-step budget）达到 `2/8` pose、平均 XY `5.81 px`、平均 angle `1.245 rad`。它只能说明短序列搜索有助于平移；由于该 budget 与 36-step greedy oracle 不同，不能与后者直接比较为 superiority，也不能作为 learned controller 证据。

当前正式修复协议冻结为：

```text
contact lease
 -> paired local sign evidence
 -> if wrong sign: max-moment excitation inside support
 -> observe XY/angle residual
 -> reject if XY budget is exceeded
 -> short-horizon task-conditioned sequence ranking
 -> fail-closed carrier / information probe
```

如果该协议在等预算、同 root-disjoint learned panel 上仍不能改善 angle closure，失败边界应明确归因于 within-contact controllability / action-effect identifiability，而不是继续增加 memory、Fourier 阶数或 commutator 特征。

## 2026-10-19 observed local Jacobian follow-up

为避免 dense action bank 下 exact-bin reuse 没有支持，runner 新增了严格 fail-closed 的 `--online-local-jacobian`。它只在同一 polygon contact region 内同时满足：至少正负 moment 观测、局部 action rank 为 2、每个 effect 有 informative angle response 时，才用 prior observed pulses 拟合局部 action-effect Jacobian。

同一 no-relocation 高角度 8-root panel 的 local-repair 结果为：effect residual `0.138`、final XY `14.19 px`、final angle `1.750 rad`、pose `0/8`。Jacobian 实际 online exploitation rate 为 `0`，因为 deployed roots 没有足够同-region paired support；这不是实现失败，而是 support-aware gate 的正确 abstention。

该结果将剩余瓶颈进一步收窄为：需要主动设计能在当前 contact region 内完成 paired excitation 的 probe schedule，并把 probe budget 纳入任务效用；仅增加训练样本、candidate density 或 predictor capacity 不足以恢复姿态闭环。

## 2026-10-19 terminal observed-effect control-flow repair

审计 trace 发现高角度 roots 从第一步就满足 `terminal_handoff=true`，旧代码因此跳过 adaptive online-effect/Jacobian 分支；这解释了此前 `online_identified_rate=0` 的一部分原因。新增 `--terminal-observed-effect` 后，paired probe 在姿态阶段真正执行：probe rate `0.930`，predicted/observed residual `0.108`，final XY `11.79 px`，但 Jacobian exploitation 仍为 `0`，final angle `1.844 rad`，strict pose `0/8`。

因此控制流 bug 已修复，但核心负结果仍存在：高频探针降低了预测残差，却没有产生可执行的 paired local chart。探针预算本身正在吞噬闭环步数，下一版必须显式优化“信息增益 / 姿态进展 / XY 漂移”的联合 utility，而不是无限重复 probe。

## 2026-10-19 stiction probe-scale audit

对 paired probe 做了两种针对性修复：

- `breakaway-paired-probe`：首次 pair 忽略小 calibration pulse，改用最大支持 moment；结果 local-repair XY `33.87 px`、angle `1.783 rad`、residual `0.357`、pose `0/8`；强 probe 离开局部 chart，反而恶化 transfer。
- 将 informative threshold 从 `0.005` 降到 `0.001`：结果与普通 terminal observed-effect panel 基本相同，local-repair XY `11.79 px`、angle `1.844 rad`、residual `0.108`、pose `0/8`、Jacobian exploitation `0`。

因此当前 stiction 不是单纯的阈值问题：弱 probe 的角度响应多数不足以稳定识别目标方向，强 probe 又改变接触 chart。可迁移的 action-effect 信息在该高角度 strata 中不存在于当前短脉冲支持内；继续调 threshold 或 moment magnitude 不应被包装成方法修复。

## 2026-10-19 extended action-support matched audit

为区分 action-support ceiling 与 learned ranking failure，在同一高角度 8-root manifest 上把 tangent bank 扩展到 `0.04/0.08/0.12/0.16/0.24/0.32/0.48/0.64`，normal bank 扩展到 `0.02/0.04/0.08/0.12/0.16/0.24/0.32`。

Privileged real-successor oracle 达到 `4/8` pose、平均 XY `8.53 px`、平均 angle `1.072 rad`；因此动作空间确实包含部分可行控制序列。相同 bank 的 learned three-arm panel 仍为 `0/8` pose：predictor angle `1.657 rad`，local repair angle `1.844 rad`，但 local predicted/observed residual 已降到 `0.077`。

这排除了“只要扩大 candidate bank 就能解决”的解释，并把负结果定位为 sequence-level action ranking / contact-conditioned transfer：模型能降低单步 residual，却没有识别出能持续积累角度收益的动作序列。

Artifacts：

- `le-wm-main/results/pusht_highangle_successor_oracle_extended_cuda_20261019/`
- `le-wm-main/results/pusht_highangle_extendedbank_terminal_focused8_cuda_20261019/`

## Literature-to-protocol update: verified local sequence ranking

The closest precedents support a specific control loop rather than a claim
that a frozen predictor is sufficient:

- **Deep Visual Foresight** ([arXiv:1610.00696](https://arxiv.org/abs/1610.00696)) and **Visual Foresight** ([arXiv:1812.00568](https://arxiv.org/abs/1812.00568)) use short-horizon visual prediction with receding-horizon re-observation.
- **Active Exploration for Robotic Manipulation** ([arXiv:2210.12806](https://arxiv.org/abs/2210.12806)) treats information gain as a planning objective under continuous contact, rather than assuming every probe is task-progressing.
- **Exciting Contact Modes in Differentiable Simulations** ([arXiv:2411.10935](https://arxiv.org/abs/2411.10935)) selects contact interactions for identifiability, motivating paired excitation inside a declared contact chart.
- **Model Predictive Robot-Environment Interaction Control** ([arXiv:2106.04202](https://arxiv.org/abs/2106.04202)) combines MPC with online system identification when the interaction model is unknown.
- **Simultaneous Learning of Contact and Continuous Dynamics** ([arXiv:2310.12054](https://arxiv.org/abs/2310.12054)) makes contact constraints part of identification; one-step prediction error alone is not the identification target.

The new matched run implemented the first conservative translation of these
ideas: a train-root-only joint two-pulse ranker, a fresh observation between
pulses, and no runtime successor access. It failed on the high-angle panel:
`0/8` strict pose, `37.22 px` XY, and `1.900 rad` angle. This rules out the
weaker explanation that re-observation plus a generic sequence ranker is
enough. The remaining method change is to make the second-pulse score
support- and task-conditioned: reuse observed effects only from the same
contact region and a rank-2 action basis; otherwise issue an
information-gathering pulse or fail closed. The ranker must use effect/utility
labels available from the declared observation protocol, not privileged
successor pose costs.

The context-matching intervention changed first-action training features to
the same imagined intermediate state used at deployment. It improved the
sequence arm (`1.900 -> 1.630 rad` angle and `37.22 -> 33.48 px` XY), but
strict pose remained `0/8`. A sequence-first/phase-second hybrid reached
`1.561 rad` and `27.43 px`, still `0/8`. Thus train/deploy covariate shift is
real but not sufficient; the residual failure is sequence-risk calibration
and contact-conditioned action-effect transfer. The next protocol must
calibrate sequence uncertainty and task utility on independent roots, with
abstention when the sequence margin is unsupported.
