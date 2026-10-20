# Verified local transfer: literature-to-protocol update

更新时间：2026-10-19

本文档把当前 RGB-only cube adapter 的负结果与已有方法逐一对应，并冻结下一轮可执行协议。文献提供的是机制先例，不是对 LeWorldModel 的实验支持；2026 年预印本不作为已解决问题的证据。

## 2026-10-19 GPU follow-up: calibrated approach repaired the interface bottleneck

The first approach-prelude implementation used raw image displacement as if
camera axes were action axes, and approached before descending.  On a fresh
8-root panel this produced `1/8` paired wins and `7/8` ties for every arm,
which was an interface/control-protocol failure rather than evidence about
the frozen predictor.  Following the RGB visual-servo precedent, the protocol
was repaired to use `descent -> calibration-only pixel/action Jacobian ->
close`, with the same mapping and 30-step prelude shared by every arm.

The fresh CUDA/EGL panel is at
`le-wm-main/results/cube_rgb_calibrated_approach_20261019_fresh8/closed_loop.json`.
On 8 root-disjoint test episodes, the verified RGB local-effect arm improved
over the paired static carrier on `4/8` roots, tied on `4/8`, lost on `0/8`,
with mean final-distance delta `-0.05925 m`.  The frozen-WM agreement-gated
arm was identical (`4/8`, `4/8`, `0/8`, `-0.05925 m`) and recorded model/local
disagreement fallbacks on every root (6--12 per root).  The fused arm was
weaker (`-0.03930 m`, still `4/8` wins and `0/8` losses).  RGB geometry alone
remained non-robust (`3/8` losses, mean `+0.01035 m`).

This is a positive closed-loop result for calibrated RGB approach plus local
observed-effect verification, not evidence that the frozen LeWM predictor has
independent causal value.  In this panel the predictor did not add measurable
decision information beyond the verified local map; the agreement gate
correctly failed closed when the two rankings disagreed.  The result should be
replicated on another environment before being generalized beyond Cube.

## TwoRoom cross-task audit: verification helps only as a fail-closed gate

The first 32-root TwoRoom audit showed that the original effect-map gate was
neutral (`0` wins, `0` losses, `32` ties). A direct inverse policy, which
replaced the planner action with the locally inverted RGB effect map, was
harmful (`0` wins, `10` losses, `22` ties`). Rolling two-step hold-out
verification reduced but did not remove the harm (`0` wins, `3` losses,
`29` ties).

The cause is that instantaneous marker displacement does not encode the
door/wall waypoint semantics needed by the long-horizon planner. The formal
controller therefore remains:

```text
frozen planner action
  -> RGB conflict/effect verification
  -> reactive fallback only when the planner action is rejected
```

The direct inverse is retained as an adversarial counterexample, not a
deployed controller. Local action-effect identifiability is not sufficient for
task planning when the task requires topological or phase-level semantics.

## PointMaze held-out consequence: support/risk gating repairs the raw WM failure

On a new 20-root PointMaze test split (`1870--1889`), with readout and visual
dynamics calibrated only on `1800--1849`, raw `wm_rgb` achieved `2/20`
successes and mean final distance `1.0186`. The calibration-only
visual-dynamics carrier achieved `14/20` and `0.2813`. Adding the RGB support/
dynamics risk gate to the frozen WM achieved `15/20` and `0.2927`, while
releasing the learned effect adapter without the gate achieved only `11/20`.
The HTSC consistency arm achieved `14/20` and `0.2810`, essentially matching
the visual-dynamics carrier.

This repairs the original raw learned-WM negative result through selective
rejection, not through evidence that the frozen predictor is causally
sufficient. The gate falls back on `78.7%` of WM decisions; the effect
consistency arm falls back on `93.3%`. The key lesson is therefore consistent
across environments: the useful object is a calibrated permission/rejection
mechanism, while the learned predictor must abstain when its action-transfer
support is not independently verified.

Artifact:
`le-wm-main/results/pointmaze_verified_effect_20261019_test1870_1889/report.json`.

## 当前结果先行

GPU/EGL 评估已通过，`4-direction gain=1` 的 root-disjoint held-out 结果为：

| 方法 | held top-1 | held pairwise |
|---|---:|---:|
| fixed action chart | 1.00 | 1.00 |
| global adapter | 1.00 | 1.00 |
| probe-fitted local adapter | 0.50 | 0.50 |

4 个 root 中 1 个被 RGB contact gate 拒绝，support coverage 为 0.75。该结果不支持“局部探针自动改善排序”；它支持更窄的诊断：在仅有两个 probe 的情况下，局部线性图是恰好插值且不可验证的高方差估计器。

对应 artifact：
`le-wm-main/results/cube_multiview_spatial_adapter_20261019_4dir_contact_gain1_panel/transfer.json`。

加入冗余 probe gate 后重跑的 artifact 为
`le-wm-main/results/cube_multiview_spatial_adapter_20261019_4dir_contact_gain1_panel/transfer_redundant_probe_gate.json`：local 更新被 4/4 次拒绝（3 次 `insufficient_probe_redundancy`、1 次 RGB contact gate），carrier fallback 的 held top-1/pairwise 均为 `1.00/1.00`。这不是创造新的预测能力，而是把原来的错误局部更新转换成可审计的 fail-closed 行为。

## 与现有工作的对应关系

| 文献 | 实际机制 | 对当前负结果的解释 | 落地改动 |
|---|---|---|---|
| [Deep Visual Foresight](https://arxiv.org/abs/1610.00696) | action-conditioned visual prediction + short-horizon MPC + repeated observation | 一次 predictor ranking 错误会被执行放大；单步 effect RMSE 不是闭环证书 | 使用 short pulse、重新观测、real/predicted successor residual gate |
| [Visual Foresight with a Local Dynamics Model](https://arxiv.org/abs/2206.14802) | 局部视觉动力学、短视 lookahead、receding horizon | 局部模型只有在当前邻域和动作支持内才可信；延长 pulse 反而跨越 contact mode | 保留 local map，但只在 support gate 通过时使用；12-step 结果不得替代 4-step 证据 |
| [Experience-Embedded Visual Foresight](https://arxiv.org/abs/1911.05071) | 用 test-time observed trajectories 做快速适应 | 当前 global adapter 与 local probe update 混合了两个不同的信息预算 | 把 probe 预算、更新收益和 frozen predictor 分开报告 |
| [ContactNets](https://arxiv.org/abs/2009.11193) | signed distance、contact Jacobian、互补/耗散结构 | RGB contact gate 只能证明接触可能性，不能恢复 mode 内 torque effect | contact 只作 permit 前置条件；不能把 contact retention 写成 pose success |
| [Contact Mode Guided Motion Planning](https://arxiv.org/abs/2105.14431) | 显式 contact mode 与连续规划联合搜索 | 8-direction 对角动作未被 probe 覆盖，属于 action-support 缺口而非 predictor 单纯欠拟合 | 用支持集内的轴向短脉冲组合替代一次性对角外推，并在每个 pulse 后重观测 |
| [PlaNet](https://arxiv.org/abs/1811.04551) / [Dreamer](https://arxiv.org/abs/1912.01603) | stochastic/recurrent latent state 与多步 imagination | 历史摘要必须服务于 action-conditioned successor，而不是只降低 latent RMSE | memory 通过 ranking regret、successor consistency 和闭环结果验收 |
| [HiP-RSSM](https://arxiv.org/abs/2206.14697) | 推断随状态变化的 hidden dynamics parameter | root-specific contact/grasp 是隐藏动力学变量；global map 不能假设跨 root 同一 | 将 root/regime uncertainty 纳入 gate；support 不足时 abstain/probe |
| [TAMPURA](https://arxiv.org/abs/2306.10603) | 不确定后果下把 information-gathering 纳入规划 | 探针不是额外诊断，而是有代价的规划动作 | 记录 probe cost、re-probe rate 和 abstention，不只记录最终排序 |

## 对当前两个负结果的处理

### 1. 4-direction：local probe variance

两个 rank-2 probe 对二维 map 只能产生精确插值，没有独立验证自由度。因此 local map 的零 probe residual 不能证明 transfer。正式策略改为：

```text
probe count < 3                  -> 不更新 local map
probe design rank < 2             -> abstain
contact gate failed               -> fallback / re-probe
held action outside support       -> 不允许 local extrapolation
```

对于四方向 bank，`fixed chart/global adapter` 是目前更可靠的 carrier；probe 只用于判定是否需要拒绝，而不是强行替换 carrier。

### 2. 8-direction：diagonal support extrapolation

三条 probe 不能认证五个 held candidate，尤其不能认证未观测对角 action。下一轮不再直接拟合对角 action 的一次性 effect，而是比较：

```text
一次性 diagonal pulse
vs.
axis pulse -> RGB re-observe -> axis pulse -> RGB re-observe
```

组合动作的每一步都必须位于已校准轴向 support；第二个 pulse 的方向和幅度只由 calibration split 冻结。评估同时报告：动作数、总 pulse 长度、re-observe 次数、ranking regret、contact retention 和 task outcome。若组合策略仍不能闭合姿态，失败应归因于 within-mode action identifiability 或 readout，而不是继续增加 memory 容量。

首轮 GPU held-root 结果已完成：direct diagonal 与 axis-composed receding 在 4 个 root 上都没有稳定 top-1 改善；post-pulse sequence adapter 的第二步 effect RMSE 为 `0.00809`，但 top-1 仍为 `0/4`。新增的 deterministic replay audit 对 12 个 root/sequence 重复行得到 source/intermediate/successor feature 最大差异 `0.0`，因此同一进程内 MuJoCo/EGL 轨迹是可复现的。此前不同进程报告之间的差异不能直接作为物理波动，必须以固定 score trace 和预捕获 initial state 重新审计。该结果尚不支持“组合动作已解决 ranking”。

进一步的跨进程审计发现：仅保存 seed 或全局 RNG 仍不能完全复现 root；variation/state 之外还存在环境初始化路径未显式序列化的状态。新增 `capture_cube_root_manifest.py`，冻结 `variation_values + qpos/qvel + goal`，并让 sequence training、two-pulse evaluation 和 replay audit 接受 manifest。manifest 仍需在正式 panel 上验证跨进程一致性；在此之前，旧 panel 的跨进程均值只作为 exploratory evidence。

随后使用冻结的 8-train/4-test manifest 重训 post-pulse adapter，并在 12 个 root 上完成两次独立 GPU 评估。两次指标完全一致：direct diagonal 为 top-1 `0.333`、pairwise `0.347`；support-preserving receding 为 top-1 `0.583`、pairwise `0.694`；paired 增量为 `+0.250` 和 `+0.347`。这是一条可保留的 action-ranking 正向证据，但仍不是姿态闭环或普适 world-model 证据；下一步需要独立任务成功率与更多 environment/architecture。

## 冻结的下一轮实验

在新 root-disjoint roots 上预先锁定三臂：

1. fixed/global carrier；
2. local probe update（当前方法）；
3. support-preserving two-pulse receding-horizon controller。

训练、calibration、planning test 完全分离。主估计量为 paired action-ranking regret；次要指标为 observed/predicted effect residual、re-probe/abstain rate、contact retention、angle closure 和 terminal success。任何只改善 effect RMSE 而不改善 ranking 的结果标记为 `prediction_only`。

## 结论边界

现有文献共同支持“局部模型 + 短脉冲 + 重新观测 + 信息获取动作”，但没有证明 frozen RGB latent 自动具有充分的 contact-mode 或 torque-effect 信息。当前实验也没有证明这一点。下一轮的科学问题不是再寻找一个更强的全局 predictor，而是检验：

> 在部署动作支持集内，观测后验证的短脉冲组合是否能把 frozen latent 的有限 action-effect 信息稳定传递到任务排序。

## 新增证据图：哪些先例真正能修复当前问题

本节只保留与当前 failure mechanism 直接对应的工作。论文中的成功结果不等于 LeWorldModel 已经复现；它们提供的是可迁移的设计约束。

| 先例 | 实际解决的对象 | 对当前负结果的直接启发 | 不可直接借用的部分 |
|---|---|---|---|
| [Experience-Embedded Visual Foresight](https://arxiv.org/abs/1911.05071) | 用刚刚观测到的新物体轨迹快速适应视觉动力学，并在控制任务中验证适应收益 | 把 adapter 从“跨 root 的固定全局映射”改成“当前 episode 的短轨迹条件化映射”；适应数据必须来自已执行轨迹 | 其输入是完整轨迹/视觉动力学，不能证明 frozen RGB latent 含有接触 torque 信息 |
| [ReOI](https://arxiv.org/abs/2506.16565) | 发现 world-model imagination 中物理不可信的视觉成分，干预观测后重新想象，并用于 action verification | 将 `predicted effect` 与 `observed RGB effect` 的残差作为 permit/reject gate；预测改善但观测残差不改善时必须拒绝该候选 | 该方法主要处理视觉 distractor，不等价于解决 action-support mismatch |
| [Balancing Exploration for Online Receding Horizon Learning Control](https://arxiv.org/abs/2010.07269) | 在 receding-horizon 控制中主动加入保持 persistent excitation 的扰动，并分析 regret/约束代价 | 探针应被设计成有限预算的 informative pulse，而不是随机补数据；记录 probe cost 与后续 regret | 理论设定是未知线性系统，不能直接给 nonlinear frozen latent 保证 |
| [Exciting Contact Modes in Differentiable Simulations](https://arxiv.org/abs/2411.10935) | 用 optimal experimental design 主动寻找信息量高的 contact modes，而不是均匀采样 | calibration action bank 应覆盖能区分 mode/effect 的方向；不能只用轴向 probes 再声称 diagonal transfer | 依赖可微模拟器和参数识别，RGB-only 部署需额外 observation gate |
| [Learning On The Job](https://arxiv.org/abs/2510.20483) | 在执行任务同时学习未知参数，按 task-relevant Fisher information 选择信息丰富的轨迹 | 选择 probe 的准则应是“减少任务相关 action-effect 不确定性”，而不是最低 latent/effect RMSE；探针要服务于当前目标 | 2025 预印本，且使用显式参数化控制模型；这里只采纳设计原则 |
| [Contact Mode Guided Motion Planning](https://arxiv.org/abs/2105.14431) | 显式枚举离散接触 mode，并在 mode 内搜索连续运动 | 把 `free/contact/stick/slide/lost` 当作不同局部 chart；跨 chart 的动作不得用单一线性 predictor 外推 | 其几何和接触模式可直接访问，不能当作 frozen RGB latent 的可得信息 |
| [On multi-step prediction models for receding horizon control](https://arxiv.org/abs/1802.09767) | 为不同预测步长直接构造多步模型，并用数据一致集合控制迭代误差 | 不再用 one-step RMSE 代表所有 horizon；每个 horizon 单独校准 residual 和 planning risk | 线性系统与显式数据一致集合假设不适用于当前视觉模型，需采用经验 calibration |

### 文献合并后的方法改动

上述先例共同指向一个比“训练更强 predictor”更窄、也更可验证的控制器：

\[
\text{RGB observation}
\rightarrow
\text{support/mode test}
\rightarrow
\text{short informative pulse}
\rightarrow
\text{RGB effect verification}
\rightarrow
\text{local update}
\rightarrow
\text{task-conditioned receding ranking}.
\]

因此下一版协议固定以下规则：

1. **先支持、后预测。** 候选 action 必须位于校准 action support；未覆盖的 diagonal action 不得由 axis-only predictor 直接外推。若任务需要 diagonal 效果，改为两个 support-preserving pulses，中间重新观测。
2. **先验证、后信任。** 对每个执行 pulse 计算 RGB successor residual。残差超过 calibration-only threshold 时，标记 `prediction_only`，拒绝更新局部图或拒绝执行下一次长 pulse。
3. **按 mode 分图。** `free/contact/sticking/sliding/lost-contact` 不能共享一个无条件的 action-effect map。RGB 无法区分 mode 时，不强行识别，回退到 carrier 或执行 information-gathering pulse。
4. **按 horizon 校准。** 对 \(h\in\{1,2,4,8\}\) 分别报告 action-ranking regret 与 successor residual；不能把 h=1 的预测优势外推为 h=4/8 的规划优势。
5. **任务效用优先。** 排序指标使用 calibration-only 的目标进展、姿态风险和 contact-retention 代理；不能用 effect norm 或 latent RMSE 作为任务效用的替代品。
6. **闭环是唯一外部验收。** 最终报告 paired final goal distance、best distance、terminal success、contact retention、probe/re-observation cost 和 abstention rate。离线 effect RMSE 只能作为机制指标。

### 对“为什么仍不能更好”的文献化解释

当前 `0/4` 或不稳定 ranking 不是单一网络容量问题，而是三个可分离的识别失败：

- **支持缺口（support mismatch）：** 训练只覆盖轴向 pulse，测试要求 diagonal macro；任何低误差拟合都不能认证未覆盖方向。
- **模式混合（mode mixing）：** 同一 RGB 局部位移可能来自不同接触模式，单个 predictor 学到的是平均 effect；Contact Mode Guided Planning 的做法是显式分 chart，而不是增大网络。
- **验证缺口（verification gap）：** effect RMSE 变小但 action ranking 不变，说明预测误差没有在任务相关方向上变小；ReOI、双控制和在线辨识都把观测后的验证/更新放入控制回路。

所以“修复成功”的判据必须是因果链闭合：

```text
support coverage / mode evidence
        -> observed-predicted effect residual
        -> action-ranking regret
        -> held-out closed-loop goal consequence
```

若其中任一箭头断裂，就只能报告对应的局部收益，不能把它包装成 frozen LeWM 已具有可靠物理因果动力学。

### 文献依据与来源状态

上述 arXiv 页面和摘要已在 2026-10-19 通过 arXiv 检索核对。2025--2026 的条目按预印本处理，只用于设计启发，不作为独立复现证据；较早的 EVF、在线 receding-horizon learning 和接触 mode planning 作为机制先例。完整论文主张仍需以各论文原文的实验设置为准。

### 文献机制的统一解释

这些先例并没有共同证明一个“更强 predictor”能够恢复姿态闭环；它们共同改变的是控制闭环的信任条件：动作必须有 support，后果必须被重新观测，局部模型必须按当前 regime 更新，信息获取动作必须计入代价。对当前 LeWM，最小可检验对象因此不是 latent prediction error，而是：

\[
\text{support/mode evidence}
\rightarrow
\text{observed effect residual}
\rightarrow
\text{task-conditioned ranking}
\rightarrow
\text{held-out consequence}.
\]

若只改善第一项或第二项，不能推出最后一项；这正是 PushT dense-bank follow-up 的结果。

### PushT dense-bank follow-up（CUDA, 2026-10-19）

同一高角度 8-root manifest 的 relocation-enabled 与 no-relocation 对照均为三臂 learned benchmark。关闭 relocation 后，predictor 的 final XY 从 `61.01` 降至 `30.69` px，local repair 从 `61.50` 降至 `25.53` px；但 predictor、fallback、local repair 均为 `0/8` strict pose success。local repair residual 为 `0.174`，仍未转化为 angle closure，且 wrong-sign events 约 `4.25/root`。结论是 relocation drift 已被隔离，但 within-contact identifiability 仍是独立瓶颈。

因此，后续实现应把 contact relocation 从姿态阶段移出，使用 contact lease、XY budget、observed successor gate 和 task-conditioned orientation utility；不得把 relocation 后的 XY 恢复计作姿态闭环收益。

### Terminal observed-effect control-flow audit

高角度 roots 默认立即进入 `terminal_handoff`，旧路径会跳过 adaptive observed-effect/Jacobian。新增 `--terminal-observed-effect` 后，paired probe rate 达到 `0.930`、residual 降至 `0.108`、final XY 为 `11.79 px`，但 Jacobian exploitation 仍为 `0`、final angle `1.844 rad`、pose `0/8`。这证明控制流遗漏已修复，同时证明“更多 probe / 更低 residual”仍不等于姿态闭环。

强 probe 与低阈值审计进一步显示：强 probe 的 residual 恶化至 `0.357`、XY `33.87 px`；阈值降至 `0.001` 仍没有 Jacobian exploitation 或 pose success。该 boundary 应作为 stiction/contact-chart non-identifiability 记录，而不是继续增加 probe 数量。

### Extended action-support oracle versus learned ranking

在同一高角度 8-root manifest 上扩展 normal/tangent bank 后，privileged successor oracle 达到 `4/8` pose、`8.53 px` XY、`1.072 rad` angle；同 bank learned panel 仍为 `0/8` pose，local repair residual 虽降至 `0.077`，angle 仍为 `1.844 rad`。因此 action support 不是唯一瓶颈，真正未解决的是 contact-conditioned sequence ranking：低单步 residual 没有转化为可积累的角度进展。

### Contact lease / active excitation follow-up

`run_pusht_mode_aware_shared_candidate.py` 现已支持 `--contact-lease`、`--orientation-xy-budget` 和 `--escalate-on-wrong-sign`。在同一高角度 8-root CUDA panel 上，contact lease 将 local-repair final XY 降至 `22.96 px`，但 pose 仍为 `0/8`；加入最大 moment escalation 后 wrong-sign 从约 `5.00` 降至 `1.625/root`，代价是 final XY 回升到 `44.60 px`，pose 仍为 `0/8`。这把 stiction sign 修复与 translation preservation 的冲突直接暴露出来。

新增 `experiments/audit_pusht_successor_beam_ceiling.py` 的 privileged horizon-2 beam audit 在 18-step budget 下为 `2/8` pose、`5.81 px` XY、`1.245 rad` angle。它是序列规划上界诊断，不是 frozen predictor 增益；后续只有在相同 pulse budget 下的 learned observed-effect beam 复现该链条，才可进入闭环主结果。

### Observed local Jacobian follow-up

为解决 exact-bin reuse 在 dense bank 下没有支持的问题，runner 增加了 `--online-local-jacobian`。在 no-relocation 同 manifest panel 上，local repair 的 observed-effect residual 降到 `0.138`、final XY 降到 `14.19 px`，但 final angle 为 `1.750 rad`、strict pose 仍为 `0/8`；Jacobian online exploitation 为 `0`，因为同一 contact region 没有满足 paired/rank-2 support。该结果说明当前需要的是主动的 region-preserving paired probe schedule，而不是继续扩大 predictor。

## 文献启发后的第一轮闭环验证（exploratory）

按“观测后验证/任务效用/回退”原则，已在 GPU 上跑通一轮独立 RGB-only Cube 闭环面板：16 个 calibration roots、8 个 root-disjoint paired test episodes、固定 20-step prelude、同一候选 action bank。artifact：
`le-wm-main/results/literature_verified_closed_loop_20261019/closed_loop.json`。

首轮结果如下，数值是相对 frozen latent-temporal baseline 的 final goal distance 变化（负值更好）：

| controller | mean paired delta (m) | wins / ties / losses |
|---|---:|---:|
| latent temporal | 0.0000 | 0 / 8 / 0 |
| support gate | -0.0186 | 2 / 4 / 2 |
| RGB geometry action | -0.0257 | 4 / 4 / 0 |
| RGB geometry temporal | -0.0257 | 4 / 4 / 0 |
| latent + RGB rank consensus | -0.0014 | 1 / 7 / 0 |
| latent + RGB one-step feedback | -0.0184 | 2 / 6 / 0 |

这轮结果给出一个比“更强 latent predictor”更具体的方向：RGB task geometry / post-action feedback 在小面板上改善了闭环 goal distance，而 latent temporal ranking 没有改善；这与 ReOI 的 action-verification 思路一致，也说明当前主要瓶颈可能位于 task readout / effect verification，而非单纯历史长度。该面板仍是 exploratory：样本量不足以声称统计显著、跨环境普适或稳定姿态闭合；下一轮必须扩大 root-disjoint panel，并预注册 paired bootstrap、terminal success、contact retention 与 abstention cost。

### Fresh-root 复核与局部修复

扩大到全新 16 个 test roots 后，上一轮 RGB geometry 的收益没有稳定复现：RGB geometry 为 `1 win / 15 ties / 0 loss`，平均变化 `-0.0030 m`；latent-temporal 为平均 `+0.0119 m`，support gate 为 `+0.0119 m`。这排除了“只要加入 RGB 几何就稳定解决”的过强解释。逐 episode trace 显示，原 ranker 在 post-prelude 阶段常常重复同一个 action 方向，导致候选排序没有形成真正的 observed action-effect update。

针对该原因，新增 `rgb_verified_effect` controller：

- 每个 pulse 后只从 RGB 估计 block displacement；
- 至少 4 条观测、二维满秩且条件数可接受后才拟合局部 action-effect map；
- 支持不足时执行固定四方向 exploratory pulse；
- 不读取 simulator state，也不使用 frozen predictor 作为动作选择器。

在另一组全新 8 roots、32 calibration roots 的 GPU panel 上，该 controller 相对 frozen static baseline 为 `5 win / 3 ties / 0 loss`，平均 final goal distance `-0.0473 m`；其 observed local map 在每个 root 有 9 次有效更新、3 次 support fallback。artifact：
`le-wm-main/results/verified_rgb_effect_20261019_fresh8/closed_loop.json`。

这条结果的科学解释必须保持精确：它支持“观测后局部 action-effect verification 能修复一部分 action ranking/闭环距离失败”，但不能支持“latent dynamics 已被修复”。因为该 arm 有意不使用 frozen predictor，当前更接近文献中的 online system identification / local visual servoing。下一步必须加入同预算的 `verified-local + frozen-WM`、`verified-local only`、`frozen-WM only` 和 oracle ceiling，并在新环境验证；否则只能把它作为 failure recovery controller，而不能作为 LeWM 内生几何收益。

### Frozen-WM 增量归因实验

随后完成了同一候选集、同一动作预算的三臂归因：`verified-local only`、`verified-local + frozen-WM fusion` 和 `verified-local + frozen-WM agreement gate`。修正三臂历史记录 bug 后，在另一组全新 8 roots 上得到：

| controller | mean final-distance delta (m) | wins / ties / losses |
|---|---:|---:|
| verified-local only | `-0.06466` | 4 / 4 / 0 |
| local + frozen-WM fusion | `-0.06436` | 4 / 4 / 0 |
| local + frozen-WM agreement gate | `-0.06466` | 4 / 4 / 0 |

artifact：[closed_loop.json](/home/gbk/code/LeWorldModel/le-wm-main/results/verified_gated_20261019_fresh8_v2/closed_loop.json)。agreement gate 在多数 pulse 上观察到 model/local disagreement（典型 root 为 9 次），因此保留 local action；它避免了此前融合臂在另一批 roots 上的退化，但没有带来额外收益。当前最强解释是：

> 当真实 RGB successor 已足以拟合局部 action-effect 时，frozen predictor 的候选排序不是增量信息，且未经验证的融合会污染局部决策。

这把问题从“如何把 frozen predictor 强行接入控制器”改成了可检验的 repairability 判别：`local effect verified + WM agreement` 才 permit；disagreement 时 `local fallback`；RGB support 不足时 `probe/abstain`。该结果仍不证明 LeWM latent 的物理因果性，反而给出一个明确失败边界：当前 frozen-WM 没有在这个闭环协议上提供可验证的额外决策信息。
