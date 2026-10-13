# 短期历史、事件记忆与接触模式：文献边界和项目落地

版本：2026-10-13  
范围：LeWM latent 的短期历史提取、Fourier/时频特征、事件记忆、接触模式和闭环规划。  
状态：文献与实验决策文档；不是论文正文，也不把相邻工作的结果当作本项目结果。

## 1. 要回答的具体问题

当前项目观察到：事件记忆相对于 current-state/action baseline 有稳定收益，但没有稳定超过历史 Fourier 控制。问题不是“再增加一个 latent feature 是否能降低离线误差”，而是：

> 对接触丰富、部分可观测的 learned latent，哪一种短期历史表示能恢复对 action-to-task transfer 真正有用的状态，并在独立闭环任务上减少错误排序、接触丢失和姿态误差？

因此必须把四个层次分开：

1. **表示层**：历史特征是否降低 held-out state/effect RMSE；
2. **可辨识性层**：历史是否恢复 velocity、contact mode、phase 或 action inverse；
3. **动力学层**：真实 successor 与预测 successor 是否一致；
4. **控制层**：短 pulse、重新观测和 mode-aware controller 是否改善独立 planning risk、contact retention 和 angle closure。

表示层的正结果不能越级成为因果动力学或规划成功证据。

## 2. 文献中的相邻解决方案

### 2.1 部分可观测和随机动力学：记忆不是简单频域压缩

- [PlaNet](https://arxiv.org/abs/1811.04551) 使用确定性与随机 RSSM，并用 latent overshooting 约束多步一致性。其关键不是把历史压成固定频率，而是让隐状态承担未观测动力学和不确定性。
- [Visual Foresight with a Local Dynamics Model](https://arxiv.org/abs/2206.14802) 采用局部视觉动力学、短视 lookahead 和 receding-horizon 执行。该范式说明在预测不可靠时，短预测—执行—重新观测通常比一次性长 rollout 更稳健。
- HiP-RSSM 类工作把隐藏动力学参数作为 latent state，用于非平稳系统；对本项目的启示是，事件记忆还可能需要表示当前动力学 regime，而不只是过去观测的摘要。

### 2.2 接触和混合动力学：显式 mode 比连续回归器更合适

- [ContactNets](https://arxiv.org/abs/2009.11193) 使用 signed distance、contact Jacobian 和互补约束表示 impact、stiction、sliding 等不连续机制。
- [Contact Optimization via Hierarchical MPC](https://arxiv.org/abs/2210.03442) 将接触位置/接触力与下层交互控制分开优化，说明接触几何不能只作为普通 pose regression 的残差。
- [D-LGP](https://arxiv.org/abs/2312.02731) 将离散任务模式与连续运动规划结合，并在模式改变时重规划；这对应本项目应显式预测 successor mode，而不是让单一 effect head 隐式平均 free/contact/sticking/sliding。
- [Real-Time Deformable-Contact-Aware MPC](https://arxiv.org/abs/2212.09234) 和几何感知接触 MPC 的共同点是把接触约束、接触位置和短时闭环放进规划变量，而不是只依赖一个静态 latent distance。

### 2.3 Episodic/retrieval memory：事件应由 cue 和边界触发

近期 latent/world-model 方向的 episodic recap、pose-conditioned retrieval、4D 几何历史和 future-token 方法，通常包含以下结构：

- 由当前观测或 pose/task cue 触发历史检索，而不是固定窗口平均；
- 对 action 或 observation change 使用 delta reconstruction；
- 在 episode boundary、phase transition 或 mode transition 处显式监督；
- 将历史用于 future prediction / action conditioning，而不是只用于重建当前 latent。

这类工作与本项目最相关的共同结论是：接触丰富任务的有效历史通常是**事件门控、模式条件和任务相关**的。它们尚不能证明 frozen RGB latent 加一个局部 adapter 就能稳定闭合姿态；其中 MemoryVAM、Mem-World、ForeTime-VLA、Ctrl-World 等近期工作在本项目中应作为预印本/作者报告结果引用，并单独核验版本与实验条件。

### 2.4 频域特征：擅长检测周期/冲击，不等于恢复混合状态

频域或时频特征在接触检测、振动分析和周期动作中常用于提取能量变化、冲击带和相位。它们对平稳、可重复、phase 对齐的检测/分类任务有优势，但接触切换往往是非平稳、稀疏、状态依赖的。固定的一阶谐波截断会丢失冲击时刻、接触持续时间、sign reversal 和 mode transition。

因此“Fourier 优于简单事件特征”并不矛盾：Fourier 可能提供更强的低维正则化，而事件特征仍然包含未被 current state 表示的增量信息。

## 3. 与本项目结果的逐项对照

### 3.1 已确认的事实

在 protocol-corrected、episode-disjoint 的 PushT temporal audit 中：

- causal-delay lifting：`10/10` split 胜出，matched RMSE 平均降幅约 `15.5%`；
- state+action time-frequency lifting：`10/10` split 胜出，matched RMSE 平均降幅约 `24.9%`；
- paired memory-4：`8/10` split 胜出；
- interaction lifting：仅 `3/10` split 胜出，平均 matched reduction 约 `-7.4%`；
- 加入完整过去 action history 后，paired memory-4 降至 `1/10`，支持“低样本/高维过拟合”解释，而不是“历史越多越好”。

fit-only adaptive selector 的独立结论是：它在 10 个 outer split 上都优于 current-state/action baseline，但没有可靠超过固定 historical first-harmonic Fourier control；因此事件记忆的正确状态是：

> `partial / conditional transfer evidence`：事件特征有增量信息，但当前没有证据表明它是比 Fourier 更好的通用短期记忆。

这些结果支持“latent 中存在可利用的短期历史”，不支持“已恢复可靠物理因果动力学”“已稳定解决姿态闭合”或“已获得视觉在线规划收益”。

### 3.3 真实跑批：horizon-aware frozen policy 与 selector stability

后续真实跑批在新的 held-out split 上复核了 horizon-aware 策略。当前已确认的定性结果为：

- `h=1`：正式策略固定为 Fourier carrier；其中一个 split 的 selector 稳定偏向 gated residual，但不改变预先冻结的跨 horizon policy；
- `h=2`：selector 在所有 split 上因稳定性不足而回退，不能把该 horizon 的局部选择当作可靠控制规律；
- `h=4`：五个 split 均回退到预先冻结的 Fourier+phase 策略；
- 因此 selector 的 split 内稳定性与跨 horizon 的正式策略必须分开报告。

这一批结果尚待完整数值表和不确定性汇总，但已经支持一个方法论修正：自动 selector 保留为 `selection-instability diagnostic`，正式控制器使用 calibration 阶段预先冻结的 horizon-aware policy。不得因为单个 `h=1` split 偏好 gated residual 而修改全局策略。

当前研究可检验的规律是：

> memory selection exhibits a horizon-dependent stability phase transition：短 horizon 可能存在局部事件收益，中间 horizon 的选择最不稳定，较长 horizon 需要预先声明的 phase-aware policy。

### 3.2 失败模式的文献化解释

| 现象 | 最可能机制 | 文献中的对应解决方向 | 本项目当前边界 |
|---|---|---|---|
| 事件特征胜 current state，但不胜 Fourier | 事件摘要未编码长期相位/连续 regime；Fourier 提供更强正则化 | RSSM/hidden regime、cue-gated retrieval、phase/time-to-transition | 还没有 mode/phase successor gate |
| interaction lifting 不稳定 | 特征维度超过 fit support，且动作接口与状态历史耦合 | action-conditioned latent、低秩/正则化 memory、PE 审计 | 低样本过拟合已定位，未解决 mode observability |
| mode 识别改善但 angle closure 不改善 | successor mode 可辨识，mode 内 torque sign/effect 不可辨识 | mode-conditioned effect、接触约束、短 pulse、receding observation | 接触保持可改善，姿态闭合仍未通过 |
| 多步 rollout 失败 | prediction-to-execution mismatch、非正规 transient、接触切换 | real/predicted transition gate、uncertainty fallback、短 horizon | 尚未得到 RGB-only 独立规划成功 |

## 4. 本项目应采用的机制结构

固定短期历史接口为

\[
\eta_t = \Phi(z_{t-L:t}, a_{t-L:t-1}),
\]

并将 memory 从“特征拼接”改写成四个可审计接口：

1. **事件门控记忆**：检测 velocity change、contact onset/loss、sign reversal、stagnation 和 phase transition；
2. **successor-mode head**：预测 \(p(m_{t+1}\mid \eta_t,a_t)\)，至少区分 free/contact/sticking/sliding/lost-contact/uncertain；
3. **mode-conditioned effect head**：预测 \(\widehat{\Delta y}_{t+1}=\sum_m p(m_{t+1}=m\mid\eta_t,a_t)F_m(\eta_t,a_t)\)；
4. **短 pulse 闭环**：只执行短 action pulse，重新读取 RGB，若 uncertainty 或 mode mismatch 超阈值则停止 torque 并进入 contact reacquisition。

Fourier 不删除，而是固定为强基线；event-aware、causal delay、GRU/SSM、event-gated SSM 和 retrieval memory 必须在相同 feature budget、相同 action interface 和相同 split 下比较。

正式策略接口冻结为：`h=1 -> fixed Fourier`，`h=4 -> Fourier+phase`；`h=2` 在 selector stability 不足时 fail-closed 回退到固定 Fourier。新的 horizon 只有在独立 calibration 协议中确定策略后，才能进入独立 test；不能使用 held-out planning 结果更新策略表。

## 5. 冻结实验协议

### 5.1 训练与评估分离

- fit：估计 PCA/basis、memory 参数、mode head、task metric 和 decoder；
- calibration：只用于阈值、uncertainty 和 fail-closed gate；
- test：root-disjoint 或 episode-disjoint，只评估预先注册指标；
- 所有 selector 必须 fit-only，不能用 held-out planning 结果选择 Fourier、事件权重或 snapshot。

### 5.2 必报指标

表示/可辨识性：held-out state/effect RMSE、event F1、successor-mode accuracy/macro-F1、action inverse risk、action ranking regret、PE 和 memory closure。

动力学/任务：real/predicted successor distance、held-out multi-horizon task error (`h=1,2,4,8,16`)、contact retention、contact-loss rate、block XY、angle error、terminal strict success，以及 paired zero/privileged/frozen RGB-only baselines。

### 5.3 预注册升级门槛

只有同时满足以下条件，才允许把事件记忆升级为有效 intervention：

1. 在所有预注册 held-out splits 上不劣于 Fourier，且在至少一个未见环境上显著优于 Fourier；
2. event/mode 指标改善必须伴随 action ranking 或 real/predicted successor gate 改善；
3. multi-step task error 和 contact retention 改善不能以 angle closure 恶化为代价；
4. RGB-only、fresh roots、无 privileged successor 查询；
5. mode accuracy 提升但 angle closure 不提升时，结论只能标为 `mechanism partial`。

若这些门槛不通过，保留最强诚实结论：事件记忆提供条件性表示收益，但 Fourier/其他 memory 仍可能是更好的正则化基线，接触丰富规划需要显式 hybrid mode 和闭环恢复。

## 6. 代码与 artifact 入口

- protocol-corrected temporal audit：`le-wm-main/experiments/empirical_spectral_audit/run_real_enriched_hybrid_audit.py`、`run_lifted_signal_information_audit.py`；
- fit-only adaptive memory selector：结果见 `le-wm-main/results/adaptive_time_frequency_memory_selector_20261013/summary.json`；
- horizon-aware frozen policy 的真实跑批汇总：待写入独立 append-only summary artifact；在完整数值表生成前，本节记录的 h1/h2/h4 结果仅作为 `partial` 证据；
- mode-aware shared-candidate adapter：`le-wm-main/experiments/run_pusht_mode_aware_shared_candidate.py`；
- 文献到机制和代码的映射：`le-wm-main/results/negative_cause_diagnosis_20261013/literature_mechanism_to_code_20261013.md`；
- 统一研究上下文：`RESEARCH_CONTEXT_20261012.md`；执行计划：`PLAN.md`。

所有旧 Fourier、event-aware、adapter 和负结果 artifact 必须追加保留，不覆盖、不删除、不通过测试集选择改写。

## 7. 当前论文可用表述

推荐：

> We find that short-term history is recoverable from frozen latent trajectories, but its utility is conditional: event-aware lifting improves over current-state/action baselines while not reliably surpassing a fixed Fourier control. Contact-rich transfer therefore requires event-gated memory, successor-mode prediction, and receding-horizon re-observation rather than frequency features alone.

中文：

> 我们发现 frozen latent 轨迹中存在可利用的短期历史，但其效用具有条件性：事件感知升维稳定优于 current-state/action 基线，却未可靠超过固定 Fourier 控制。因此，接触丰富的 transfer 需要事件门控记忆、successor-mode 预测和短视闭环重新观测，而不能仅依赖频域特征。

禁止：

- “事件记忆已经解决了接触模式识别”；
- “Fourier 失败，因此事件特征是正确记忆”；
- “held-out latent RMSE 改善证明物理因果动力学”；
- “mode accuracy 提升证明姿态闭合或 frozen visual planning 已解决”；
- “近期文献的成功结果可以直接外推到本项目 frozen RGB latent”。
