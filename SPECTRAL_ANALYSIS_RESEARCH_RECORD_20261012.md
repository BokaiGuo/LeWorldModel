现有的研究结果在我们主线的谱分析上的推进如何呢







21:15






用时 1分钟 42秒



谱分析主线已经有明显推进，但现阶段最强结论仍是“严格的诊断框架与识别边界”，还不是 learned world model 的普适模型选择准则。
层次	当前进展	结论状态
同谱不同传输	已证明 predictor spectrum 不足以决定 covariance-relative transfer；得到精确的 \(\sin(2\theta)\) 关系	已成立
受控诊断 benchmark	9 个同谱 predictor、128 次重复；alignment selection win rate = 1.0，Spearman \(\approx0.999\)，8-step transfer 差异稳定	已成立，但仅限受控线性族
Gauge 理论	已证明普通 Euclidean commutator 非 \(GL(d)\) 不变；metric-aware 版本具有协变性；covariance-only whitening 的 alignment 退化为恒等映射	已成立
非正规扩展	Schur off-diagonal、resolvent、transient amplification 已纳入；固定 spectrum 且 commutator 为零时，Jordan coupling 仍可改变有限步增益	已成立，且否定 commutator 单独作 certificate
非线性扩展	已加入 local Jacobian 与 Taylor remainder bound	已成立，但仅为局部分析
learned checkpoint	有弱到中等的 alignment–transfer 相关性，但跨架构、环境和 protocol 不稳定；最新严格 matched benchmark 为 0/6 个可识别 matched strata	尚未成立
LeWM/规划	能检测 action-correlated latent structure，但没有证明物理因果动力学或规划收益	尚未成立


最关键的正结果来自：
[受控 covariance–predictor benchmark](/home/gbk/code/LeWorldModel/le-wm-main/results/covariance_predictor_diagnostic_benchmark_20261008/diagnostic.md)
它证明了：
在固定 predictor spectrum、固定 eigenspace family 和固定 operator parameterization 的条件下，covariance-relative alignment 可以提供 spectrum 之外的 multi-step transfer 信息。

但真实 learned checkpoint 的结果明显弱得多。已有 probe 实验中 alignment–held-out transfer 的 Spearman 大约只有 0.23–0.26，且不同环境、架构和 horizon 的收益不一致。更严格的最新审计甚至没有形成足够的 MSE+spectrum matched stratum：
[matched learned transfer benchmark](/home/gbk/code/LeWorldModel/le-wm-main/results/matched_learned_transfer_benchmark_20261011/matched_learned_transfer_benchmark.md)
因此当前不能声称：
- alignment 已普遍预测 learned-world-model 规划性能；
- commutator 是通用 transfer certificate；
- learned checkpoint 中已经验证了条件性模型选择准则。
主线目前最有价值的科学贡献已经清晰收敛为：
Eigenvalue spectra alone are non-identifying for covariance-relative latent transfer. A valid analysis must specify the latent metric and jointly report alignment, Schur geometry, and finite-horizon amplification.

这比单纯提出“交换子很重要”强得多，因为已经完成了：
1. 同谱干预；
2. matched one-step / multi-step 对照；
3. gauge-aware 理论；
4. 非正规 counterexample；
5. 非线性 remainder 控制；
6. learned checkpoint 的负结果审计。
当前真正缺的不是更多 PushT 消融，而是跨架构、跨环境的可识别 matched learned-model 实验。按照 [PLAN.md](/home/gbk/code/LeWorldModel/PLAN.md) 的门槛，在获得至少两个架构、两个环境、多个 seed 的有效 matched strata 之前，论文最稳妥的定位仍是：
a gauge-aware, non-normality-aware covariance–predictor transfer diagnostic framework with a validated controlled mechanism and an unresolved learned-model generalization boundary.

换句话说：理论主线已经成型，受控实证已经有决定性结果；尚未闭合的是从“受控 operator-pair 诊断”到“真实 learned model 的稳定 transfer-risk prediction”。


21:17







进行完整完全的论文调研，看看别人的研究是否做过类似问题，他们提出的解决的方法是什么，达到什么样的效果，我们的创新点要怎么建立







21:18






用时 4分钟 31秒



总判断
完整调研后，原先“latent alignment 能解释并解决 world-model planning 失败”的强叙事已经不能成立。近期工作已经分别解决了：
- latent prediction 与 planning 的错配；
- action-effect 不可辨识；
- latent cost 与真实任务进展的错配；
- contact mode 与 precondition；
- 非正规动力学和 planner-reachable distribution；
- frozen visual encoder 上的任务适配。
因此，我们的创新点不能再定义为“发现 alignment 对规划重要”，而应收缩并强化为：
一个 gauge-aware、metric-relative、non-normality-aware 的 covariance–predictor diagnostic framework，用于判断谱信息何时不足以识别 latent transfer，并解释现有 decision/action diagnostics 为什么仍然不能自动形成可靠规划。

一、最接近的谱分析与表示几何工作
1. Representation alignment 与 spectral geometry
工作	已解决的问题	方法与效果	与我们的差异
Latent Functional Maps, 2406.14183	两个 learned representation space 如何建立对应	用谱几何构造 functional map，支持跨空间 correspondence、stitching 和 retrieval	比较的是两个 representation space，不是固定 \((\Sigma,P)\) 下 predictor transfer
Spectral-Grassmann Wasserstein, 2509.24920	如何比较不同 dynamical systems 的谱结构	联合使用 eigenvalues 与 spectral projectors，构造 sampling-rate invariant 的 Wasserstein metric	与我们“不能只看 eigenvalues”高度相关，但目标是系统间距离，不是 covariance-relative transfer
Principal angles / spectral projectors	子空间关系在重复特征值和基变换下如何稳定表示	用 projector overlap、principal angles 替代单个 eigenvector angle	它们是标准工具；我们的新增点只能是把它们组织进 covariance–predictor pair 的可证伪诊断
ResDMD, 2111.14889	如何避免 Koopman spectrum 的 spectral pollution	用 resolvent、pseudospectrum 和残差提供谱估计误差控制	解决谱估计可靠性，不研究协方差相对 eigenspace alignment


2. 非正规算子与谱可靠性
近期工作进一步说明，仅仅补充 commutator 仍然不够：
- Understanding Latent-Dimension Scaling through Spectral Reliability, 2610.11866 用 eigenpair residual 检测 spurious eigenmodes，发现 latent one-step prediction loss 下降并不保证谱结构可靠。
- 经典 non-normal operator literature 已经明确：相同 eigenvalues 可以对应完全不同的 Schur off-diagonal coupling、pseudospectrum 和 transient amplification。
这支持我们目前的 Jordan counterexample，但也说明：
“同谱不同 transfer”本身并不是新的矩阵分析事实；真正需要证明的是 covariance–predictor pair 诊断在 learned model 中有额外预测价值。

二、最近工作已经直接处理 latent planning 失败
这是对我们创新性影响最大的部分。
1. Prediction error 不等于 planning quality
A Control Theory of Predictability in Latent World Models, 2607.10362 明确指出：
- data-averaged prediction error 不一定预测 control success；
- planner 访问的是 planner-reachable distribution，而不是训练分布；
- control error 可分解为 on-manifold residual 与 off-manifold divergence；
- non-normality 会放大 finite-horizon error；
- planner-reachable fidelity 比普通 validation MSE 更能预测控制。
这与我们的结果高度一致，但它已经提供了更直接的 control-theoretic 解释。因此我们不能再声称“首次发现 prediction spectrum 不足以保证 planning”。
2. Action-identifiability 与 real/predicted transition mismatch
ATM: Why Latent World Models Can Fail to Plan, 2606.09028 是当前最重要的竞争工作之一。
它提出 Action-Consistency Transfer Matrix，比较：
- real encoded transitions；
- model-predicted transitions；
- real-domain inverse action identifiability；
- predicted-domain inverse action identifiability；
- real/predicted domain 间的 inverse transfer。
主要结果：
- 在 TwoRoom、PushT 和 OGBench-Cube 上，true-transition action-identifiability 比 standard prediction loss 更能跟踪 planning；
- 对 success 差异超过 5% 的候选模型，pairwise ranking accuracy 达到 98.81%；
- 模型可能在 predicted transition 中高度可解码 action，却无法迁移到真实 transition。
这几乎直接解释了我们当前的负结果：
adapter 中 action-correlated signal 存在，但 predicted effect 与真实执行 effect 的关系不一致。

3. Action-discriminative world model
AD-WM, 2609.30264 使用：
- residual latent dynamics；
- predictor-level action recovery；
- mutual-information/action-discrimination regularization；
- counterfactual MPC 诊断。
报告结果：
- OGBench-Cube hard-start success：3.7% → 52.0%；
- 五个模拟环境中四个提升；
- Franka basic pick-and-place：42.2% → 71.1%；
- factual prediction error 与 success 排序不一致；
- CEM-aligned elite regret 更接近真实控制表现。
这比我们当前的 local adapter 更接近真正有效的解决方案。它的核心不是“再换一个 latent metric”，而是直接提高候选 action effect 的可区分性。
4. Decision-aligned latent geometry
近期还有多项工作直接修复“latent cost 排序错误”：
- Temporal-Distance-JEPA, 2607.25337：用 trajectory temporal order、cross-trajectory negatives 和 rollout consistency 学习 directed progress cost。报告 TwoRoom 成功率达到 100%，并改善 OGBench-Cube 和 PushT。
- Predictive but Not Plannable: RC-aux, 2605.07278：加入 multi-horizon prediction 与 budget-conditioned reachability supervision，直接改善 LeWM-style planning。
- Latent World Models with Monotone Planning Costs, 2608.09073：用 rollout loss 和 monotone cost ranking，报告 orientation error 相比 DINO-WM 降低约 2.7×。
- Criterion-Aligned Auxiliary Losses, 2610.01224：将 success criterion quantity 作为训练辅助目标；PushT 和 Cube 分别提升约 3.5 和 3.4 个百分点。
- D-JEPA, 2609.24749：学习 candidate futures 之间的 decision-relevant ordinal relation，报告 PushT success 87.89%，RoboTwin 平均提升 15.04 个百分点。
- AnisoWM, 2609.37441：直接针对 LeWM 的 isotropic latent geometry，引入可学习 anisotropic covariance；四个 visual-control 环境均优于 LeWM。
因此，“latent geometry 必须与 planning task 对齐”已经被多个工作分别证明。我们的贡献不能停留在这个层面。
三、接触丰富操作文献如何解决我们遇到的姿态闭合失败
现有调研中的证据已经比较清楚：
1. Contact mode 显式建模
- ContactNets, 2009.11193：用 signed distance、contact Jacobian 和 complementarity-inspired loss 建模冲击、sticking、sliding。
- Contact Mode Guided Planning, 2011.01454：把 contact mode 作为离散规划变量。
- Convex Hybrid MPC, 1710.05724：离线搜索 mode sequence，在线执行连续控制。
- Learning Preconditions of HFVCs, 2206.12728：学习 action/skill precondition，报告 success 51.5% → 73.2%。
共同解决方式是：
\[
\text{mode estimation}
\rightarrow
\text{precondition}
\rightarrow
\text{mode-conditioned effect}
\rightarrow
\text{short pulse}
\rightarrow
\text{re-observation}
\rightarrow
\text{recovery}.
\]
这正好解释为什么我们目前的 global effect regressor、KNN 和 local Ridge 都不稳定：它们试图用一个连续函数同时表示 free、approach、sticking、sliding 和 lost-contact。
2. 视觉为何经常不够
近期 visual-tactile 工作的共同结论是：
- 接触点；
- 摩擦；
- slip；
- force/torque transient；
- 局部形变；
- action history；
往往不能从单帧 RGB 稳定恢复。
例如：
- VT-WAM, 2607.02503：显式预测 visual-tactile future dynamics，六个接触任务平均 success 71.67%。
- PSR, 2609.21753：用 multimodal sensorimotor history 建立 predictive representations，报告整体 success 91.7%。
- PhaForce, 2603.08342：contact-phase predictor + slow planning + fast correction，平均 success 86%。
这说明我们当前 RGB-only learned adapter 的负结果很可能不是简单的 feature capacity 不够，而是存在真实的 partial observability 和 action-excitation 不足。
四、因果动力学和 world model 的相关结论
1. 预测不等于因果
Learning Counterfactual World Models, 2609.05834 将失败称为 counterfactual collapse：
模型能生成视觉上合理的未来，但不能区分不同 intervention 的行为后果。

其方法是：
- recurrent belief-state encoder；
- action-conditioned latent dynamics；
- counterfactual contrastive objective；
- perceptual-alias negatives。
报告：
- Occluded Push：65.1% → 74.6%；
- Aliased Maze：67.3% → 78.9%；
- Deferred Kitchen exploitative failures：18.4% → 9.7%；
- counterfactual separability 与 planning success 相关系数 \(r\geq0.94\)。
2. Controlled-world-model identifiability
On the Identifiability of Controlled World Models, 2607.22430 是理论上与我们最接近的新工作之一。它指出 controlled world model 需要同时满足：
1. representation identifiability；
2. transition identifiability；
3. spectral separation；
4. non-degenerate conditional action excitation。
其理论结论包括：
- transition prediction error bound 与 spectral separation margin 有关；
- counterfactual error 会随 weakest action-excitation margin 变差；
- 在非线性观测下验证了这些条件。
它直接提醒我们：
我们不能从 covariance–predictor geometry 单独推出 causal dynamics；必须把 action excitation、transition identifiability 和 intervention coverage 纳入理论或实验。

五、我们的原始创新点哪些保得住，哪些已经被削弱
已经不能作为主要创新点
以下表述应放弃：
- “首次发现 prediction error 不等于 planning success”；
- “首次证明 latent geometry 影响 planning”；
- “首次提出 action-effect ranking 重要”；
- “commutator 是 world-model transfer certificate”；
- “alignment 可以普遍选择最佳 learned checkpoint”；
- “frozen latent + local adapter 可以稳定解决 PushT”。
近期 ATM、AD-WM、D-JEPA、AnisoWM、Temporal-Distance-JEPA 和 Control Theory of Predictability 已覆盖这些方向的关键部分。
仍然可以建立的创新点
创新点一：covariance–predictor pair 的谱可识别性边界
我们的核心对象不是一般的 latent geometry，而是：
\[
(\Sigma, P, M)
\]
其中：
- \(\Sigma\)：latent covariance；
- \(P\)：predictor 或 local Jacobian；
- \(M\)：预先声明的 external metric。
我们证明：
\[
\operatorname{spec}(P)
\not\Rightarrow
\mathcal T(\Sigma,P,M).
\]
这和 ATM、AD-WM 的 action-identifiability 不同：它们研究 action-induced transition 是否可辨识，我们研究 predictor spectrum 是否足以识别 covariance-relative transfer。
创新点二：full-\(GL(d)\) gauge-aware no-go
已有 representation alignment 通常默认一个 Euclidean coordinate system。我们的理论明确区分：
1. Euclidean commutator 在一般可逆重参数化下不保持；
2. metric-aware commutator 在 metric 协变变换下保持；
3. covariance-only whitening 不能产生非平凡 full-\(GL\) alignment；
4. 必须引入 decoder/action/task-space metric 才能得到非退化诊断。
这是一个真正有辨识意义的 no-go result，应成为论文的理论核心。
创新点三：alignment 与 non-normality 的正交分解
我们不是把所有谱失败都归因于 commutator，而是区分：
\[
\text{covariance misalignment}
\quad\neq\quad
\text{non-normal transient amplification}.
\]
具体包括：
- covariance-relative alignment；
- principal angles / projector overlap；
- Schur off-diagonal energy；
- resolvent/pseudospectral sensitivity；
- finite-horizon transient gain；
- nonlinear local-Jacobian remainder。
这比单一 alignment score 更严格，也区别于只做 action-identifiability 的工作。
创新点四：从“正面方法”转为“诊断边界”
真实 learned checkpoint 的失败不是缺点，而是可以形成一个重要边界：
受控同谱实验中 alignment 能识别 multi-step transfer，但在真实 learned checkpoint 中，由于 off-manifold divergence、action-transition mismatch、metric mismatch 和 non-normal amplification，alignment 不能单独升级为 planning certificate。

这条结论与近期工作是互补的：
- ATM 解释 real/predicted transition mismatch；
- AD-WM 通过 action discrimination 修复 action structure；
- 我们解释为什么即使 action signal 存在，spectrum/alignment 仍然不足以完成 transfer identification。
六、建议重新建立的论文创新叙事
推荐主线改为：
We study when spectral summaries of learned latent predictors are insufficient to identify covariance-relative transfer. We establish a gauge-aware and non-normality-aware diagnostic framework, prove a full-\(GL\) no-go result for covariance-only alignment, and validate a fixed-spectrum mechanism in controlled experiments. In learned world models, we show that this diagnostic boundary persists: action-correlated latent structure and local predictability do not automatically imply cross-domain intervention fidelity or closed-loop planning.

中文对应：
本文研究 learned latent predictor 的谱摘要何时不足以识别其相对于潜变量协方差的迁移行为。我们建立具有规范协变性和非正规性分解的 covariance–predictor diagnostic framework，证明仅依赖协方差无法得到非平凡的 full-\(GL\) alignment，并通过同谱干预实验验证谱外几何对多步传输的影响。进一步在 learned world model 中发现，action-correlated latent structure 和局部可预测性并不自动转化为跨域干预保真度或闭环规划收益。

七、必须补做的竞争性实验
要让这个创新点达到 oral 级别，至少需要正面对比近期最近邻，而不是只做自己的 ablation。
实验 A：统一 diagnostic comparison
在同一 frozen LeWM/JEPA backbone、同一 train/validation/test split、同一 action bank 上比较：
- MSE；
- variance；
- covariance alignment；
- commutator；
- principal-angle/projector overlap；
- Schur non-normality；
- transient gain；
- inverse action risk；
- ATM cross-domain transfer；
- decision-metric alignment；
- reachability score；
- candidate elite regret。
主要问题：
\[
\text{哪一个指标在控制 prediction MSE 后，仍能预测 held-out planning risk？}
\]
实验 B：同谱、同 MSE、同 stage 的 learned intervention
当前真实 learned benchmark 是 0/6 个 identifiable strata，说明不能继续把已有结果包装成 positive learned evidence。
需要构造：
- 同一 learned checkpoint；
- 同一 latent dimension；
- 同一 predictor spectrum；
- 同一 one-step MSE；
- 同一 action distribution；
- 只改变 covariance-relative orientation 或 task metric；
- 测试 multi-step transfer、ATM、elite regret 和 planning risk。
如果做不到真实 checkpoint 的严格 matched strata，应诚实保留 synthetic/semisynthetic 结论，不能升级为 learned-model selection criterion。
实验 C：把 ATM 作为外部 metric，而不是竞争方法
将真实 action-effect metric 定义为：
\[
M_{\mathrm{action}}
=
\text{cross-domain inverse-transfer geometry}.
\]
然后测试：
\[
\text{alignment under }M_{\mathrm{action}}
+
\text{Schur geometry}
+
\text{transient gain}
\]
是否比单独的 ATM、单独的 commutator、单独的 MSE 更能预测 held-out planning risk。
这才可能形成我们的独立增量：
ATM 识别 action-transition transfer；我们测量该 transfer 是否还能被 covariance-relative spectral geometry 解释。

实验 D：非正规性必须进入 planning-risk 预测
固定 eigenvalues、alignment 和 one-step MSE，仅改变：
- Schur off-diagonal；
- resolvent norm；
- finite-horizon transient amplification。
检验其对：
- h-step latent error；
- planner-reachable error；
- candidate rank inversion；
- elite regret；
- closed-loop success；
的影响。
这可以把“非正规扩展”从理论附录提升为真正实证贡献。
实验 E：跨环境留一验证
至少包括：
- PushT；
- TwoRoom；
- PointMaze；
- OGBench-Cube 或 Reacher。
训练阶段只使用部分环境，测试时执行 leave-one-environment-out。若 alignment 只在 PushT 有效，结论必须降级为 task-specific；如果 Schur/transient/ATM 组合能跨环境预测，才有资格称为 conditional transfer-risk diagnostic。
最终创新性判断
目前最准确的 gap verdict 是：
原始强 gap：不成立
“没有人研究 latent geometry 与 planning 的关系”已经被近期工作覆盖。
修正后的 gap：部分成立
仍未被现有工作完整解决的是：
在明确指定 gauge 和 external task metric 后，如何联合区分 covariance-relative alignment、non-normal transient amplification、action-transition transfer 和 off-manifold divergence，并判断 predictor spectrum 何时足以、何时不足以识别 learned latent transfer。

这才是我们应当坚持的创新点。
当前最强的论文标题方向可以是：
When Spectra Are Not Enough: A Gauge-Aware Covariance–Predictor Diagnostic for Latent Transfer

而不应再使用：
universal world model、planning certificate、causal latent dynamics 或 universal transfer criterion。

已有本地证据图也支持这一重新定位：
- [完整文献证据图](/home/gbk/code/LeWorldModel/le-wm-main/results/negative_cause_diagnosis_20261012/literature_evidence_map_20261012.md)
- [接触机制审计](/home/gbk/code/LeWorldModel/le-wm-main/results/negative_cause_diagnosis_20261012/literature_mechanism_audit_20261012.md)
- [真实 learned matched benchmark](/home/gbk/code/LeWorldModel/le-wm-main/results/matched_learned_transfer_benchmark_20261011/matched_learned_transfer_benchmark.md)
结论是：谱分析主线仍然有价值，但价值已经从“提出一个能解决规划的 alignment 方法”转为“建立一个严格说明谱信息、表示几何、非正规性和规划风险之间不可互相替代的诊断框架”。


21:22







继续进行更广的文献调研，识别出我们真正能够解决的问题与insight。从机器视觉的上取样到下取样，从信号系统的拉普拉斯变换，时域转频域再逆变换。最后就是ai中升降维度解偶，本质都是一个东西，把复杂信息扩张到新空间，进行更精准提取。在平面上里两个点在同一位置，你永远无法找到他们的分界线，让他们映射到空间中，点与点的间隙立马清晰可见，再转映射到平面，现在你可以看到他们之间的关系了。将这样的思想应用在我们的研究里面，和谱分析进行结合







21:27






用时 51秒



你的类比抓住了一个很重要的共同结构，但需要把它从直觉提升为严格的研究对象：
\[
\text{原始观测}
\xrightarrow{\;\Phi\;}
\text{扩张/升维表示}
\xrightarrow{\;\mathcal{O}\;}
\text{可分离的动态结构}
\xrightarrow{\;\Pi\;}
\text{任务空间结果}.
\]
其中：
- \(\Phi\)：delay、Fourier/Laplace、kernel、特征交叉或神经 lifting；
- \(\mathcal{O}\)：在新空间中进行谱分解、子空间分解或动力学估计；
- \(\Pi\)：decoder、task readout 或 action-effect projection。
关键限制是：升维不会凭空创造信息。它只有在满足稳定嵌入、历史可观测性或 action excitation 时，才能把原空间中的混叠结构分离出来。
一、相关文献揭示的统一规律
1. 动力系统：delay/Hankel lifting 解决部分可观测性
Takens delay embedding 和 Hankel-DMD 的核心思想是把
\[
x_t
\]
扩展为
\[
\eta_t=[x_t,x_{t-\tau},\ldots,x_{t-(L-1)\tau}],
\]
使原本不可观测的动态状态在历史空间中变得可区分。
代表工作：
- Delay-coordinate maps and Koopman spectra：delay coordinates 可以逼近 Koopman 离散谱子空间；
- Hankel DMD / HAVOK：在 time-delay coordinates 中进行线性动力学建模和 MPC；
- Stable Embedology：说明 delay embedding 不仅要拓扑可逆，还要具有稳定的几何保真度；
- Robust approximation of stochastic Koopman operators：指出在随机系统中，delay lifting 可能产生偏差，不能自动保证谱可靠。
这直接对应我们的问题：
单帧 frozen RGB latent 可能无法区分“接触位置、速度、摩擦状态、接触模式”；历史 lifting 有可能恢复这些隐藏变量，但必须通过 held-out observability 和 action-effect transfer 验证。

2. Koopman/EDMD：升维后把非线性转成线性算子
Koopman 方法用观测函数 \(\phi(x)\) 把非线性系统写成近似线性形式：
\[
\phi(x_{t+1})\approx K\phi(x_t).
\]
控制版本进一步加入：
\[
\phi(x_{t+1})
\approx
A\phi(x_t)+Bu_t.
\]
相关工作已经证明：
- 合适的 observable lifting 可以改善非线性预测和控制；
- kernel-EDMD 可以给出状态依赖的误差界；
- learned Koopman eigenfunctions 可以提升闭环控制；
- 但 dictionary closure、谱污染、输入矩阵 \(B\) 估计错误会破坏控制。
尤其是 Control-Coherent Koopman Modeling 明确指出：如果输入作用矩阵 \(B\) 在升维空间中被错误估计，预测器的谱即使看似合理，也会导致错误控制。
这比单独看 predictor eigenvalues 更接近我们当前的核心问题：
\[
\text{spectrum}(A)
\quad\text{不能替代}\quad
\text{action reachability}(B)
\text{ 和 task observability}(D).
\]
3. 信号系统：频域分解本质上也是一种结构化 lifting
Fourier/Laplace 变换把时域信号映射到频域：
\[
x(t)\rightarrow X(s)
\quad\text{或}\quad
x(t)\rightarrow X(\omega).
\]
在 LTI 系统中，频率基可以把卷积和微分转化为乘法，因此谱结构具有直接物理意义。
但对 learned nonlinear world model，必须注意：
- 频率不是表示空间的内在属性；
- 基函数选择依赖观测、窗口和采样率；
- 非平稳接触事件会产生 broadband/transient response；
- 非正规算子的 transient amplification 不会被 eigenvalue spectrum 单独描述；
- action-conditioned dynamics 需要研究 transfer function，而不是只研究 state spectrum。
因此我们不应简单说“latent eigenvalues 就是信号频率”。更准确的做法是分析：
\[
G_h^\Phi
=
D_\Phi A_\Phi^{h-1}B_\Phi,
\]
即 action 到 task-space outcome 的有限时域 transfer operator。
4. 机器学习：升维可以改善线性可分性，但不保证因果可分性
kernel trick、random features 和深度 representation learning 都利用了同一个思想：
\[
x\mapsto \phi(x),
\]
使原空间中不可分的结构在新空间中变得可分。
但 disentanglement 文献反复说明：
- 无监督 disentanglement 一般不可识别；
- 仅仅看到 factor separation，不代表得到 causal variables；
- 需要 intervention、multiple views、weak supervision 或 action labels；
- action-conditioned representation 比静态 invariance 更可能保留可控因素。
这和我们的 frozen LeWM 结果完全一致：
latent 中检测到 action-correlated direction，不等于该 direction 能稳定预测真实 torque effect。

二、对我们研究最重要的新统一对象
当前研究不应该只分析：
\[
(\Sigma_z,P_z).
\]
更合理的对象是 action-lifted operator tuple：
\[
(\Phi,\; C_\Phi,\; A_\Phi,\; B_\Phi,\; D_\Phi,\; M_{\mathrm{task}}),
\]
其中：
- \(\Phi\)：lifting map；
- \(C_\Phi\)：lifted-state covariance；
- \(A_\Phi\)：无动作的局部 latent dynamics；
- \(B_\Phi\)：action-effect operator；
- \(D_\Phi\)：从 lifted state 到任务变量的 readout；
- \(M_{\mathrm{task}}\)：外部声明的任务度量。
可以定义：
\[
\eta_t
=
\Phi(z_{t-L:t},a_{t-L:t-1}),
\]
并拟合：
\[
\eta_{t+1}
=
A_\Phi \eta_t+B_\Phi a_t+\epsilon_t.
\]
任务变量为：
\[
y_t=D_\Phi\eta_t.
\]
于是，真正与规划相关的有限时域 action transfer 是：
\[
\mathcal{G}_{h}^{\Phi}
=
D_\Phi A_\Phi^{h-1}B_\Phi.
\]
这比单独研究 \(A_\Phi\) 的 eigenvalues 更接近 PushT：
- \(A_\Phi\) 描述历史状态如何传播；
- \(B_\Phi\) 描述 action 如何进入 latent；
- \(D_\Phi\) 描述哪些变化真正对应 block XY、angle 和 contact；
- \(M_{\mathrm{task}}\) 决定哪些方向有任务意义。
因此我们的核心问题可以重新定义为：
在什么条件下，经过历史和 action lifting 后，谱结构、eigenspace alignment、Schur geometry 和 finite-horizon gain 能够识别真实 action-to-task transfer？

这比“alignment 能否直接解决规划”更加精准，也更可能形成真正的创新。
三、文献与我们结果共同揭示的四个 insight
Insight 1：升维首先解决的是 observability，不是 planning
delay lifting 可能把隐藏速度、接触持续时间和 mode transition 暴露出来，但它不能自动提供正确 action ranking。
因此需要区分：
\[
\text{state separation}
\neq
\text{action-effect identification}
\neq
\text{planning success}.
\]
这解释了为什么当前局部 temporal observer 能提高 contact retention，却没有稳定解决姿态角排序。
Insight 2：应分析 action-tangent spectrum，而不是只分析 state spectrum
原有分析主要围绕：
\[
\operatorname{spec}(P_z).
\]
但规划真正关心的是：
\[
B_\Phi,\quad
A_\Phi^hB_\Phi,\quad
D_\Phi A_\Phi^hB_\Phi.
\]
也就是说，关键对象不是“哪些 latent modes 存在”，而是：
哪些 spectral modes 能被 action 激活，并且能通过 task readout 形成可执行的物理结果。

这可以称为 action-tangent spectral transfer。
Insight 3：Fourier/Laplace、delay lifting 和 learned lifting 的共同问题是 basis validity
所有 lifting 都可能产生三类错误：
1. aliasing：不同物理状态落在相同 representation；
2. leakage：一个 mode 混入多个任务因素；
3. non-normal amplification：谱稳定但有限步响应巨大。
因此 lifting 后不能只报告 eigenvalues，还应报告：
- stable embedding distortion；
- inverse/action recovery；
- covariance-projector overlap；
- Schur off-diagonal energy；
- transient gain；
- task readout error；
- held-out intervention fidelity。
Insight 4：真正的“分离”必须在外部任务空间中验证
如果两个点在原空间重叠，升维后分开，这是 representation-level separation。
但我们最终必须证明：
\[
\|\Pi(\Phi(x_1))-\Pi(\Phi(x_2))\|
\]
对应真实任务差异，而不是只在 latent 空间中人为变远。
因此必须使用：
- block XY；
- wrapped angle；
- contact retention；
- action inverse risk；
- real/predicted transition transfer；
- independent closed-loop outcome。
否则只能说“latent separability 提高”，不能说“物理可控性提高”。
四、我们真正能够解决的科学问题
目前最有希望、也最不容易与已有工作重叠的目标是：
学习或选择一种 action-history lifting，使 frozen visual latent 中原本混叠的接触动力学在 lifted space 中变得可辨识，并用 gauge-aware spectral diagnostics 判断这种可辨识性是否能转化为 held-out action-to-task transfer。

它包含三个可证伪命题：
命题 A：历史 lifting 是否改善可观测性
比较：
\[
\Phi_0(z_t)=z_t
\]
与：
\[
\Phi_L(z_t)
=
[z_t,z_{t-1},\ldots,z_{t-L}],
\]
检验 mode classification、action recovery 和 contact-state observability 是否提高。
命题 B：action lifting 是否改善 action-effect 可辨识性
加入：
\[
\Phi_{\mathrm{act}}
=
[\Phi_L(z_t),a_t,z_t\otimes a_t].
\]
比较不同动作在 lifted transition space 中的 separation：
\[
\mathcal I_{\mathrm{act}}
=
I(A_t;\Delta\eta_t\mid \eta_t),
\]
或用 held-out inverse risk / ATM cross-domain transfer 估计。
命题 C：谱几何是否能预测 transfer
在 action-effect quality 和 one-step MSE 尽量匹配后，比较：
- alignment；
- principal angles；
- Schur non-normality；
- transient gain；
- controllability/observability Gramian；
对 held-out multi-step action-to-task transfer 的预测能力。
这三个命题比“commutator 是通用 certificate”严格得多，也比“再训练一个 adapter”更有理论价值。
五、建议建立的完整实验框架
阶段一：受控系统验证
使用具有已知真实状态和 action effect 的非线性系统，比较：
1. raw state；
2. delay/Hankel lift；
3. Fourier/Laplace features；
4. polynomial/kernel lift；
5. action-history lift；
6. mode-conditioned lift。
固定真实 dynamics spectrum 或构造同谱 operator family，测量：
- one-step MSE；
- multi-step transfer；
- action inverse risk；
- observability Gramian；
- controllability Gramian；
- Schur gain；
- finite-horizon task transfer。
目标不是证明“升维永远有效”，而是找出：
哪些 lifting 条件足以让 spectrum-relative transfer 变得 identifiable。

阶段二：LeWM frozen latent
对每个 episode 构造：
\[
\eta_t=
[z_{t-L:t},a_{t-L:t-1},\Delta z_{t-L:t-1}],
\]
然后拟合：
\[
A_\Phi,\ B_\Phi,\ D_\Phi.
\]
所有 metrics 只在 train split 计算；validation 只做阈值选择；test 只评估：
- action recovery；
- contact-mode prediction；
- block XY effect；
- angle effect；
- real/predicted transition transfer；
- held-out planning risk。
阶段三：跨环境
至少比较：
- PushT；
- TwoRoom；
- PointMaze；
- OGBench-Cube 或 Reacher。
如果某个 lift 只在 PushT 有效，只能称为 contact-specific observable lift；如果在不同环境中都能改善 action-to-task transfer，才可以称为 conditional general-purpose diagnostic。
阶段四：与近期方法正面对比
必须加入：
- MSE-only；
- raw latent spectrum；
- covariance alignment；
- ATM；
- action inverse risk；
- AD-WM-style action recovery；
- reachability/temporal cost；
- AnisoWM-style covariance shaping；
- lifted spectral diagnostic。
我们的优势不应是“性能最高”，而应是：
能够解释为什么某一种 lift 在某个环境有效、在另一个环境失效，以及失败来自 observability、action excitation、metric mismatch 还是 non-normal amplification。

六、最终创新点建议
最稳的创新表述不是：
我们提出一种升维方法解决 world-model planning。

而是：
我们提出一个 action-history lifted covariance–predictor framework，用于区分表示可分性、action-effect 可辨识性、谱几何传输和任务空间可观测性。该框架证明并实证检验：升维可以改善局部 action separability，但只有在满足 stable embedding、action excitation、task-metric specification 和 finite-horizon stability 时，谱信息才可能预测真实 transfer。

这形成四个明确贡献：
1. 统一 lifting 视角：把 delay、频域变换、kernel feature 和 learned latent 视为可比较的 observable lifting；
2. action-tangent spectral analysis：从 state spectrum 转向 \(D_\Phi A_\Phi^hB_\Phi\)；
3. gauge-aware/non-normal diagnostic：区分坐标变化、协方差错位和 transient amplification；
4. failure localization：判断负结果究竟来自 latent aliasing、hidden contact mode、action-excitation 不足、readout mismatch 还是 planner off-manifold drift。
最终能真正解决的问题是：
frozen visual latent 中的接触动力学是否可以通过历史—动作 lifting 被稳定地分离，并且这种分离是否能够在外部任务度量下预测 held-out intervention transfer？

这既吸收了机器视觉、信号系统、Koopman lifting 和 disentanglement 的共同思想，又不会声称升维本身创造了因果物理信息。它将我们的谱分析从“特征值不够”推进到更具体的结论：
\[
\boxed{
\text{可用的谱分析对象不是 }P_z，
\text{而是 action-lifted }(C_\Phi,A_\Phi,B_\Phi,D_\Phi,M_{\rm task}).
}
\]
已有本地调研证据可作为基础：
- [文献证据图](/home/gbk/code/LeWorldModel/le-wm-main/results/negative_cause_diagnosis_20261012/literature_evidence_map_20261012.md)
- [接触机制审计](/home/gbk/code/LeWorldModel/le-wm-main/results/negative_cause_diagnosis_20261012/literature_mechanism_audit_20261012.md)
- [matched learned benchmark](/home/gbk/code/LeWorldModel/le-wm-main/results/matched_learned_transfer_benchmark_20261011/matched_learned_transfer_benchmark.md)


21:27







还有就是寻找对我们研究有用的数学工具帮助我们的研究







21:28






用时 43秒



真正有用的数学工具，不是继续增加普通的 eigenvalue 或 commutator，而是把研究对象扩展为：
\[
\eta_t=\Phi(z_{t-L:t},a_{t-L:t-1}),
\]
\[
\eta_{t+1}=A_\Phi\eta_t+B_\Phi a_t+\varepsilon_t,
\qquad
y_t=D_\Phi\eta_t .
\]
其中：
- \(\Phi\)：历史、频域、kernel 或 action lifting；
- \(A_\Phi\)：lifted state dynamics；
- \(B_\Phi\)：action-effect operator；
- \(D_\Phi\)：task-space readout；
- \(y_t\)：block XY、angle、contact mode 等外部变量。
谱分析真正应该研究的是：
\[
D_\Phi A_\Phi^{h-1}B_\Phi,
\]
而不是单独研究 \(A_\Phi\) 的特征值。
最值得引入的数学工具
1. Delay embedding、Hankel matrix 与 Koopman lifting
对应当前的 hidden contact state 和历史不足问题。
构造：
\[
\eta_t=
[z_t,z_{t-1},\ldots,z_{t-L}],
\]
再对 Hankel matrix 做 SVD 或低秩分解。
它可以帮助我们：
- 恢复单帧 latent 中缺失的速度和接触持续时间；
- 将 mode transition 转化为历史空间中的线性可分问题；
- 分离慢变化、周期变化和瞬时接触变化；
- 检验当前失败究竟来自 representation aliasing 还是 dynamics regression。
可参考：
- Delay-coordinate maps and Koopman spectra
- HAVOK Model Predictive Control
- Stable Embedology
需要注意：delay lifting 只有在稳定嵌入成立时才有意义，不能自动创造因果信息。
2. Controlled Koopman / EDMD 与 bilinear lifting
普通 latent predictor 只拟合：
\[
\eta_{t+1}\approx A_\Phi\eta_t.
\]
我们的 PushT 问题必须拟合：
\[
\eta_{t+1}
\approx
A_\Phi\eta_t+B_\Phi a_t
+\sum_j a_{t,j}B_{\Phi,j}\eta_t .
\]
这可以表达：
- action 对 latent 的直接作用；
- action 与当前接触状态的交互；
- 不同 torque 在不同 mode 下产生不同 effect。
关键工具包括：
- controlled Koopman operator；
- EDMD with control；
- bilinear system identification；
- kernel EDMD；
- control-coherent Koopman model。
其中 Control-Coherent Koopman Modeling 特别说明，输入矩阵 \(B\) 的错误估计会导致预测和控制同时失效。
这正好对应我们当前的核心瓶颈：
不是缺少一个更复杂的 \(A\)，而是没有可靠估计 action-effect operator \(B_\Phi\)。

3. Observability Gramian 与 controllability Gramian
这是目前最值得加入主线的工具。
有限时域 observability Gramian：
\[
W_o(H)
=
\sum_{h=0}^{H-1}
(A_\Phi^h)^\top
D_\Phi^\top M_{\mathrm{task}}
D_\Phi A_\Phi^h .
\]
它回答：
latent 中哪些方向能够被 task readout 观测到？

有限时域 controllability Gramian：
\[
W_c(H)
=
\sum_{h=0}^{H-1}
A_\Phi^h B_\Phi
M_a
B_\Phi^\top
(A_\Phi^h)^\top .
\]
它回答：
哪些 latent 方向能够被 action 实际激活？

真正与规划相关的方向，必须同时满足：
\[
\text{observable}
\cap
\text{controllable}
\neq \varnothing .
\]
这可以精确解释当前现象：
- covariance tail 可能具有较大方差，但不可由 action 激活；
- latent 中存在 action-correlated direction，但无法被 task readout 解码；
- angle signal 存在，但对应方向的 controllability 很差；
- translation direction 和 torque direction 可能落在不同子空间。
建议新增：
- observability eigenvalues；
- controllability eigenvalues；
- joint reachability-observability score；
- task-space Hankel singular values。
这比单独看 latent covariance spectrum 更接近“可规划性”。
4. PBH rank test 与 persistent excitation
要判断 action-effect 是否可辨识，必须加入 system identification 中的 persistent excitation 条件。
令回归向量为：
\[
\psi_t=[\eta_t^\top,a_t^\top]^\top .
\]
要求：
\[
\lambda_{\min}
\left(
\frac{1}{N}
\sum_{t=1}^{N}
\psi_t\psi_t^\top
\right)
\geq \alpha>0.
\]
如果 \(\alpha\) 很小，说明当前 trajectory 没有充分激发系统，Ridge、KNN 或 torque calibration 得到的 action sign 不可信。
这可以把“local Ridge 失败”从经验描述变成可检验判断：
不是模型一定错误，而是当前接触轨迹没有提供足够的 action excitation 来识别 mode-conditioned \(B_\Phi\)。

还可以使用 PBH test：
\[
\operatorname{rank}
\begin{bmatrix}
\lambda I-A_\Phi & B_\Phi
\end{bmatrix}
=d
\]
判断某个动态 mode 是否可控；以及：
\[
\operatorname{rank}
\begin{bmatrix}
\lambda I-A_\Phi\\
D_\Phi
\end{bmatrix}
=d
\]
判断该 mode 是否可观测。
这会给姿态闭合问题一个非常清楚的诊断分解：
- 不可控：action 无法激活该姿态方向；
- 不可观测：RGB latent 无法识别该姿态方向；
- 可控但不可观测：需要额外 sensing 或 history；
- 可观测但不可控：需要改变 action bank；
- 二者都满足但仍失败：检查 non-normality、metric 或 planner。
5. Riemannian metric、pullback metric 与 generalized eigenproblem
当前最关键的 gauge 问题是：普通 Euclidean alignment 依赖坐标。
可以使用 decoder/task readout 的 pullback metric：
\[
G_\Phi(\eta)
=
J_{D_\Phi}(\eta)^\top
M_{\mathrm{task}}
J_{D_\Phi}(\eta).
\]
它表示 latent 中的距离在 task space 中实际对应多大的变化。
然后将谱分析改写为 generalized eigenproblem：
\[
A_\Phi v_i
=
\lambda_i G_\Phi v_i,
\]
或在 \(G_\Phi\)-内积下计算：
\[
\langle u,v\rangle_{G_\Phi}
=
u^\top G_\Phi v.
\]
这样得到的 alignment 是：
- 与坐标规范协变的；
- 与 task-space 意义相关的；
- 不会因为 covariance-only whitening 而退化；
- 可以直接连接 block XY 和 angle 的敏感方向。
这比继续使用 latent covariance 作为唯一 metric 更合理。
6. Schur geometry、pseudospectrum 与 Kreiss constant
对于 learned predictor，普通 eigenvalues 不能描述 transient behavior。
应该加入：
Schur off-diagonal energy
\[
A_\Phi=Q(T_\Phi)Q^\top,
\qquad
T_\Phi=\Lambda_\Phi+N_\Phi .
\]
其中 \(N_\Phi\) 的大小刻画 non-normal coupling。
Resolvent norm
\[
R_A(z)=\|(zI-A_\Phi)^{-1}\|.
\]
Finite-horizon amplification
\[
\Gamma_H
=
\max_{1\le h\le H}
\|A_\Phi^h\|_{G_\Phi}.
\]
Kreiss-type quantity
\[
K(A_\Phi)
=
\sup_{|z|>1}
(|z|-1)\|(zI-A_\Phi)^{-1}\|.
\]
这些指标可以解释：
- 同样 spectrum，为什么 rollout 差异很大；
- 为什么 one-step MSE 很小但 multi-step planner 失效；
- 为什么 commutator 很小也不代表 finite-horizon transfer 稳定；
- 为什么某些 latent mode 会产生短期放大。
当前已经完成的 [gauge/non-normal audit](/home/gbk/code/LeWorldModel/le-wm-main/results/transfer_certificate_20261009/gauge_nonnormal_audit.md) 和 [intrinsic transfer audit](/home/gbk/code/LeWorldModel/le-wm-main/results/transfer_certificate_20261009/intrinsic_transfer_theory_audit.md) 可以作为这一部分的基础。
7. Koopman mode residual、closure error 与 pseudospectral residual
升维后最危险的问题是：新增维度可能只是冗余或噪声。
应定义 lifting closure residual：
\[
r_{\mathrm{closure}}
=
\left\|
\eta_{t+1}
-
\Pi_{\mathcal{H}_\Phi}
\left(
A_\Phi\eta_t+B_\Phi a_t
\right)
\right\|.
\]
并为每个候选 eigenpair \((\lambda,v)\) 计算：
\[
r_{\mathrm{eig}}(\lambda,v)
=
\frac{\|A_\Phi v-\lambda v\|}
{\|v\|}.
\]
这可以避免把：
- 高方差噪声；
- 非稳定 eigenmode；
- 延迟坐标冗余；
- 训练分布外的 spurious mode；
误判成可用动力学。
ResDMD 和近期 Spectral Reliability 都支持这一方向。
8. Action-identifiability 与 conditional inverse risk
对于 action-conditioned world model，仅用 forward prediction 不够。
可以定义：
\[
\mathcal R_{\mathrm{inv}}
=
\inf_g
\mathbb E
\left[
\|g(\eta_t,\eta_{t+1})-a_t\|^2
\right].
\]
或者使用条件互信息：
\[
I(A_t;\Delta\eta_t\mid \eta_t).
\]
这两个量分别衡量：
- action 是否能从 transition 中恢复；
- transition 是否真正携带 action-specific information。
但必须进一步区分：
\[
\mathcal R_{\mathrm{inv}}^{\mathrm{real}},
\qquad
\mathcal R_{\mathrm{inv}}^{\mathrm{pred}},
\]
以及 cross-domain transfer：
\[
\mathcal R_{\mathrm{inv}}^{\mathrm{real}\leftarrow\mathrm{pred}},
\qquad
\mathcal R_{\mathrm{inv}}^{\mathrm{pred}\leftarrow\mathrm{real}}.
\]
这正是 ATM 的数学基础，也是我们最应该加入的外部 action metric。它可以与我们的 covariance alignment 联合，而不是与 ATM 重复。
9. Counterfactual consistency 与 causal representation
如果要讨论“物理因果动力学”，需要从相关性升级到 intervention consistency。
对同一状态 \(\eta_t\) 执行动作 \(a\) 和 \(a'\)，要求：
\[
\Delta y_{\mathrm{pred}}(\eta_t,a)
-
\Delta y_{\mathrm{pred}}(\eta_t,a')
\]
与真实环境中的：
\[
\Delta y_{\mathrm{real}}(\eta_t,a)
-
\Delta y_{\mathrm{real}}(\eta_t,a')
\]
保持一致。
可以定义 counterfactual transfer error：
\[
\mathcal E_{\mathrm{cf}}
=
\mathbb E
\left[
\left\|
\left(
\hat y_{t+1}^{a}
-
\hat y_{t+1}^{a'}
\right)
-
\left(
y_{t+1}^{a}
-
y_{t+1}^{a'}
\right)
\right\|_{M_{\mathrm{task}}}^{2}
\right].
\]
这比普通 prediction MSE 更接近因果 action effect。
需要明确：
- action correlation 不是 causal effect；
- inverse action recovery 不是完整 causal identification；
- 必须有 same-state matched interventions；
- 必须报告 action support 和 confounding boundary。
10. Robust reachability、tube MPC 与 conformal calibration
要将 latent error 推进到 planning risk，需要使用误差传播工具。
如果：
\[
\|\epsilon_t\|\leq \delta_t,
\]
且局部线性算子为 \(A_t\)，则有限时域误差可界为：
\[
\|e_H\|
\leq
\sum_{j=0}^{H-1}
\left\|
A_{H-1}\cdots A_{j+1}
\right\|
\delta_j.
\]
如果加入 Schur/transient gain：
\[
\|e_H\|
\leq
\sum_{j=0}^{H-1}
\Gamma_{H-1-j}\delta_j.
\]
然后将它映射到 task space：
\[
\|e_H^y\|_{M_{\mathrm{task}}}
\leq
\|D_\Phi\|_{M_{\mathrm{task}}}
\|e_H\|.
\]
这可以直接服务于：
- candidate rejection；
- contact-safe tube；
- pose closure safety margin；
- uncertainty-aware fallback；
- planner-reachable risk。
conformal prediction 可用于校准 \(\delta_t\)，但必须注意：ensemble dispersion 不能自动称为 calibrated uncertainty。
最值得优先实现的五个工具
按对当前研究的实际价值排序：
第一优先级：有限历史 action lifting
\[
\eta_t=\Phi(z_{t-L:t},a_{t-L:t-1})
\]
解决单帧 latent 对接触模式和速度不可观测的问题。
第二优先级：\(A_\Phi,B_\Phi,D_\Phi\) 三元系统辨识
不要再只拟合 latent predictor，要同时估计：
- dynamics；
- action effect；
- task readout。
第三优先级：observability–controllability Gramian
用数学方式判断姿态方向到底是：
- 不可见；
- 不可激活；
- 还是被 non-normality 放大。
第四优先级：persistent excitation 与 inverse-risk gate
在 action ranking 前先判断当前状态是否有足够支持：
\[
\lambda_{\min}
\left(
\frac{1}{N}\sum\psi_t\psi_t^\top
\right)
\ge \alpha .
\]
不满足则拒绝 learned torque ranking，进入 fallback。
第五优先级：metric-aware transfer spectrum
最终分析：
\[
D_\Phi A_\Phi^{h-1}B_\Phi
\]
的谱、alignment、Schur geometry 和 transient amplification，而不是只分析 \(A_\Phi\)。
可以形成的新理论命题
最值得发展的不是“commutator 是 certificate”，而是下面这个命题：
对 action-lifted latent system，predictor spectrum 只有在同时满足 stable lifting、action excitation、task observability、metric specification 和 bounded non-normal amplification 时，才可能识别有限时域 task transfer。

可写成条件链：
\[
\boxed{
\text{stable lifting}
+
\text{persistent excitation}
+
\text{observability}
+
\text{controllability}
+
\text{metric specification}
+
\text{bounded transient gain}
}
\]
否则：
\[
\operatorname{spec}(A_\Phi)
\not\Rightarrow
D_\Phi A_\Phi^{h-1}B_\Phi .
\]
这条结论可以把目前分散的负结果统一起来：
- local Ridge 失败：persistent excitation 不足；
- angle ranking 失败：\(B_\Phi\) 与 \(D_\Phi\) 不匹配；
- learned adapter 不稳定：mode-conditioned observability 不足；
- frozen planning 失败：off-manifold reachability 和 transient gain 未控制；
- commutator 不可靠：缺少 \(B_\Phi,D_\Phi\) 与 finite-horizon information。
因此，最有潜力的研究方向是：
从 covariance–predictor spectral analysis 推进到 action-lifted, task-observable, non-normal transfer analysis。

这不是简单增加特征维度，而是用数学工具判断“升维后哪些方向真的可观测、可控、可迁移、可规划”。


21:29







再继续往下推进，为我们这个研究加入更多的数学物理工具







21:31






用时 6分钟 40秒



寻找相关的物理和数学的研究，为我们这个研究搭建牢不可破的数学和物理框架













可以，但必须避免“数学工具堆叠”。最有价值的方向是把当前对象从
\[
(\Sigma,P)
\]
扩展为一个具有投影、记忆、控制输入和任务读出的数学物理系统：
\[
\boxed{
(\Pi,\;K_{\mathrm{memory}},\;A,\;B,\;D,\;M_{\mathrm{task}})
}
\]
其中：
- \(\Pi\)：从完整物理状态到 frozen latent 的投影；
- \(K_{\mathrm{memory}}\)：未观测变量引起的记忆核；
- \(A\)：lifted latent dynamics；
- \(B\)：action-effect operator；
- \(D\)：task-space readout；
- \(M_{\mathrm{task}}\)：任务度量。
这比继续增加普通特征或单个 commutator 更接近数学物理本质。
1. Mori–Zwanzig：解释 frozen latent 为什么缺少接触动力学
当前 frozen latent 可以看作完整物理状态 \(x_t\) 的投影：
\[
z_t=\Pi(x_t).
\]
投影后，系统通常不再是 Markov 的。严格的 reduced dynamics 应写成：
\[
z_{t+1}
=
A z_t
+
B a_t
+
\sum_{\tau=0}^{t}K_\tau z_{t-\tau}
+
\xi_t,
\]
其中：
- \(A z_t\)：当前可见状态的 Markov 部分；
- \(B a_t\)：动作作用；
- \(K_\tau\)：被投影掉的隐变量产生的 memory kernel；
- \(\xi_t\)：unresolved orthogonal dynamics。
这直接解释了我们目前的失败：
单帧 latent adapter 假设 \(K_\tau=0\)，但接触位置、速度、摩擦和接触模式都被压缩进了 memory term。

相关工具：
- Mori–Zwanzig latent Koopman closure
- Regression-based projection for learning MZ operators
- Partial Observation with Mori–Zwanzig
建议新增的模型不是简单 concatenation，而是：
\[
\eta_{t+1}
=
A\eta_t
+
B a_t
+
\sum_{\tau=1}^{L}K_\tau\eta_{t-\tau}
+
\epsilon_t .
\]
需要比较：
1. Markov adapter：\(K_\tau=0\)；
2. finite-memory adapter：\(K_1,\ldots,K_L\)；
3. low-rank memory kernel；
4. mode-conditioned memory kernel。
主指标应包括：
- multi-step latent transfer；
- action inverse risk；
- contact-mode prediction；
- angle/XY effect ranking；
- closed-loop pose closure。
这里的理论创新潜力是：
当前谱分析只研究 \(A\) 的谱，但投影后的 transfer 实际由 \((A,K,B,D)\) 共同决定。

因此：
\[
\operatorname{spec}(A)
\not\Rightarrow
\text{finite-horizon task transfer}
\]
即使 \(A\) 的 spectrum、covariance 和 commutator 都固定，memory kernel 变化仍然可以改变规划结果。
2. Lie group 几何：正确处理 PushT 的位姿变量
PushT 的 object pose 不是普通欧氏向量，而是：
\[
g_t\in SE(2).
\]
当前直接回归：
\[
(x_t,y_t,\theta_t)
\]
容易造成：
- angle wrap discontinuity；
- world-frame 与 contact-frame 混淆；
- 平移和旋转尺度不一致；
- 同一个局部动作在不同姿态下的作用不一致。
应该使用 body-frame twist：
\[
\xi_t
=
\log(g_t^{-1}g_{t+1})
\in\mathfrak{se}(2),
\]
并用群的 adjoint 变换处理不同坐标系：
\[
\xi^{\mathrm{world}}
=
\operatorname{Ad}_{g_t}
\xi^{\mathrm{body}}.
\]
这样 action-effect 不再是静态的：
\[
\Delta(x,y,\theta),
\]
而是：
\[
\Delta \xi_t
=
B_{\mathrm{contact}}(g_t,m_t)a_t.
\]
这对姿态排序尤其重要，因为 torque sign 的作用通常依赖：
- agent 与 block 的相对位姿；
- 接触点位置；
- block 当前朝向；
- 当前 contact mode。
应直接比较：
- Euclidean pose regression；
- wrapped-angle regression；
- \(SE(2)\) logarithm；
- body-frame twist；
- contact-frame Lie algebra effect。
如果 Lie-group 表示能在相同 action bank 下稳定提高 angle-effect ranking，这会是一个实质性修复；如果不能，则说明瓶颈确实在接触模式/不可观测性，而非坐标表达。
3. 混合系统与互补约束：把 contact mode 从连续回归中分离出来
PushT 更接近 hybrid dynamical system：
\[
m_t\in
\{
\text{free},
\text{approach},
\text{sticking},
\text{sliding},
\text{lost-contact}
\}.
\]
连续状态满足：
\[
x_{t+1}=F_{m_t}(x_t,a_t),
\]
而模式由 guard condition 决定：
\[
m_{t+1}
=
G(m_t,x_t,a_t).
\]
对于接触，可以使用互补条件：
\[
0\leq \lambda_n
\perp
\phi(q)\geq0,
\]
其中：
- \(\phi(q)\)：signed distance；
- \(\lambda_n\)：法向接触力。
摩擦约束可以写成：
\[
\|\lambda_t\|
\leq
\mu\lambda_n .
\]
在 RGB-only 场景中，我们不能直接声称这些物理变量可观测，但可以学习：
\[
q(m_{t+1}\mid h_t,a_t)
\]
和 mode-conditioned operators：
\[
A_m,\quad B_m.
\]
最合理的系统形式是：
\[
\eta_{t+1}
=
A_{m_t}\eta_t
+
B_{m_t}a_t
+
\epsilon_t .
\]
谱分析也应变成 mode-specific：
\[
\operatorname{spec}(A_m),
\quad
\mathcal C_m,
\quad
\Gamma_{H,m}.
\]
这会解释为什么全局 commutator 或全局 alignment 在 PushT 上不稳定：
不同 contact mode 的 operator 被混合以后，得到的全局谱可能没有明确物理意义。

真正应该检验的是：
\[
\text{mode-conditioned alignment}
\rightarrow
\text{mode-conditioned action transfer}.
\]
4. Port-Hamiltonian 与 dissipativity：验证“物理合理性”
如果要进一步讨论 latent 是否具有物理动力学，可以引入 port-Hamiltonian 结构：
\[
\dot x
=
\bigl(J(x)-R(x)\bigr)\nabla H(x)
+
G(x)u,
\]
其中：
- \(H(x)\)：能量函数；
- \(J(x)=-J(x)^\top\)：保守互联结构；
- \(R(x)\succeq0\)：耗散；
- \(G(x)u\)：外部 action port。
能量变化满足：
\[
\dot H
=
-\nabla H^\top R\nabla H
+
y^\top u.
\]
这可以提供比 MSE 更强的物理审计：
- 无 action 时能量是否非增；
- action 输入是否对应合理能量注入；
- friction 是否体现为耗散；
- rollout 是否出现非物理能量爆炸；
- 不同 contact mode 是否对应不同能量交换规律。
相关方向：
- Port-Hamiltonian Neural Networks
- CIPHER: latent port-Hamiltonian dynamics from partial observations
- Port-Hamiltonian Koopman synthesis
但这里必须谨慎：
PushT 当前没有直接的力、速度和能量观测，因此 port-Hamiltonian 不能直接成为主方法，只能先作为 privileged simulator audit 或物理一致性辅助目标。

最稳妥的实验是：
1. 用 simulator state 计算真实机械能/接触功；
2. 比较 latent predictor 的 decoded energy increment；
3. 检查 energy residual 是否比 MSE 更能预测 pose failure；
4. 再尝试用 energy/passivity auxiliary loss 训练新模型。
5. Transfer operator 与 resolvent：把谱分析推进到有限时间响应
当前研究已经使用 Schur geometry 和 transient amplification，但可以更进一步研究 transfer/resolvent：
\[
\mathcal R(z)
=
(zI-A)^{-1}.
\]
对带 memory 的模型，应写成：
\[
\mathcal R(z)
=
\left(
zI-A-\widehat K(z)
\right)^{-1}.
\]
这说明有限时域响应不仅由 eigenvalues 决定，还取决于：
- Schur off-diagonal；
- memory kernel；
- action input \(B\)；
- task readout \(D\)。
真正与规划相关的 transfer function 是：
\[
\mathcal H(z)
=
D
\left(
zI-A-\widehat K(z)
\right)^{-1}
B.
\]
建议报告：
- \(H_\infty\)-style gain；
- resolvent norm；
- pseudospectrum；
- finite-horizon \(H_2\) gain；
- mode-conditioned gain；
- task-space gain。
这一方向可以把论文从：
eigenvalues do not determine alignment

推进到：
eigenvalues do not determine finite-horizon action-to-task response.

6. 能量方法与 Lyapunov 工具
如果不想完整学习 Hamiltonian，也可以使用更弱、更稳妥的 energy/Lyapunov 工具。
寻找一个 task-relevant scalar：
\[
V(\eta)
\]
使得在安全接触模式下：
\[
V(\eta_{t+1})-V(\eta_t)
\leq
c(a_t,\eta_t).
\]
或者要求：
\[
V_{t+1}
\leq
\rho V_t+\gamma\|a_t\|^2,
\qquad
\rho<1.
\]
这类 inequality 可以用于：
- 检测 rollout 是否稳定；
- 判断 torque pulse 是否进入危险区域；
- 构建 contact recovery 的安全门；
- 给 planner 提供比 latent distance 更稳健的约束。
相比直接要求“latent 学出真实物理能量”，Lyapunov 方法只要求：
latent dynamics 在任务相关区域内满足可验证的稳定性或耗散性不等式。

这更可能在 learned latent 上成立。
7. Fokker–Planck / transfer operator：把不确定性建模为分布传播
当前 adapter 大多输出 point prediction：
\[
\hat \eta_{t+1}=F(\eta_t,a_t).
\]
但接触动力学天然具有 mode uncertainty。可以改为传播条件分布：
\[
p(\eta_{t+1}\mid \eta_t,a_t).
\]
在连续时间中，对应 Fokker–Planck：
\[
\partial_t p
=
-\nabla\cdot(f p)
+
\frac12\nabla^2:(\Sigma_\epsilon p).
\]
实际不需要完整求解 PDE，可以近似使用：
- Gaussian mixture mode model；
- particle rollout；
- ensemble transition kernel；
- Wasserstein distance；
- distributional reachability tube。
关键不是输出一个 ExtraTrees dispersion，而是评估：
\[
W_2
\left(
p_{\mathrm{pred}}(\eta_{t+1}\mid\eta_t,a_t),
p_{\mathrm{real}}(\eta_{t+1}\mid\eta_t,a_t)
\right).
\]
这可以直接测量：
- 预测分布是否覆盖真实 contact outcome；
- 哪些 action 的 distribution shift 最大；
- 是否应该触发 fallback；
- uncertainty 是否与实际 pose failure 相关。
8. 多尺度与 coarse-graining：连接升降维、频域和 world model
你的“上取样、下取样、时域转频域、升降维”的直觉，可以通过 coarse-graining 和 renormalization 形式化。
定义不同时间尺度的 lifting：
\[
\Phi_\Delta(z_t)
=
[z_t,z_{t-\Delta},z_{t-2\Delta},\ldots].
\]
对每个尺度估计：
\[
A_\Delta,\quad B_\Delta,\quad D_\Delta.
\]
真正重要的不是某一个尺度的 spectrum，而是谱和 transfer 是否具有跨尺度一致性：
\[
\mathcal T_h^{(\Delta)}
=
D_\Delta A_\Delta^{h-1}B_\Delta.
\]
如果一个 latent mode 只在单一采样频率下出现，而在相邻尺度消失，它可能是：
- aliasing；
- sampling artifact；
- spurious eigenmode；
- training-stage artifact。
如果某个 action-effect mode 在多个 \(\Delta\) 下都保持，则它更可能是稳定的 physical mode。
这可以形成一个新诊断：
multiscale spectral transfer consistency。

它比单纯报告一个 latent spectrum 更接近信号系统和数学物理中的 scale robustness。
最值得优先加入的数学物理工具
按当前项目的实际价值排序：
第一优先级：Mori–Zwanzig memory
直接解释 frozen RGB latent 的 partial observability 和 local adapter 失败。
第二优先级：\(SE(2)\) Lie-group pose geometry
直接改善 PushT 中 angle、body-frame 和 contact-frame 的表示问题。
第三优先级：hybrid mode / complementarity
直接处理 contact mode switching，避免让单一连续 predictor 学习不连续动力学。
第四优先级：resolvent + finite-horizon transfer
把当前谱分析从 eigenvalue/alignment 推进到真正的 action-to-task response。
第五优先级：observability–controllability–energy 三联诊断
判断一个 latent mode 是否：
- 能被观察；
- 能被 action 激活；
- 能以物理合理方式传播。
建议形成的新统一理论对象
最终可以把当前研究写成：
\[
\boxed{
\mathcal W_\Phi
=
\left(
C_\Phi,\,
A_\Phi,\,
K_\Phi,\,
B_\Phi,\,
D_\Phi,\,
M_{\mathrm{task}},\,
\Gamma_H
\right)
}
\]
并提出：
\[
\operatorname{spec}(A_\Phi)
\not\Rightarrow
\mathcal H(z)
\not\Rightarrow
\text{planning success}.
\]
要使谱信息具有任务解释力，至少需要：
\[
\boxed{
\text{stable lifting}
+
\text{memory control}
+
\text{action excitation}
+
\text{task observability}
+
\text{mode validity}
+
\text{bounded transient gain}
}
\]
这会把我们的研究真正推进到一个更强、也更有数学物理深度的方向：
不是证明“谱分析能解决 world model”，而是建立一个投影—记忆—模式—能量—谱几何统一框架，判断 learned latent 中哪些动态结构是真正可观测、可控、可迁移和物理一致的。

近期文献中，Mori–Zwanzig latent closure、control-coherent Koopman、port-Hamiltonian latent dynamics 和 LeWM 的 Fourier physical structuring 都已经分别处理了其中某一部分；我们的潜在创新在于把它们统一到 covariance–predictor–action–task transfer 的诊断框架中，而不是简单重复某一种物理先验。






要把这项研究做成“数学和物理上站得住”的工作，关键不是继续增加指标，而是把研究对象从
\[
(\Sigma,P)
\]
升级为“有限记忆、混合模式、输入—输出”的动力学实现。这样才能同时处理谱、动作、任务空间、接触模式和坐标规范。
一、建议确立的核心对象
完整系统写成：
\[
x_{t+1}=F_{m_t}(x_t,a_t,\xi_t),\qquad
o_t=h(x_t)+\nu_t,\qquad
z_t=E(o_t),
\]
其中 \(x_t\) 是真实物理状态，\(m_t\) 是接触模式，\(z_t\) 是 frozen visual latent。
对 latent 做历史—动作 lifting：
\[
\eta_t=\Phi(z_{t-L:t},a_{t-L:t-1}),
\]
并使用有限记忆的混合动力学：
\[
\eta_{t+1}
=
A_{m_t}\eta_t+
\sum_{\ell=1}^{L}K_{m_t,\ell}\eta_{t-\ell+1}
+
B_{m_t}a_t+
N_{m_t}(\eta_t,a_t)+w_t .
\]
任务变量满足：
\[
\delta y_t \approx D_{m_t}\delta \eta_t,
\qquad
M_y\succ 0.
\]
对 memory kernel 做 companion-state augmentation 后，真正应该研究的对象是有限时域 action-to-task transfer：
\[
\mathcal T_{m,h}
=
M_y^{1/2}
D_m
\bar A_m^{h-1}
\bar B_m
M_a^{1/2}.
\]
频域表达为：
\[
\mathcal G_m(e^{i\omega})
=
M_y^{1/2}D_m
\left(
e^{i\omega}I-\bar A_m-\widehat K_m(e^{i\omega})
\right)^{-1}
\bar B_mM_a^{1/2}.
\]
这条主线把你提到的几个直觉统一起来：
- 机器视觉的上下采样：改变观测分辨率；
- 延迟嵌入与升维：恢复被投影掉的历史状态；
- 拉普拉斯/傅里叶变换：把时间域动力学转为 resolvent/frequency response；
- AI 中的 lifting：寻找更接近可辨识动力学的坐标；
- 逆映射：检验新的表示是否真的保留了 action-to-task transfer。
但必须强调：升维不会自动创造物理信息。它只有在满足 observability、persistent excitation 和 closure 条件时才有效。
二、最重要的新数学工具：最小 realization 与 Hankel 几何
这是目前最值得加入主线的工具。
对于局部线性系统：
\[
\eta_{t+1}=A\eta_t+Ba_t,\qquad y_t=D\eta_t,
\]
其输入—输出 transfer function 为：
\[
G(\lambda)=D(\lambda I-A)^{-1}B.
\]
在 controllable 和 observable 的 minimal realization 条件下，如果两个 realization 具有相同的 transfer function，则它们至多相差一个相似变换：
\[
A'=TAT^{-1},\qquad B'=TB,\qquad D'=DT^{-1}.
\]
因此：
- eigenvalues 是 gauge-invariant 的；
- \(D A^{h-1}B\) 是 gauge-invariant 的；
- transfer function 是 gauge-invariant 的；
- 普通 covariance alignment 和 Euclidean commutator 不是一般 gauge-invariant 的。
这给你们现有理论提供了更强的结论：
真正可识别的对象不是 latent eigenvector，而是 minimal realization 所表示的 action-to-task transfer equivalence class。

进一步定义有限时域 controllability 与 observability Gramian：
\[
W_c(H)=
\sum_{h=0}^{H-1}
A^hBM_aB^\top(A^h)^\top,
\]
\[
W_o(H)=
\sum_{h=0}^{H-1}
(A^h)^\top D^\top M_yDA^h.
\]
二者共同决定有限时域 Hankel 几何：
\[
\mathcal H_H
=
W_c(H)^{1/2}
W_o(H)
W_c(H)^{1/2}.
\]
其特征值对应有限时域 Hankel singular values。它们比 latent variance 更接近你们真正关心的对象：
一个 latent 方向是否既能被动作激活，又能被任务变量观测到。

这可以直接解释当前负结果中“高方差方向不一定包含动力学信息”的现象。
三、把现有诊断量重新分层
当前 alignment、commutator、Schur 和 transient 不应被视为同等地位。
第一层：坐标无关的 transfer quantities
这些应该成为主指标：
- Markov parameters：\[
  D A^{h-1}B;
  \]
- transfer function：\[
  D(\lambda I-A)^{-1}B;
  \]
- finite-horizon Hankel singular values；
- \(H_2/H_\infty\) transfer norms；
- inverse action risk；
- real/predicted transition discrepancy；
- hybrid pathwise transfer。
第二层：声明 metric 后的几何量
这些是条件性诊断：
- covariance–predictor alignment；
- principal angles；
- projector overlap；
- metric-aware commutator；
- task pullback geometry。
第三层：非正规与鲁棒性量
这些解释 horizon-dependent failure：
- Schur off-diagonal energy；
- resolvent norm；
- pseudospectrum；
- finite-horizon transient amplification；
- \(H_\infty\) gain；
- constrained joint spectral radius。
因此论文中的核心逻辑应改为：
\[
\text{latent covariance geometry}
\longrightarrow
\text{reachable-observable subspace}
\longrightarrow
\text{action-to-task transfer}
\longrightarrow
\text{finite-horizon risk}.
\]
而不是：
\[
\text{eigenvalue spectrum}
\longrightarrow
\text{planning success}.
\]
四、接触任务必须使用混合系统工具
PushT 不是单一连续系统，而是一个 hybrid dynamical system：
\[
m_t\in
\{
\text{free},
\text{approach},
\text{sticking},
\text{sliding},
\text{lost-contact}
\}.
\]
每个模式有不同的局部动力学：
\[
\eta_{t+1}
=
A_{m_t}\eta_t+B_{m_t}a_t+\epsilon_t,
\]
并且模式转移满足：
\[
m_{t+1}
=
T(m_t,\eta_t,a_t,o_{t+1}).
\]
对于一条模式路径 \(m_{0:H}\)，真正的 transfer 是：
\[
\mathcal T_{m_{0:H}}
=
D_{m_H}
A_{m_{H-1}}
\cdots
A_{m_0}
B_{m_0}.
\]
这说明“全局平均 spectrum”可能完全掩盖 contact-mode transition 的失败。
应增加：
- mode-conditioned \(A_m,B_m,D_m\)；
- mode transition confusion matrix；
- constrained joint spectral radius；
- pathwise transfer gain；
- mode-switching 后的 error amplification；
- lost-contact 后的 recovery reachability。
适合的理论来源包括：
- Goebel, Sanfelice & Teel, Hybrid Dynamical Systems；
- Stewart & Trinkle 的接触互补动力学；
- Posa 等人的 contact-implicit trajectory optimization；
- Jungers, The Joint Spectral Radius。
对应研究上的关键结论是：
当前 learned adapter 失败未必是连续回归器容量不足，而可能是将多个不可交换的 contact modes 错误压缩成了一个连续 operator。

五、引入非正规系统的完整工具链
单独报告 commutator 不够。应采用：
\[
A=Q(\Lambda+N)Q^*,
\]
其中 \(N\) 是 Schur off-diagonal coupling。
至少同时报告：
\[
\Gamma_H=\max_{1\le h\le H}\|A^h\|_{G},
\]
\[
R(\lambda)=\|(\lambda I-A)^{-1}\|_{G},
\]
以及 \(\varepsilon\)-pseudospectrum。
物理解释是：
- spectrum 描述渐近模态；
- Schur coupling 描述模态之间的非正交耦合；
- resolvent 描述外部扰动的放大；
- transient amplification 描述有限 horizon 内的真实风险；
- planning 通常更接近有限 horizon，而不是 asymptotic stability。
相关基础来源：
- Trefethen & Embree, Spectra and Pseudospectra；
- Schmid, “Nonmodal Stability Theory”；
- Kreiss matrix theorem；
- robust control 中的 \(H_\infty\) analysis。
这可以将你们的主理论升级为：
predictor spectrum 不足以决定 transfer；在非正规系统中，还必须测量 reachability、observability、resolvent 和 finite-horizon amplification。

六、Mori–Zwanzig 是解释 frozen latent 失败的核心物理工具
frozen visual latent 是完整物理系统的投影，因此通常不是 Markov：
\[
z_{t+1}
=
Az_t+Ba_t+
\sum_{\ell=1}^{L}K_\ell z_{t-\ell+1}
+\xi_t.
\]
其中：
- \(K_\ell\)：隐藏速度、摩擦、接触持续时间和历史状态；
- \(\xi_t\)：未解析动力学和观测噪声；
- \(A\)：瞬时 Markov 部分；
- \(B\)：当前动作的直接作用。
应严格比较：
1. Markov adapter；
2. finite-memory adapter；
3. low-rank memory kernel；
4. mode-conditioned memory kernel；
5. history encoder；
6. action-history interaction model。
如果 finite-memory closure 显著降低 real/predicted transfer error，说明之前的负结果主要来自 partial observability；如果 memory 改善预测却不改善 action ranking，则说明瓶颈转移到了 action excitation 或 mode inference。
核心文献：
- Lin et al., Data-driven learning for the Mori–Zwanzig formalism；
- Lin et al., Regression-based projection for learning Mori–Zwanzig operators；
- Gupta et al., Mori–Zwanzig latent space Koopman closure for nonlinear autoencoder。
链接：
- Mori–Zwanzig data-driven learning
- Regression-based Mori–Zwanzig
- MZ latent Koopman closure
七、物理结构工具：能量、耗散与无量纲化
Port-Hamiltonian 结构适合做物理一致性审计：
\[
\dot x=(J-R)\nabla H(x)+Gu,
\]
其中：
\[
J=-J^\top,\qquad R\succeq0.
\]
理想情况下：
\[
\dot H
=
-\nabla H^\top R\nabla H+y^\top u.
\]
对你们的 RGB latent，不能直接声称它是 Hamiltonian state。但可以在 privileged simulator state 上检验：
- prediction 是否产生非物理能量；
- torque pulse 是否违反耗散；
- contact recovery 是否导致能量爆炸；
- learned adapter 是否产生不稳定闭环。
可参考：
- Greydanus et al., Hamiltonian Neural Networks；
- Roth et al., Stable Port-Hamiltonian Neural Networks；
- Linares et al., Controlled oscillation modeling using port-Hamiltonian neural networks。
另外建议加入 Buckingham-\(\Pi\) 无量纲分析。PushT 中应显式考虑：
\[
\Pi_1=\frac{vT}{L},\qquad
\Pi_2=\frac{\tau T^2}{mL^2},\qquad
\Pi_3=\mu,\qquad
\Pi_4=\frac{\Delta t}{T}.
\]
这可以区分：
- 只是 pixel scale 改变；
- 只是动作 pulse duration 改变；
- 真正的摩擦与惯性变化；
- 采样率导致的谱 aliasing。
这比直接跨环境比较 latent eigenvalues 更具有物理意义。
八、最终建议形成的理论框架
建议把整篇研究的数学骨架固定为：
\[
\boxed{
\text{Projection}
\rightarrow
\text{Memory}
\rightarrow
\text{Hybrid Modes}
\rightarrow
\text{Minimal Realization}
\rightarrow
\text{Reachability/Observability}
\rightarrow
\text{Transfer Geometry}
\rightarrow
\text{Non-normal Risk}
}
\]
对应的核心对象为：
\[
\boxed{
\mathfrak T_H
=
\left\{
M_y^{1/2}D_m
\bar A_m^{h-1}
\bar B_mM_a^{1/2}
\right\}_{m,h}
}
\]
而不是单一的 covariance alignment 或 commutator。
论文真正可以争取的新颖表述是：
We formulate learned latent planning as a gauge-aware, finite-memory, hybrid input–output realization problem. This framework separates coordinate-dependent covariance geometry from similarity-invariant action-to-task transfer, and further distinguishes partial observability, contact-mode switching, non-normal amplification, and physical admissibility as independent failure mechanisms.

中文可以写成：
我们将 learned latent planning 形式化为一个具有规范自由度、有限记忆和混合接触模式的输入—输出 realization 问题。该框架区分坐标依赖的协方差几何与相似变换不变的 action-to-task transfer，并进一步将部分可观测性、接触模式切换、非正规放大和物理可容许性分解为相互独立的失败机制。

九、优先级
最值得立即加入研究的不是所有工具，而是以下四个：
1. Minimal realization / Hankel singular values
   解决 gauge、variance 与 planning relevance 的根本问题。
2. Finite-memory transfer function
   解决 frozen latent 的 hidden-contact-memory 问题。
3. Constrained joint spectral radius
   解决 contact-mode switching 不能由单一 spectrum 描述的问题。
4. \(H_\infty\) / resolvent / robust reachable tube
   解决有限 horizon 内 prediction error 被非正规动力学放大的问题。
现有工具包已经覆盖 Mori–Zwanzig、Gramians、PBH、\(SE(2)\)、hybrid mode、Schur、transient、energy 和 multiscale consistency，可从 [MATHEMATICAL_PHYSICS_TOOLKIT_20261012.md (line 1)](/home/gbk/code/LeWorldModel/MATHEMATICAL_PHYSICS_TOOLKIT_20261012.md:1) 查看；下一步应把它们统一到 minimal input–output realization，而不是继续平行增加诊断量。
当前最窄、最强的科学主张应是：
在满足有限记忆 closure、persistent excitation、minimality 和 mode coverage 的条件下，planner-relevant transfer 由 reachable-observable action-to-task realization 决定；latent spectrum 和 covariance alignment 只能作为该 realization 的条件性诊断，而不能单独作为 transfer certificate。

这比“commutator 是通用证书”更严格，也比“升维能够自动解决规划”更有可能形成真正的理论贡献。


21:38
