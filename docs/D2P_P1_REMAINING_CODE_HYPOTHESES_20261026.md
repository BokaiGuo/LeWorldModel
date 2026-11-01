# D2P P1 尚存代码假设：独立静态审计（2026-10-26）

## 范围与结论

本轮依用户要求仅阅读 Reacher goal metric/fullQ、PushT nuisance suffix/metric/GN 相关源码、测试源码及小型分析回执；仅新增本文件。没有执行测试、数组重评分、神经推理、GPU 查询/作业、训练、下载或 HDF5 读取。没有重做目标 agent 干预，也没有触碰正在进行的 Reacher metric/reference 或 model-only pool 工作。

已完整读取 `code-debugging/SKILL.md`，按 LogicError 的“单位—形状—索引—目标函数”路径审计。技能引用的 `/home/gbk/.claude/skills/code-debugging/references/debug-patterns.md` 不存在，未猜测其内容。以下“排除”均限于指定路径；测试只读，不能写成本轮测试通过。

**尚未发现足以确认为本批 P1 失败根因的明确实现 bug。** 当前优先级是明确正则中心、局部代理与实际非线性目标的合同，而非继续解释成 Jacobian 转置错误。两项尤其不能混称：

- Reacher GN 是 delta 阻尼；PushT GN 是绝对 atanh 参数能量；新的 PushT array compiler 又是 delta ridge。各自公式可以正确，但不能以同一个 `.01` 名字宣称同一控制代价。
- PushT nuisance runner 只改变已执行 prefix 后的 suffix 度量/求解，没有接入 array compiler 的 prefix/stage/own-V 选择链。它不是完整 D2P compiler 修复的实验。

以下代码位置均相对于 `/home/gbk/code/LeWorldModel/le-wm-main/`，行号对应本轮读取版本；并行工作可能继续修改共享源码，复核时须对照已封存 implementation。

## H1：Jacobian 转置、方向符号或 Q 重复加权

**状态：所读公式中已排除简单符号/转置错误；没有排除所有神经路径的局部误差。**

- `experiments/spectral_reachability/reacher_d2p_goal_encoder_metric.py:21–33` 构造 `W=sqrt(Q)` 并左乘向量/响应；不是直接以 Q 乘残差再平方。
- `reacher_d2p_goal_metric_compiler.py:20–29,57–61,74–84` 分别生成 `WJ`、`W(target-own_zero)` 和加权 terminal；目标也在第48行用同一 W 变换。
- `pusht_d2p_goal_nuisance_metric.py:16–24` 构造正交投影 P。`preflight_pusht_d2p_p1_reconfiguration.py:151–164` 使用 `J=P*dF/dp`、`r=P*(goal-F)`、`Vh.T diag(s/(s²+lambda)) U.T r`；不是把输出左奇异向量当动作方向。
- `pusht_d2p_full_metric_suffix.py:38–42` 为 `J.T@(goal-prediction)-lambda*p`，符号与绝对参数正则相符。
- 已有测试源码 `tests/test_reacher_d2p_goal_metric_compiler.py:10–18` 检查批/单样本和 `W@A`；不是本轮重新运行。

**最小区分实验（未执行）：** 非神经、非对称矩阵 `A=[[1,2,3],[4,-1,.5]]`，固定非对角 SPD Q，逐项比较残差梯度 `2*A.T@Q@(A*p-g)` 与实现链；同时检查 `W.T@W=Q`。不要重跑真实根评分。该实验只区分代数接线，不检验模型物理 fidelity。

## H2：动作 Jacobian 单位混用、tanh 链式导数遗漏

**状态：两条现有 neural caller 的显式映射已覆盖；跨 compiler 坐标等价与训练统计来源未排除。**

- Reacher `reacher_d2p_masked_action_chart.py:14–32,51–52` 的变量是 reduced raw torque，扩展到五步十坐标后仅归一化一次；未启用驱动器为字面 raw zero。
- PushT `preflight_pusht_d2p_p1_reconfiguration.py:83–100` 的变量是 p，依次 `tanh(p)`、重复五个二维 command、按 supplied mean/std 归一化；`_suffix_terminal:121–133` 在求 Jacobian 的 callable 内完成映射，AD 包含 tanh 与 std 链。
- `pusht_d2p_packet_contract.py:90–115` 明确 caller 负责统计来源；不能以公式存在证明 supplied public statistics 就是权重训练统计。
- `pusht_d2p_nuisance_program_scoring.py:3–6,37–43` 只接收 caller-supplied Jacobian，无法知道其是 raw、normalized 还是 atanh 坐标。

**最小区分实验（未执行）：** 无模型仿射 packet encoder fixture，给五步不同系数和非单位 std，在 `p=0` 与非零 p 分别手算 `dF/dp = dF/da * diag(1-tanh²(p))`，再比较 raw/normalized/p 坐标下经换元后的同一个二次型。不得把重复二维 command 的两自由度误称十个独立 command；不得以 CPU 神经模型代替。统计 provenance 另查小型配置/manifest，不读全量 HDF5。

## H3：nominal action 与 delta 正则中心不一致

**状态：差异已确认；“错误”取决于预声明目标，不能直接判为 bug 或历史反转根因。优先级最高。**

- Reacher `reacher_d2p_masked_programs.py:35–48`：令 `H=J.T@J+lambda*I`、当前点 n，QP 右端为 `J.T@r+H@n`。其准确目标为 `||r-J*(u-n)||²+lambda*||u-n||²`，约束的是绝对 u 在 box 内；不是 `lambda*||u||²`。非线性接受只看终态 MSE（第43–49行）。这是阻尼 GN，不是绝对动作能量优化。
- PushT `pusht_d2p_full_metric_suffix.py:21–25,35–52` 准确优化 `||P*(F(p)-g)||²+lambda*||p||²`；方向中有 `-lambda*current`。不能“修复”为 Reacher 公式而不先改变协议。
- 新 array scorer `pusht_d2p_nuisance_program_scoring.py:75–89` 优化 `||r-PJ*delta||²+lambda*||delta||²`，返回 `nominal+delta`。`tests/test_pusht_d2p_nuisance_program_scoring.py:70–80` 甚至显式锁定非零 nominal 的 delta ridge；这不是遗漏 nominal 的偶然迹象。
- 当前 nuisance suffix runner 导入的是 `pusht_d2p_consistent_metric_suffix`（第13–16行），并没有调用 array scorer，不能把 array scorer 的中心差异归因到已完成 suffix 实验。

**最小区分实验（未执行）：** 取标量 `F(p)=p, g=1, n=.5, lambda=1`：绝对 ridge 解为 `.5`，delta ridge 解为 `.75`。分别对 box 内两点写出目标值与正常方程，先让协议选择目标；无需神经推理或旧根重评分。若要求 absolute action ridge，Reacher QP 右端应是 `J.T@r+J.T@J@n`，但本轮禁止修改，且当前 fit 注释只声明 box GN。

## H4：stage / own V / common residual 接线混淆

**状态：Reacher fullQ 主选择链已静态排除 common residual 冒充 own V；两种 selector 必须分账。**

- `reacher_d2p_goal_metric_compiler.py:63,76–79` 分开计算 common residual 与各 prefix 自身 `target-own_zero`，同一响应传给两个不同 gain receipt。
- 同文件第90–98行用 W 计算 prefix 与 terminal cost；第114行主 `decision=native.select(...)`，另存 historical common rule。
- `reacher_d2p_proposal_native_selection.py:29–48` 主链取 `own_response_gain.bounded_quadratic_objective`，总分 `prefix_cost + objective/192`；不是拿 common gain 或末态成本替代 own V。第52–55行还分别检查首段非贪心、终态和 own-V 证书。
- `pusht_d2p_nuisance_program_scoring.py:75–89` 的 stage 来自 prefix state，own residual 来自 nominal suffix terminal，同一 P，所有误差和 ridge 均除 D；没有发现漏除192或把 prefix residual 当 suffix residual。

**最小区分实验（未执行）：** 两候选不同 own-zero terminal、相同 J，构造 common-residual 排序与 own-residual 排序相反的解析 fixture；断言两个决定写入不同字段，且主决定只依赖 own 字段。已有测试源码 `test_pusht_d2p_nuisance_program_scoring.py:58–67` 有 own residual 隔离，不必重新计算真实 bank。

## H5：有界局部 own-V 与实际后缀搜索优化的不是同一个函数

**状态：合同差异未排除；不是 Q 漏接。**

- Reacher `reacher_d2p_goal_metric_compiler.py:73–84` 的 own-V 在 zero suffix 线性化，并通过 bounded-response QP 加 `.01*||u||²`；`bounded_response_qp.py:83–87` 给出实际公式。
- 同一 compiler 随后执行 GN→CEM→GN。`reacher_d2p_masked_suffix_search.py:15–20,28–35,38–50` 以非线性 terminal MSE 搜索/保留 incumbent，不包含 own-V 的动作 ridge；GN 是 H3 的 delta 阻尼。第90–98行储存的最终 terminal cost也不是局部 V。
- 因而“同 Q”成立不意味着“同名 reach cost 与实际优化代价完全相同”。QP 证书只证明局部 box quadratic 的数值正确，不证明非线性后缀可达性或 C2。

**最小区分实验（未执行）：** 无神经一维曲率 fixture `F(u)=u+c*u²`，固定两个 prefix 的偏置、同一 W；解析列出 zero 线性 V、实际执行 terminal MSE、带 absolute ridge 的实际目标。判定差异来自曲率还是正则中心，不改 beta、margin 或筛真实根。该实验不是另一次 Reacher reference/metric bank audit。

## H6：PushT nuisance 只修 suffix，却被当作完整 stage/own-cost compiler

**状态：尚未实现完整接线，已确认；不是已有 suffix runner 违背其声明。**

- `run_pusht_d2p_nuisance_suffix_bank.py:52–59` 明说全部 prefix 固定、只重新编译 observed suffix，不是 full prefix/stage/compiler repair。
- 第82–104行读取旧64个 prefix，执行后观测、再调用 `solver.solve`；没有生成 nuisance spectral prefix，没有计算 `stage+beta*own_V` 决定。
- `pusht_d2p_consistent_metric_suffix.py:21–32` 把同 P 接到线性初始化及 GN；raw legacy score 被显式标作不用于选择。
- `pusht_d2p_nuisance_program_scoring.py:30–44,99–113` 是独立数组模块；其存在不等于 neural caller 已提供自身状态/Jacobian provenance、非线性检查和封存选择。

**最小区分实验（未执行）：** 先做零推理调用图/AST 合同检查，确认 full compiler caller 是否真实调用 array scorer 并封存 stage/own-V/最终动作；目前所读 runner 没有。下一步若要实现，应另立协议，先解析 fixture，并保留旧 runner 不动；不能直接以旧 suffix 结果打开 P1。

## H7：prefix/suffix history 错一拍或重复 bootstrap

**状态：现有索引链静态一致，且既有 exact replay/split 门有覆盖；非零 history 的独立语义验证仍不能仅靠共用 helper 排尽。**

- Reacher `reacher_d2p_masked_action_chart.py:35–48` 用 `history[:,1:]+future[:1]` 预测实际当前 boundary 的后继；`reacher_d2p_masked_programs.py:16–23` 推进后同步滑动 image/history。
- fullQ compiler 第85–89行对同完整动作要求 split 与 direct **逐位相同**；这排除两条路径分段不一致，但共享同一错位 convention 时仍可能共同出错。
- PushT `preflight_pusht_d2p_p0.py:139–147` 声明实际 boundary `F1,F2,F3` 与 incoming `A0,A1,A2`；P1 `_advance:114–118` append 新 A 后截最后 H 项，即 `A1,A2,A3`。
- nuisance runner 第98–104行先执行完整 prefix，后用 `frames[1:]+[rgb]` 与 `history[1:]+[prefix]`；`_suffix_terminal:127–130` 再 append suffix，索引应为 `A2,A3,A4`，没有再造 bootstrap。
- 第101–102行 successor RGB 与旧 bank 精确绑定；第115–119行 identity suffix action/final RGB 精确重放。当前 PushT warmup 全零（P0 第118行），对“交换两个不同历史 action”的敏感性弱。

**最小区分实验（未执行）：** 无神经 tracing predictor，把每帧和每个 packet 赋独一标号，非零且互异 history，逐调用记录 image/action 对；不要再跑真实轨迹。检查 prefix→observed suffix 与 direct path 的精确标号序列，而非仅比较两条共用 recurrence 的数值。

## H8：nuisance 投影数值秩、平方度量与原物理目标不一致

**状态：P 左乘两次导致错误平方的简单猜想已排除；输入有限 goal span 的适用范围和浮点秩风险未排除。**

- `pusht_d2p_goal_nuisance_metric.py:16–24` 用 float64 SVD 和 float64 epsilon 判秩，P 对称幂等，故 `||P*r||²=r.T*P*r`，不是一般 SPD Q 错误平方。Reacher 用 sqrt(Q)，见 H1。
- goal embeddings 实际来自 float32 neural encode，runner 第83–86行以 float64 构 P 后再转 float32。接近退化的 goal 差分可能被 float64 判秩计入微小量化方向；现有代码未依据输入误差界证明秩稳定。不能据 outcome 调大 SVD tolerance。
- 只投影五张给定目标图的四维差分空间，不等于全局 agent invariance，更不等于 block-pose metric；模块第1–5行已声明此限制。
- `analyze_pusht_d2p_nuisance_suffix_bank.py:69–80,106–118` 分开 own/common/identity/pose 指标；不存在把 `forecast.nonlinear.predicted_terminal_vector`（投影坐标）当 raw vector 的已读错误，第76行明确检查它等于 `P@predicted`。

**最小区分实验（未执行）：** 非神经近共线 goal fixture，在固定 float32 roundoff 范围内给同一 goal set 做扰动，报告完整奇异谱、P 对称/幂等误差与子空间夹角，不选最有利秩，不重评分真实 bank。若存在显著秩不稳定，先归为数值合同问题，再独立预注册处理规则。

## H9：返回 receipt 丢失参数，导致绝对能量/饱和坐标无法完全复核

**状态：审计可观测性缺口已确认，不是已确认动作执行错误。**

- `pusht_d2p_full_metric_suffix.py:58–64` 返回参数给 caller，但 receipt 仅存 objective 和 terminal。
- `pusht_d2p_consistent_metric_suffix.py:28–32` 对外输出的是 `tanh(params)` action，不封存最终 params 或 `sum(params²)`。
- `analyze_pusht_d2p_nuisance_suffix_bank.py:78–83` 对严格内部 action 可用 atanh 重建能量，但对 float32 饱和为 ±1 的坐标只能检查非负能量下界，无法独立重建完整 objective。饱和并不意味着原参数无限；不能据此自动判 bug。

**最小区分实验（未执行）：** 无模型 `p=10` 等固定参数 fixture，展示 float32 tanh 取整如何丢失可逆性；未来新协议若需要完整证据，应另存 p、参数能量及接受点 prediction。不要修改既有 report，也不补造旧见证。

## 小型回执交叉核对与停止条件

- 已只读 `results/reacher_d2p_full_goal_metric_compiler8_analysis_v2_20261026.json`：第3–4行列出1632次独立 QP fits、912物理分支；这意味着不应重复该已完成数值复算。本轮未读取124MB predictions或 source HDF5。
- 已只读 `results/pusht_d2p_nuisance_suffix_bank4x16_analysis_v1_20261026.json` 的 summary：own-metric跨臂成本不能当 common-objective改善；在共同 nuisance metric 下均差约−0.063998，但 pose cost均差约+2.930957（13改善、23恶化、28平）。四根而非64个独立根。仅引用既有结果，不重新分析或重跑。
- 分析器源码第121–136行要求原 report 的11门、3968预测样本和128分支；本轮没有重跑分析器或重新验证哈希，因此不把这些历史 gates 表述为本轮现场重验。

**建议的最小后续工作：** 先在协议层书面固定 H3 的正则中心以及 H5 的局部 V 与实际后缀目标关系；随后如获授权，仅新增非神经解析 fixtures（H1/H2/H3/H7）区分公式接线。不要把 nuisance suffix 的有限结果升级为完整 compiler/P1，不复制现有目标干预、真实 bank评分、QP reference复算或 model-only pool builder。

本轮没有达到“明确 bug”证据标准，故没有修改任何共享实现。若后续 fixture 证明某实现违反其预声明目标，应先报告精确反例、受影响 caller 与历史 receipt 范围，再由用户决定修复；不得借机训练、降低阈值、换成功定义或挑 outcome 根。
