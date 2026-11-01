# D2P：执行前缀后控制能力改变的因果审计

## 独立评分验证已完成；完整编译器另行验证

202654000–007独立八根、232既有源episode排除的评分验证现已完成，
403920个主机CUDA预测样本、456个真实分支、23项完整性门及独立复算通过。
它只重评原动作和后缀，不是完整Q一致编译器。两种模式对原固定greedy的
全驱动/仅肩/仅腕关节终距均差分别为−0.00412/−0.00546/−0.00599和
+0.00249/−0.20227/+0.07573；未稳定普遍获益，不事后选模式或删除损害根。

完整Q一致编译器202655000–007的v2现已完成805536个主机CUDA预测样本、
912物理分支、18门和1632独立QP审计，三任务均无完整模型见证。
对原固定greedy关节终距均差+0.003010/+0.001808/+0.016933，不构成修复。
逐程序覆盖归因保留全部408程序，真实关节与原RGB的首段/终态合取候选
分别16/29/21个，尚不含真实能力改变，不用于oracle选择。
原封存native/common/greedy/zero角色的能力探针也已完成58单元、3116真实未来、
6522主机CUDA预测样本，12门全通过，1044个独立拟合。common见证1/2/6个中
固定Q真实RGB能力改善0/2/5个，原关节C2诊断及同根完整要件仍均0，未打开P1。
入口与失败修正归档见
[独立目标度量验证](D2P_GOAL_METRIC_PROSPECTIVE_CONFIRMATION_20261026.md)。

## 最新补齐：新八根能力、目标重放与目标编码器度量（2026-10-26）

以下已完成结果优先于历史“新根能力探针尚缺”的状态，不回填旧P1。
`reacher_d2p_root_metric_capability8_v1_20261026_cuda.json`完成新八根、三任务、
四模型封存角色的57个去重单元、3074条物理未来、6433个主机CUDA预测样本；
12项完整性门全部通过。独立分析完成684个原空间与342个固定Q拟合，共1026个。

|任务|模型见证|原物理关节C2诊断通过|固定Q真实RGB能力双尺度通过|全部要件同时通过|
|---|---:|---:|---:|---:|
|全驱动|0|0|0|0|
|仅肩|3|1|2|0|
|仅腕|5|0|3|0|

后三列限定模型见证。仅肩202653000/002有双终态和固定Q真实RGB能力改善，
却无相应真实关节能力增益；202653007有原关节C2诊断，却无固定Q RGB双尺度
增益。不得跨根拼接要件。全根/全任务/全角色图已生成并逐图检查，位于
`results/reacher_d2p_root_metric_capability8_figures_v1_20261026/`。

`reacher_d2p_metric_task_alignment8_v1_20261026.json`保留全部408程序。
每任务1088对排序中，模型RGB/真实RGB反转502/322/376对，真实RGB/关节反转
67/212/275对。腕部损害根202653000的模型与真实RGB都偏好损害动作；其有关节
限位维未错误环绕。因此“预测RGB排序失败”不足以独自解释该损害。

`reacher_d2p_goal_metric_pixels8_v1_20261026_cuda.json`主机CUDA审计8个声明
目标的直接物理渲染及32条去重轨迹：目标RGB/编码与逐包动作/RGB/编码/qpos
精确一致，11项门全通过。损害根腕部像素MSE从0.00626614升至0.00661576，
关节终距从0.742601升至1.663395，但冻结latent代价改善；陈旧目标或错误重放
不是该实例的解释。像素也不是普遍有效的关节成功代理，不替换原任务定义。

进一步实际测试仅用原目标RGB和冻结编码器输入Jacobian的固定候选：
`Q=.01*(J_E J_E^T+.01I)^-1`，同目标三任务共用Q，无训练/关节标签/调参。
`reacher_d2p_goal_encoder_metric8_v1_20261026_cuda.json`完成1536个CUDA输入
VJP、104次编码器forward、408个证书QP，14项门全通过；固定方向有限差分
最大相对误差低于0.001。全部矩阵先封存再评分，评分不读取物理未来。
独立NumPy/SciPy分析复算408个QP与全部48个两模式选择，随后按根身份接入
原物理完整动作库，输出`reacher_d2p_goal_encoder_metric8_analysis_v1_20261026.json`。
v3分析进一步验证编码器源码归档绑定、保存分析器SHA，并补全部原固定控制
的配对对照：`reacher_d2p_goal_encoder_metric8_analysis_v3_20261026.json`。
v1/v2保留，v2是与v1同算法的重复复算，不提供新证据。

|任务|仅改gain：见证/双终态合取|一致Q评分：见证/双终态合取|一致Q关节均差对自身greedy|
|---|---:|---:|---:|
|全驱动|5/1|4/1|+0.001934|
|仅肩|0/0|0/0|0|
|仅腕|1/0|2/0|-0.005234|

一致Q评分对**原固定greedy**的关节终距均差分别为：全驱动+0.001934、仅肩
-0.269365、仅腕-0.088017；对原flat CEM+GN incumbent分别为-0.004656、
-0.027121、-0.185875。均为同一已知八根完整动作库重评，未宣称等预算独立
确认。仅肩虽无reconfiguration见证，新greedy本身已改变，不能称没有控制
收益，也不能把这种收益称C2通过。腕部202653000损害动作被回退避免，但未
恢复reconfiguration。这是待新根验证的评分合同候选，不是全面修复。

相关广覆盖回归1502项通过（49.20秒）；纯数值测试不构成神经CPU实验或效能
证据。全根图manifest的10个归档/图像/说明文件及输入分析SHA均已重新核验。

分母均8；合取含模型见证、真实首段劣于zero、真实RGB和关节终态胜greedy，
**未包含本候选真实能力探针，不能称完整C2**。一致Q模式自身greedy可能改变，
不可冒称对原固定greedy或等预算baseline的改进。此处只重评原17动作及原后缀，
没有重新优化后缀，更不是新根校准或正式P1。该候选不足以修复问题；不支持
“理论已被否定”，也未确认新的因果代码缺陷。后续须区分评分合同、模型响应
与物理任务能力，而非调阈值、删根或用物理outcome选程序。

## 运行前协议

2026-10-26更新：完整库v1/v2均因第七根同状态RGB单像素分叉严格停止，
不构成C2负结果。完整库v3以显式`fixed-warm-render-v1`在主机CUDA重跑已完成：
408单元/22576真实未来/47192预测样本，13项完整性门全部通过。
原52单元逐位对齐，保留全部根与全部17程序以及原P1门槛。
新增两次真实draw/两次完整状态检查的逐根收据门；详情见
[渲染审计](D2P_RENDER_REPLAY_AUDIT_20261026.md)。完整库测量通过不等于P1通过。

### 完整库归因结果

独立复算全部4896个有界拟合，原52单元再核对；全部统计/图像产物及归档源码
SHA匹配，两张完整八根图已目视。入口：
`results/reacher_d2p_prefix_capability_all17_analysis8_v3_warm_20261026/`。
另完成全部408程序的原selector谓词重建及1632个响应—残差两因素组合拟合。

|任务|物理诊断候选数|包含候选的根数|旧模型判据拒绝数|仅因common-gain拒绝|
|---|---:|---:|---:|---:|
|全驱动|3|2/8|3|3|
|仅肩|20|5/8|20|13|
|仅腕|7|3/8|7|2|

这里的诊断候选明确要求：首段真实RGB劣于zero，终态关节距离和RGB均优于
greedy，关节共同残差有界增益在两固定尺度均优于zero和greedy。它不是新增
P1成功定义。所有30个候选的真实RGB共同增益都未同时胜控制，实际历史
model-FD也未恢复这些RGB增益见证。仅肩7个、仅腕4个另有首段model/physical
符号分歧；各原因不互斥。完整结果拒绝“库中没有任何物理诊断候选”的归因。

四种响应/残差组合在全部30候选上均无正的候选—控制point-estimate margin；
各任务最接近零者仍为负数，不能以数值证书间隙解释拒绝。原408个QP独立
复现，gain最大绝对差4.945377440890297e-12。因而仅纠正预测后继漂移或共同
残差偏置，不足以修复当前判据；RGB欧氏能力代理与关节任务能力之间的度量
合同失配需要显式处理。此证据既不证明所有预测正确，也不否定D2P理论。
不能直接用qpos/oracle增益替换部署分数，不能按这30个候选挑根或重设阈值。

拒绝归因与两因素产物分别为
`results/reacher_d2p_capability_rejections_all17_8_v3_warm_20261026.json`和
`results/reacher_d2p_capability_factorial_all17_8_v3_warm_20261026.json`。
两者均保持`P1_admission=false`、`physical_labels_used_for_deployment=false`。

### 固定模型根响应度量候选（尚非修复效能）

为检验原提案显式Q的缺项而不直接用qpos/readout代替RGB，增加一个固定候选
`model-root-response-metric-v1`：Q=.01*(J0*J0^T+.01*I)^(-1)，J0仅取原模型
在zero前缀之后预测的后缀Jacobian。两个.01沿用原有正则化尺度，不设CLI调参
或多候选择优。它是输出坐标预条件，不声称已经校准成物理任务度量。

`audit_reacher_d2p_root_metric.py`先封存定义、源码及全部输入SHA，再读取完整
八根/17程序的保存数组，对模型预测及两个尺度真实RGB响应各拟合一次，共
1224次。模型度量构造不读取任何真实后继、qpos、能力标签或outcome；真实
探针仅离线评估该候选，保留全部408程序与原zero/greedy控制。31项相关数值
合同测试已通过，包含SPD公式、正交坐标变换、零响应及输入不变性。
该候选不改变当前部署策略、原成功定义、根集合或P1门，不能把point-estimate
margin当作原证书通过。

完整1224拟合现已完成，协议、全部输入、归档生成器SHA及408程序身份已复核。
入口：`results/reacher_d2p_root_metric_all17_8_v1_20261026.json`。只在原物理
诊断候选中统计以下点估计，不选取有利根，也不改变原诊断标签：

|任务|原物理诊断候选|新度量model正margin|新度量真实RGB双尺度正margin|两者同时|
|---|---:|---:|---:|---:|
|全驱动|3|0|0|0|
|仅肩|20|8|9|5|
|仅腕|7|1|4|0|

原欧氏度量在这些候选上均无正margin。固定预条件在仅肩有局部一致性改善，
但全驱动/仅腕没有同时成立者，不能称通用度量修复。当前只改变能力代理的
数值诊断，没有把首段、后缀优化和成功指标整体改写为新度量。所有原model
首段/终态判据仍须保留；仅看5个同时成立者不等于实际selector会选择它们。
正式采用前需要CUDA证书化评分、完整模型选择及独立新根校准/闭环证据。

### CUDA证书化评分及实际选择核验

`reacher_d2p_root_metric_score.py`已将上述固定度量接入模型专属评分接口：
仅投影原模型字段，保留首段/终态代价和全部原selector比较条件，不读取真实
状态或后继标签。主机CUDA完成24个度量SVD、408个有证书QP，8项完整性门
全部通过；全部旧selector精确复现，无新神经推理或VJP。产物：
`results/reacher_d2p_root_metric_certified_score8_v1_20261026_cuda.json`。

独立`audit_reacher_d2p_root_metric_decisions.py`按root identity连接全部8根、
三任务，不以物理最优候选替换实际选中动作；独立SciPy点解逐一落入408个
CUDA增益证书区间（允许1e-8相对数值容差）。完整实际选择结果为：

|任务|模型见证根|实际选中动作满足原物理诊断|满足新度量RGB双尺度增益|两者同时|
|---|---:|---:|---:|---:|
|全驱动|0/8|0|0|0|
|仅肩|4/8|3|2|2|
|仅腕|3/8|0|2|0|

物理列只对模型见证根计数。原“仅肩有5个两者同时的候选”不等于实际能选出
5根；完整模型选择实际只留下2根。仅腕虽然生成3个模型见证，却没有选中
符合原物理诊断的动作。该结果拒绝“完成评分证书就解决了P1”，也不把剩余
误选归结为一般理论失败。产物：
`results/reacher_d2p_root_metric_decisions8_v1_20261026.json`。

### 独立新根固定评分验证（已完成）

202653000–007在仅检查HDF5元数据后一次性冻结；8个源episode互异，与此前
224个源episode不重叠，没有按outcome重抽样。新脚本
`run_reacher_d2p_root_metric_confirmation.py`在原冻结模型上保留3任务/17前缀、
同一固定CEM+GN后缀，以及原欧氏selector、zero、greedy、flat GN、flat CEM控制。
两种选择共享同一完整动作库；只干预能力评分，不改首段/终态指标或门槛。
使用`fixed-warm-render-v1`，先封存全部根的预测和决定再执行456真实分支。

每任务预测预算16,830 = 原compiler 2,482 + 17*(前缀1 + Jacobian前向2 +
CEM 386*2 + GN 4*8*2 + split/full检查5)，共403,920个predict样本。
该预算没有声称完全计算匹配。新度量另有408个CUDA有证书QP。
独立分析器`analyze_reacher_d2p_root_metric_confirmation.py`保留所有新根和
三任务，复算模型QP，检查封存动作/执行哈希，输出配对RGB和关节终态后果。
此轮不重新测量真实局部能力增益，不能凭终态改善自动放行P1。

主机CUDA已完成全部403,920个预测样本和456条真实执行分支，17项完整性门
全部通过。独立分析器重新拟合408个模型增益，封存协议、预测、全部root
sidecar、源episode排除和归档源码SHA均核对通过。结果入口：
`results/reacher_d2p_root_metric_confirmation8_analysis_v2_20261026.json`。

|任务|新评分模型见证|旧评分模型见证|新选择满足真实首段+双终态合取|全8根关节终距相对greedy均差|
|---|---:|---:|---:|---:|
|全驱动|0|1|0|0（全部回退）|
|仅肩|3|0|3|-0.270743|
|仅腕|5|0|1|+0.100748|

负均差有利于新评分。仅肩3胜/0负/5平，描述性配对bootstrap区间
[-0.655816,-0.001290]；真实RGB终态成本均差-0.139489，区间
[-0.314454,-0.004832]。3个选择均保留真实首段劣于zero，不把直接进展
冒充重配置。例202653000的关节终距1.580106→0.454785，202653002为
1.572911→0.542610；相同后缀动作库、原模型、原门槛，仅能力评分改变。
这是独立新根上评分干预的局部修复证据，不是一般理论否定或正式P1通过。

仅腕真实RGB终态5胜/0负/3平，均差-0.052236，但关节距离4胜/1负/3平、
均差+0.100748，区间[-0.025604,+0.339187]。不能隐去损害根202653000：
RGB成本1.838393→1.781598同时关节距离0.742601→1.663395。另3个模型
见证不满足真实首段非贪心。该反例再次显示RGB代理、物理任务和模型首段
校准不是同一合同；固定Q并未整体解决它们。新物理能力探针尚未补齐，
不能用这4个首段+终态合取根追认C2/P1，也不据此开放P2/P3/P5。

分析v1复用了旧bootstrap helper的文字标签，误把RGB成本比较标为joint、
把新根标为known roots。已修复分析器标签并生成不可变v2，增加对应回归；
v1保留作审计，所有数值不变。这是报告元数据错误，不解释真实排序反转。

```bash
.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_root_metric_confirmation \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_recentered_confirmation_suffix_fresh8_20261026_cuda.json \
  --seed 202653000 \
  --output results/reacher_d2p_root_metric_confirmation8_v1_20261026_cuda.json
```

最新相关广覆盖数值/逻辑回归1472项通过（48.90秒），不是效能或P1证据。

继续检验C2，而非把终态改善本身认作重配置成功。使用上一轮已知的新八根
202652000–007、官方原始冻结Reacher权重、相同目标和三种精确执行器限制。
这不是新的独立效能队列，不改变P1阈值、任务或beta，也不训练任何模型。

每个任务保留四个模型决定的前缀角色：旧common-residual候选、非线性可行代价
选择、同库直接进展greedy、零前缀。仅按相同prefix索引去重，所有角色仍保存，
不得只测那个已知阳性根或只测有利候选。

先封存全部根/任务/角色映射、源码与输入SHA，再执行全部物理探针。每个前缀：

1. 从完整原始根快照执行相同五步动作，要求首包完整192维RGB表示、qpos、动作
   哈希及图像哈希与原报告逐位一致。
2. 在模型预测的后继重新计算原完整Jacobian，要求与原封存响应逐位相同。
3. 改用真实RGB/action历史，在同样字面raw零动作切点重新计算两包预测和完整
   自动微分Jacobian。历史从真实执行更新，不把预测状态或归一化动作写成真实历史。
4. 在真实后继状态的每个可用raw动作坐标，固定±0.1、±0.25两尺度，测量全部
   十步真实未来及其192维表示和双关节坐标；同时保存实际历史下的模型FD端点。
   禁用关节保持字面raw零，动作仅归一化一次。零未来重复两次必须完全重放。
5. 独立数值分析比较AD/model-FD、actual-history-model/physical-FD、模型预测后继/
   真实观测后继的响应，以及完整RGB和尊重限位/周期拓扑的物理关节响应。
   对所有前缀用共同zero-prefix真实未来残差比较能力，再分别保留own-state残差。
   有界线性拟合只是局部诊断，不能升级为非线性全局可达性或成功率证明。

所有神经计算在授权主机CUDA运行，禁止CPU神经fallback；NumPy/SciPy仅复算
已保存数值。物理状态只用于恢复/评估，不进入模型、selector或新的部署动作。
物理探针不得成为planner的oracle。前向样本、完整Jacobian数与VJP行数分别核账。

## 运行与证据

新增`audit_reacher_d2p_prefix_capability.py`；输出首次使用，不覆盖历史报告。
首轮四角色审计已完成：52个去重单元、2864条真实十步未来、5988个主机CUDA
predict样本、104个完整Jacobian、19968行逻辑VJP；11项完整性门全通过。
独立分析器从所有端点重建完整响应并完成624个SciPy有界拟合，原前缀首包及原
预测后继响应与上游完整回执逐位一致。两个全根图已目视检查；最新入口为
`results/reacher_d2p_prefix_capability_analysis8_v3_20261026/`，旧版本保留。
七项图/统计产物及生成器SHA已复核；相关广覆盖回归**1219项通过**。测试只证明
逻辑合同，不代替上述真实CUDA测量，也不改变P1状态。

不开放P1/P2/P3/P5。以下首轮命令已完整执行：

```bash
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_prefix_capability \
  --reference results/reacher_d2p_recentered_confirmation_suffix_fresh8_20261026_cuda.json \
  --selection-reference results/reacher_d2p_recentered_confirmation_mixed16_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_prefix_capability8_20261026_cuda.json
```

## 首轮结果与边界

### 响应与共同残差分离的两因素审计

新增`audit_reacher_d2p_capability_factorial.py`，避免把旧selector与真实历史下
诊断的差异统称为“模型响应错误”。对全部程序分别拟合四种固定组合：预测后继
响应/模型共同残差、预测后继响应/真实共同残差、实际历史响应/模型共同残差、
实际历史响应/真实共同残差。始终保留原zero与greedy控制，不重新挑有利控制。
第一种必须独立复现原有有界QP证书区间；其余只作evaluator替换，真实未来
残差不得进入部署选择，也不能以点估计margin替换原证书阈值。

新审计的数值单元与已有独立归因测试61/61通过。已用v3首个完整根202652000
三任务/全部17程序做204个数值拟合，原51个QP全部复现，gain最大绝对差
4.547473508864641e-13。首根运行只验证新归因工具；随后已基于完成的全17/八根
报告完成1632个四组合拟合，拒绝部分报告的合同保留，完整结论见本页上方。
未更改任何运行中封存依赖、任务、阈值或模型参数；P1仍关闭。

0.1尺度AD/model-FD中位相对误差，全驱动/仅肩/仅腕分别0.001098/0.000835/
0.001148；模型AD与真实RGB-FD余弦中位数0.984378/0.983881/0.991726。
0.25尺度相应AD/model-FD误差0.006904/0.004910/0.007381，保留有限尺度非线性
与像素量化的影响，不把非零误差称作梯度代码错误。预测后继换成实际历史后，
Jacobian中位相对变化约4.00%/4.52%/3.43%，没有证据支持“预测/实际历史完全相同”。

关键全驱动根202652000虽然原common候选index9首段非贪心且终态胜greedy，
但0.1尺度真实关节共同残差可达增益比例为0.922498，低于greedy的0.970456与
zero的0.970868；0.25尺度仍为0.932970对0.970456/0.970871。因此终态小幅获益
不是本测量下的控制能力提高见证。不能拿该根填补C2。

全驱动common候选相对zero的RGB共同增益比例差约+0.304（8胜），真实关节共同
增益比例差却为-0.006899（1胜/7负）。两种度量、残差和正则化单位不同，此差异
不能直接证明哪种理论错，但明确禁止用RGB局部增益代替当前关节成功任务的能力
改变。仅肩feasible-selector相对greedy真实关节增益比例差+0.004299，3胜/0负/
5平，两尺度一致；RGB差值区间跨零，且被选前缀仍不是相对zero非贪心。

新增机制诊断显式分开四项：真实首段劣于zero、真实终距优于greedy、真实终态RGB
优于greedy、真实关节共同增益在两尺度均优于zero及greedy。它只定位缺失环节，
不是事后设定的新P1通过合同，始终`P1_admission=false`。

## 全候选库覆盖与qualification失配

为避免把“没有选出见证”误写成“库中没有可用程序”，新增
`audit_reacher_d2p_bank_coverage.py`，对三批全部24根、三任务、17程序、三种后缀
求解器逐一复算已保存的物理RGB与关节终态。所有真实标签仅用于事后归因，不能
用于部署选择、修改阈值、定义新任务族或新outcome根。

|任务|旧GN4库内存在物理两阶段获益候选的根|CEM后GN4后|模型识别这些候选的根|旧common准入/选中的根|
|---|---:|---:|---:|---:|
|全驱动|4/24|18/24|9/24|1/24|
|仅肩|2/24|19/24|18/24|0/24|
|仅腕|2/24|19/24|16/24|1/24|

这里“物理两阶段获益”要求真实首段劣于zero且终态RGB、关节误差同时胜同库
greedy，仍可能是很小的改善，不证明直接梯度停滞、物理能力改变或统计优势。
该审计证明后缀求解与qualification/选择存在需要分解的缺口，不把模型已识别
但被qualification排除的候选直接称为预测失败。原第一批没有封存proposal-native
决定，分析器保留缺失，分母为16，而不事后填充成24个预声明决定。

完整记录：`results/reacher_d2p_bank_coverage24_20261026.json`。

## 历史v1/v2失败现场（保留，不追认）

以下记录中断当时的状态及诊断，不是当前v3状态；v3完整通过及独立归因见页首。

仅测四个selector角色仍可能遗漏真正的能力改变动作。因此已扩展到全部8根×
3任务×17原始前缀，不按任何物理outcome筛程序，固定相同尺度、时域和动作限制。
原52个单元必须完整逐位复现，否则fail-closed；其余单元只是补齐库内能力地图，
不能反向选择新部署策略。

完整资源合同：408单元、22576条真实未来、47192个CUDA预测样本、816个完整
Jacobian、156672行逻辑VJP。该轮主机CUDA任务在第七根开始后严格停止：
2026-10-26 14:06:35触发`physical zero future replay not exact`，两次同状态零未来的
192维表示不同。前六根完整sidecar保留；主报告未生成，不记全库通过，也不能
把本次未完成归为机制负结果。当时下一步用独立分层审计区分完整物理状态、RGB和
相同输入CUDA编码的确定性，不放宽逐位检查。
输出：`results/reacher_d2p_prefix_capability_all17_8_20261026_cuda.json`。

独立分层CUDA复查已完成：根202652006的51个程序/任务单元，每个零未来重放4次，
共204条真实未来、408次CUDA编码及153个跨重放比较。完整快照、关节、当前与
历史RGB、192维表示全部逐位相同，且相同输入的重复编码全部一致。输入SHA及
生成器SHA已复核。它没有复现长运行错误，不能宣称错误已修复。
记录：`results/reacher_d2p_zero_replay_layers_202652006_20261026_cuda.json`。

已修正审计器的失败可观测性漏洞：原先抛异常丢失具体候选及物理/RGB分层现场，
现在在严格停止前保存`.failure.json`（根/任务/程序、两端完整状态、RGB/history
哈希、表示最大误差、调用计数和协议SHA）。不改变动作、正常路径模型调用数、
单元字段、逐位比较或准入规则。随后用全新v2产物路径从头完整主机CUDA重跑，
不拼接失败批次的六根、不覆盖旧记录。v2完成前仍不记全库通过。
输出：`results/reacher_d2p_prefix_capability_all17_8_v2_20261026_cuda.json`。

更新：v2在2026-10-26 14:30:24再次严格停止，新增失败回执已抓到
202652006/full_drive/index16的完整状态一致而终态RGB分叉。单根原探针顺序
复现确认仅一个绿色通道量化单位、九组派生几何一致；v2未生成完成报告。
后续受控渲染干预见[渲染分叉审计](D2P_RENDER_REPLAY_AUDIT_20261026.md)。

分层复查命令（已完成，神经计算仅主机CUDA）：

```bash
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_zero_replay_layers \
  --reference results/reacher_d2p_recentered_confirmation_suffix_fresh8_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --seed 202652006 --repeats 4 \
  --output results/reacher_d2p_zero_replay_layers_202652006_20261026_cuda.json
```

v2重跑使用下列同一完整命令，仅将`--output`明确改为上述`_v2_`路径；其余参数、
八根及52原单元对齐合同均不变。

```bash
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_prefix_capability \
  --reference results/reacher_d2p_recentered_confirmation_suffix_fresh8_20261026_cuda.json \
  --selection-reference results/reacher_d2p_recentered_confirmation_mixed16_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --all-programs --alignment-reference results/reacher_d2p_prefix_capability8_20261026_cuda.json \
  --output results/reacher_d2p_prefix_capability_all17_8_20261026_cuda.json
```

## 全库结果后的拒绝原因归因合同

新增独立 `audit_reacher_d2p_capability_rejections.py`，不修改运行中的CUDA审计或
任何部署selector。它先重新验证完整能力报告及原52单元逐位复现，再以显式
root identity连接原封存模型决定，逐项重建全部408个候选的四个原始准入条件：
非零程序、模型首段非贪心、模型终态胜greedy、共同残差增益的下界胜两对照上界。
原qualified集合、greedy和native索引必须精确复现，否则停止归因。

每个候选同时保留完整真实能力诊断及两尺度RGB增益比较。输出明确区分：
真实诊断候选却被哪个模型条件拒绝，以及模型准入却不满足真实诊断的候选。
这不把诊断合取变成新P1规则，不使用真实标签改选择器、阈值、任务、beta或根。
即使没有候选，也只代表此固定程序库、时域、局部测量和已知八根的边界，不能
宣称排尽代码错误或否定一般理论。

归因还显式区分模型/真实首段符号、真实关节/RGB共同增益、实际历史模型/真实RGB
增益，以及旧预测状态或参照残差/实际历史FD增益的分歧，保留非互斥原因。
任何这类事后分歧本身都不会生成`code_defect_confirmed=true`：仍需要对应的因果
代码修正与新根验证。新增纯数值回归反例也证明，完全准确的动力学加一个非等距
编码器，就足以让未加任务度量的latent gain反向排序；这不是本批真实效能证据，
也不授权用真实outcome拟合新Q或调整现有P1门槛。

已在原八根封存预测上独立复建408条模型判据，精确复现全部24个selector决定；
其中共同增益条件分别拒绝全驱动126/136、仅肩133/136、仅腕125/136条候选。
这些分母包含zero/greedy等全部原程序，各拒绝原因可重叠；拒绝频繁本身不是bug
证据，当时须等待完整物理对照后归因。新增及相关定向测试53项通过，相关广覆盖
回归1239项通过（47.94秒），`git diff --check`通过；这些仅证明逻辑合同。

前五根sidecar的临时完整库诊断发现部分四项合取候选，不能把原四个selector角色
的零见证外推为全库零见证。例如202652000仅肩index3/11均通过临时物理合取，
模型首段/终态也识别，却仅被common-gain拒绝。但index3的0.1尺度真实关节增益
0.527668高于greedy0.520035，其真实RGB增益0.059711反低于greedy0.273079；
模型actual-history-FD亦为0.023501对0.276117。此例说明任务关节度量与RGB代理
存在具体错位，不能简单称作自动微分错误。该预览不替代失败全库审计，不用于
开启P1、改阈值、选择部署程序或新根；现已由v3完整复核，结果见页首。

本轮已执行的完整v3拒绝归因命令（旧v2命令已废止，因其无完成主报告）：

```bash
.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_capability_rejections \
  --capability results/reacher_d2p_prefix_capability_all17_8_v3_warm_20261026_cuda.json \
  --reference results/reacher_d2p_recentered_confirmation_suffix_fresh8_20261026_cuda.json \
  --output results/reacher_d2p_capability_rejections_all17_8_v3_warm_20261026.json
```

本轮已完整执行的独立有界复算及全根图命令：

```bash
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_prefix_capability \
  --report results/reacher_d2p_prefix_capability_all17_8_v3_warm_20261026_cuda.json \
  --output-dir results/reacher_d2p_prefix_capability_all17_analysis8_v3_warm_20261026
```
