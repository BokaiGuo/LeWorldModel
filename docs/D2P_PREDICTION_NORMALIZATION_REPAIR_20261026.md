# Reacher：原六根排序反转的预测头归一化统计根因

## 本轮结论

用户引用的原六根 margin 反转不是已经证实的理论负结果，也不能概括为原始
发布模型的必然边界。固定原六根、目标、动作、真实后继及视觉坐标，只将
joint-repair 检查点换回官方原始检查点，排序一致从4/6恢复为6/6。进一步固定
joint-repair的全部学习参数，只恢复预测投影头BatchNorm运行统计，同样恢复6/6。

这是对原失败根的受控因果定位，不是通过改变根、目标或阈值获得的结果。
恢复后的模型将原两个动作正确识别为**贪心动作**，所以它们不再是有效的非贪心
重配置见证；不能把“纠正错误见证”写成“P1通过”。整个研究仍未闭环。

## 精确输入合同

- 原根：20274800..20274805，全部保留；两个原反转为20274802/20274804。
- 原六根目标均合法，不受此前wrist限位目标修复影响。
- 原回执：`results/reacher_d2p_frozen_source_validation_20261025_cuda/p1_calibration/report.json`。
- 原排序：同目录`p1_fidelity.json`；本轮重新执行全部12个候选/neutral物理包，
  root/goal/warmup动作与后继RGB身份精确重放，原四种cost按原容差重现。
- 原回执绑定的是joint-repair `d68e2da61d4b64fa01ddb5c476cc6fc975ffa4ff750a1efe4822a4516cdf009d`，
  **不是**官方原始 `eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469`。
- 两模型observed successor与goal的完整192维视觉坐标逐位相同，最大差为0。
- 全部神经运算使用主机RTX5060Ti/CUDA，无CPU回退、无qpos读出、无新动作选择。

## 检查点和数值精度隔离

证据：`results/reacher_d2p_original_margin_checkpoint_v3_20261026_cuda.json`，
48个实际predict样本，所有完整性门通过。FP64只改变action encoder/predictor/
pred_proj动力学计算，视觉encoder/projector保持原FP32坐标，不能称为全网络FP64。

| 根 | joint-repair预测margin | 官方原始预测margin | 实际RGB margin |
|---|---:|---:|---:|
|20274800|+0.0007704932|+0.0548417587|+0.0038779480|
|20274801|+0.1622865193|+0.2031818471|+0.1210102701|
|20274802|+0.0057020035|-0.0724168624|-0.0508830592|
|20274803|+0.0143770591|+0.1927024552|+0.1708968314|
|20274804|+0.0050629611|-0.0221809047|-0.0176635345|
|20274805|+0.0004319620|+0.0231796783|+0.0260475059|

margin为candidate减neutral成本，负值表示candidate更贪心。表内用同一组FP32
latent向量做FP64代数，故最后几位与原float32成本直接相减略有不同。两种
前向精度均为joint-repair 4/6、官方6/6。FP64动力学引起的最大margin变化分别
3.16e-7/3.45e-7，不能解释原反转。候选平均latent L2误差2.712646→1.061919，
neutral为2.564883→0.863910；不据此声称独立效能或统计显著性。

上游`Embedder.forward()`内部强制`.float()`，所以直接`.double()`不是有效的
双精度前向。审计采用相同Conv1d/MLP代数去除这个cast，并要求FP32动作编码
与原生路径逐位相同；该门全部通过。没有修改安装库或部署前向。

## 2×2因果定位：参数与运行统计分离

证据：`results/reacher_d2p_original_margin_bn_20261026_cuda.json`。24个根/variant
单元、48个实际predict样本。两套未干预基线完整latent逐位复现；每个参数家族
内只交换以下非学习buffer，所有参数及其他buffer保持不变：

```text
pred_proj.net.1.running_mean
pred_proj.net.1.running_var
pred_proj.net.1.num_batches_tracked
```

| 学习参数 | 预测头运行统计 | 排序一致 | 候选均值L2 | neutral均值L2 |
|---|---|---:|---:|---:|
|joint-repair|joint-repair|4/6|2.712646|2.564883|
|joint-repair|官方原始|6/6|1.400867|1.157387|
|官方原始|joint-repair|5/6|2.488928|2.220750|
|官方原始|官方原始|6/6|1.061919|0.863910|

这证明恢复base统计足以消除这两个已知反转，也表明统计与参数有交互；不能说
学习参数完全无影响，更不能承诺只换统计对所有未见根都成立。诊断variant只在
内存构造，没有覆盖原权重，也没有将post-hoc修复冒充training-free主线。

代码原因：三条repair trainer在每个训练epoch对`pred_proj.train()`，使共享
BatchNorm在不同的source-transition、candidate与neutral批次上连续更新统计。
训练时各forward用当前batch统计，部署singleton使用混合后的running统计；这
不是与发布检查点相同的预测投影合同。冻结encoder/projector并不能冻结pred_proj
的非参数统计；只哈希named_parameters也不能检测该类变化。

## 已实施的代码防护

新增`reacher_d2p_training_normalization.py`，接入dynamics、counterfactual、joint
三条训练入口。新默认`--batchnorm-policy frozen-base-statistics`：

1. 视觉encoder/projector始终eval。
2. 动力学保持训练和梯度，dropout仍按训练合同工作；仅BatchNorm固定eval统计。
   BN affine参数不被偷偷冻结。
3. 每epoch核查running_mean/var/count逐位不变，漂移直接fail closed。
4. manifest显式记录normalization合同、所有buffer名称及不变性。
5. 历史训练复现必须显式`--batchnorm-policy legacy-domain-updates`。原结果和
   检查点不改写；旧命令使用新默认会产生不同的训练协议。

主机CUDA真实训练池集成验证已完成：
`results/reacher_d2p_training_normalization_guard_20261026_cuda.json`。
原独立训练pool各取同样8条训练样本，调用实际counterfactual trainer，各执行一次
optimizer step。新默认三个预测BN统计均不变，legacy均变化；两者梯度有限、学习
参数实际更新，视觉参数/buffer逐位不变、原checkpoint不变。所有门通过。
这是实现验证，不是完成重新训练或证明泛化改善，不将该训练batch准确率写成
研究效能。当前training-free主线继续使用官方原始模型。

随后补齐实际transition与joint训练循环的主机CUDA集成验证：
`results/reacher_d2p_transition_normalization_guard_20261026_cuda.json`。
完整93GB官方source SHA与原训练manifest一致，按元数据固定选取训练episode
0/1/2/3各两条clip，排除当前/历史184个评估episode。两条trainer分别在两种
BN模式执行一次真实optimizer step，96个predict样本，所有门通过。加上上述
counterfactual验证，三条实际训练路径共6个optimizer step/128个predict样本：
新模式统计全部不变、旧模式三个预测头buffer全部变化；视觉完整state不变，
参数实际更新、梯度有限。该检查不进行完整重训或新checkpoint泛化声明。

## 独立复核、图与失败记录

两个独立analyzer从保存的完整向量重新计算所有cost、margin、误差分解与排序，
按root identity连接，验证原基线和不变的视觉坐标。全部根均保留，无结果筛选。
全六根PNG/PDF、LaTeX图注、summary、notes和SHA manifest位于：

- `results/reacher_d2p_original_margin_checkpoint_figures_v2_20261026/`
- `results/reacher_d2p_original_margin_bn_figures_v3_20261026/`

上述图包均已逐图目视检查，两个当前图包各7个产物的SHA manifest校验通过。
BN后处理额外要求完整的9项严格gate，并独立复核上游检查点对照，不接受只留下
`cuda_only=true`的部分回执。新增18项全六根/四variant反篡改测试覆盖缺根、
重复身份、错序配对、部分gate、非布尔gate、改变视觉坐标、非逐位原基线与改写
summary。它们只是证据完整性测试，不是神经预测效能。另有3项source元数据与
分块哈希回归；当前广覆盖757项通过。
v3仅增加严格回执复核，不改变向量或科学数值；v1/v2图包保持原样。

第一次检查点审计因相对路径被loader解释为cache模型名在加载前中断；第二次
因上游Embedder的强制float cast在FP64阶段中断。两次protocol/implementation
均保留，v3换新路径完成，不能把这些新审计入口错误说成原六根排序的根因。
第一版精度图上下边界过紧，v2增加margin并保留第一版；BN图v2增加顶部留白，
均不改科学结果。

新增transition/joint集成审计首次在任何协议/结果写出前复用了小文件`read_bytes`
哈希函数，93GB source触发MemoryError，尚未训练。已改用现有分块SHA实现，
加禁止无界读取的回归后重跑完成；此新入口错误不属于原P1反转原因，亦未中断
并行的新根实验。

## 与合法目标新根确认及P阶段的关系

合法目标确认入口已要求显式`--goal-contract`，默认旧周期目标的隐式路径已移除。
新`joint-feasible-v1`绑定限位wrapper、原render事务及每根goal admission。
旧证据只能解释为`legacy-periodic-replay`，不能被重新标为合法目标任务。
8个新根202647000..007来源episode唯一，排除历史176个episode；该独立确认与
原六根归一化定位是两项实验，不能混写。新根运行现已完整完成：136词/128条
真实分支、398540个CUDA predict样本，14项完整性门全部通过；6个新witness
中5个真实RGB非贪心，202647003仍反转。它使用官方原始权重，不能被BN干预
自动解释；无目标反射，不作为限位干预收益。全根图/独立复核与数值见目标修复页。

本轮已经解释并消除了原六根在指定checkpoint中的两个排序反转，尚未证明所有
P1负结果均被修复。更不能从官方正确识别原贪心动作推出非贪心机制收益。
P1仍需合法任务合同下新鲜preflight、独立校准与outcome；P2/P3/P5不借此打开。

该新八根剩余反转随后完成多包历史误差定位：官方原生rollout与手写递推在全部
64个192维包边界逐位相同，而真实已执行历史的单包预测消除诊断反转。进一步
实际因果RGB waypoint跟踪完成64条物理分支，改善多数终态但未恢复原非贪心
条件。这与本页BN修复是两项不同实验；不能把单步evaluator诊断当部署oracle，
也不能把执行反馈收益当BN修复泛化。新八根202648000..007四阶段确认现已完整
完成：192条完整分支、401878个主机CUDA predict样本，完整性门全通过；native
初始计划反馈均距0.063315→0.027490，8胜/0负，七个模型witness经反馈后相对
固定neutral均真实非贪心。但动作已改，原方案仍有001反转、旧批003仍未修复，
对同反馈强对照CI跨零，不是原动作排序修复或P1通过，更不是本页BN修复效能。
见[多包历史与因果反馈修正](D2P_WAYPOINT_TRACKING_REPAIR_20261026.md)。

## 复现

所有神经入口须主机CUDA，输出及sidecars须不存在。后处理不进行神经计算。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_original_margin_checkpoint \
  --reference results/reacher_d2p_frozen_source_validation_20261025_cuda/p1_calibration/report.json \
  --fidelity results/reacher_d2p_frozen_source_validation_20261025_cuda/p1_fidelity.json \
  --receipt-policy results/reacher_d2p_joint_repair_v1_20261024_cuda/checkpoint \
  --official-policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output /absolute/new-checkpoint-audit.json
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_original_margin_checkpoint \
  --report /absolute/new-checkpoint-audit.json --output-dir /absolute/new-checkpoint-figures
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_original_margin_bn \
  --crossover /absolute/new-checkpoint-audit.json --output /absolute/new-bn-audit.json
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_original_margin_bn \
  --report /absolute/new-bn-audit.json --output-dir /absolute/new-bn-figures
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_training_normalization \
  --repair-manifest results/reacher_d2p_joint_repair_v1_20261024_cuda/checkpoint/repair_manifest.json \
  --output /absolute/new-normalization-integration.json
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_transition_normalization \
  --repair-manifest results/reacher_d2p_joint_repair_v1_20261024_cuda/checkpoint/repair_manifest.json \
  --exclusion-protocol results/reacher_d2p_feasible_goal_confirmation_fresh8_20261026_cuda.protocol.json \
  --output /absolute/new-transition-joint-normalization-integration.json
```
