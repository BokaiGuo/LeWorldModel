# Reacher：多包预测历史误差定位与因果 RGB 跟踪修正

## 当前判定

原六根的两个反转已经由预测头 BN 运行统计的受控交换定位，见
[归一化代码修复](D2P_PREDICTION_NORMALIZATION_REPAIR_20261026.md)。本页处理另一项
问题：使用官方原始冻结权重、合法目标时，新根202647003仍出现四包前缀排序反转。
不能把两批实验、两种checkpoint或两种修复混为一谈。

新证据将剩余反转定位到多步自由滚动的预测历史误差：项目手写滚动与官方原生
`model.rollout`在全部64个完整192维包边界向量上逐位相同，最大差为0；不能将
这一反转归因于这条已检验的rollout索引/分段实现。用已经执行的真实历史做单包
预测时，全部八根的终包排序符号正确。但这只是evaluator定位，不是部署修复，
因为规划时不能预先获得未执行分支的未来RGB。

随后实际实现并执行了因果修正：先封存原模型生成的全部中间状态，逐包只读取
本分支已经观察到的RGB/action history，使用冻结模型跟踪下一个封存状态。
全部8根、4类原计划、2种执行方式共64条30步物理分支全部完成，384个包边界、
5536个主机CUDA predict样本，11项完整性门通过。它减少了平均执行偏差，也
改善了多数根的终态，但**没有修复原非贪心前缀条件，不开放P1**。

固定这套反馈规则后，另八个来源隔离新根202648000..007的四阶段主机CUDA确认
现已完整完成。native初始计划的重放/跟踪均距为0.063315/0.027490，8胜/0负；
七个原模型witness经反馈后均相对固定neutral保留真实非贪心条件，包括新根001
的执行反转恢复。这是因果执行修正的新根支持，**不是原动作排序已修好**；旧根003
仍未修复，对同反馈强对照的CI也跨零，不开放P1。详见下方独立确认。

后续还完成同八根全部64个真实prefix后继的双尺度全坐标物理响应审计，5184条
未来执行、10496个CUDA predict样本及448个独立增益复核。它没有建立相对greedy
的稳定控制能力改变优势；原反馈执行改善与统一探针未来是不同动作合同，不互相
替代。首版新审计的严格身份失配保留且未被后续复现，不将它直接归为渲染bug。
详见[真实响应和重放身份审计](D2P_FEEDBACK_PHYSICAL_RESPONSE_AUDIT_20261026.md)。

## 固定动作的误差定位

权威证据：
`le-wm-main/results/reacher_d2p_packet_rollout_fidelity8_v2_20261026_cuda.json`。
v1保留；v2增加官方原生rollout对照，共256个predict样本。没有改动作、目标、
根、权重、统计buffer或评估阈值，全部8根candidate/neutral四包前缀的真实
RGB/history/action边界均精确重放。10项完整性门通过。

根202647003的candidate减neutral终包RGB目标成本margin为：

| 预测/观察方式 | margin |
|---|---:|
| 原自由滚动 | +0.0165028572 |
| 每包已执行历史的单包预测（evaluator-only） | -0.0105063915 |
| 真实后继RGB | -0.0516061783 |

自由滚动仅此根反转；observed-history诊断八根均无反转。全16条固定路径的平均
latent L2误差，随包数分别为：

| 包 | 自由滚动 | 已执行真实历史单包预测 |
|---|---:|---:|
|1|0.999529|0.999529|
|2|1.294935|1.019495|
|3|1.449846|0.962683|
|4|1.543522|0.941970|

这些数据支持预测历史误差累积的定位，不证明单步模型无误，也不支持“理论被否定”
或“所有代码问题已穷尽”。官方action Embedder只做逐token的kernel-1卷积及MLP，
不是跨token平滑造成该差异；没有改安装库。

### 精确区分公共状态偏移和动作相对响应误差

独立NumPy后处理还在全8根×4包×2种预测上验证了完整192维恒等式：
`observed_margin - predicted_margin = neutral_baseline_bias + response_linear_error + response_quadratic_error`。
最大恒等式残差7.66e-16；这里使用真实neutral未来只作evaluator归因，不能作为部署输入。
以下从保存的FP32完整向量用FP64代数重算，末位可与GPU端先求均值再相减的数字
相差约1e-7；未改变模型前向、成本定义或阈值。

| 根 | 原预测margin | 实际margin | 公共neutral偏移项 | 动作响应线性项 | 动作响应二次项 |
|---|---:|---:|---:|---:|---:|
|202647000|+0.014770|+0.009216|+0.010531|-0.018260|+0.002176|
|202647001|-0.073169|-0.047376|+0.074310|+0.015674|-0.064190|
|202647002|-0.060172|-0.003700|+0.091638|-0.002319|-0.032847|
|202647003|+0.016503|-0.051606|+0.010450|-0.069240|-0.009319|
|202647004|+0.012511|+0.001055|+0.002291|-0.004956|-0.008791|
|202647005|+0.014404|+0.018991|-0.003972|+0.008807|-0.000248|
|202647006|+0.031737|+0.032067|-0.044370|+0.043657|+0.001043|
|202647007|+0.071535|+0.050027|+0.068538|-0.022070|-0.067975|

003的响应误差合计-0.078559，公共偏移+0.010450反而部分抵消。即使oracle
使用真实neutral后继、保留模型动作响应，margin仍为+0.026953，仍与真实负号
相反。这进一步说明简单平移校正不足；不意味着可以跳过实现审计或否定理论。
对应全根两图、完整64项分解及SHA manifest位于
`results/reacher_d2p_packet_margin_error_figures_20261026/`，两图已目视且7项
产物SHA通过；12项精确恒等式/非有限输入合同回归通过。

## 因果反馈修正合同

入口：`experiments/spectral_reachability/audit_reacher_d2p_waypoint_tracking.py`。
它是新执行诊断，不替换原P1默认部署路径。

1. 保留202647000..007全部根及native dispatch、同库greedy、neutral prefix、
   完整六包CEM四类计划。使用官方原始权重SHA
   `eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469`。
2. **只从pre-prefix `.selection.json`取初始计划**：动作词的4包参数加其当时
   planned 2包commands；CEM使用当时完整6包参数。禁止使用真实prefix执行后
   `.suffix.selection.json`里的native commands作为初始计划。解码后的30步
   raw动作与原报告各`*_carry`完整见证逐位相同。
3. 从原根生成6个完整192维模型waypoint，所有32条初始计划和waypoint在任何
   新物理执行前统一封存；不投影到qpos，不以真实结果选waypoint。
4. 比较原动作完整重放与反馈跟踪。第一包严格不改，从第二包起只用本分支已经
   执行得到的真实RGB/action history。每包直接有界command GN固定4次、damping
   0.01，带SPD box-QP证书和非线性模型下降检查；无tanh反解截断。
5. 每包动作/输入RGB身份/冻结目标先写不可覆盖selection，执行回执哈希链接
   到下一包selection。完整链共384次先选后执行。未来neutral分支图像、物理
   qpos/qvel和目标关节状态不提供给planner；物理量只用于执行器恢复和终态评价。
6. 实际predict调用逐样本计数，独立后处理按32条初始计划的21次多步预测、384
   次局部预测与全部GN回执重新核账为5536。权重、全部buffer和eval状态不变。

## 已知八根结果

证据：`le-wm-main/results/reacher_d2p_waypoint_tracking8_20261026_cuda.json`。
初始selection SHA为
`3a8f627cce045c8178b246bad157a31fe3f8b92f3d92aa7cd0b0f985774e4398`。
全部32条carry基线的完整动作及终态指标重新复现；全部第一包的真实RGB逐位相同。

| 原始计划 | 开环重放平均关节距离 | 同计划反馈跟踪 | 反馈相对重放胜/负 |
|---|---:|---:|---:|
|native dispatch|0.044391960|0.024081641|7/1|
|同库greedy|0.050626128|0.033513114|5/3|
|neutral prefix|0.063695234|0.029145045|8/0|
|完整六包CEM|0.053331959|0.042998323|5/3|

native跟踪末包平均waypoint L2偏差1.966333→1.245317；四类计划的误差均下降。
原反转根202647003的终态关节距离0.034000→0.005798。但是根202647002由
0.016425恶化到0.053757，不能删掉；native对自身重放的描述性95%CI
[-0.039195,+0.000896]跨零。对同样反馈greedy为5胜/1负/2平，均差-0.009431，
描述性CI[-0.026065,+0.002662]也跨零，不宣称稳定机制收益。

最关键的未解决点：反馈后的003根相对**原neutral动作重放**第四包margin为
-0.025958，仍是实际贪心，原重放margin为-0.051606。
`zero_tracking`会产生非零补偿动作，不是P1的零动作基线；native tracking减
zero tracking的margin为-0.036769，只是同反馈控制器的策略比较。004根在该
策略比较为-0.003404，但相对固定neutral重放为+0.009383，**不能据前一个数
声称004失去非贪心条件**。初版文字混淆这两种参照，已由原完整向量独立复算纠正；
没有改动动作、物理结果、封存分析器或当时正在运行的新根确认。这是新报告的口径
修正，不是此前P1反转根因。

反馈改变了动作，不能用旧动作的模型margin与新动作后继比较后声称“排序准确率”；
这里只检验真实执行是否保留原计划的非贪心意图，003仍未满足。全根实际margin如下：

| 根 | 原计划重放减固定neutral | 跟踪减固定neutral | 跟踪减同样反馈zero计划 |
|---|---:|---:|---:|
|202647000|+0.009216|+0.005467|+0.006758|
|202647001|-0.047375|-0.052464|-0.059713|
|202647002|-0.003700|-0.002901|+0.001273|
|202647003|-0.051606|-0.025958|-0.036769|
|202647004|+0.001055|+0.009383|-0.003404|
|202647005|+0.018991|+0.019433|+0.005969|
|202647006|+0.032067|+0.067609|+0.056038|
|202647007|+0.050027|+0.077651|+0.077561|

修复终态误差不等于通过P1；也不把“反馈zero”与“真正neutral”混为一个方法。
该双参照表另有不可覆盖回执
`results/reacher_d2p_waypoint_reference_margins8_20261026.json`，入口
`audit_reacher_d2p_waypoint_reference_margins.py`重新验证上游384条因果哈希链后生成。
新增5项参照合同测试包含两参照会得出相反符号的反例，不允许将反馈zero标签
替代固定neutral，也不允许标记为same-action model fidelity。

## 过去观测残差能否直接修复开环排序

另做同一官方模型、同一8根和同一candidate/neutral四包动作的纯预测对照：
`results/reacher_d2p_packet_causal_residual8_20261026_cuda.json`。从根之前完整三帧
RGB和三条已执行动作，预测已观察root，取observed-minus-predicted的192维残差。
固定系数1，不拟合、不访问候选未来RGB，预声明比较终点加一次及每步加一次。
所有修正预测先统一写入不可覆盖`.predictions.json`，之后才与既有真实后继连接。
136个CUDA predict样本、9项完整性门全通过；原16路径全部64个原始预测向量逐位
复现，参数和全部buffer不变，没有新增物理效能声明。

| 路径 | 第四包平均latent L2 | 终包排序反转根 |
|---|---:|---|
|原始自由滚动|1.543522|202647003|
|终点加单位过去残差|1.228917|202647003、202647004|
|每步加单位过去残差|2.464126|202647003、202647004|

原反转根003的模型margin反而从+0.016503变为+0.028121/+0.123825，真实为
-0.051606。降低平均latent误差不保证改善候选排序；过去一次转移的公共偏移
不足以修复本批反转，不能据此改部署或挑选新的残差系数。该负结果只否定这两条
固定修正，不否定研究理论，也不宣称所有可能代码缺陷被排除。

独立完整坐标后处理和全根两张PNG/PDF位于
`results/reacher_d2p_packet_causal_residual_figures_20261026/`；两图已目视，7项
产物SHA通过。新增13项数据合同测试通过，覆盖原基线逐位性、固定系数、根身份、
完整向量及首步两种修正相同；它们不是CPU神经实验。

## 独立复核、图与回归

两个NumPy分析器独立重算完整坐标的cost、margin、L2偏差，按root identity连接；
反馈分析器核验全部384条selection/execution哈希链、执行动作SHA、观察前沿、
未改第一包、原carry基线和实际compute总量。所有根和失败分支均保留。

- `results/reacher_d2p_packet_rollout_fidelity_figures_v2_20261026/`：全八根逐包margin，
  全16路径×4边界误差；严格完整gate及原生rollout核验。
- `results/reacher_d2p_waypoint_tracking_figures_20261026/`：全八根四类计划终态距离，
  四类计划完整六包平均执行偏差。

每包均含PNG/PDF两图、summary、notes、LaTeX图注和SHA manifest。上述四图已
逐图目视，两个当前图包各7个产物SHA通过；旧packet图包同样7项通过并保留。
新增83项数据/frontier/完整坐标回归全部通过，不作为CPU神经效能。
包含这些测试的广覆盖840项已通过；另补齐4项完整pipeline的先封存后执行、
子命令失败即停、source元数据冲突拒绝和CUDA失败不回退测试，也全部通过。
pipeline测试首轮有一项把带/不带`-u`的命令固定下标当作模块名，已改为按`-m`
定位；这是新测试的断言错误，没有改动或中断封存的真实运行，也不是历史排序根因。
包含pipeline和残差新测试的广覆盖已重新运行，857项通过；另12项精确margin
误差分解测试通过。随后将上述全部新测试纳入整体重跑，最新广覆盖869项通过
（22.53s）；这些仍是代码/证据合同测试，不替代主机CUDA结果。
将双neutral参照合同测试纳入后，广覆盖874项通过；最终再次重跑为874项通过
（22.64s）。

## 新根独立确认：四阶段全部完成

固定反馈规则不调参，使用新八根202648000..007，episode分别为8596/1920/4900/
213/6159/2029/9356/5486；全部唯一且与此前184个episode隔离。新结果产生前已
写入`results/reacher_d2p_waypoint_confirmation_fresh8_20261026_cuda/pipeline.protocol.json`，
绑定上述根、排除集、官方权重、所有入口SHA和完整四阶段命令。
四个阶段已全部完成且进程exit 0；`pipeline.json`状态为completed，6项完整性门
全部通过。原方案编译136词、执行128条完整物理分支，396330个CUDA predict样本、
14项门通过；反馈比较64条完整分支、384包，5548个CUDA predict样本、11项门
通过。两阶段合计192条完整分支、401878个predict样本，神经计算均使用主机
RTX 5060 Ti CUDA，没有CPU回退、后训练或权重/buffer改写。

八根goal admission均为`joint-feasible-v1`，全批无目标反射。运行期间五个封存
入口的SHA保持不变；外层协议及planning/tracking两份回执SHA已独立核验。
两个图包共四张图已逐图目视，planning的6项及tracking的7项产物SHA全部通过。
通用子分析器保守沿用`known roots / not confirmatory`的scope；不事后改写封存
脚本或回执。此次来源隔离、固定规则先于新结果的证据由外层协议提供，以下CI
仍仅作描述，不冒充独立正式校准、预注册效能检验或等资源机制比较。

### 原方案与因果反馈是两种不同执行合同

原方案会在真实prefix后重新求解suffix。其native/greedy/neutral/CEM平均终态
关节距离分别为0.034584/0.044122/0.046553/0.022075。native对greedy为6胜/
1负/1平，差-0.009538、描述性95%CI[-0.017222,-0.002681]；对CEM为4胜/4负，
CI跨零。七个模型witness中只有六个真实RGB非贪心，202648001仍反转，原P1
不能开放。

反馈对照只使用prefix执行前封存的初始完整计划；其replay复现原方案的`*_carry`，
不是上段重新求解suffix后的结果，不能混用两个分母计算反馈收益。

| 原初始计划 | 完整开环重放均距 | 同初始计划因果跟踪 | 跟踪相对重放胜/负 |
|---|---:|---:|---:|
|native dispatch|0.063314754|0.027490108|8/0|
|同库greedy|0.065490239|0.033205716|7/1|
|neutral prefix计划|0.073363090|0.025031340|8/0|
|完整六包CEM|0.073556396|0.034570227|8/0|

native自身差-0.035824646，描述性95%CI[-0.045331850,-0.027331346]；末包平均
waypoint L2偏差2.739933→1.513370。相对同反馈greedy为5胜/2负/1平，差
-0.005715608、CI[-0.014174040,+0.002968218]，跨零；相对同反馈neutral为4胜/
4负，对同反馈CEM为5胜/3负，CI也都跨零。因此当前证据支持反馈减少初始计划的
执行漂移，不支持已经确立重配置独有优势；反馈与重放计算不等资源。

### 七个新witness保留执行意图，不是同动作排序准确率

独立双参照回执`reference_margins.json`重新验证上游全部因果链，按root identity
连接完整向量。下表第四包margin为candidate成本减固定原neutral重放成本，
正号表示此处所检验的真实非贪心条件；另列同反馈zero比较以避免混淆。

| 根 | 原模型margin | 原计划重放减固定neutral | 跟踪减固定neutral | 跟踪减反馈zero |
|---|---:|---:|---:|---:|
|202648000|-0.030242205|-0.037822723|-0.020128012|-0.033048153|
|202648001|+0.005676985|-0.010045528|+0.027185321|+0.009880066|
|202648002|+0.040612459|+0.011660337|+0.030294895|+0.028342724|
|202648003|+0.015085459|+0.012826681|+0.009526014|+0.006743431|
|202648004|+0.020090103|+0.020635605|+0.000103235|+0.010405779|
|202648005|+0.017183065|+0.006808519|+0.045842409|+0.022449493|
|202648006|+0.009823799|+0.008375883|+0.007240534|+0.000814199|
|202648007|+0.026066542|+0.021336555|+0.034721136|+0.027665615|

001原动作的符号仍反转，但反馈程序的真实执行恢复为正；七个模型witness在反馈
后7/7满足该执行条件。000是原本就无witness的fallback，保留其负号、不从全批
剔除；004正margin仅0.000103235，不将符号正确夸大为鲁棒裕量。

**动作已经改变**，因此不能把7/7写成原模型同动作排序fidelity，也不能拿旧模型
margin和新动作后继混算校准。旧批202647003仍未满足该条件；本批也尚无独立
正式calibration/outcome及等资源优势，P1/P2/P3/P5不因此开放。BN统计污染与目标
限位属于已经找到的代码/协议缺陷；剩余动作响应误差属于待解决模型/执行合同问题，
既不能说“所有代码问题已排除”，也不能在缺少新受控证据时断言必为代码bug或
理论不成立。

## 复现

所有神经入口必须在主机CUDA运行，禁止CPU回退；输出目录须不存在。完整确认
单命令会先核验旧证据，再封存新协议，依次执行四阶段，不隐式跳过失败命令：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_waypoint_confirmation_cuda \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --alignment-reference results/reacher_d2p_state_native_selector_all136_20261026_cuda.json \
  --disjoint-reference results/reacher_d2p_feasible_goal_confirmation_fresh8_20261026_cuda.json \
  --pilot-tracking results/reacher_d2p_waypoint_tracking8_20261026_cuda.json \
  --seed 202648000 \
  --output-dir results/reacher_d2p_waypoint_confirmation_fresh8_20261026_cuda
```

上述为实际已完成的存档命令；该目录及回执已经存在，再执行将拒绝覆盖。
复现应使用新的输出目录，不能覆盖证据或把相同根重放称为再次独立确认。
四阶段之后的独立双参照审计命令如下，同样要求输出文件不存在：

```bash
.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_waypoint_reference_margins \
  --report results/reacher_d2p_waypoint_confirmation_fresh8_20261026_cuda/tracking.json \
  --output results/reacher_d2p_waypoint_confirmation_fresh8_20261026_cuda/reference_margins.json
```

最终874项代码/证据合同回归命令（不含CPU神经效能实验）：

```bash
.venv/bin/python -m pytest -q \
  tests/test_reacher_d2p*.py tests/test_audit_reacher_d2p*.py \
  tests/test_analyze_reacher_d2p*.py tests/test_train_reacher_d2p*.py \
  tests/test_run_reacher_d2p_causal_dynamics_repair_cuda.py \
  tests/test_bounded_response_qp.py tests/test_d2p_stage_admission.py
```

原提案C1--C3、三组实验及完整P阶段仍按
[闭环审计](D2P_RESEARCH_CLOSURE_AUDIT_20261025.md)执行；不把本次诊断或新根pilot
替代合法任务合同下独立校准/outcome、可控能力改变机制、强规划器等资源对照、
动态horizon/预算或跨模型复制。
