# P2 midpoint older2 来源与同Q/native目标分栏（saved-only）

## 独占写集与问题

仅新建 `agent_p2_midpoint_history_lineage_20261028_v1` source/tests/本 innerdoc。旧 midpoint、suffix、gap、MAIN 文档与封存结果不修改。worker 不运行模型（CPU/GPU）、physics、网络或子智能体；没有训练、拟合、阈值更改、计划更改、按结果筛根。本分析重用 MAIN strict 与独立 suffix 数值审计，不重做其8NPY误差审计。

问题不是“再修正两项更早 imagined latent”，而是先证明它们究竟从哪里来；再分开原预测→actual terminal teacher Q 的误差与 actual teacher Q→native任务目标的差异。固定原4 known roots × joint/composition 共8项，exact ties、零、负结果、未知和工程失败不删。

## 输入权威与闭包

固定引用 MAIN suffix：

- CUDA report `results/agent_p2_midpoint_teacher_suffix_20261028_v1_cuda_MAIN01/report.json`，SHA `08bec783a5791c15a03396131df4693e3d4c63c43c9523c38f3e499d65e9fe66`。
- strict `results/agent_p2_midpoint_teacher_suffix_20261028_v1_consume_MAIN01.json`，SHA `36864d57aa6914f0d09ff4e947ef141ccf0037a2fc23f9d7e3f414fb217d04ac`。
- MAIN 不导入producer的独立算术审计 `results/audit_p2_suffix_MAIN_20261028_v1_MAIN01.json`，SHA `90c79b18ecf23f1cb9bb9298e6147ecb4a3e393d8f92c51c7b7a13396af6366d`。这里只认证并复用，不重复其误差复算。

旧 suffix 三SHA仍严格绑定 `ed4954…eae7 / 855e8c…8d63f / b8850b…ace5d`（完整值在新源码 FROZEN）。`suffix.verify` 只读取旧 prepare/source/input 闭包并重建既有输入，不调用 run；它含上游 midpoint 的 saved-only 检查，不是运行新神经或重复 suffix 独立算术审计。

对本任务额外读取的原 compiler `source_seal/manifest.json` 按 compilation report 中已封存 SHA认证，再按manifest认证其 args 与每份 archive 的size/SHA。**追踪 archive 源码，不导入旧 compiler 或其 torch/gym 依赖，也不拿当前 dirty 源码替代历史源。** 原 public LeWM encode/predict/rollout 源码从 suffix 闭包认证后用AST读取，不导入。所有追加输入在读取结束再验SHA。NEW输出只 exclusive-create，不覆盖原结果。

## older2 的精确来源

封存路径是：

```text
compiler run
  -> physical._reset_and_warmup(history_len=3)
  -> p0._collect_root_anchored_rgb_context
  -> _collect_causal_rgb_context + 实际env.render
  -> p1._encode_context -> model.encode
  -> compiled.input_context.emb
  -> 原root predictor_context
  -> selected paid segmented midpoint context
  -> teacher context（只替换最后一项）
  -> 实际suffix CUDA predict.input_0.npy
```

封存 collector 连续执行三个真实zero-command packet，每包5个独立step。BoundaryHistory保留F0/F1/F2/F3；causal helper先返回F0/F1/F2，root-anchored helper再移位加当前render，得到 **F1/F2/F3，即warmup5/10/15**。它不是先做imagined bootstrap，也不是规划起点0/5/10三张图。

|midpoint slot|原初始slot|原warmup step|相对规划起点|来源|
|---|---:|---:|---:|---|
|older0|1|10|−5|真实RGB encode所得、原值保留|
|older1|2|15|0|真实RGB encode所得、原值保留|
|last|新项|prefix后|+5|paid预测；新suffix任务仅用actual z5替换此项|

本分析逐项核对：`compiled.input_context.emb[0] == initial.latent_history`；`paid_mid.latent_history[:2] == initial.latent_history[1:]`；teacher older2与其完全相等；paid native buffer和实际 CUDA predict input逐数组精确相等（原float32）。每项输出向量digest、初始/paid/teacher context digest、真实input NPY receipt、原history stack RGB SHA和goal RGB SHA。

动作也核完整：原raw history确有3个zero command（不是缺失补0），按原mean/std/time-major5×2得到原3tokens；初始between history为A1/A2，midpoint带A2/A3，suffix outgoing为A2/A3/A4。raw prefix5形成A3、sealed suffix5形成A4；不额外drop/shift，不把未执行后缀当真实状态。

**证据强度限制：** 原 compiler source seal没有递归包含packet helper；本任务的packet helper来自之后已认证的物理prepare闭包，而不是追称为原 compiler transitive seal。原compiler留下 encoded context JSON和后续physical的history stack SHA，但本消费闭包没有原始逐帧pixel→encoder input/output原生receipt。因此“历史代码路径为observed encode”和“数值传递完全相同”有直接证据；“每帧原生编码执行已独立证明”“表征完全正确”没有。相关字段为null/unknown，不造receipt，也不把封存代码当执行证明。

结论边界：**older2不是两项预测rollout状态；不能把仍有终态误差解释成‘那两项尚未替换的imagined历史误差’。** observed origin不排除编码表征失真、batch/transform效应、公共API或动作时序协议问题、suffix动力学预测误差。

## 同Q与native目标：保存分栏，不造统一分数

每臂输出：原open-loop predicted Q、actual-z5 suffix predicted Q、同原Q/goal的actual terminal teacher Q；另列原native7/jointXY/objectXY/agentXY/velocity/raw angle/wrapped诊断和原官方success。原goal vector、Q projection内容和metric ID逐项与原compiled及physical active metric精确匹配。没有新Q、goal或动作。

四根分别比较：

1. original/suffix预测Q与actual teacher Q排序是否一致；
2. actual teacher Q与native7/jointXY/objectXY排序是否一致；
3. native7²的agentXY²、objectXY²、raw angle²、agent velocity²贡献（纯代数非因果）；
4. 原官方success保留布尔，不将距离偏好改称成功。

各差值仅在同单位指标内做C−J，不拿latent Q减native距离。Q为原冻结投影latent MSE；native7是未归一的混单位7维范数；jointXY包含agent+object位置。object-only不是完整原任务。wrapped angle仅诊断，原native成功谓词不改。

本轮只读实际saved运行显示以下描述性关系（J=joint，C=composition）：

|root|actual terminal Q偏好|native7偏好|jointXY偏好|suffix预测是否match同Q|
|---|---|---|---|---|
|0|J|C|C|F|
|1|J|C|C|T|
|2|J|J|C|F|
|3|C|J|J|T|

因此本4known根中，同Q与jointXY排序均不同；原strict的全8MSE下降和新2/4同Q排序match，并不意味着native任务改善。全部8官方success仍false；不改变旧0/4记录，也不据此推断理论不可能、泛化或P1/P2通过。以上不是再做MAIN的NPY误差审计，而是将其认证成本与旧native分栏连接。

## 下一项固定协议：明确代码/协议与预测误差能区分到哪一层

`next_protocol(jobs)`只读取8项输入身份，不读取分数、MSE、排序、success或真实outcome；RAM测试会改变这些额外字段，要求输出协议完全不变。以下是给MAIN审读的事前候选协议，**没有在本交付中实现/启动新的CUDA producer**，不是已有结果或新增门槛。

最小A：独立minimal native adapter以相同saved teacher full3/3outgoing tokens，原模型/normalizer/Q/精度，对固定全8项各执行1次AE+predict（内含predictor/pred_proj）。新增每模块8次，0encoder/projector/AD/physics；原paid信号/搜索/编码/物理费用仍继承另列。完整保存每层input/output，逐项比较旧原生数组与新独立路径，保存首个分歧，不按误差选子集。

- 精确不一致只定位到所测路径的实现/runtime/precision差异，不立即称科学反例。
- 完全相等只排除这条被测执行路径分歧，**不独立证明共享history/action协议正确，不证明剩余误差纯由动力学引起**；重复性检查没有预测收益结论。若MAIN已有覆盖完全相同原生路径的独立执行证据，应复用，不重复付费。

协议B：原public `LeWM.rollout` cached-emb分支与manual native recurrence，在同真实连续observed history `z[-5,0,5]`上比较。public输入cached emb `[1,1,3,192]`、真实pixels `[1,1,3,3,224,224]`、outgoing action_sequence `[1,1,3,10]`、history_size=3；代码中`T-H=0`但loop为`n_steps+1`，所以**恰好1个真实suffix packet预测**，不是0步，也不再拼一个future token导致两次预测。

固定每项public/manual各1次，分别8个predict和8个AE，共16predict/16AE/16predictor/16pred_proj，不padding。cached真observed latent闭包可用时0encoder；若要独立核原逐帧encode而无receipt，完整三帧复核是24encoder samples+24projector samples，不能写成0，也不能把重编码当新预测收益。

**B当前确切未决依赖：** consumed closure中没有得到同history stack下warmup10、warmup15/规划root、actual step5的完整逐帧原生encode receipt与真实RGB triplet闭包。不能拿root0图复制填充旧时刻、用终态替代、凭几何近似、或无声rerender。MAIN先核是否别处已有精确资产；没有则需另行授权完整计量的重建，新增physics数目前unknown，不填0。重建身份不匹配是协议/重建失败，不能当负动力学结果。

若public/manual输入与输出闭合仍对真实terminal有误差，只能描述在该观测表征与协议下的模型/表征条件误差；actual teacher Q对native目标不一致仍单独存在。不能通过调整Q、成功阈值、只测roots0/2或换动作“改善结论”。新根/freshness、独立P1机制证据仍缺，科学准入全部false。

## 执行与回归

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。MAIN只读分析并保存NEW输出：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_midpoint_history_lineage_20261028_v1 --output results/agent_p2_midpoint_history_lineage_20261028_v1_MAIN01.json
```

省略`--output`则仅stdout；无prepare/run/GPU隐式入口。输出为新的saved分析，不是新的物理/模型实验报告；worker仅运行stdout读取和RAM/saved tests，没有写MAIN结果。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_midpoint_history_lineage_20261028_v1.py tests/test_agent_p2_midpoint_teacher_suffix_20261028_v1.py tests/test_agent_p2_midpoint_teacher_20261028_v1.py tests/test_agent_p2_native_composition_gap_20261028_v1.py
```

覆盖：older槽/compiled/root/RGB/action/normalizer/paid digest/native buffer/dtype/null/bool变异拒绝；原goal/Q/physical metric替换拒绝；完整4×2分母和join；同Q对齐但native不齐的RAM反例；object-only不替代full task；unknown不补0；next协议不读outcome；源AST读取不执行；冷子进程精准拦截torch/gym/physics导入；exclusive输出不覆盖。

首轮真实测试为34pass/1fail/1error：新分析器误按旧plans列表位置连接，实际旧plans的root1/3两臂顺序反向。已按`(root_slot, arm)`身份连接修复，仅改新source；修正后36tests通过5.60s。随后增加goal/Q与public ABI回归，最终联合终态与三SHA见交付回复，保留首轮失败，不抹除、不改旧资产。

完成标准是新三文件、RAM/saved回归、命令、SHA后冻结，不是研究goal完成。冻结后本v1不再追加；后续只能新版本。
