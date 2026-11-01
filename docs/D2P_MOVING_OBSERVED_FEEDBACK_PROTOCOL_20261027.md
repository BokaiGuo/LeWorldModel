# Moving：固定旧任务的step5真实观测反馈协议

状态：设计交付，尚未实现本协议的live driver，未prepare、未运行模型或物理。MAIN独占源码审查、准入、实际HOST CUDA/物理与保存数组独立复算；不设置专用审计agent。本文不改变P1/P4门槛，`P1_admission=false`、`P4_admission=false`、`independent_confirmation=false`。只新建本文，不改任何sealed文件或共享旧文档。

## 1. 唯一问题与不可变项

在相同旧moving任务、相同任务goal、相同前5步、相同总10步控制窗口下，step5用**本次执行自身新RGB**更新context，再重编译后5步，是否改变模型预测、真实latent Q或原生任务结果？答案允许变好、持平、变差或运行失败。不能预设反馈能修复模型，也不能把训练数据factual保真、构造witness可达或teacher强制结果当成闭环收益。

固定source_slot `[0,1]`，不替根、不按原/新outcome过滤；原两种prefix来源为 `line_search_GN`、`full10_conditional_D2P`。按root-major固定四个配对层：`(0,GN),(0,D2P),(1,GN),(1,D2P)`。每层的原full10 plan与前5 raw命令在新实验开始前封存。后续gradient/GRASP三臂物理结果不参与选臂/选prefix；二者不是本文主实验新增臂。

固定以下来源身份（均为Python根`/home/gbk/code/LeWorldModel/le-wm-main`下路径）：

|来源|路径|外部SHA256|
|---|---|---|
|moving source|`results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json`|`c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`|
|原编码|`results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json`|`e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`|
|原moving planner|`results/moving_history_planners_cuda_20261027_v1_cuda_main01/report.json`|`cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0`|
|原moving physical（仅历史评价参考）|`results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json`|`a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf`|

planner实际路径须MAIN从原own-consume链确认与上述SHA一致；路径不存在/身份不符则prepare拒绝，不能搜索另一个“近似”report顶替。task00/01必须通过moving encoder的自有`load_task`，不能调用旧zero-only loader或改全局接口。两task SHA分别为`a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`、`101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。

原192模型、transform、encoder/projector、goal latent、identity Q、public normalizer、packet logit球全部不变；训练normalizer provenance仍unknown。不训练、不拟合Q/readout，不换阈值。输入render固定原`with_target=False`；不混入并行paired-overlay的新显示合同。

H3是三帧历史context，不是未来15步。所有主分支从原控制root起最多执行10个public step：前5固定，后5固定或重规划。原source实际warmup10另外付费，不能藏进“只有10步”的叙述。

## 2. 主实验：三种模式、同prefix、同10步

每个固定配对层独立从原recipe启动三次，顺序固定`O→I→R`，共12个执行分支。不把12当12个独立任务，独立任务数仍只有2；GN/D2P是同根不同prefix，模式是配对干预。

|模式|step1–5|step5用于suffix compiler的context|step6–10|合法含义|
|---|---|---|---|---|
|O：open-H2 continue|原sealed prefix5|不重编译；真实step5仅guard/评价|原sealed suffix5|原开放H2基线，本轮新执行对照|
|I：imagined-context recompile|同一个原prefix5|原plan封存的**模型预测**prefix latent；动作history必须为本次实际执行值|新固定H1 ridge suffix5|matched计算对照，不是observed feedback|
|R：fresh-observed recompile|同一个原prefix5|本次step5新RGB编码得到的actual prefix latent；同一个真实动作history|同算法、同预算的新H1 ridge suffix5|仅此模式构成本文的一次观测反馈决策|

三者均在自己的环境连续执行，不reset/restore到保存prefix；前5步之间也不调用旧未来帧。旧physical suffix10/旧prefix5图不得作为R部署输入，旧terminal更不得当新目标。旧历史结果只能在所有新命令封存后进入评价域。

matched配对还要求三模式至step5的实际raw、goal身份及fresh RGB像素SHA一致；隔离evaluator另核状态/snapshot与原合法prefix轨迹一致，不能把这些状态给compiler。若prefix重现不一致，该层的配对前提失败，仍保留三模式固定分母与已发生账，不重置到保存state、不换prefix、不事后放宽容差。PNG像素相同与encoder批次浮点roundoff是不同检查：后者只能用预先冻结的数值一致性规则诊断，不能倒推“观测精确同值”或容忍不同PNG。

I采用原selected plan已封存、与自己前5 literal raw动作对应的想象prefix，不另取别的bank候选；取不到严格绑定的预测context即拒绝该协议prepare，不能以真实prefix或zero latent补齐。I与R都新编码各自step5作为guard/评价，但I compiler明确看不到该actual latent。O也执行相同step5编码以便评价；因此I/R的新增编码成本一致，O的控制算法成本仍更低，不称O与I/R全pipeline等算力。

主因果比较是同层`R−I`：排除“多做一次suffix solve”本身；`R−O`是整个新增观测+重编译流程的改变，不能单独归因于观测。即便前5路径逐位相同，三次执行仍分别保留完整receipt与编码，不去重样本成本。

## 3. Context、控制权限及真实时间顺序

令原task历史为`[z0,z5,z10]`，实际source raw warmup为`W[10,2]`（前5 `[.1,0]`、后5 `[0,.1]`的float32）；原sealed prefix为`P[5,2]`。计划相对step5即source绝对step15，不得混淆时间编号。

重规划context为：

`latent_history = [z5,z10,z_prefix]`

`raw_history = concat(W[5:10], actually_executed_P)`

`action_history = public_float32_normalize(raw_history).reshape(2,10)`

R的`z_prefix`必须来自本次fresh RGB；I来自原同prefix模型预测。两者历史动作都是真实已执行raw，不能把想象动作/zero padding当实际history。每次suffix forward实际输入3个latent、两历史action tokens及当前suffix token，time-major 5×2；不再除100、不clip/tanh历史、不重拟合normalizer。

严格事件序列：

1. 新prepare冻结四层、三模式、原任务/plan/goal/model/source SHA、固定solver与预算；不能预先产生R的新suffix。
2. 每执行分支fresh env，执行原reset/placement与真实warmup10，guard每个新RGB，核原history0/5/10、原raw与来源身份；同goal绑定仅由evaluator完成，不改变poststate。
3. 先封存原prefix5命令，执行step1–5并逐步fresh RGB guard。step5先durable保存PNG文件/像素SHA、actual raw与无缺口history，再编码、保存原float32 native192 NPY及model/source绑定。
4. R仅在`prefix_observed_and_encoded`完成后可compiler请求；I用独立的imagined输入包，O不进入compiler。所有suffix候选及新动作的计算均记attempt/return。
5. 新suffix选择/动作seal必须durable完成，写入当前episode_id、step5 observation/context SHA及剩余horizon=5；收到该seal后才能dispatch step6。R真实时间因果必须满足`observe5 < encode5 < compile < seal < step6`。
6. 连续执行后5步，逐步guard；step10只编码/评价，不再重规划。step5→6之间绝不reset、set_state、替换goal、借旧endpoint或加步数。

planner输入只含允许的RGB/latent历史、实际raw history、原任务goal/Q和source/model身份。hidden state、pose、native success、witness/证书、未来RGB、物理terminal与其他模式结果只在隔离evaluator域；不得传入部署selector。状态/snapshot可用于防作弊guard与评价，但只能拒绝，不能据此更换动作/根/solver。sourcepack不因“已持有”而对compiler开放witness字段。

## 4. 最小suffix机制：匹配工作量，不冒称再跑完整GN/D2P

参考已封存`moving_history_observed_suffix_cuda_20261027_v1`的H1 bounded-ridge核与五阶段账；本文**不是直接承认其saved-prefix结果等于live部署**。新runtime须独立接口/新schema，不能修改旧sealed publisher/physical core以绕过时间因果。

每个I/R分支固定执行：

1. 当前context下零raw nominal：1 predict/action_encoder/pred_proj。
2. 同context零logit点新J：1 AD-forward predict/action_encoder/pred_proj；J为192×10，不能复用原root J或别的模式J。
3. 原`bounded_ridge(Q@J,Q@(goal−nominal),damping=.002,cap=.75√5)`，1 SVD；无额外调参、line search或训练。
4. 原suffix同context非线性评价：1 forward，仅诊断，不作为动态fallback。
5. 新ridge suffix非线性评价：1 forward。
6. 同context新suffix独立H1 replay：1 forward；全context须匹配，非仅最后latent近似。

这共5次forward，因为第2项AD-forward本来就包含在账内。候选表固定含zero、old suffix、new ridge三个；zero复用已付的nominal，不加隐含forward。新ridge raw在其评分前封存。**固定派发new ridge**，不以旧/新模型分数或真实反馈挑候选，不因等于旧动作而补选。I/R遵循同一派发规则与固定分母，名称应体现`H1_ridge_after_{GN,D2P}_prefix`，不能把它称为重新执行了原完整full-H2 GN或完整D2P bank搜索。

如果MAIN未来要测试“step5完整迭代suffix optimizer/bank重编译”，须另起协议和预算，不能在本文已有5-forward账内偷偷增加候选/AD或替换算法。

## 5. 全量独立预算

主实验不复用新RGB编码，也不把旧源获取成本记成零。新编码均batch1；原task已有root历史和goal latent不重编码。所有模式每次执行分别在step5、step10编码：2 encoder+2 existing projector。

|新增工作|每个O执行|每个I/R执行|主实验12执行合计|
|---|---:|---:|---:|
|encoder / image projector各|2|2|24|
|predict API / predictor / action_encoder / pred_proj各|0|5|40|
|其中AD-forward（已含在predict）|0|1|8|
|192×10 Jacobian|0|1|8|
|SVD|0|1|8|
|functional.jacobian API|0|1|8|
|内部batched autograd.grad API|0|1|8|
|数学VJP directions（非API次数）|0|192|1536|
|控制public steps（最多）|10|10|120|
|真实warmup public steps|10|10|120|

故public-step总cap为240，不能只报120；prefix实际控制60步、suffix实际控制60步。predict/predictor是同一次调用的API/子模块两种receipt，不相加为80；J API与其内部grad是同8次Jacobian构造，不能漏AD-forward或假称16个独立J/1536次API。单次forward有3context tokens，主实验predict/action各120个token位置。新编译器必须逐个计量所有候选、AD-forward、replay和异常attempt；失败不退款，不retry。

旧GN/D2P计划来源成本另列：每root GN78 predict、D2P66 predict及各自AD/math；两root共288 predict（以及原paid encoder8/projector8、source构造/旧物理等），不是新预算内的免费真实计算。物理三模式复用封存plan时只记一次已发生来源成本，并同时报告单次部署等价成本归属，不能通过摊销声称本实验同D2P总算力。I复用旧想象prefix必须明确cache reuse，不装成新增forward；R不借旧observed prefix。

新live driver工程cap建议预先写入prepare：每个独占CUDA/物理job内部wall600s，allocated3GiB/reserved4GiB；全主实验新增输出≤64MiB（含PNG/NPY/partial/journal/report），prepare≤256KiB且Q以原task引用而非12份复制；每次写保留200MiB，普通写另留256KiB给final。不得因负结果、超时或磁盘不足减候选/步数或提高cap补跑。拆job时必须列出固定job数量、各自cap与全局累计账，不把600s误说整个分阶段实验总耗时。

物理账必须包含env创建/constructor隐式reset、显式reset/placement、set_state、goal render/bind、每public step内部physics、fresh render；不能只hook后半段。设计上固定12次env创建，显式reset recipe两次/执行；建议保守全链工程cap：reset36、set_state72、goal_render36、goal_bind12、physics_step2688、explicit_render288，全部attempt/return独立计量。每执行正常计划保存warmup0..10及plan0..10共22个fresh PNG，主实验正常264张；多余render也需在cap内计账。这些是**上限，不是实测数**，也不是按错误的固定“每public step只有一次physics”推算。MAIN实现审读时必须核constructor/reset原生路径能否被完整计量；不能覆盖入口的计量缺口应在prepare前补接口，不能伪报cap合规。实际早停只减少completed，不抹掉attempt或分母。

## 6. Teacher-only与已有saved-prefix证据的隔离

teacher-only指把已执行轨迹的真实prefix latent作为模型context，预测**该轨迹实际执行**的suffix successor，只计算factual误差，不发布新控制命令。它可以诊断保真，不能据此说已经部署反馈，不能把替换latent后预测变好说成控制收益。

已有saved-prefix suffix compiler即便生成新动作，也只是已完成prefix上的离线重编译；若后续新物理严格从root重放、精确核prefix后执行新suffix，可称known-prefix条件控制重放证据，不能省略其saved-source假设，不能等同本文`observe本次→compile本次→step6`的live因果链。

默认主实验不额外启动teacher神经任务。若MAIN另行预注册teacher诊断，固定全部12个新执行episode，待全部控制seal/执行完成后，每个episode用自己的actual step5 latent与自己的actual suffix5做一次H1 factual预测、标签为自己的actual terminal10；新增12 predict/action_encoder/pred_proj、0AD、0新编码（复用已计24个编码），独立report，不改变主实验控制或分母。不能用O的旧suffix预测对照R的新terminal冒充同动作factual标签。zero/mask、不同goal或未来teacher token不属于本诊断预算，需另声明；不把真实未来帧喂给selector。

## 7. 最小新接口与durable握手

建议driver只提供以下四个窄接口，具体实现stem须MAIN另分配；本文不生成可执行producer或占位结果：

|接口|允许输入/输出|拒绝条件|
|---|---|---|
|`load_moving_task_and_prefix`|外部SHA task/plan，parent_task_id、slot/arm、prefix5/原suffix5、旧imagined prefix、原goal/Q/modelbinding|zero schema、task/plan不一致、root/goal换绑|
|`capture_encode_own_step5`|本episode新PNG、文件/像素SHA、实际raw ledger、encoder receipt→float32 native192及array-entry SHA|旧episode图、错时间点、无step5 public return、非CUDA或model漂移|
|`compile_remaining5(mode, context_packet)`|只含允许context/原goal/Q/actual history/身份；返回固定五阶段完整receipt、new raw5、selection seal|witness/state/terminal/outcome入包，旧J，候选或预算缺口|
|`admit_and_execute_suffix5`|同episode observation seal、compiler own-consume seal、精确5条raw→step6..10连续执行receipt|跨episode、原goal/history改变、step6早于seal、reset/restore或超步数|

context_packet至少绑定：`episode_id/source_slot/prefix_arm/mode/parent_task_id/task_export_sha256/parent_selection_sha256/actual_prefix_commands_sha256/raw_history_sha256/latent_context_sha256/goal_Q_modelbinding_sha256/remaining_public_steps=5`。R再强绑定新PNG/file/pixel/latent NPY/array-entry SHA及encoder source SHA；I绑定原imagined-prefix来源和新actual编码仅评价标记。model请求白名单不包含物理state、原生reward/success、witness、未来标签或其他episode结果。

所有source/version/prepare/model/numerics、请求输入数组和返回原float32预测、J/solve原返回精度、选择动作、step6前seal、逐步fresh RGB guard均durable保存。异常保留first_error与secondary、固定12rows、各phase not_started/failed_partial及已发生资源；不补假capability、假second batch或成功状态。若真实native success/terminated/truncated在10前发生，按原环境停止，不强行补10步；保留状态并按实际长度与固定分母报告，未执行suffix/编码不能假填完整结果。guard/计算失败也不得悄悄回退旧suffix并计R成功。

## 8. 报告与解释门

按两个root、两prefix来源、三模式完整列出：原生success与原官方wrapped误差、物体/agent分量及独立wrapped-angle诊断、实际物体位移/旋转、同goal的真实Q5/Q10、剩余suffix实际raw、原/新suffix是否相同、模型预测Q及对应实际suffix的factual误差、全部资源及错误。native success保持原环境定义，额外wrapped诊断不暗改其判据；hiddenstate仅在隔离evaluator算标签。

主配对表至少给同层`R−I`、`R−O`与`I−O`，包含所有反例和缺测原因。不用latent Q改善替代native成功，不把两root的多臂/多次执行当独立统计确认；不按结果加seed、换root、挑模式或调阈值。不同suffix导致的terminal不能当成同一factual反事实标签；只在同层实际结果上讨论策略干预差异。

即使R更好，也只支持这些旧任务/固定prefix上的一次observed反馈校准；不能推出非贪心、独立能力改变、完整P1机制通过或“反馈普遍修复”。即使R失败，也不单凭旧稀疏响应或teacher误差判物理不可控。独立确认、强基线与原P1所有前提仍另行保留。

## 9. 可执行前置与本次交付

MAIN后续最小实现顺序：纯RAM验证原task/goal/history→mock完整O/I/R时间因果和预算→冻结新source/prepare及own consumer→MAIN真实prefix观察/编码/重编译/连续后5→MAIN独立保存数组/账目复算。mock必须覆盖R偷用旧PNG、I偷看actual latent、跨episode seal、漏AD-forward、旧J复用、step6抢跑、早停/超cap/close异常；不是新增专用审计任务。

本次只交付设计文档，不授权或执行上述任一步真实工作。experiment-design技能用于先锁控制变量、matched比较与固定分母；其通用训练/调参/多数据集模板不适用于本用户约束，未据此扩展实验。个人goal的complete仅代表本文设计交付，不表示协议已部署、实验已完成或P1完成。
