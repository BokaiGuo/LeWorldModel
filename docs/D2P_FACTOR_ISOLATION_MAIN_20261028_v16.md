# D2P：分层排因与可执行修改（2026-10-28）

本轮主线程优先研究，不以增加诊断文件数量当作研究闭环。以下是当前有证据的因素清单，不声称穷尽所有未知因素；“本批排除”不等于其它模型/环境永久排除。代码/协议故障、预测误差、表示—任务失配和科学前置分别记录。

**当前结论（以下各节含保留的历史快照）**：保护性v2已完成主机CUDA→真实
full10物理→终态RGB编码→完整strict→MAIN独立NPY复算。8/8模型与真实编码Q
均改善，jointXY 7改善/1变差，成功仍0/8（仅2 known roots）。已修复优化参照
和无下降保护的实现缺口；日志信封/阶段goal两项验收代码错误也已在新版本
完整验收。新P1实际80对仍17次预测→编码反转，不能将验收错误冒充全部反转
的根因。raw有限差分480请求和24矩阵独立复算完成，本批没有粗大导数接线
证据；P3完整同反馈20步仍无任务一致收益。详情与完整SHA见页末终态。

## 新实际干预：真实中间 RGB 上重预测同一后缀

固定原 contextual planner 的全部8计划：2个已见根×off/on×joint/conditional。原前缀、后缀、goal192、identity Q、模型、控制10步均不变。新主机 CUDA 编码真实checkpoint0/5/10共24次，并在实际中点history上执行8次原生predict。没有新增优化器、训练或物理；这次干预改变预测条件，尚未改变实际控制。

- prepare SHA：`b1c7f465f427f4da3236c64be245c6da60d118316ff796649632a69385c017c8`。
- CUDA report SHA：`b57fa03560696c3c0b75fe842ec8263d6ec3961341d6736a4f6e782d1f0e92ea`。
- 完整 strict SHA：`3d3d020c113959aec25f4e310b8512b814f7a78976f2bd7d5e58e0e435027f4b`；全量重建上游physical/planner及本次原生模块/数组/事件，不是只核状态。
- MAIN独立原NPY算术复算：`le-wm-main/results/main_p2_feedback_effect_20261028_v1_MAIN01.json`，SHA `1206ba545dfd11e64ca0bc36d54c308e9c099905c25dc73ab9f79eed96688b61`。此derivative不再重建上游闭包，明确依赖刚完成的固定strict。
- 新桥47测试及MAIN独立复算6测试联合53项通过；errors=[]，冻结模型before/after相同，elapsed63.985秒（包含前置复验）。

| 固定检验 | 实际结果 | 能与不能说明什么 |
|---|---|---|
| checkpoint0重编码对原root latent | 8/8 MSE精确为0 | 支持本批原始图像/预处理/加载一致；不证明后继预测正确 |
| actual-mid同suffix终态预测MSE | 7/8下降，index5略上升 | 前缀条件误差有实际影响，不是全部差异的解释 |
| 对实际terminal Q的绝对误差 | 6/8下降 | 预测MSE与目标排序并非同一检验 |
| 四槽原预测→实际Q排序 | 3/4同向→反馈4/4同向 | 本固定小校准面板修复1处排序；非新根泛化或P1通过 |
| 实际Q→jointXY / objectXY / angle排序 | 4/4、3/4、2/4同向 | 完整任务桥仍有缺口，不能用整体XY掩盖object/angle |
| 原生任务成功 | 0/8，未产生新物理执行 | 不能把重新预测称为成功控制 |

四槽composition−joint的原/反馈/实际Q差：

| root/condition | 原预测差 | 实际中点预测差 | 实际编码差 |
|---|---:|---:|---:|
| 0/off | −0.02253415 | +0.36485372 | +0.45935311 |
| 0/on | −0.26121115 | −0.31630200 | −0.25887010 |
| 1/off | +0.02468555 | +0.15072514 | +0.02842507 |
| 1/on | −0.16448977 | −0.31288334 | −0.63581346 |

## 因素清单：逐项证据与尚缺的干预

| 层 | 可能影响因素 | 当前证据/状态 | 修改或下一检验 |
|---|---|---|---|
| 数值执行 | CUDA/CPU、TF32/autocast、模型参数改变、API/cached接线 | 本批真实主机CUDA、MATH SDP、TF32/autocast关闭；原生完整buffer封存；模型未变 | 保留硬守卫，不CPU fallback、不放宽容差；不能把其它模型的1ULP失败并入此批 |
| 导数与局部近似 | detach、动作坐标/归一化链式系数、AD与有限差分不一致、局部方向外推 | 全8锚点、3固定步长、10 raw坐标正负共480次主机CUDA已完成；MAIN独立重算24个192×10矩阵与strict一致；h=.01相对AD误差0.0000371092–0.0000877794 | 本批未发现粗大符号/转置/尺度接线错误；不等于真实动力学正确或全局导数认证；继续区分raw导数与float64 tanh proposal surrogate |
| 视觉输入 | root/goal陈旧、overlay、颜色遮挡、预处理/CLS/patch混淆 | 本批8root重编码精确相同；此前archive/live overlay差异真实存在；P5目标改色被遮挡，不是encoder不变性 | 按各模型原合同分栏；目标mask不可由遮挡图猜造 |
| 动作与时间 | raw/normalized尺度、重复归一化、动作history、native3 vs shifted3、frame15/20混用 | 本次cp raw10与warmup+实际prefix逐位一致；suffix5 bits原封不动；只比native输出最后一行；cp0/5/10明确对应绝对10/15/20 | 在真实反馈驱动中保持相同buffer和时间验证，不让state或物理标签进planner |
| 训练输入溯源 | 公开动作normalizer与实际checkpoint训练统计是否相同 | 当前mean/std的源码及归一化次数已绑定，但moving编码器明确`training_provenance_verified=False`；实际checkpoint config只给结构，不含训练统计。既有全档案动作审计已完成，约3.3%差异不能证明训练normalizer错误 | 保持原合同；不按收益换统计，不把“代码用同一常数”误写成“官方训练溯源已排除”。确证需官方训练版本/统计绑定，不重复全档案诊断 |
| 环境重放 | 起点、goal、RNG、reset、实际执行动作、接触/physics回调 | 原full10 physical严格消费完成；此前特定renderer RNG接线错误按新consumer解释，旧失败保留 | 新控制动作必须重新真实执行并核共同prefix，不用预测改善替代物理证据 |
| 可观测性 | 相同静态RGB隐藏速度/接触/角速度，有限history是否充分，goal单帧不含目标速度 | 实际native snapshot含body velocity/angular velocity；真实3帧/history已核，仍不能由此推出预训练模型能恢复全部隐藏动力学；同状态renderer×goal全80对已验算，匹配后仍4对渲染排序反转 | 需区分完整动力学重放与模型信息充分性；状态只给评估器，不能偷接planner；匹配诊断不替代另一renderer的完整闭环 |
| 前缀误差传播 | 全imagined上下文造成后缀条件偏离 | actual-mid同suffix干预7/8预测MSE改善、排序3/4→4/4，例外保留；实际RGB反馈重编译/物理/终态编码v1及保护性v2均已完成 | 新v2真实编码Q全8改善但XY仍1损害、angle3损害，继续区分条件预测、优化和表示—任务三层；不能将此批改善转授给旧P1反转 |
| 动作条件响应 | shared/reference bias、相对响应与二次项共同作用 | 原P1固定76可比中70有相消；共同/相对/二次项对完整12反转的必要计数为5/4/5，非互斥 | 不能部署pairwise“删项修正”；需可实现的真实反馈/同状态动作对照 |
| 目标与指标 | 单帧看不到goal速度；agent位置主导；物体角度/接触被弱化；native成功与distance合同不同 | 原P1 distance36反转中34涉及速度项；但object/angle22/33反转仍在；新P2angle仍2/4反向 | 各原生轴分开报告，不删除速度改旧gate，不事后按物理收益选Q |
| 选择/优化 | stage+reach拒绝有益暂退、局部SVD不是全局可达、候选不足/局部极小、无条件替换合法incumbent | 原zero局部解3/8比原后缀模型Q差；新incumbent线性化+固定候选下降guard实际CUDA/物理/终态编码/strict均完成，8/8模型与实际Q下降，jointXY 7改善/1变差 | 已修优化参照和无下降保护缺口；两因素尚未独立消融，不将编码下降当任务成功；保持候选/优化预算预先固定，不调margin绕门 |
| 预算/评估 | 只执行H3第一包却判断完整策略、付费信号遗漏、predict相同但总资源不同 | 原full48和第一包结果保留；新P3双方均完整public20、每5步自身RGB反馈，4episodes/16nodes及原strict完成；每臂132predict/1536grad | 已排除该新批的首包/反馈不等混杂，但object/angle仍退化、成功0/4；predict额度相同不称等FLOPs，原consumer额外加固边界仍保留 |
| 泛化/统计 | 两known根/off-on重复当独立根、按结果选根、来源遗漏 | 本批2known roots，非8独立；4reserved根缺历史authority | 不打开保留根、不自签fresh；现有根仅用于校准和代码/机制归因 |

## P1三项误差分解不是单根因证明

正式固定80对结构诊断已analyze并独立verify：
`le-wm-main/results/agent_p1_t3_structure_diagnostic_20261028_v1_MAIN01.json`，SHA `03eb182af71d4cfd4acfe3ee25ef1ad8e253ce3cd80e82f37a2eeba2abcf888e`。80对保留76可比/4未知，8种代数组件组合全部列出，不选最好mask。MAIN联合原分解与P3测试88项通过。

共同参考是每对固定`e_j`，不是已估计的跨动作统一bias；70对有相消也不能证明任何一项没有作用。代数项删除不一定对应一组全局一致、可部署的endpoint校正。因此下一步是实际反馈后重编译/执行，而不是将观测误差回填模型评分。

**不可跨批偷换归因**：上述P1固定40槽的teacher context本来就在绝对第15步，原分解明确绑定实际teacher输入与对应动作history；P2的新8计划原来从自己的imagined中点编译。P2实际中点干预的改善不能被写成“P1的12处反转已经由前缀误差解释或修复”。P1在真实teacher条件下仍有反转，需要继续核完整history来源、动作条件响应、隐藏动态状态及表示/任务桥；两批证据各自保留。

## P3同预算第一包：完成核账，不替代完整策略实验

MAIN完整重建已执行48槽闭包后写出：
`le-wm-main/results/main_p3_paired_quota_outcomes_20261028_v1_MAIN01.json`，SHA `1b3818a20276d84672d9b309750ea95161ae8661d1f863d49ce64d7dc7975d2b`。

24child对中18对有双方物理结果，17对同实际5步端点；adaptive jointXY 2好/15坏。fixed/adaptive成功行2/1包含重复轨迹，不作独立成功率或显著性。所有拒绝/未到达/早停保留。原H1/H3仅执行首5步，不足以否定或证明完整闭环C3；新完整public20驱动正在独占实现。

## 同时间全patch任务桥

新`agent_p5_taskmetric_patch_bridge_20261028_v1`完整消费frame20 v4，固定四词×256patch×384channel、12有向对；正式saved analyze/独立verify通过，SHA `f960d9f3b537a62cb98c3bd70c45380bdc4e9c8717e0c70edbb852120073b8ad`，MAIN29测试通过。保持目标mask拒绝、各区域aggregate=null；global-attention token坐标不能冒作局部因果区域，不据coverage挑patch或拟合Q。零动作位置正确但角度差π/4，仍被uniform feature代价偏好，提示应继续检验任务桥，不是换指标即可宣布解决。

## 正在实施的修正与完成标准

### 新物理干预已完成，不再仅是重预测

实际中间观测上重新求解8个suffix已完成HOST CUDA：40 predict、8 Jacobian/8 SVD、1536数学VJP；模型hash前后相同。新完整10步物理报告SHA `cd1accce54cbf6c48195490f8cd588dc7c4d701b1c6bc2f32844ab4d5a351603`，160 public / 1632 physics / 528 PNG，8/8 horizon_completed、errors=[]。全闭包strict已完成，回执SHA `c191b8076469f78fb7e02c793e0b061bcf6706266b1af6e90f3944748db2e11d`。

MAIN另从新旧原报告的final state7、goal7独立计算全部原生轴，逐值与strict比较；每条共同prefix5的checkpoint/state/snapshot/PNG完全相同。独立结果 `le-wm-main/results/main_midfeedback_physical_effect_20261028_v1_MAIN01.json`，SHA `057f1a9c19c89a2dd5e41200289901fdc457fc88d848db89adc26904883299e4`；本derivative依赖上述固定完整strict，不再次重建全部上游。

| 新suffix相对原suffix | 改善 | 变差 | 相同 |
|---|---:|---:|---:|
| joint XY | 6 | 2 | 0 |
| agent XY | 6 | 2 | 0 |
| object XY | 5 | 3 | 0 |
| agent velocity | 5 | 3 | 0 |
| wrapped angle | 5 | 3 | 0 |

新旧native成功均0/8；只是2个已见根的8计划，非8独立样本。不得将本批反馈控制改善用于解释原P1的12处teacher15排序反转。

新8终态RGB桥现已HOST CUDA和完整strict完成：prepare SHA `7f8ae7b12427a8ed61caeda865951028eba52b787709aed38e3b86941fc0a923`，实际report SHA `1c50b9c08ce7b80dfc5764e2f648caa789caa7cb1267735954be1959425aa92a`，strict SHA `79acff0d3729ec4ad4c1c340db93bab81a576a66d3136eb46c2a9e0fb193e15c`。8 encoder/8 projector、新predict/AD/physics=0、errors/blocked=[]、模型未变；342.750秒包含完整上游重建。MAIN独立从原NPY复算所有actual Q、predicted Q和endpoint MSE，与strict逐值相同。

新动作相对旧动作的效应分解：预测Q变化→实际编码Q变化在index0/3反向（其余6同向）；实际编码Q变化→native jointXY变化在index0/3/5反向（其余5同向）。实际编码Q 5改善/3变差，而native jointXY 6改善/2变差。故**优化基准问题、动作条件模型误差和表示—任务桥失配同时存在**；只修防劣选择不能保证后两层消失，不能将2 known roots的校准面板当独立P1确认。

### 新定位：零点局部求解且无下降守卫

`moving_history_observed_suffix_cuda_20261027_v1.py:260–288`实际在零动作求nominal/Jacobian，对该局部线性模型做一次有界ridge，随后无条件dispatch新解；old suffix只有诊断作用。原协议如此，并非回执接线错误，但它**不保证非线性模型目标下降，也不保证优于已有后缀**。本次模型Q相对同actual-mid旧后缀在index0/4/5变差，变化+0.009484/+0.263945/+0.010692；其中0/4的真实jointXY也变差，5的真实jointXY反而改善，三者都保留。

因此已拆出可修改的优化因素：后续新版本应保留合法原后缀incumbent、使用固定模型目标选择/下降保障，同时核原float64→tanh→float32与新float32 chart边界。不按真实收益选择、不原位改封存v1，不将“模型单调不退步”当成真实任务单调或P1保证。新版本正独占实现。

**进一步独立定位**：MAIN从原nominal/Jacobian/logits原NPY展开局部线性、非线性模型及二次余项，原8行全部保留；`le-wm-main/results/main_suffix_linearization_audit_20261028_v1_MAIN01.json`，SHA `44e558fe97f43c694beae6519167c7c43c9d0f6181878536f56a6b2fa58072db`，3项定向测试通过。8/8相对zero的局部正则目标下降，8/8非线性模型目标也下降，但3/8不如已有incumbent。故本批最窄证据是**优化参照过弱且无条件替换原方案**，不是“ridge符号错”或“曲率使它连zero都不如”。float32 logit舍入最大约1.66e-8–2.84e-8，本审计仅量化，不据此排除全部chart语义问题。独立算术依赖固定完整compile strict `f31377e78981fb4a84c04fb4a790dcbe1b3a23e00c7202a7d73bf9311c043094`，不新增神经/物理，不修改科学阈值。

MAIN已实现 `main_suffix_incumbent_guard_20261028_v1.py`：原生预测向量重算原identity Q；同context SHA/同goal、合法原raw5和外部已验证domain/prediction才允许比较；严格更优才换，平局保旧，不接受native labels字段。11项回归通过。guard本身不认证预测来源或动作球合法性，由实际compiler负责；不承诺真实任务单调。

在固定compile/strict SHA上实际调用此guard的全8回顾复算，保旧index0/4/5、其余5用new；逐值Q等于原封存预测。再关联已验收物理分支，jointXY差为0/−8.93039/−54.14455/−23.93581/0/0/−16.71180/−5.88592。此为**实现后历史配对重放**，不是新在线策略执行、预注册新根确认或成功率提升；它避免了本批两处整体XY损害，也拒掉index5的一处真实收益，因此不能当成普适任务守卫。v2实际编译/交接集成继续，旧v1不改。

完整v2编译/strict/原native handoff已交付并MAIN全源码/测试/doc审读：合法原incumbent处求raw Jacobian、固定4步长候选、完整模型Q比较、独立replay、未改善保旧；模型输入不含物理终态/labels。MAIN独立54项联合测试通过（含guard11），实际prepare已提交；未把测试或提交当作新CUDA/物理结果。v2原始source SHA `e459cdc5de507353873346d6f86bc223afed291982904c4c4c6ce9bdfb7eaeab`，旧v1及原结果保留。

### 目标区域假设和真实schema错误

P5旧全196像素interior patch规则得到0个完整patch，失败保留。另行冻结无阈值/不读候选的规则：按goal唯一颜色的988个精确fill像素给全部256patch面积权重。历史四词same-time分析SHA `fa487e9947c5c56ade9c46809f4013597f7cbc6d639ac7db1dd866244aa185b0`：actualQ→coverage一致2/6变3/6，仍3反转；新加权prediction→actual5/6一致。零词仍较低actualQ，不能据此宣称目标度量已修复；不继续按coverage调权、不称独立确认。area/diagnostic合计13测试通过。

P4新桥v1的原recipe真实文件是三字段gzip envelope，直接索引recipe/witness会KeyError，平铺假fixture曾掩盖此问题。新v2改调用原`read_prepared`完整有界解包/semantic SHA/source验证，使用真实envelope fixture；MAIN全源码/测试/doc审读并独立56项通过。此为尚未生产执行的下游代码缺陷，不是历史P1负结果解释。缺历史cert/approval的保留四根仍不打开。

1. P2：真实mid5重编译及配对完整10步物理已完成并完整strict，终态RGB核查与带incumbent下降保障的新优化版本继续；保留全部8计划及失败。
2. P3：同控制20步的固定/自适应策略，每5步实际RGB反馈、原目标、完整资源向量，不再用H3第一包代理完整策略。
3. P1：最小可实施因果对照准备，用已见根校准鉴别条件误差；独立新根、非贪心能力变化、强基线收益仍是科学闭环的必要项。

当前已有实质预测归因进展，没有证明完整P1/C2或研究闭环。不预先保证所有负结果最终都能归结为代码，也不在尚有协议/实现疑点时归因理论。

## 更新：保护性优化已实际执行；P1新面板捕获验收代码错误

保护性v2已完成主机CUDA编译（SHA `363925ff3714c6c9a5fbd2ec6c59e86dbcb63fe685cde1945d292ec1a9196117`），全部8计划模型Q严格优于合法incumbent；56 predict、8 Jacobian、1536数学VJP、无SVD，180.342秒。完整compile consume完成，回执SHA `8e8ae7f7e3b85680044600a6c1fabf028037353c62370c58f97cba297687a7b7`。新物理执行已完成，报告SHA `039e23ce87980e5a66f38cb47bc50c106690d17d2adb7169a4d4bed0adb70956`、errors=[]；其全闭包strict仍在运行，未将运行完成等同科学验收。v2终态RGB适配器MAIN独立36测试通过，实际新编码尚未执行。

P1同实际20步起点、五词延伸到25步的主机CUDA/原生物理实验已结束：40分支、1000 public、10160 internal physics、88 encode、40 predict；实际报告SHA `2e7794d32602ca2b045202099fedb6b99ac5988d6255ee68870032219530084c`、status=complete_unadmitted、errors=[]。原v2 consume实际失败，位置630行 `all trajectory frames and terminal triplet join global journal`，未写成功回执。MAIN检查发现journal fresh_observation含`index/previous_sha256/event_sha256`，内存terminal_triplet不含这三项，因此直接canonical全字典比较不能相等。新consumer须先认证完整链/事件种类，再仅剥离已验证的日志封装比较全部原生payload；不得删除状态、RGB、动作、时间或其它未知字段来凑通过。封存v2及失败保留，当前不从尚未严格验收的新面板宣称任何P1排序结论。

新增MAIN独立原生状态复算器 `main_safeguarded_physical_effect_20261028_v1.py`，7测试通过；固定全部8行、原/新报告SHA、共同prefix5、全部原生轴和实际选中候选，等待完整strict回执。另行准备固定三步长中央差分审计，复核raw Jacobian全10列/192输出，不能用AD forward一致性替代导数验证。

### v2保护性优化的实际原生效果已验收

上述物理全闭包strict现已完成，回执SHA `7e71cf504691cb1b132c437267319f3291a62b0f6c9f839f0652b42a974f8de8`。MAIN独立从原新state7/goal7重算，与严格回执逐项相同；独立结果SHA `5bbb09ac688e4592aecbd07aaed3d4f88dd04d95c4551c5479b325cbb904de6c`，路径 `le-wm-main/results/main_safeguarded_physical_effect_20261028_v1_MAIN01.json`。

| 新v2相对原后缀 | 改善 | 变差 | 相同 |
|---|---:|---:|---:|
| joint XY | 7 | 1 | 0 |
| agent XY | 7 | 1 | 0 |
| object XY | 6 | 1 | 1 |
| agent velocity | 8 | 0 | 0 |
| wrapped angle | 4 | 3 | 1 |

新旧成功仍为0/8，仅2 known roots，非独立确认。完整共同prefix5状态/snapshot/PNG完全相同。本干预同时改了incumbent线性化和候选下降守卫，不能声称单独证明某一个改动的因果贡献；8/8模型Q下降也不保证真实任务单调。真实终态编码桥prepare正在重建完整上游，尚未产生新的v2终态预测→编码排序结论。

P1 saved v3精确信封兼容后，继续被原v2的第635行native-label检查拒绝：原生产warmup400步发生在任务goal bind之前，逐步标签400/400与placement native_goal精确相同、不是最终task goal；prefix200/suffix400均与task goal精确相同。原验收将全部1000步都按最终row.goal计算，错误混淆目标绑定时序。未放宽原生距离/成功阈值；正在新版本逐步认证placement→warmup→goal_bind→控制动作的真实目标，不跳过原标签检查。两次验收失败均保留，不算科学负结果，更不用于解释历史12处模型排序反转。

P3新驱动MAIN已完整审读源码803行/测试340行/doc60行，独立54测试通过；新prepare SHA `77b0872ef13a6f3447226c60024109f785d6508fba882aa68ee42107265bfc8a`，主机CUDA+物理运行已提交，仍未核终态。双方每5步实际RGB反馈，总控制请求20步；付费信号及全部资源分轴报告，不称等FLOPs或P3通过。

### P1精确协议修复及新动作响应对照已完成

新saved-only v4已MAIN完整源码/测试/doc审读；与固定有限差分适配器联合57测试通过。实际MAIN消费已完成，SHA `5a5c1948dd37ff0725447887218f6d57e59aaae54a3f1df28925fb8fe32b01a8`，路径 `le-wm-main/results/agent_p1_prospective_action_effect_20261028_v4_MAIN01.json`。40 terminal joins、1000 native calls（400warmup/200prefix/400suffix）、88像素到原224 preprocessing逐位相同；全部80新对可用。仅适配已认证信封和唯一原native_labels调用的阶段目标，完整原v2代码其它语句保留；逆AST回归通过。原v2/v3失败及源码/结果均保留，不宣称它们直接通过。

MAIN新增独立标量有理数复算（不复用原effects函数）逐项重建完整二次恒等式、动作响应和renderer差分，与完整strict逐项精确相同；3测试通过。结果 `le-wm-main/results/main_p1_action_response_analysis_20261028_v1_MAIN01.json`，SHA `e20beaf937022f031d32cea7e990cfb56df284c5c7637ad47f1ee04dd3f4867e`：

| 新实际20→25面板诊断 | 计数 |
|---|---:|
| model→actual编码严格排序反转 | 17/80 |
| 反转中删除共同零动作投影项后不再严格反转 | 1 |
| 反转中删除动作条件响应余项后不再严格反转 | 16 |
| 两贡献项异号 | 48/80 |
| actual编码→joint XY严格反转 | 7/80 |
| actual编码→wrapped angle严格反转 | 28/80 |
| 同物理终态、固定原goal，切换renderer后的actualQ排序反转 | 7/80 |
| 非零词的相对响应误差非零 | 32/32 |
| 同终态renderer切换改变相对编码动作响应 | 32/32 |

新面板各起点是完整实际20步重放后的同一物理snapshot、RGB及raw history，五词预测全部在新25步outcome前封存；原80/76/4 teacher15面板与其12反转不被替换。新结果将共同偏移解释的局限收窄为**明显动作条件依赖**，但删除代数项不是一套可部署校正，也不是隐藏速度、接触、视觉域或某个模型实现的唯一根因证明。固定原goal的renderer诊断不是renderer-matched goal任务评价，不用它事后挑渲染条件。模型所选8条成功仍0/8，2known roots不作独立确认。

### P3完整同反馈20步已真实执行并完成原严格验收

实际主机CUDA/物理报告SHA `cdf46362e7e446105edf15f8f13f3113b9a7481b93acbd81338e43f2c5640157`，saved full consumer回执SHA `bc21c769d42fb11b958c2de65ecbd24b225312794049208259d739b3122ae305`。4/4实际控制20步，16/16节点封存；每臂132predict/1536grad，完整pipeline528predict/6144grad、20encoder、120public、1216physics。固定H1；adaptive第一根4次H3，第二根H3/H1/H1/H3，调度实际不同。只称相同predict额度，绝不等FLOPs。

两known根adaptive减fixed原生distance分别−126.62417/−35.74558，但object position分别+10.14364/+19.52423、wrapped angle分别+0.38869/+0.46692，native成功仍0/4。故完整同反馈驱动消除了“只执行首包/反馈不等”的解释，却没有建立任务一致收益或C3机制。原consumer对physical event kinds/census的额外攻击面加固尚未完成；本记录不是新的科学门或安全证明。

v2新终态RGB桥prepare完成，SHA `b099c494f49caec7d19cc3180e013c4b9f08c8768c39bdf0d7d788f77897ead4`，主机CUDA执行中；固定三步长raw FD prepare完成，SHA `dd9127c1feb721b5c232482990f0bfbd19b3f6d8d0d2b737740687eeb3b98baa`，全8×3×10×2请求主机CUDA执行中。未把已启动写成完成或排除导数错误。

**MAIN额外源码核查边界**：官方安装LeWM `predict`原样返回各位置下一embedding，`rollout`使用最后一项；训练路径`ctx_act[:history_size]`与数据frameskip=5打包顺序相符，当前直接full3调用未见额外内部动作shift。该只读核查不等独立时钟实验或训练来源认证。另安装PushT `_get_info`的接触平均分母写为`int(1/dt*control_hz)`，而`step`实际子步数为`int(1/(dt*control_hz))`；在当前dt=.01/control_hz=10时分别1000和10。此影响`n_contacts`数量解释，不改变真实step积分；planner输入白名单不包含它，不能用作17反转根因。未修改封存环境或据此重写旧接触标签；既有“非零接触”只能按其原含义引用，不能当精确平均接触数。

### 全raw Jacobian独立中央差分已完成，不再只验AD forward

实际主机CUDA 480请求全部返回，固定8锚点×3步长×10坐标×正负，0缺侧、0新AD/视觉/物理/训练；报告SHA `f555758ec008cab87cdc375c8295a7940dc5c0354bd27049a1aa7787a5b945fa`，244.900秒含完整上游验证。严格回执SHA `55e66b58ff263454cee662465cb402cdcb4af3621a2b7add7541cffdb42f4738`。MAIN另从原signed raw请求、完整实际NPY、封存AD矩阵逐元素重算24个192×10差分矩阵及全部指标，逐值与strict相同；独立结果SHA `469af2de172d118e1f995101a147289c77247e44856d6cb72c051ddfda718011`，路径 `le-wm-main/results/main_raw_fd_recompute_20261028_v1_MAIN01.json`。

| 固定raw步长 | 相对AD矩阵L2误差（8锚点范围） | 方向余弦范围 |
|---|---|---|
| 0.01 | 0.0000371092–0.0000877794 | 0.9999999965–0.9999999993 |
| 0.001 | 0.000217081–0.000455634 | 0.9999998962–0.9999999768 |
| 0.0001 | 0.00223718–0.00454111 | 0.9999897140–0.9999974975 |

真实float32 spacing的独立secant也全部保存/重算，未事后选最佳步长、设置科学阈值或改动作合法域。未发现本批raw Jacobian的整体符号/转置/显著尺度接线错误；不将局部数值一致写成全局AD正确性认证、真实动力学一致性或P1通过。MAIN本轮原生/动作效应/FD独立复算联合13测试通过，终态新NPY算术器另3测试通过；不把重叠测试累加。

新v2全部8终态主机CUDA编码已完成，报告SHA `9a63b5a3db263a3f9accdc8d54faa31b40787c362ca52e5e29ca793f785c7941`，572.219秒（包含重复全上游验证），errors/blocked=[]、模型未变；完整strict仍在执行。新增神经只有8encoder/8projector，无新predict/AD/物理。尚不据此宣称预测/编码/物理三层排序修复。

### 最新终态：保护性v2的真实编码严格验收与独立复算已完成

上述“strict仍在执行”为保留历史快照。完整消费现已exit0，回执
`le-wm-main/results/agent_p2_midfeedback_terminal_rgb_20261028_v2_consume_MAIN01.json`，
SHA `a66d4187cdbf714509a8bc931d12ef6c51410603da8c4445c7884bbde2e143e0`。
MAIN从全部8个实际gzip NPY独立计算新actual Q、predicted Q及endpoint MSE，
与严格回执逐值相同；原后缀Q明确继承完整strict，不冒充重新独立重编码。
结果 `le-wm-main/results/main_safeguarded_terminal_analysis_20261028_v1_MAIN01.json`，
SHA `6b925dcc6fa831535a42757118563e1db81cd2e4108035210e894561a4e62c2a`。

| 保护性v2相对原incumbent | 固定全8实际结果 |
|---|---|
| 模型Q / 真实编码Q | 均8改善、0变差；变化符号没有反转 |
| 真实编码Q变化 → jointXY变化 | index4反转；其余7项同向 |
| 真实编码Q变化 → wrapped angle变化 | index4/5/6反转；index0角度不变 |
| 实际原生成功 | 新旧均0/8，仅2 known roots |

index4的预测Q下降0.005716919、真实编码Q下降0.001287304，
但jointXY增加4.646164、wrapped angle增加0.02027225。
这是一个**预测改善确实迁移到真实编码、但未迁移到物理任务**的具体例子，
不能再将该行物理损害一概称为模型预测排序反转；也不能据此断言编码器无解。
本批同时改变线性化点与下降选择，两因素贡献尚未独立拆分；后续独占消融
先核已有候选/分支，缺失反事实不得从别批outcome补造。

**进一步收窄guard归因**：新8行最终selected_index为`[2,1,3,2,3,1,1,1]`，
没有一行选index0原后缀；全部候选最优值严格小于原后缀Q。因此在这批已固定
v2四提案上，“四提案argmin”与“原后缀+四提案argmin”最终动作相同，
不能把新v2的7/8 XY改善单独归功于guard保旧。guard仍提供模型目标防劣保障，
在旧v1回顾面板保旧三行；但新v2收益需在提案公式、线性化点和候选搜索层
继续区分。逐候选日志中的`selected=incumbent`可能指前一个较优新候选，
不等于最终保留index0；本段依据最终literal选择与模型Q，而非该字符串计数。

本轮已确认并修改的实现/协议问题包括：优化参照过弱且无条件覆盖合法方案、
日志信封与payload错误比较、warmup使用最终任务goal验标签；它们分别有真实
生产干预或完整saved验收证据。后两项解释验收失败，不解释旧P1全部排序反转。
仍开放的因素为checkpoint训练normalizer来源、匹配renderer的goal对照、隐藏
动力学可观测性、任务一致目标、独立根及强基线。不存在“所有潜在因素已排除”
或“理论被否定”的结论。P1/P2/P3科学准入仍为false。

主线程另补终态算术回归：同向编码改善不推出物理改善、角度平局不计反转、
根/condition/arm错接必须拒绝、strict或编译Q不符必须拒绝、原生float32身份
必须保留。本轮10文件联合回归实际231 passed（72.99秒），含P1阶段修复、
P2保护性编译/终态/有限差分及P3同反馈驱动；它们是代码验证，不是231项
独立科学实验。`git diff --check`另行核查，旧失败及封存产物不覆盖。

### renderer × goal匹配对照已落盘并重新完整验算

MAIN完整审读新source/tests/doc，并独立联合回归33 passed（renderer22+factorial11，
6.91秒）。两known根原四份off/on目标RGB及完整encoder/CLS/projector/latent桥
已有主机CUDA封存，故不重复神经实验。新saved分析保留全部80对、原动作及
原selected word，精确计算完整2×2终态renderer/goal renderer；两项交互恒等式
的有理数余项均为0。目标物理state7完全相同，on的环境outline pose仍是原默认，
不把它冒充constructed goal位置。

实际分析 `le-wm-main/results/agent_p1_renderer_matched_goal_20261028_v1_MAIN01.json`，
SHA `ae6dfabb047047090d14ac9f9ab00adf4547a9c09184d7070ace050b703d885b`；
从全部固定输入重算并精确比较的消费回执SHA
`1e1bae0a2e6776fa72dbc9b159309aa081e2f553e4392e31ee76843ed3773089`。
复用原完整P1 strict authority，新增独立认证四目标链，不冒充重复原完整物理消费。

| 全80对固定动作诊断 | 原匹配 | 换终态renderer、原goal | 终态与goal同时匹配 |
|---|---:|---:|---:|
| 与原匹配排序的严格反转 | — | 7 | 4 |
| 编码排序→jointXY严格反转 | 7 | 12 | 7 |
| 编码排序→原native distance严格反转 | 36 | 43 | 38 |
| 编码排序→wrapped angle严格反转 | 28 | 30 | 30 |

三列各40个物理angle ties均保留，不计反转。由此确认原“换renderer但沿用原goal”
诊断存在目标不匹配的混杂，已用新factorial计算补齐；然而**匹配后仍4对渲染
排序反转，且未消除表示—物理任务失配**。不能把7→4称为P1预测反转17→4，
二者不是同一计数；没有改选动作、训练、拟合、打开科学门或新增物理收益。

MAIN另在不导入上述诊断模块的独立只读算术中，从原report的80对两端向量与
四目标直接计算双线性式 `sum((zi-zj)*(zi+zj-2g))/192`，全部320个精确Fraction
margin及sign与新分析逐项相同（实际exit0，all_exact=true）。这不是新模型调用
或再次物理执行；其作用是避免仅用同一个consumer函数复现同一算术错误。

### 优化两因素的最小补齐清单已实际生成

MAIN完整审读新的factorial source/tests/doc，生产prepare已完成：
`le-wm-main/results/agent_p2_safeguard_factorial_20261028_v1_prepare_MAIN01.json`，
SHA `9ad5980d7943f11dc5e6d53c285f37dc4a86e69620d136036bcdd656d3e9d469`。
实际重建两版原compile闭包、绑定已完成的物理strict身份及全部8分支，
保留32cells/64proposal槽；统一v2提案公式，仅改变zero/incumbent线性化点和
是否将原后缀加入同四候选argmin。旧ridge与新gradient的直接比较只列为混杂
资产，不追认为干净消融。

incumbent侧32提案已有精确预测；zero侧32提案没有相同raw/context的现成预测，
故生成32条literal native输入请求，status=`blocked_missing_model_candidates`。
这是未执行候选的工程状态，不是科学负结果。`new_compute.phase=planned_not_executed`，
`actual_new_compute=null`；没有把计划32预测写成完成，没有新增AD/视觉/物理。
缺失侧不猜选择、不拿别的动作outcome补物理结果。主机GPU只读检查为RTX5060Ti、
空闲15474MiB、利用率0%；最小durable CUDA调用/严格返回验收适配器继续独占实现，
尚未启动这32次预测，绝不CPU替代。

新fixed32适配器投入CUDA前，MAIN独立联合测试真实捕获`1 failed/46 passed/8 errors`：
请求payload也叫`index`，与日志链序号重名导致`dict()` TypeError；日志writer与
reader使用的hash ABI不一致，又导致失败final journal无法验收。生产尚未启动，
这些是真实新增代码接缝，已要求改为独立request_index并统一原canonical/hash链，
保留完整意图/输入/返回验收；不改validator以凑通过。它们不能被追认为旧P1
排序反转的原因。超时后final预算路径也须有真实回归，避免丢失部分失败记录。

### 最新：固定32预测已主机CUDA完成、strict通过且MAIN独立复算

新适配器三件套已完整审读并冻结；接线/原生journal ABI/超时final修复后，
MAIN独立runner+factorial联合57 passed（6.98秒）。此计数与之前renderer+factorial
33项含重叠，不累加成独立总测试数。回归通过后，下面完整生产流程现均已完成：

- runtime prepare SHA `4177fc6b4805030e85902833d13c4da92ee02bbb341fe01915b4cfc9919adfd9`，实际verify通过32请求；
- 主机CUDA报告 `le-wm-main/results/agent_p2_safeguard_factorial_cuda_20261028_v1_cuda_MAIN01/report.json`，SHA `380772fc6f6ed6e10c0c1e23d855f7db75dd7ecb3056f56a2f0c9915170916d6`；
- 完整返回数组/事件/输出census/选择strict回执 `le-wm-main/results/agent_p2_safeguard_factorial_cuda_20261028_v1_consume_MAIN01.json`，SHA `8a59a189feaedf3b215b7aa9245c3b027228e2a51cfef32a47df286c664e5d89`。

实际32/32 predict、32/32 action_encoder、32/32 pred_proj，inclusive predict API32
单列；新AD/SVD/视觉/物理/训练=0，errors=[]，模型before/after完全相同，elapsed
8.673454秒。执行只接受刚由MAIN完整重建的固定factorial prepare SHA，并在前后
重hash其全部上游bindings；明确`full_upstream_rebuilt_here=false`，不重复双
consume_compile，也不冒充本wrapper重新进行了原全部物理消费。新增预测都在
主机CUDA，saved重算/NumPy回归不算CPU神经实验。

MAIN另在**不导入实验模块**的独立只读程序中，读取32个原NPY重算全部模型Q，
再从原goal/旧context和四候选全量重算32个stable argmin及literal raw bits；
逐值与strict相同（实际exit0：independent_NPY_Q=32、independent_stable_selections=32、
all_exact=true）。不是只核已有summary。

| 同公式、同四alpha的干净模型消融 | zero线性化 | 原incumbent线性化 |
|---|---:|---:|
| 不含保旧时，比原后缀Q更优 | 7/8 | 8/8 |
| 加保旧后最终动作改变 | 1/8（index4） | 0/8 |
| 加保旧后选中原后缀 | 1/8 | 0/8 |

两个guard设置下，incumbent线性化相对zero线性化均为6行模型Q更优、2行更差
（index3/6），不是普适更优；不改变公式、候选数和目标后仍能测到线性化点
的实际影响。zero侧index4若无guard，模型Q从原0.138612309变到0.168202476；
guard精确保留原方案，防止这一次**模型目标**恶化。两种anchor的guard off/on
共享同四候选预测，guard额外forward=0；继承的旧AD/候选费用另列，不称等FLOPs。

**仍未填补的物理反事实**：本批只新增预测，没有执行新zero侧控制分支。不能
将zero模型Q较小写成真实XY/角度改善，也不能将既有v2物理7/8改善归到某个
单因素；旧v2的物理与终态编码结论保持原范围。下一最小控制对照是合法新
zero-anchor选择的同根、同prefix完整物理执行，并仅按精确动作身份复用已验
收分支；不重选动作、不借其它raw序列outcome。P1科学门依然未过，没有理论
no-go结论。checkpoint normalizer训练溯源、任务一致表示和独立确认仍开放。

### 续推：补齐真实物理2×2，而不是继续合并误差项

本轮MAIN新增独立消费器
`le-wm-main/experiments/spectral_reachability/main_anchor_guard_physical_factorial_20261028_v1.py`，
直接读取三组真实native state与实际dispatch journal；不导入环境/模型模块。
32cell逐一按完整raw10位串及同task/root/goal/arm、共同prefix5位串、真实中点
snapshot/PNG匹配。guard复用由动作身份决定，绝不按哪条真实效果好决定。
逐轴给出anchor、guard及交互差分；物理distance/XY/velocity/angle统一越小越好。
若zero物理结果不存在、动作不同、返回不足10步或中点不同，明确拒绝出完整表。
新增14项合成接线回归通过；与既有guard/factorial消费者联合78 passed（7.26秒，
之后新增prefix逐位检查的14项也重新通过）。这些仍是测试，不是新增物理结果。

待新的zero物理8分支执行并strict通过，MAIN才填写完整真实2×2。当前不能把
旧v2的7/8 jointXY改善归给某一个因素，也不能把模型guard当作物理收益保证。
另两条独占实现并行准备训练normalizer来源manifest与P3 saved消费者加固，
不训练、不更改归一化常数、不改门槛，不以缺失来源认证推断模型一定有错。

### 最新：zero侧真实物理已完成，干净2×2效果与优化归因边界

新增adapter三件套经MAIN完整审读。发现metadata文件未fsync的实现错误后，
改为文件仍open时flush/fsync、关闭后同步目录，补真实回归；不是旧P1根因。
首次prepare `f228bb8a...` 遇到智能体最后补测造成tests SHA从`40dcbd81...`
变为`a04262fe...`，verify及run均在环境/输出目录创建前拒绝；旧prepare保留。
它是闭包漂移工程失败，不是物理或科学负结果。冻结确认后生产MAIN02：

- prepare `le-wm-main/results/agent_p2_safeguard_factorial_physical_20261028_v1_prepare_MAIN02.json`，SHA `7ebd434aaff49d66d65181186569f9511827c263c003d1856c573f99adafae70`，独立verify通过；
- 主机原生物理report `le-wm-main/results/agent_p2_safeguard_factorial_physical_20261028_v1_MAIN01/report.json`，SHA `999f8278dbda931a2bed1ba1aa0ec202d47ec8444ad090b58f28ce88c74f4a2f`；
- 完整strict回执 `le-wm-main/results/agent_p2_safeguard_factorial_physical_20261028_v1_consume_MAIN01.json`，SHA `9c125a32268899ece92144d95382529aa2a2f406de36cfd88c05828d4e5ac00d`；
- MAIN独立state/dispatch32cell分析 `le-wm-main/results/main_anchor_guard_physical_factorial_20261028_v1_MAIN01.json`，SHA `cb1827b8b1e65805766b1e53ffca2f7045a9b9d05ddd547914bae3d93accfa30`。

实际8/8完整10步，160public、1632physics、528新鲜PNG、errors=[]，wall20.564秒，
成功0/8。神经调用0是因为动作已由上一轮主机CUDA预测并封存；原生物理与
saved NumPy算术不是CPU神经降级。新zero终态还未编码，不猜真实latent Q。
原native execute/assemble及exactmid gate不改；外层明确原contract的历史
compiler计数仅兼容ABI，不是新计算或新的旧compiler receipt。

| 同提案/候选/任务的真实消融 | jointXY优/劣/同 | objectXY优/劣/同 | angle优/劣/同 |
|---|---:|---:|---:|
| 无guard：incumbent anchor减zero anchor | 5/3/0 | 5/2/1 | 3/4/1 |
| 有guard：incumbent anchor减zero anchor | 4/4/0 | 4/3/1 | 3/4/1 |
| zero anchor：guard开减关 | 1/0/7 | 1/0/7 | 0/1/7 |
| incumbent anchor：guard开减关 | 0/0/8 | 0/0/8 | 0/0/8 |

四设置均成功0/8，只有2known根。zero侧index4保旧使modelQ从0.168202476
下降至0.138612309、jointXY下降9.359959、objectXY下降17.566769，但angle
增加0.038628、velocity误差增加272.029595、native distance增加260.283239。
因此**模型目标保旧不等于全物理目标保旧**；换线性化点的真实效应也并非一致
正向。旧v2的7/8 jointXY改善不能全归给anchor或guard；不能声称单一优化代码
修正已解除P1。zero终态RGB桥准备中，新增神经实验仍须主机CUDA。

MAIN最新联合36 passed（9.51秒，adapter22+独立分析14，含重叠不累加）；
测试不计作科学样本。

### 训练归一化来源：值合同已核，训练身份仍未建立

MAIN完整审读新配置来源source/tests/doc，独立12 passed（0.66秒），正式manifest
及全量rebuild verify均exit0：

- `le-wm-main/results/agent_p1_configuration_provenance_20261028_v1_MAIN01.json`，SHA `43841acd6b2f0f293d041b2c2b871076b425cc3e090e1adfa21e96cfece7dd57`；
- verify SHA `06c0c8516d770da3bbb4ce127529211d5fd9e2e0554406db6eb21c77f5cad853`。

checkpoint/config文件、cache对象、公开/JEPA/current compiler/封存task的mean/std
float32 bits均绑定并一致；本工具未反序列化权重、未读dataset数组、未训练、
未执行神经或physics。当前train/config快照不能冒充该checkpoint的历史training
run；README数据集名称与下载revision也不是训练内容身份。十项缺失链路（run、
历史代码与resolved config、不可变数据/episode split/filter、动作转换、导出scaler、
weights绑定、H5→Lance转换）逐项保留，`training_identity.verified=false`。
这收窄“归一化已排除”的错误说法，不证明常数错，也不授权根据结果换统计。

### P3已有实际结果的saved验收加固已落盘

MAIN完整审读新v2 source/tests/doc，最新v1+v2联合120 passed（80.40秒）；
实际完整saved消费exit0，回执
`le-wm-main/results/agent_p3_same_feedback_physical20_20261028_v2_consume_MAIN01.json`，
SHA `84a007329c7813fde4c2ed332e233e7d8cf4afabb5216f2da58381f02855104c`。
新增全量event ABI与所有资源逐episode census、嵌套attempt/return、未知/多余
事件拒绝及PNG→float32预处理独立逐bit复算。实际128 observation/guard、
80dispatch/control_return、20encode、16seal、80handler、1560attempt/return
均闭合，20张预处理数组每张150528值精确一致；没有新增模型或物理。
原v1未在receipt封存预处理文件SHA，新v2如实说明这是当前saved值重绑定，
不追授生产时attestation。P3原object/angle退化与成功0/4保持，不因consumer
通过改成C3成立；这次没有发现上述缺口导致实际控制损害的证据。

### zero8终态主机CUDA与独立原NPY三层分解已完成

新terminal桥三件套MAIN完整审读、独立39 passed（1.29秒），明确冻结后再
prepare，避免上次开发期测试漂移。prepare SHA
`6615271511a3010f4b06e7136305091ba7c533269b7c04997653dbe39d599f94`，
独立verify通过；真实主机CUDA report
`le-wm-main/results/agent_p2_safeguard_factorial_terminal_rgb_20261028_v1_cuda_MAIN01/report.json`，
SHA `7fe0bb86560d20b5dfce10b14d5e8e90847b15f77f78e402105d277ef84aced2`。
实际8encoder/8projector、predict/AD/physics=0、errors=[]、模型前后完全一致，
elapsed10.177秒。完整预处理/hidden257/CLS/projector/原生192、双ledger和
输出census strict回执SHA
`1dcd057eeaf06f9f5156fe27ae7c44811e19a2c48fcd375af5e5a607cba3a1a4`。
没有把legacy receipt schema投影为新authority，没有globals patch或CPU神经。

MAIN新增不导入实验模块的独立NPY工具及9项回归（0.16秒），读取全部原
control/zero/incumbent实际latent NPY，再由cell向量重算32预测Q、32实际Q，
逐值与accepted strict相同。独立结果
`le-wm-main/results/main_anchor_guard_latent_factorial_20261028_v1_MAIN01.json`，
SHA `20fc92f83d1f1e2282a8a61f0c97717b1235e6833ea4ef4636c1b2484833148a`。

| 固定8分支对照 | 预测→真实编码排序反转 | 真实编码→jointXY反转 | 真实编码→angle反转 |
|---|---:|---:|---:|
| anchor变化、guard关 | 2 | 3 | 2 |
| anchor变化、guard开 | 2 | 4 | 2 |
| zero anchor的guard开关 | 0 | 0 | 1 |
| incumbent anchor的guard开关 | 0 | 0 | 0 |

最后一行全部同动作，不能把“0反转”宣传为新可靠性证据。anchor反转位置1/3：
branch1预测差−0.000400804、实际Q差+0.007839663；branch3预测差+0.006181149、
实际Q差−0.105050733，而jointXY增加14.003626。zero guard的branch4预测Q
差−0.029590167、实际Q差−0.015461422，但angle/velocity/distance损害保持。
这将本批问题明确拆为**动作条件预测误差**与**表示—物理目标失配**两层，
既不是共同偏移就可消除，也不能以原优化缺口修复声称P1通过或理论no-go。
训练输入身份与隐藏动态可观测性仍未排除，任务一致表示／真实非贪心能力机制
和独立确认继续是研究缺口；没有按这些outcome重选动作、Q、根或阈值。

本轮最终MAIN相关五文件联合98 passed（11.91秒），P3 v1/v2联合另120 passed
（80.40秒）；不与前面重叠回归累计。`git diff --check`通过。本轮属于新增真实
物理/CUDA与可区分两层误差的实质进展，完整研究目标仍active，不标complete。

### 新增：完整候选选择评分的精确分解

后续 MAIN 新增全888候选的独立模型侧分解，结果SHA
`6724b8f9275b97d22c2811065ca74d55ccb9467870b86828d2ce753ed6e51655`。
旧 stage+ridge 与完整非线性 terminal witness 在4条件中3处选择不同；
舍入代价效应至多3.28e−9，非线性代价效应可达0.356。首段惩罚与有限步
近似共同改变排序，不能称统一偏移，也不是新物理实验或唯一根因证明。
完整终态所选仍全是相对root不退步；库内最佳预测先退步候选仍未胜过原
选择。因此新评分不能被写成P1已解。代码、精确公式、全部主要因素族与
修改边界见 `docs/D2P_SELECTOR_CAUSAL_FACTORS_MAIN_20261028_v1.md`。
新selector、真实执行适配与feedback gradient MPC分支正在并行实现；
此段不是它们已经运行完成的证明。
