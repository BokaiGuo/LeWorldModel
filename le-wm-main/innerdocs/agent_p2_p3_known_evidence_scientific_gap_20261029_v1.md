# P2/P3：known证据闭合后的科学前置差距表

2026-10-29。有界证据快照，不是新gate、准入证书或事前预注册。仅新本文；未改旧文件、调用NN/physics/fullstrict、重跑分析器或重复Pauli三层差分代码。采用documentation-and-adrs技能区分证据、决策及未解决条件。后续新receipt可能补齐工程缺项，应按其实际SHA另记，不能把本文快照当永久状态。

## 1. 本次直接核对的证据

以下产物路径相对`/home/gbk/code/LeWorldModel/le-wm-main/`，本轮只读检查SHA及必要字段，不重新颁fullstrict。

|证据|路径|实际SHA256|
|---|---|---|
|六槽完整RGB receipt|results/agent_p2_six_target_rgb_20261029_v1_consume_MAIN01.json|c16d99ba82c8d3adeb68515e2035cc191bd060ab6c273e07ff21cef9a23d870b|
|六槽已认证物理标量|results/agent_p2_six_target_physical_scalar_summary_20261029_v1_MAIN01.json|89922172a5a26f42a3048b3a12abf1733a6e40dc7b634949d7c358cbae4163cb|
|六槽Fraction归因|results/agent_p2_known_six_margin_attribution_20261029_v1_MAIN01.json|4c0db1d8e7606afbe944467ca3e2c6bee6ddd4a5a6815dd77b9022068f67b104|
|full30同时刻标量|results/main_p2_full30_same_time_summary_20261029_v1_MAIN01.json|d9b97b5c351bcee90f1d2bde5598d2d2c6469dd7bcceb09b9a66117cad00fa16|
|full30 Fraction归因|results/agent_p2_full30_margin_attribution_20261029_v1_MAIN01.json|a820a5fe8dcb3170d8defe4a2cb5eb2fe48c89f69045b0e6d266f25cbef41aac|
|P3 secant模型交付|results/agent_p3_secant_delivery_20261029_v1_consume_MAIN01.json|6f583507a9e5ce6d0f859032344fc303527c250b5633fa8d3482ded3615e4fb6|
|P3 goal-conditioned actual-cp5交付|results/agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1_consume_MAIN01.json|b50268a2018e6da1fbc46a4d19d701cd4a9a575cfaf581807cf29c01861d7ba8|

合同解释依据：外层`/home/gbk/code/LeWorldModel/docs/D2P_P2_COMPARISON_CONTRACT_20261026.md`（SHA `07635f5ebf9c6b86308d0ec56b3b9ea0cb44b69b2c30988de95f65ebaadbdbcc`）；actual-cp5局部合同`innerdocs/main_p3_actual_cp5_R_delivery_protocol_20261029_v1.md`（SHA `deed88fd70073c1aaca96d32c81968b7f02a7a3841e3d8a7be00d4e4d573275f`）。现状索引为外层`docs/D2P_DOWNSTREAM_PARALLEL_MAIN_20261029_v2.md`，本次读取快照SHA `fc6634ae501d4047bb28d87f89a73e6e4c71fcee5e7aa49e584a46e461a318be`；主记录会续写，此SHA只锚定本次快照。

## 2. 四类“过不了”必须拆开

|前置类别|当前能确认什么|尚不能确认什么；如何报告|
|---|---|---|
|数据/实现完整性|本次六槽native/paid/RGB及两份独立标量归因已有各自完整保存认证；RGB24/24位置、24/24pair，encoder/projector各24次，新predict/AD/physics为0，cp0精确闭合|不能说“所有现行runner均认证”；新ABI错误、写入路径/保存cap/consumer故障是对应工程断点，不是模型负收益。原relative输出被拒无文件/无physics属于此类|
|freshness/校准|receipt明确known_roots=2、fresh=false；可以继续已见任务的工程/描述性校准|重复reset、不同臂/时刻、换goal/renderer不产生新独立根。历史loaded-bytes闭包、完整reset/exposure ledger及批准范围不能由数值推导代替；未取得相关authority前不签独立确认。旧P4缺项须按原请求补事实，不另造字符串证书|
|真实非贪心机制|可以检测原资格、模型见证是否交付、实际动作/时点是否一致，并逐轴描述有限真实结果|模型资格/whole-segment parity不等真实任务收益；未激活、no_witness、合法拒绝、未实现R调度、已执行无益是不同状态。P1未获科学通过，不借工程success放开P2/P3|
|强基线统计|可以报告共同cap、完整资源向量及已见任务的全分母配对差|不能由predict数、相同horizon或共同cap宣称equalFLOPs/实际等算力；unknown费用不补0。没有充分独立根和预定统计设计，不把24/14pair作独立样本签强基线优势或置信区间|

这里不把旧Reacher preflight或其单runner当成现行PushT model-library/freshness/机制合同；上述四层分别需要自身来源与权限。理论存在性既不签freshness，也不保证这批真实收益。

## 3. P2/P3可继续与不能升级：直接审读表

|线索/对照|可继续检验的有限命题|不能升级的命题；还缺什么|
|---|---|---|
|P2六目标共同controller|在相同root/history/原任务goal下，固定六目标产生的已付费actual-token预测、实际编码和native轨迹怎样不同；tracking-Q与task-Q分栏|不是纯“target质量效应”：prior物理时距仍null、目标可能不可达、controller/模型误差及生成支持/预算混杂。跨不同tracking target的loss不得排名|
|P2六槽RGB三层|本次receipt允许同动作同时间model→encoded→native检查；MAIN记录24pair中model-taskQ→actual-taskQ有3个严格反向，actual-taskQ→jointXY有1个，→native7距有7个；cp0六pair ties保留|不能唯一归因encoder普遍失配、动力学或renderer，也不能由cp0精确推出未来预测正确。此处引用MAIN已完成分解，不另写重复消费者|
|P2六槽平方margin|24pair总平方差负7/零6/正11/null0；位置/总距反向6处，速度平方项在算术上可压过位置/角项|不是将速度抹去就会改善控制的因果实验；不能改Q/成功指标或挑好轴签胜。Fraction只精确表示保存标量，非精确物理|
|P2 full30 transfer vs joint|完整28cp/14pair；joint−transfer总平方差负5/零8/正1/null0，root0 cp5/10/15/25位置平方增加被速度平方下降抵销|joint生成支持/计算更多，不是组合机制公平优势；root1同值不删。full30原生成功0/4仍如实报告；六槽full15原生成功0/6也不是总体不可能性证明|
|P3 actual-cp5 spectral/goal-conditioned bank|严格按各自资格与固定bank保留所有组；goal-conditioned实际四组均no_witness，可以做拒绝原因分解|不能写成已有可交付R被physical代码阻止，或全动作空间不存在R；不能降原`immediate>zero AND reach<zero`门槛取winner|
|P3 secant变体|已认证root0/H1 model_delivery_verified；另三组no_witness。该单组可进入其既定first5原生guard接缝的MAIN后续验收|不能说“P3所有bank都无witness”。secant新变体不冒充旧spectral8等价；模型见证不是physical release，更不是完整R-aware tree/C3资源调度收益|
|P3 shared-panel/共同cap|按各自有限支持域解释已实施horizon、候选和资源；分开未激活/合法拒绝/工程错误/已执行不利结果|单parent、progress-only horizon应用或共同cap不能替代原完整宽/深/类型/R orchestrator。真实多轴预算与调度作用须单独验收，不能借P2六槽RGB移交科学pass|

P3 secant的下一物理接缝在独占既有任务中，本轮不再开runner，也不将一个model-delivered组替换另外三个no_witness。本文未重审其全部原始IO或执行physics，只使用已给实际receipt的身份与交付状态。

## 4. 独立性、原生成功和方向：不可合并

六槽是两根×三臂；24cp与24pair共享根、动作前缀及时间，不是24个独立试验。full30四branch/28cp/14pair也只有两已见根，并与六槽复用原来源；不能把两个实验拼成四个独立根。新env、signed-bit不同动作、换renderer或不同计费输出都不解除历史exposure。

令d=原state7−原goal7，顺序agentXY/blockXY/raw-angle/agentvelocity。保留：

- native7_distance=norm(d7)，reward=−norm(d7)；平方为jointXY²+raw-angle²+agentvelocity²。原goal velocity非零时也不替换为0。
- 原native success为norm(d[:4])<20且min(abs(d[4]),2pi−abs(d[4]))<pi/9；它包含agent+block XY，但**不含velocity阈值**。原min-angle可能出现负值的行为保留，wrapped atan2诊断并列，不悄悄修复合同。
- block接近目标不等jointXY合格；速度改善可使native7距下降但不促成native success。六槽root0 program的block误差约1.244、jointXY约94.551，仍native_success=false；不能称object成功等于任务成功，也不能称此处仅速度阻止success。
- 同时间cp对照与native早停event-time success分栏；缺cp保null，不沿用最后状态、不对短执行补零。每个原生成功标签和位置/角度/速度均独立保留，不混单位择优。

## 5. 最小可检验下一步：只补缺失见证，不再造实现

**优先工程slice：消费既有Pauli full30 RGB交付，不新增第二套代码。** 既定`innerdocs/agent_p2_two_program_full30_rgb_protocol_20261029_v1.md`已固定4×7=28位置、4anchor/24forecast及14个同时间pair；当前有界文件名核查未见该full30 RGB的actual consume，不能把协议/六槽RGB冒充完成。本轮不要求MAIN重跑六槽或full30物理；待已有负责人交新实际receipt及SHA，MAIN按其原28位置协议验收，不因本次归因更换cp、目标、动作或正常化。

可检验命题：在原full30各自封存raw30下，cp0新编码是否精确匹配原anchor、cp5..30 actual-token模型与真实编码的task-Q方向是否一致；再与本次原native全轴并列。transfer在cp15是不中断A→B接缝，不reset；joint有不同动作，不将跨臂差当同动作因果。同根原goal192/normalizer/Q、全前缀rawbits与真实时刻必须认证。缺图null、24forecast与4anchor分别计账；不把H3/别的轨迹替代缺输出。此前已发生的native结果使该桥属于封存动作后的诊断，不倒签盲确认。

新增成本只归原28图HOST编码桥：最多28encoder+28projector，predict/AD/physics/新goalencode=0；保留其既定64MiB/600s和实际IO/wall范围，不在本文另授生产或扩cap。**如果actual full30 RGB已由MAIN随后交付，本slice自动转为接其receipt，不重复运行。** 该步骤可区分长时预测/编码/物理断点，仍不能开放P1/P2/P3。

真正的科学升级另有先后条件：先由MAIN补原历史/exposure与真实根authority，冻结采用模型/selector/目标分布/根顺序/完整失败分母/资源向量及统计方案，再在独立数据上检验原任务收益和强基线；不从这两根的反向pair挑新配置，不借事后协议消除既有exposure。P3还需其R-aware真实激活、causal pre-dispatch历史guard和对应物理结果，以及实际预算约束的完整调度对照；有witness不自动完成这些环节。

本文没有新producer、测试或生产命令；文档检查仅SHA/字段/公式/路径和空白检查，不能计作新科学实验或fullstrict。当前只交科学差距表与上述最小接线次序，既有代码/合同/封存保持不变。
