# Spectral Reachability Atlas：冻结视觉世界模型的动态可达几何

## 阅读入口：最新审计优先于历史流水（2026-10-26）

候选度量进一步验证已完成：保持原五goal差分投影固定，对物体x/y±16与角度
±0.15、未见推杆对角位置做44次主机CUDA编码；9门及44个独立最小二乘复算
通过。物体编码变化能量保留中位86.8%（最差72.7%），未见推杆中位0.28%。
这是目标表示层面的正面前提，不是物体控制信息/预测排序/物理收益保证，
尚未打开P1；正式方法还需明确goal集合输入、全链一致和独立物理验证。
见[目标推杆干预后续敏感性验证](D2P_PUSHT_GOAL_AGENT_INTERVENTION_20261026.md)。

最新主机CUDA目标干预已完成：20次目标编码、9门及960个成本独立复算通过。
物体目标/动作固定，只平移目标RGB的推杆，full192 one-shot/GN4固定库
terminal argmin在16干预中改变4/2次，确认物理物体任务之外的目标约束会
影响评分。它不是原selector改变或控制修复证据。有限goal集合的nuisance
投影候选已实现，但尚缺物体敏感性、全链一致编译与新根物理效能验证。
见[目标推杆干预](D2P_PUSHT_GOAL_AGENT_INTERVENTION_20261026.md)。

PushT最新保存数组审计：全2336736动作的sample std比公开常数高约3.3%，
尚无权重训练版本绑定，未更换部署归一化。全部192真实分支中，三种求解器
分别51/47/40条仅推杆运动而物体终点姿态精确未变，且29/33/28条仍有真实
编码RGB改善。物体目标与RGB代理的差异已有具体执行证据，但尚不构成因果
根因或理论否定；既有4个封存selected后缀均未改变物体，原P1继续关闭。
见[PushT动作统计与物体响应](D2P_PUSHT_ACTION_STATS_AND_OBJECT_RESPONSE_20261026.md)。

最新完成：完整Q一致编译器及真实能力探针之后，已补696个同度量/残差合同
独立拟合，并完成32根纯模型条件主机CUDA状态准入与全量独立复算。
149184预测样本、15完整性门通过，但单执行器平台且采样逃逸合格根仅1个，
不足冻结的8根，最终准入列表为空；不补根、不改阈值、不开P1后续阶段。
PushT当前可见资产尚不能认证官方checkpoint训练normalizer，不能以公开示例
常数冒充确切溯源。相关回归1591项通过，研究未闭环且不宣告理论失败。
详见[机制合同与模型条件队列](D2P_MECHANISM_CONTRACT_AND_MODEL_COHORT_20261026.md)。

最新追加：新独立八根57单元真实能力测量与8目标/32轨迹精确重放已完成；
目标编码器输入co-metric已实际完成1536主机CUDA VJP和408证书QP，独立
复算及原固定基线配对对照见[前缀能力审计最新节](D2P_PREFIX_CAPABILITY_AUDIT_20261026.md)。
仅肩/仅腕已知库重评出现控制收益，但完整能力合取与正式P1仍未通过，不能
以诊断完成替代原研究闭环；也不将代理不一致外推为理论失败。

**C2缺失环节已进一步分解**：新增52个真实前缀后继单元的2864条未来、5988个
主机CUDA预测、11门全部通过和624份独立有界拟合。当前四角色的终态获益不自动
对应关节任务的控制能力提高，且RGB/关节共同可达增益可方向不同。
24根全部17程序事后覆盖检查发现，后缀搜索改善后，18/19/19根各有真实非贪心且
双终态指标更好的候选，但旧common仅选中1/0/1根。未将事后候选用于部署或修改
P1。全17前缀首轮在第七根出现零未来重放非逐位一致而严格停止；第七根51单元
独立分层CUDA复查204条未来、408次编码全部一致，尚未复现该长运行问题。
审计器已补保存失败现场，不放宽规则，并从头启动全新v2完整CUDA重跑。
部分完整库预览出现被旧共同增益条件遗漏的物理能力诊断候选，同时存在RGB与
真实关节代理度量错位；不能把selector未选出见证等同理论失败。原52单元仍必须
逐位复现，v2未完成前不记全库通过。
见[前缀能力与选择缺口](D2P_PREFIX_CAPABILITY_AUDIT_20261026.md)。

**新定位到代理评分造成的具体误选**：官方冻结模型、16已知根三任务816个程序，
240条真实分支/35088个主机CUDA预测/15门完整通过，1632个新QP独立复算。
已知全驱动坏根202650004只把零切点线性代价替换为已有非线性可行动作代价，
不增加搜索，即从真实终距1.654116降至0.024495；固定重线性化进一步至0.018549。
这证明一处实现代理失败，不是理论负结果。但收益由单个已知根主导，三任务仍
没有合格的相对zero非贪心被选根，不开放P1。新8根来源隔离确认现已完成：
另240分支/35088预测/15门通过，1632个QP独立复算，旧八根五种执行逐位复现。
新仅肩只换评分均距0.697934→0.560880，2胜/0负/6平；同优化native对greedy
3胜/0负/5平，但仍非相对zero非贪心，不能开放P1。全驱动/腕无稳定新增优势。
24根stage代价下界检查仅2根全面排除非贪心，不能把该问题泛化为全部根因。
见[零切点代理因果审计](D2P_RECENTERED_REACH_AUDIT_20261026.md)。

**上一包误差补偿已做完整因果对照**：同16根三任务三固定程序四执行方式，
576条真实分支/29376个主机CUDA样本/12门全通过；原三方式432分支1296包的
动作、192维预测、真实观测及求解receipt逐位复现。补偿纠正了同动作诊断004
的预测margin，却改错另一根，实际执行004仍劣于同反馈greedy；三任务相对
同预算目标重规划的距离/RGB区间均跨零。不将该办法设默认，不开放P1，也不
将剩余失配归为理论失败。相关1150项回归通过，见
[因果残差对照](D2P_CAUSAL_RESIDUAL_AUDIT_20261026.md)。

**逐包定位与反馈执行修正完成**：全16已知根、三任务144条固定程序的432个包与
原真实执行/官方rollout精确一致，1728个CUDA预测样本；关键004在真实历史单步
预测下仍反转。另完成全部432条因果反馈分支、19728个CUDA样本和1296组执行
回执独立核验，原前缀不动。全驱动路标跟踪均距0.127644→0.113647，12胜/4负；
关键004对同反馈greedy转为真实终态更优，但非原同动作排序修复，肩/腕无稳定
收益，不开放P1或据此否定理论。新增指标拓扑错误已修，432实测端点不受影响；
相关广覆盖1131项通过。见[逐包反馈修正](D2P_MASKED_PACKET_FEEDBACK_REPAIR_20261026.md)。

**固定前缀后缀搜索已确认数值不足**：全八根×三任务×17前缀的主机CUDA干预
1224条真实分支、349320个predict样本、16门全通过，原408分支精确重放，816个
QP独立复算。仅改善后缀为保留原解的CEM后GN4，全驱动/仅肩/仅腕全前缀均距
0.779198/0.897793/0.861826→0.147112/0.650112/0.711909，不能将旧搜索失败
归为理论结论；但旧选择器整体未改善，两个新增模型候选仍有一个真实RGB反转。
原提案own-state reach cost与旧common-residual gain不是同一合同，已实现独立
且结果前封存的 `prefix_MSE + own_V/192` 选择器，不回填P1。新八根确认还复现
官方真实源状态被硬限位起点检查错误拒绝；显式v2保留软限位源状态而不放宽目标，
16根源状态/渲染审计全部通过，15个原有效上下文逐位不变。同八新根CUDA基线与
后缀确认现已全部完成，456+1224条物理分支、59568+349320个predict样本、14/16门
全通过，408基线分支逐位重放，另816个QP复核。新任务全前缀均距全驱动/仅肩/
仅腕0.699495/0.842140/0.839171→0.073673/0.589849/0.762679，前两任务八根均
改善；own-state选择并未产生非贪心见证，旧判据新增全驱动候选有真实终点反转。
独立审计另修复flat-CEM singleton兜底与后缀初始NaN/形状fail-open漏洞，全部
16根三任务97392个CUDA样本、10门差分回归保持原动作/预测逐位不变，不能冒称
本批反转根因。三张全根图/9项SHA均复核；P1仍关闭，最新1072项回归通过。详见
[后缀求解与残差合同审计](D2P_MASKED_SUFFIX_SEARCH_AUDIT_20261026.md)。

**严格单执行器新任务预检完成**：旧5%力矩不能等同关闭执行器。新增reduced raw
chart，未启用关节为字面原始零，过去历史不屏蔽、未来归一化一次。八个独立新根、
两时域两幅度，976条真实未来与2272个主机CUDA predict样本、12门全通过，384个
有界线性拟合独立复算。幅度0.25两包物理最小响应奇异值全驱动/仅肩/仅腕均值
0.489252/0.032425/0.024709；这是短时条件变差，不是不可控性或重配置成功。
另一批与全部200个旧episode隔离的新八根的完整动作见证审计亦已完成：456条
真实分支、59568个predict样本、14门全通过；408个程序分段/整段预测逐位相同。
三任务均0/8合格重配置，native回退greedy；全驱动flat GN/CEM+GN incumbent均距
0.865052/0.028534，不能由搜索失败推断模型无能力。首版新writer身份错误已修复，
完整重跑与独立归因确认全部456分支/1368包、预测及动作不变，仅身份写入修正，
失败版保留，不冒充旧P1根因。两组全根双图及SHA复核通过；最新989项合同回归
通过，不调P1门或宣布理论失败。详见
[严格执行器任务与程序审计](D2P_EXACT_ACTUATOR_TASK_AUDIT_20261026.md)。

**C2真实后继响应已直接测量**：同八根全部64个真实prefix后继，双尺度二十维
command正负探针共5184条真实未来、10496个主机CUDA predict样本，12项完整性
门通过；448增益独立复核、双图/产物SHA核验，最新906项合同回归通过。模型内部
AD/FD接近不等于真实RGB响应准确；相对greedy未建立稳定qpos控制增益，普通全
驱动任务尚不能当合格controllability-gap family。反馈终态改善仍有效，但不等于
独有能力改变机制；不开放P1，也不宣布理论失败。首版审计的一次严格重放失配
未在512边界隔离与修订版复现，根因仍待定，未放宽阈值或删失败证据。详见
[真实动作响应与重放审计](D2P_FEEDBACK_PHYSICAL_RESPONSE_AUDIT_20261026.md)。

**剩余反转已做固定动作误差定位及真实反馈干预**：官方原生与手写rollout全64个
192维边界逐位相同，未发现这条路径的索引差异；真实已执行历史单包预测消除
003的诊断反转，不能当部署oracle。先封存全部原计划/waypoint、再只用本分支
已观察RGB固定4次GN跟踪的64条真实分支已完成，5536个CUDA predict样本、
全部门通过。native均距0.044392→0.024082（7胜/1负），但原非贪心条件仍未
恢复，不开放P1。新八根202648000..007与历史184个episode隔离，确认协议在
结果产生前封存，四阶段现已全部完成：192条完整分支、401878个CUDA predict样本，
完整性门全通过。native初始计划均距0.063315→0.027490，8胜/0负；七个模型
witness经反馈后相对固定neutral均保留真实非贪心条件，新根001恢复。但动作已改，
不能当同动作排序fidelity；原方案仍有001反转，旧批003仍未修复，对同反馈强对照
CI跨零，P1仍不开放。四张新图及产物SHA均已核验，最新874项合同回归通过。见
[因果反馈修正与证据边界](D2P_WAYPOINT_TRACKING_REPAIR_20261026.md)。

**合法任务合同新根确认已跑完**：官方原始冻结权重、来源隔离的8根202647000..007，
显式goal admission、136词/128条真实分支、398540个CUDA predict样本，14项完整性
门全部通过。native dispatch/同库greedy/CEM均距0.032802/0.032761/0.024858，
对greedy为2胜/4负/2平，描述性CI跨零；6个模型witness中5个真实RGB非贪心，
合法目标根202647003仍反转。本批没有目标反射，也没有joint-repair模型，不能
冒称限位干预效能或BN修复泛化。所有根保留、独立后处理和两张图/manifest均已
核验，P1仍不开放，不能上升为理论否定。见
[合法目标新根确认](D2P_GOAL_JOINT_LIMIT_REPAIR_20261026.md)。

**原六根margin反转已有代码级因果定位**：原回执使用joint-repair而非官方原始
权重。固定全部六根、目标、动作、真实RGB与完整192维视觉坐标，官方原始排序
为6/6，joint-repair为4/6。保留joint-repair全部学习参数、仅恢复预测头的三个
BatchNorm运行buffer，同样恢复6/6；反向统计交换令官方变为5/6。FP64动力学
不改变排序。因此此前对这六根“代码问题已排除、冻结模型边界”的归因过宽。
三条repair trainer已默认固定base BN运行统计，并在主机CUDA真实训练样本的
实际训练循环中逐条验证：两种模式共6个optimizer step，新模式统计不漂移、
参数仍更新、梯度有限。全六根图与严格向量后处理已复核，相关广覆盖757项通过。
原两动作被正确判为贪心，纠正错误见证
不等于P1成功；training-free主线仍用官方原始权重，不借此打开P2/P3/P5。
见[预测头归一化根因与修复](D2P_PREDICTION_NORMALIZATION_REPAIR_20261026.md)。

**最新发现的代码根因**：目标事务将有限位wrist当成周期关节环绕，两个最严重根
202644006/202646001的目标分别+3.030815/+3.132241 rad，超出实际±160°范围。
这不是checkpoint可靠性结论，更不是理论失败。新增明确版本的限位合法目标
wrapper，全部16原始根/32条CUDA分支已完成，73728个predict样本、全部门通过。
两个坏根同样CEM设置的距离分别0.831375→0.075630、0.753711→0.086552，
其余14根图像/轨迹/动作/终态精确不变。保持完整偏移幅度、不删根，这是显式
新任务合同，不回填旧P1。最新广覆盖684项通过；其他合法目标排序反转仍待解决。
用户引用的原6根margin校准已另做精确上下文/目标元数据重放，全部目标合法；
原反转20274802/20274804不受此修复影响，不能将两个后续坏根修复冒称原问题全解。
见[目标关节拓扑修复](D2P_GOAL_JOINT_LIMIT_REPAIR_20261026.md)。

**独立确认及逐包反馈均已完成**：来源隔离8新根的136词/128条分支全门通过，
413302个predict样本；新dispatch/greedy均距0.176866/0.176877，描述性CI跨零，
4个witness中3个首段真实RGB非贪心。另8已知根逐包反馈共40次30步执行，
129480个predict样本，全部重放门通过；开环/冷反馈/carry反馈均距
0.152007/0.117004/0.135600，增加反馈未修复上述越界目标。最新广覆盖658项
通过，图和复现见[逐包反馈审计](D2P_PACKET_FEEDBACK_AUDIT_20261026.md)。

**新增全词评分/执行合同审计**：8根136词、128条主机CUDA真实分支全部完成；
旧优化器成本/轨迹/动作和历史完整终态精确复现，所有完整性门通过，实际predict
样本353414。统一后缀评分改变6/8个决定，但新dispatch均距0.128201，旧runtime
为0.124054；对同样runtime的greedy为3胜/3负/2平且CI跨零，6个新witness仅
2个真实RGB首段非贪心。此候选干预不充分解释失败，不接入默认部署，不宣称
P1或理论负结果。排除168个旧episode的新8根独立确认已在主机CUDA完成，
结果见上方最新条目。此原审计当时相关回归596项通过，见
[统一数值后缀合同](D2P_STATE_NATIVE_SUFFIX_CONTRACT_20261026.md)。

**同轮选择目标消融已完成**：取消非贪心优先，按全17词模型终点取最小，8根
32条主机CUDA分支全通过原基线/重复决定精确回放。3胜/1负/4平，但最难根
0.800530→0.919481、真实终态RGB也恶化，整体均距0.128201→0.136998。
不把它部署成修复或用于绕过P1；相关广覆盖617项通过，详细全根结果仍见上述页。

**同轮搜索覆盖审计已完成**：8根32条主机CUDA分支、804992个predict样本，
原完整CEM轨迹和native终态精确重放，全部门通过。固定128微批，将128×3/16elite
扩为1024×12/64elite，完整六包开环计划均距0.152007→0.123128（7胜/1负）；
同native前缀后缀搜索反而0.128201→0.142566（3胜/5负）。最难根仍未解决，
不能把模型目标下降等同物理修复。此为已知根联合搜索干预、非等资源、非逐包
CEM-MPC，不修改P1。最新广覆盖632项通过，证据及两张全根图仍见上述页。

**新增后缀盆地审计**：全部8根、相同prefix/相同初始动作的48条主机CUDA分支
已完成；原动作、模型预测、真实终态精确复现，command前向、Jacobian链式法则
与FD通过。根202643006局部距离1.476133，同模型CEM后缀为0.028642，说明该根
严重失败是可被搜索修复的数值盆地限制。单换box或多迭代未解决该根，模型成本
选多起点的全根均距为0.035700（原0.215253），但描述性CI跨零。8个新episode、
四类prefix、192条主机CUDA分支的独立确认亦完成，107432个predict样本、全部
完整性门通过。dispatch均距0.361358→0.124054（6胜/2负）；但修复greedy/CEM
分别为0.121340/0.123306，dispatch对二者区间均跨零。七个witness仍有一个首段
RGB反转。回归535项通过；部分数值问题修复不等于P1或研究闭环完成。
见[动作坐标与多起点修复](D2P_SUFFIX_ACTION_CHART_AND_MULTISTART_20261026.md)。

**最新代码因果修复**：已确认选前缀时的imagined suffix被丢弃、执行后从零重编译
会破坏部分完整计划。8旧根全136词的408条主机CUDA后缀干预全部完成且精确复现
原cold；cold/warm模型成本incumbent将均距0.458002降至0.124733。排除此前152个
episode的8新根/80条物理分支确认亦完成，均距0.467475→0.215253，对cold为3胜/
2负/3平，对同样修复greedy为5胜/0负/3平。完整CEM incumbent仍为0.039051，
五个witness仍有一个真实RGB首段反转。回归482项通过；不把部分代码修复写成正式
P1或C1--C3闭环。小样本CI敏感性与所有失败分支见
[计划连续性修复](D2P_SUFFIX_CONTINUITY_REPAIR_20261026.md)。

**此前两项主机CUDA检验**：24个既有任务/根单元的自身残差/Jacobian交换对照已
完成，旧残差、113个响应、决定及物理对照均精确复现；新诊断未带来修复，未接入
部署。另落地冻结Jacobian生成有限动作词，在8个新episode上测完17词和两类平坦
对照，共152条30步真实分支，四包前缀逐包更新RGB/动作历史。6个模型witness仅
3个保持真实RGB非贪心；dispatch均距0.458002，仍弱于完整CEM的0.120545。
全部词/顺序对保留，不调阈值、不选择性删根，旧反转、正式P1和研究闭环仍未解决。
详见[残差锚点与动作顺序实验](D2P_RESIDUAL_ANCHOR_AND_ACTION_WORDS_20261026.md)。

**本轮新增代码溯源**：32根四分支五档raw动作插值的640次主机CUDA执行全部
通过接口与精确回放检查，021/027首段反转仍保留。固定4096-episode、排除此前
144个episode的三帧RGB历史检索，与同库随机包的2048个候选单元对照已全部完成；
排序一致率95.51%→95.65%，配对区间跨零，不能宣称修复。旧近零动作与字面零
的微小残差仅改变其他两根，不解释原反转。另找到并修复P1 consumer信任旧true
标志/空gate的准入漏洞，以及P1→P2触零CI误放行；四份历史P0正例仍通过严格复核。
神经/编译/执行函数AST不变，最新相关回归403项通过。原P1和整套研究仍未闭环，
不把代码/协议修复、审计结束或checkpoint局部误差写成理论否定。

**最新选择器与物理机制审计**：40 根固定候选一致求解对照、完整128候选新8根
（与前120个episode隔离）及后继物理控制探针均已完成。新8根选出6个witness，
真实RGB首段非贪心6/6、qpos仅3/6；全根对同库greedy为3胜/3负/2平，均差
-0.011168、描述性CI跨零。逐候选一致求解并未确立稳定重配置优势，计算亦不匹配。
物理探针进一步发现：同一greedy目标残差下，仅2/6个witness在两个固定尺度均
改善有界后续响应；不能把更好的优化后终点直接命名为控制能力改变。24条完整
动作见证均精确回放。新模型侧共同目标响应资格的另一组新8根已完成（与此前128个
episode隔离）：0个witness、全部精确回退greedy，不能冒充P1修好。同根固定决定
因果重放已完成：对仅看终点的策略3胜/1负/4平，CI跨零，仍会拒绝一个真实好分支。
进一步发现新1000步box-QP在真实latent上未收敛（投影残差约1e-3），已实现带
误差证书的active-set QP，完成相同22个模型矩阵的主机CUDA复核：最大目标误差上界
4.285461e-13，但8根选择完全不变。因此该数值缺口不解释本批零放行，亦不解释历史P1。
新的certified三任务矩阵24单元及三组物理几何已全部完成：全驱动/弱肩/弱肘，
同一8个新来源episode、与此前136个隔离，逐任务FD/动作限制/完整CEM-MPC。
全驱动0 witness、
后者1 witness且实际RGB/qpos首段非贪心、同根终态优于greedy/CEM；弱肩对greedy
全根均差-0.003254，对CEM为4胜/4负，单根受益不是整体优势。弱肘0 witness、
全部greedy。唯一弱肩病例双尺度物理响应相对greedy更好，但不如zero，仅有部分
机制支持。三任务合并审计通过，不计作P1通过。另修复P0→P1触零CI误放行门：
严格负区间并绑定完整性gate；神经/执行函数AST未变。此漏洞不解释历史负结果。
最新相关逻辑回归358项通过。
40根审计中剔除的4个旧witness实际均不如greedy，旧两个首段RGB反转仍保留。

**新增因果证据，覆盖下文旧 CEM 判断**：固定 32 个根和相同首段，仅恢复 CEM
搜索中保存的尾段、无需更多搜索，两种 CEM 输出均 32/32 优于旧公共 DLS 后缀；
完整 CEM tail 对旧 dispatcher 为 30 胜 / 2 负。旧 CEM 对照其实搜索三段、只执行
首段再换 controller，不能当作完整 CEM-MPC 强基线。Dispatcher 同首段的实际 RGB
CEM 后缀也将均距从 0.257405 降到 0.033826（31 胜 / 1 负），但增加了计算。
这确认了一项执行协议问题；首段 RGB 排序、P1 和完整研究闭环仍待解决。
证据：`reacher_d2p_cem_execution_contract32_20261025_cuda.json`；详见下述审计页。
谱后缀修复的固定 1/2/4/8 次 SVD 对照也已完成：dispatcher 均距分别为
0.257405/0.183632/0.133556/0.101150，8 次对单次 29 胜 / 1 负 / 2 平，但尚不及
observed-CEM。所有首次迭代动作/终态精确重放，模型未训练，首段保持不变。
这是已验证的后缀求解不足，不是 P1 非贪心排序已解决；合并代码逻辑回归 283 项通过。
进一步将 coarse-stop 的最小非零步长从 1/16 扩展为下探到 1/65536（仅停步时尝试），
5/5 个 dispatcher 停步分支都找到了模型下降且真实终态改善，其他27根动作不变；
均距降到 0.083214，仍非 P1 pass。8 个新 episode 的三版本确认及真正 CEM-MPC
现已全部完成；与前 112 个 episode 隔离、完整 93 GB source SHA 重新通过。
Dispatcher single/coarse-8/fine-8 均距为 0.226112/0.118124/0.057419，fine-8 对
原单次 8 胜 / 0 负；真正 CEM-MPC 为 0.032064。唯一重配置根在 greedy 后缀充分
优化后失去原优势，因此这是后缀实现修复的新根支持，不是 P1 pass 或完整研究闭环。
该阶段 289 项代码逻辑回归通过；后续新增作业状态以本页开头为准。
另已完成官方/runtime RGB 交叉审计，不支持渲染迁移是官方动作预测误差的主要来源；
CEM 历史最优保留单独改善 20 根模型目标，却没有改善整体物理终态。以上均为主机
CUDA、原始冻结 checkpoint、已知根诊断，不是新效益或理论否定。

后文保留大量旧协议、旧 checkpoint 与中断记录，用于溯源，不代表这些判断仍是当前全局结论。
原提案 C1--C3、三组实验和 P0--P5 的完整缺口表见
[研究闭环审计](D2P_RESEARCH_CLOSURE_AUDIT_20261025.md)；该清单不降低原有研究范围。
Reacher 当前证据见 [原始 RGB 与代码修复审计](REACHER_D2P_SOURCE_NATIVE_RGB_AUDIT_20261025.md)：
原始冻结模型的 32-root RGB 排序复核已完成；继续发现并修复了 DLS 过冲接受、预算低报和
`mj_forward` 改写 snapshot warm-start 的问题。因此“所有代码解释已排除”不是已成立结论。
小阻尼下已观察到一个真实非贪心两段收益案例，但其整体 8-root 中性回退策略仍差于贪心。
独立来源 episode 的 32-root 完整确认已在精确状态修复后完成：6 根有 witness，5 根两段
终态优于两个对照，但 qpos 对 zero 的首段非贪心仅 1/6，全根 neutral fallback 仍劣于 greedy。
正式 P1、等预算强基线与完整
Planning Compiler 闭环仍未完成。不得把诊断成功或完成运行当作阶段通过。

最新继续审计：32/32 evaluator-target RGB 与绑定目标精确一致，self-goal encoder 代价均为零；
但实际观测的 RGB/qpos 局部几何仍有反转，不能混写两坐标。另把“无重配置 witness”与
“执行 neutral prefix”拆开，新增纯模型 progress fallback；旧 32-root 分支重放相对 greedy
均差由 +0.465292 变为 -0.033304，但这是开发后重放，不是新 efficacy。固定 32 个新根、
与此前 72 个 source episode 隔离的主机 CUDA 全根执行与 CEM 对照已完整完成，所有完整性
gate 通过。新 dispatcher 对 greedy 为 4 胜 / 3 负 / 25 平，均差 -0.013170，95% 描述性
CI [-0.048198,+0.020525] 跨零；对 zero 为 32 胜，对这份探索性 CEM 为 29 胜 / 3 负。
回退修复得到新根证据，但重配置的增量收益和计算效率均未确立；dispatcher 的 VJP 工作
与耗时明显高于两个对照，不能声称完整等预算优势。eligible 首段相对 zero 的真实 RGB/qpos
非贪心为 5/7，原 P1 仍不自动解锁。最新代码回归 125 项通过，完整细节见上述审计页。

进一步完成 128 个固定首段的主机 CUDA 同根回放：RGB/action/observed costs 精确复现，
仅有最大 4.77e-7 的 singleton/batch 模型分数差，不解释两个排序反转。只用过去已观测
transition 的 unit innovation 将 mean latent L2 降至 1.094625，但 eligible 排序仍为 5/7，
不采用其作为开门规则。审计函数单根 batch 校验漏洞已修复，但原实验未触发它。
另已完成 8 个新根的三种未来动作限制任务（与前 104 个 episode 隔离）：24 单元的动作、
FD 和重放完整性均通过，full/weak-shoulder/weak-elbow 的 witness 为 0/2/2，仍非正式 P1。
后缀置零消融的 72 条完整轨迹精确重现原分段终态；两例需要后缀控制才能反超 greedy，
另有受益例主要来自被动运动，不能算成控制能力改善。新排序误差分解表明，两个反转根
即便有 0.996 的响应余弦，目标投影误差仍可反转排序，future-zero oracle anchor 也修不好。
全 10 维、双尺度的 1,280 次真实动作干预已完成，未发现新增梯度实现错误证据；有限尺度
曲率、渲染量化和预测响应差异不混写成理论失败。最新相关逻辑回归 244 项通过。

## 研究对象与边界

本计划将 LeWorldModel 的后续研究从“对规划是否可信的诊断或拒绝”转为一个
training-free 的规划问题：冻结视觉世界模型的动作诱导几何应当决定规划器如何
搜索、组合动作并到达未来状态。暂定方法名为 **Spectral Reachability Atlas
(SRA)**。

给定当前 RGB 上下文、目标 RGB 与冻结的 action-conditioned world model，SRA 只
优化动作序列。它不训练或微调 encoder、predictor、adapter、policy、critic、
subgoal network 或 task readout。所有 learned-model 运行必须在主机 CUDA 上完成；
CUDA 不可用是 fail-closed 的环境阻塞，绝不能以 CPU 结果替代。

这条路线与旧的 Decision-Spectral / GDD-WM repair 证据链分开保存：旧链的
offline C1/C2/C3 审计不构成 SRA 的 Jacobian、MPC 或 RGB 闭环结果；反过来，SRA
也不能改写旧链的负结论。

## 从 Harness 到 Planning Compiler / Runtime

工程上的 world-model harness 可以负责连接 provider、候选生成、预测、评分与执行；
它本身不是本文的方法贡献。本文的研究对象是一个 **Planning Compiler**：在模型完全
冻结时，使用同一有限时域动力学对象把任务与当前 RGB 编译为带动作见证的临时控制程序。
暂定论文名为 *World Models as Planning Compilers: Synthesizing Action Programs from
Frozen Visual Dynamics*，方法名为 Dynamics-to-Programs（D2P）。

SRA 是 D2P 的局部动力学后端，而不是独立地以“对 latent 做 SVD”作为贡献。一个
program 至少包含动作 chunk、预测后继、局部 action response、horizon、预测执行代价
和它的生成动作见证。所有 program 仅是本次任务的临时数值对象，不会训练为 skill
library、macro-action model、policy、critic 或 subgoal generator。

| 编译产物 | 由冻结模型计算 | 不能替代的验证 |
|---|---|---|
| Goal-progress program | `G_H` 的 task-weighted SVD 与有界阻尼更新 | 与匹配预算的 CEM、GD、Gauss--Newton、GRASP 比较真实执行结果 |
| Reconfiguration program | 短 action chunk 之后重新计算的 `V_reach` 和可达子空间 | 动作短期不接近目标、却改善后续真实可控性和闭环任务结果 |
| Composition program | 带动作见证的 progress/reconfiguration chunk 串联并重规划 | 相比无组合 SRA 与普通 latent-subgoal 的额外闭环收益 |

这一区分避免将普通 harness、额外采样、一般 BPTT、任意 latent waypoint 或训练的
high-level proposal 误写成 D2P。D2P 的 kill experiment 是 reconfiguration program：
若它不能在物理任务中超过相同预算的非组合谱规划和强优化基线，项目应停留在规划优化
诊断，不宣称程序合成或新的 planning runtime。

## 核心机制

对长度为 `H` 的动作序列 `U`，在冻结模型预测终点处定义局部 action-to-latent
算子：

\[
\mathcal G_H(z_t,U)=\frac{\partial \hat z_{t+H}(U)}{\partial U}.
\]

在名义动作轨迹附近，`G_H` 的右奇异向量给出可独立操纵的动作程序组合，左奇异
向量给出它们在未来 latent 中能改变的方向。对任务度量 `Q` 和动作尺度 `R`，使用
`~G_H = Q^{1/2}G_HR^{-1/2}` 的局部 SVD。第一版固定 `Q=I`，不把任何训练的
readout 伪装成任务几何。

若终点残差为 `r=Q^{1/2}(z_g-\hat z_{t+H})`，局部阻尼最小二乘值为：

\[
V_{reach}(z,g)=\|r_\perp\|^2+
\sum_i\frac{\lambda}{\sigma_i^2+\lambda}(p_i^\top r)^2.
\]

该量是局部、动作约束忽略时的解析 surrogate；它不是全局可达性证明，也不是新的
控制理论定理。方法贡献只有在它能选择有利于后续可达性的状态、并在真实 RGB
闭环中改善任务结果时才成立。

对于离散动作对，SRA 使用实际 frozen rollout 的顺序效应
`c_ij(z)=F(F(z,a_i),a_j)-F(F(z,a_j),a_i)` 产生 Action-Order Primitives。它是
离散 action-order effect，不能被误称为线性矩阵 commutator；在光滑连续系统中才
可另行讨论与 Lie bracket 的近似关系。

## 先排代码与协议，再解释负结果

每个阶段必须先完成以下排查。任一项失败都归类为实现、环境或协议失败，不作为
理论反例：

1. CUDA 真分配、GPU 名称、Torch/CUDA 版本和运行命令完整记录；不得 fallback。
2. 输入 RGB/proprio、action normalization、checkpoint/config SHA 和 root split
   全部绑定到产物。
3. 模型参数 `requires_grad=False`，运行前后参数哈希一致；优化器只能持有动作。
4. 对 action Jacobian 检查非零、有限，并与中心有限差分一致；不从 encoder 的
   `no_grad` 误推断 action path 不可微。
5. 候选动作的 shape、time axis、history padding 和 action-order 逐项审计；现有
   PointMaze wrapper 的历史 action padding 是已知高风险路径。
6. 目标、超参数、chart 搜索和停止准则只能由 calibration roots 冻结；outcome roots
   只做一次性评估。
7. 真实成功/代价由环境执行得到；world-model latent 目标仅可作 planner objective，
   不可替代 physical label。若 raw `states/actions` 不能被 evaluator 逐状态回放，
   evaluator 标签一律失效，不能把该失败误写成方法负结果。

## 阶段和解锁门槛

| 阶段 | 可执行对象 | 通过标准 | 失败解释与后续 |
|---|---|---|---|
| P0 | PointMaze 已发布 checkpoint 的 action Jacobian；CEM、vanilla action-GD、透明的 gradient-and-restart comparator | CUDA、冻结哈希、非零 Jacobian、中心差分一致、action-order effect 均通过 | 先修 action layout/rollout/normalization/checkpoint 问题；P0 未通过时 P1--P5 全部阻塞 |
| P1 | Spectral Cost-to-Go 与 latent-distance 的 root-disjoint 中间状态选择 | 候选选择不读取 outcome；单独的真实 successor label 支持比较 | 只报告中间 surrogate 不报告在线收益；未胜出是方法负结果而非模型训练失败 |
| P2 | Action-Order Primitives、多个 local reachability charts 的搜索 | 对照动作顺序确有非交换预测效应；chart transition 用冻结选择规则 | 禁止使用矩阵 commutator 或 privileged state 充当 primitive |
| P3 | horizon-dependent spectral expansion | 长时域 gain 与 forward/backward 预算、latency 一并报告 | 若更长 horizon 只增噪声/成本，记录为尺度边界 |
| P4 | PointMaze、Reacher、PushT RGB-only receding-horizon 闭环 | 同一 checkpoint、匹配优化预算、根级配对统计、环境 success/cost | 每个环境独立结论；PushT 接触失败需分解为接近、接触、推送、闭合，而非笼统负例 |
| P5 | 其他冻结模型和任务类别 | 预先声明的跨架构复制且保持训练-free 约束 | 不将单一 checkpoint 的结果概括为通用世界模型定理 |

P1--P5 不是“写好代码就完成”：它们分别以前一阶段的硬门槛和真实环境标签为前提。
在前提不具备时状态应为 `blocked_upstream`，不能以合成数据、adapter 结果、CPU
fixture 或 privileged oracle 伪造完成。

### D2P 的下一条可启动路线：Reacher admission track

PointMaze 的旧 SRA P1/P2 负结果已被 script-identity repair 后的独立 panels 取代：
P1 的方向不稳定，P2 没有稳定 primitive 信号。因而仍不能把同一、尚未稳定的 mechanism
换名为 D2P 后重跑。D2P 的第一个新实验必须转移到独立的冻结 Reacher checkpoint；它有
RGB-only live evaluator、固定的 action normalization、真实 episode success/qpos-distance
labels 和主机 CUDA 路径。现有 40-episode paired CEM 结果只说明该 checkpoint 存在一个
可运行的 frozen-model baseline，**不是** D2P 结果，也不得用于选择 D2P 参数。

该 admission track 的冻结协议如下：

| 对象 | 固定内容 | 禁止的替代 |
|---|---|---|
| Backbone | `/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm`；encoder/predictor 全程冻结、参数 hash 前后相同 | GDD teacher、adapter、readout 或任何后训练权重 |
| Runtime input | 当前 RGB、评估器渲染的 goal RGB、历史和 action；`qpos/qvel` 只允许计算环境 outcome | 将 target state、simulator state 或物理标签注入 policy/program generator |
| D2P program | 由当前 rollout 的 action Jacobian/SVD 生成的 bounded action chunk、action witness、predicted successor、horizon 与 compilation receipt | 固定 pulse/recovery bank、learned macro-action、latent waypoint、事后选出的候选 |
| Comparators | 匹配 model-call/backward/candidate budget 的 CEM、action-GD、Gauss--Newton、GRASP-style 和无组合 SRA | 仅与 random 或额外算力的搜索器比较 |
| Evaluation | root-disjoint calibration/outcome seeds；paired live success、final/min qpos distance、wall time、model calls 与 backward calls | 用 latent improvement、elite rank 或 evaluator-only surrogate 替代真实 outcome |

Reacher D2P-P0 的唯一目标是检验 **goal-progress program** 的实现和短时物理后果；其
闭式更新与同一线性化点、同一阻尼和同一约束下的 local Gauss--Newton 完全等价。因此，
把“击败 Gauss--Newton”写成通过条件会构成无效比较：P0 必须先验证两者产生相同动作，
再与匹配模型调用/反传预算的 CEM 与 vanilla action-GD 比较真实 outcome。它只在
action-to-latent Jacobian 的非零/有限、中心有限差分一致、真实历史 action prefix 被
绑定、动作 shape 正确、模型 hash 不变和 fresh paired live evaluation 全部通过时产生
结论。若只得到低秩动作参数化收益，论文只能称为 planning optimization。只有在 P0 通过后，才允许启动
reconfiguration P1；P1 必须额外证明短期不贪心 action chunk 会在重新编译后改善真实
后续可控性，并超过无组合 SRA 与强优化器。PushT 仍不是该路线的首个 kill experiment，
因为 contact-state observability 与 runtime successor verification 尚未闭合。

## 当前实现与产物约定

`le-wm-main/experiments/spectral_reachability/run_sra_p0_pointmaze_cuda.py` 是 P0 的
CUDA-only 可审计入口，配套 launcher 为
`le-wm-main/experiments/spectral_reachability/run_sra_p0_pointmaze_cuda.sh`。它以真实
PointMaze RGB history、发布 checkpoint 及其原生 preprocessing 运行；记录 checkpoint
hash、参数前后 hash、Jacobian sketch、finite-difference 对照和三个 optimizer 的同
目标结果。其 gradient-and-restart comparator 必须明确标为 GRASP-style，直到获得
可审计的外部 GRASP 实现和匹配配置，不能称为逐实现复现。

P0 只建立“冻结模型的动作路径可用”的工程事实，不建立 SRA 有效性、物理预测正确性
或闭环成功的科学结论。

### 已执行的代码审计与 CUDA 证据（2026-10-23--24）

- **P0：通过工程门槛。** 在 RTX 5060 Ti / CUDA 13.0 上，四个独立 PointMaze roots
  的 Jacobian sketch Frobenius norm 为 `0.691--1.241`；中心有限差分中位相对误差为
  `0.0028--0.0110`；动作反序的 latent effect 为 `1.270--3.897`。checkpoint
  与模型参数 hash 在运行前后相同。回执为
  `le-wm-main/results/sra_p0_pointmaze_20261023_cuda/report.json`。
- **旧 P1/P2 输出撤回。** 初版 evaluator 将一个 archive action 错执行为一次积分，
  同时遗漏 `MazeEnv.step` 的动作前速度截断；XML 还遗漏了 ground collision mask。
  这些是代码/协议错误，故 v1/v2 结果不能作为方法负结果。v3 已被带 backend
  provenance 的 v4 结果替代。

#### Raw-transition 回放合同

CUDA-gated 审计固定 20 个 roots、9 个帧、`+5/+10` horizons，共 360 个原始
`states/actions` 转移。修复后的高层 replay 始终先截断速度到 `[-5,5]`，再对每个
动作执行 5 个 MuJoCo steps。

- MuJoCo 3.1.6 无法满足该合同，最大 state L2 error 为 `1.17553`，所以它不能为
  P1/P2 产生标签。
- 隔离的 MuJoCo 2.2.0 backend 满足 **position-only** 合同：最大 position L2 为
  `2.169257e-6 < 3e-6`，但完整状态最大 L2 为 `2.387018e-5 > 2e-6`。因此绝不声称
  full-state equivalence。
- 为排除“2.2.0 版本本身造成标签”的解释，已从镜像恢复 MuJoCo 2.1.2 与
  `mujoco_py==2.1.2.14`，在同一主机 CUDA gate 下复跑完全相同的 360 个转移。其
  position/state maxima 和均值逐项等于 2.2.0 的结果；官方 runtime 二进制 SHA-256 为
  `1ebd4a5f6126cf440dad2fb641bb54e905799eaf90b093eb04111f6ca32f0fcd`。该复现仍只
  支持 position labels，并不证明 archive 的原始生成 runtime 已被确定。

对应回执为
`results/sra_pointmaze_replay_mujoco220_position_contract_20261023_cuda/report.json` 和
`results/sra_pointmaze_replay_mujoco212_position_contract_20261024_cuda/report.json`。
它们是 host-CUDA-gated 的代码/环境合同证据，不是 learned-model CPU 替代实验。

### 当前阶段结论（以 provenance-bound 回执为准）

#### 当前 P 阶段分支账本（Reacher 更新至 2026-10-25，其余按既有回执）

本项目中存在名称相同、但不能互相替代的 P 阶段序列。后续命令必须先选择正确分支；不得把一个
分支的成功、失败或现成 runner 迁移为另一个分支的证据。

| 分支 | 已验证到的边界 | 当前未完成阶段 | 为什么不能直接继续 |
|---|---|---|---|
| PointMaze-SRA | P0--P3 的输入/物理标签审计通过；P4 的 fresh RGB-only closed-loop panel 已封存 | P5 跨 checkpoint/任务复制 | P4 只覆盖一个冻结 PointMaze checkpoint、生成但合同已审计的新 root distribution、固定五臂 bank 与四个 model blocks；它不自动构成 Reacher、PushT 或跨架构结果。 |
| Reacher-D2P | 历史 48-root P0 与 model-only preflight 保留；原始冻结模型 source-native 32-root RGB 排序复核完整完成；DLS 接受规则、完整状态 warm-start 与预算记录已修复，另完成新 dispatcher 的 32-root CUDA 诊断 | 新合同的正式 P1 preflight / calibration / outcome、等资源强基线、P2--P5 | 不能再声称所有代码解释已排除。新 dispatcher 7 个 witness 的真实 RGB/qpos 对 zero 非贪心为 5/7；对 greedy 的全根均差 -0.013170、CI 跨零，且计算成本明显更高。回退拖累已得到同根分支证据，重配置增量和 controllability-gap 任务前提仍未确立。历史 P0/preflight 不可跨新 solver/context SHA 复用；诊断成功不自动解锁 P2。 |
| TwoRoom-D2P | source-supported P0、root-anchored P1 与同可行域 comparator 的 CUDA/replay 合同均通过 | P1 mechanism calibration，以及 P2/P5 | 旧 raw-zero 机制门曾把 source-supported prefix 与 OOD control 作正式比较，现已由 sealed same-bank greedy 修复；修复后的 8-root calibration 仍有 3/8 physical order reversals，且 24-branch exact audit 排除了 packet recovery、五步执行、successor RGB、原生 rollout、目标语义和 root context 错误。它是该冻结 checkpoint 的 model-to-physical ranking boundary，不是 D2P 或谱理论的一般反例。 |
| PointMaze effect-adapter/HTSC | strict RGB closed-loop panels 已完成 | 不是 SRA 的 P4 证据 | 它的 `visual_dynamics` 通过 calibration-only latent-to-position readout 评分，不执行 SRA 的 action-Jacobian/spectral chart。它可作为独立基线/工程证据，不能替代 SRA 的机制检验。 |
| PushT-D2P / 跨模型 | time-major action packet 与 rollout bootstrap 对齐修复后，PushT P0 通过、P1 24-root outcome fail-closed | PushT P2--P5；全项目 P5 独立冻结模型复制 | PushT 的 23 个模型 witness 有 5 个在真实物理 prefix 上翻转为更贪心；CUDA、same-root zero、successor recompile、independent replay 与 root-disjoint gates 均通过。现有 TwoRoom/Cube adapter/teacher 结果不满足 training-free D2P，不能充作 P5。 |

该账本的作用是区分**尚缺实现/协议**、**已实现但尚无足够新合同证据**与**已经测试但不满足
机制门**。PointMaze 的 P4 已有受限分支证据，不能再将其 controller 笼统写作未实现；
Reacher 的新诊断也不能当作正式 P1 通过。以上均不构成谱理论或世界模型理论的一般反例。

#### Reacher-D2P complete-state/preflight 合同修复与 fresh P0/P1（2026-10-25）

对仍为负的 Reacher receipt 再次做 runtime 审计后，执行 snapshot 补齐了 MuJoCo complete
state 与 model geometry 的分离保存；在为了重渲染 goal RGB 临时改写状态后，必须重新恢复
完整 state，而不能只恢复 qpos/qvel。这一修复有 unit-test contract，且不会将 evaluator qpos
暴露给 planner。P1 preflight 的 witness threshold 也已从旧 runner 的隐式默认值改为显式绑定。

同一审计还发现 P1 preflight 的 witness threshold 可在 preflight CLI 中声明、但旧 P1 runner
硬编码默认值，导致更强 preflight 不能被其消费。runner 现显式绑定
`min_model_witness_roots` 于 preflight、calibration 和 outcome receipt；这避免弱化或混淆
preflight 门槛。

在上述输入与重放合同下，主机 CUDA 的 fresh 48-root P0 outcome
`results/d2p_reacher_p0_goalmarker_confirmation_n48_20261025_cuda/report.json` 中，
D2P goal-progress minus CEM 的 final-qpos-distance mean 为 `-0.270831`，paired bootstrap
95% CI 为 `[-0.437139, -0.124588]`（40/48 D2P wins）。CUDA、冻结 hash、相同 planner/
execution root RGB、physical action box 和 D2P=local-Gauss--Newton gates 全部通过。因此旧
Reacher P0 negative receipts 仅保留为发现过期输入/状态合同缺陷的诊断，不能作为当前 P1 的
上游否定。

P1 使用新的 root-disjoint preflight（4/4 model witness；所有 finite-difference、RGB root
identity 与 future-score gates 通过）后，在 6 个 calibration roots 上得到 6/6 model-side
reconfiguration witnesses。然而同根 raw-zero physical control 显示仅 3/6 witness 是真实
non-greedy。独立 CUDA first-chunk fidelity audit 同时确认 raw/normalized packet inversion、
warmup 与 successor replay 均精确，却只有 3/6 candidate-vs-zero model goal-cost orders 与
实际 successor RGB encoder orders 一致。权威 receipts 为
`results/d2p_reacher_p1_goalmarker_contract_preflight_n4_20261025_cuda/report.json`、
`results/d2p_reacher_p1_goalmarker_qpos_calibration_n6_20261025_cuda/report.json` 与同目录
`first_chunk_fidelity.json`。

为避免把 renderer 假设误记作机制解释，新增的 host-CUDA source RGB contract audit
`results/reacher_d2p_source_rgb_contract_v2_20261025_cuda/report.json` 从官方 HDF5 的 24 个
`qpos/qvel/target_pos` roots 出发，仅把 target geometry 在 source 位置与当前 fingertip 位置间
交换。每个样本的两张 runtime RGB 完全相同（最大 mean-absolute pixel error 为 0），说明该
environment 的 target 默认透明：marker mutation 不会进入冻结模型的像素输入。独立 source
temporal audit 则确认声明的 packet shift `0` 是严格最优（mean latent MSE `0.0381412`，shift
`-1/+1` 分别为 `0.751267/0.737448`），故一 packet 时间错位也不是此失败的代码解释。

因此此 checkpoint 的 Reacher P1 outcome 继续 fail-closed：不得通过改变 margin、改用 outcome
qpos 或重复 calibration roots 绕过。这个结果把当前缺口精确限定为**冻结 checkpoint 对
action-conditioned one-token successor order 的可靠性**；它既不否定 P0 的 D2P goal-progress
结果，也不构成“谱/D2P 理论失败”。

#### Reacher-D2P action-normalization contract repair（2026-10-24）

对 Reacher 的“冻结 runtime receipt”追溯到
`audit_frozen_cross_environment_prediction.py:_action_stats` 后，发现它将每个五步 action
block 先拼接成 10 维，再对 10 个位置分别拟合均值和标准差。原始训练入口则调用
`get_column_normalizer(dataset, "action", "action")`：它先在 HDF5 的二维物理 action
列上拟合 z-score，随后 `HDF5Dataset` 才把五个连续 action 按 time-major 次序拼成 10
维 token。二者不是等价的坐标变换。

公开 `quentinll/lewm-reacher` 训练 HDF5 的二维统计为 mean
`[3.57e-05, -5.06e-04]`、unbiased std `[0.57741, 0.57718]`，因此正确的 10 维
receipt 必须严格重复该 pair 五次。旧 runtime receipt 的最大 mean 偏差为 `0.10783`，
最大 std 偏差为 `0.03522`，并且五个 packet slot 并不相等。它足以让同一物理 action
在训练与 P0/P1 runtime 中进入 action encoder 的数值坐标不同。

`build_reacher_d2p_training_action_stats.py` 现在从已验证的官方 HDF5 manifest 生成
不可覆盖 receipt；`train_reacher_d2p_dynamics_repair.py` 将训练源的二维统计按五步重复，
并且对任意 position-specific receipt fail closed。官方 CUDA pipeline 的训练、P0、P1
preflight、calibration 和 outcome 统一使用该 receipt。先前使用旧 receipt 的 Reacher
P0/P1 数值（包括“冻结模型到真实 RGB successor 的排序失配”）全部降级为
`superseded_protocol`：它们不能再支持模型理论负结论，也不能用于选择新实验参数。

同时，官方 archive 在 `num_steps=4, frameskip=5` 下产生 1,820,000 个 clip；旧修复
trainer 的 12 个全 sweep 需要 682,500 个 batch，且没有中间 checkpoint，容易被误读为
模型没有进展。新 trainer 记录固定的随机 source batch budget（每 epoch 512 train / 128
validation）；它只用于选择 training-source checkpoint，后续 scientific admission 仍由
fresh、root-disjoint、host-CUDA 的 P0/P1 receipts 决定。

#### PointMaze-SRA P4 fresh-archive repair（2026-10-24）

在为 RGB-only P4 补充新的 roots 前，对通用 extension generator 的审计发现了两个会制造
伪负结果的实现缺陷：它把一个记录的高层 `MazeEnv.step` action 当成单个 MuJoCo tick，且将
原始 `float64` state/action 静默写成 `float32`。前者遗漏了 action 前的速度截断和每 action
五个 integrator ticks；后者使从量化初态重放 99 个 action 后的终态最大偏差达到 `4.26378`。
因此第一次生成的 `data/point_maze_sra_p4_hlcontract_20261024` 及其
`results/sra_pointmaze_p4_hlcontract_archive_transition_audit_20261024.json` 是失败的
**数据合同审计**，不得用于任何 P4 方法结论。

`extend_pointmaze_state_action_pool.py` 现与
`pointmaze_physical_evaluator.py` 共用高层转移合同（速度限幅 5、raw action 限幅
`[-1,1]`、每 action 五 ticks），保留 `float64`，并在 manifest 绑定 evaluator/XML hash 与
MuJoCo version。新的不可变 archive
`data/point_maze_sra_p4_hlcontract_float64_20261024` 提供 roots `2000--2207`；逐 root
重放审计的最大终态误差为 `0.0`，receipt 为
`results/sra_pointmaze_p4_hlcontract_float64_transition_audit_20261024.json`。以历史 58 份
receipt 进行的容量审计确认 208/208 个新 RGB roots 未被消耗：
`results/sra_pointmaze_p4_fresh_root_capacity_20261024.json`。它们的 RGB 在只读取旧
roots `1896--1899` 作 renderer calibration 的条件下单独 materialize 到
`data/point_maze_sra_p4_hlcontract_rgb_float64_20261024`；不读取 planning outcome。

随后直接在主机 CUDA（非 sandbox/非 CPU fallback）上，以该 fresh RGB archive 的
roots `2000--2003` 运行冻结 PointMaze P0 preflight。CUDA、冻结参数 hash、四个非零
Jacobian、中心有限差分（最大 root-median `0.002743`）和 action-order effect 均通过，receipt
为 `results/sra_p4_fresh_rgb_p0_preflight_20261024_cuda/report.json`。这只解除 P4 的
**数据与模型输入前置条件**；RGB-proprio proxy、matched-budget receding-horizon SRA
controller、latency accounting 和 root-disjoint outcome panel 仍必须在此 archive 上单独实现
与验证，不能由 P0 admission 或其他 effect-adapter controller 替代。

#### PointMaze-SRA P4 RGB-only closed-loop closure（2026-10-24）

`LegacyPointMazeRuntime.step_block()` 还曾遗漏 archive 合同要求的 action 前速度限幅和 raw-control
clip；现已与 `replay_recorded_actions()` 逐 action 终态等价。P4 controller 每次只使用当前
renderer RGB、goal RGB 和由**校准 roots** RGB marker 得到的 proprio proxy；simulator state 只用于
reset、stepping 和 evaluator label。spectral 与 immediate 两臂都在同一五臂 bank 的 25 个两步
frozen-model rollouts 上打分：前者取最优 continuation cost，后者从相同两步 tensor 读取首
successor cost；两臂固定执行四个 model blocks，因此逐 root candidate budget 相同且记录 latency。

首个 r2 144-root run 没有绑定 P4 runner SHA-256，故只保留为诊断。修复后 r3 receipt 同时绑定
runner、runtime、RGB calibration manifest 和 state-extension manifest。r3 roots `2416--2479` 只用于
RGB/proxy calibration，roots `2480--2623` 为 144 个 root-disjoint outcome episodes；transition 和
fresh-capacity audit 均通过。主机 RTX 5060 Ti / CUDA 13.0 的 receipt
`results/sra_p4_rgb_closed_loop_r3_full_20261024_cuda/report.json` 通过 CUDA、冻结参数、输入
privacy、根分离、finite outcomes、文件 identity 和每 root 匹配 25×4 two-step rollouts gates。
reducer `results/sra_p4_rgb_closed_loop_r3_full_summary_20261024.json` 给出
spectral-minus-immediate final distance `-0.467328`，95% CI `[-0.571319,-0.366668]`，two-sided
sign-flip `p=1.0e-5`；minimum distance 为 `-0.285013`，95% CI `[-0.354026,-0.218409]`；success
为 24/144 对 10/144。因此 **PointMaze-SRA P4** 在此封存协议下闭合：two-block reachable-set
selector 超过相同计算预算的 immediate selector。该结论不泛化到 Reacher、PushT、其他 checkpoint、
训练分布外状态或 P5。为防止 r3 roots 被以后续 panel 重用，post-outcome capacity receipt
`results/sra_pointmaze_p4_r3_post_outcome_capacity_20261024.json` 标记该 archive 的 208/208 RGB
roots 都已被 calibration 或 outcome provenance 消耗。

| 阶段 | 代码/协议状态 | 科学状态 | 可用结论 |
|---|---|---|---|
| P0 | 通过：修复 archive-to-encoder 输入契约后，CUDA、发布 checkpoint、action shape、有限差分、参数冻结均已重新核验 | 工程前提成立 | checkpoint 有可微、非零 finite-horizon action response；不是闭环收益证据 |
| P1 | 通过：receipt binding、动作归一化、batch/time axis、two-call tree、chart composition，且 archive RGB/proprio 只在 encoder 内各标准化一次 | **两个独立面板均支持 reachable-set selector** | spectral-minus-immediate regret 为 `-0.095590`、`-0.053619`；合并 `-0.074604`，但小样本 pooled CI `[-0.352513,+0.221041]`，故只构成 fixed-bank mechanism admission，而非闭环效应量结论 |
| P2 | 通过：receipt binding、动作支持范围、batch/time axis、目标投影以及 archive-to-encoder 输入契约均已重新核验 | **有正的 goal-directed action-order signal** | 合并 global Spearman `+0.288175`；root-mean `+0.318095`，95% CI `[+0.112381,+0.512381]` |
| P3 | P0--P2 的修复后机制门槛已通过；两组 root-disjoint 的 fixed-tree receipt 及 batch-invariance 审计已完成 | `fixed_tree_boundary_mapped` | H2 有稳定的 fixed-tree improvement；H3/H4 仍不确定，且绝不得将该诊断写成 RGB 闭环收益 |
| P4 | 通过（仅 PointMaze-SRA branch）：fresh RGB archive、RGB-proprio proxy、matched 25×4 two-step budget、latency 和 root-paired live outcomes 均绑定并通过 | `pointmaze_closed_loop_supported` | spectral 相对 immediate 的 final/minimum distance 均为负且区间不跨零；结论限于该冻结 checkpoint/新 archive/四 block protocol |
| P5 | 尚未执行 | `pending_independent_frozen_model_or_task` | 不将单 checkpoint、单一生成 root distribution 的 P4 结果泛化为跨模型结论；Reacher-D2P 的独立上游 P1 仍关闭 |

旧 P1/P2 回执
`results/sra_p1_pointmaze_v4_bound_mujoco220_position_contract_20261023_cuda/report.json`
与
`results/sra_p2_action_order_v4_bound_mujoco220_position_contract_20261023_cuda/report.json`
保留为首次发现代码/环境问题的历史记录；它们未记录运行时 endpoint wrapper 与 replay
backend 的脚本 identity，不能继续承担稳定效果结论。

#### PointMaze script-identity repair 与独立 panels（2026-10-24）

`audit_sra_pointmaze_replay_legacy_backend_cuda.py` 现在把 replay backend 和 candidate
endpoint wrapper 的 SHA-256 写入 position contract；`pointmaze_physical_evaluator.py`
在每次 P1/P2 前比对当前两个脚本的 SHA-256，不一致即 fail closed。起初的 fresh roots
`1864--1879` 中 root `1877` 超过预注册 `3e-6` position threshold（`3.9036e-6`），
因此该 panel 没有被用于 learned-model 评价。随后只按 raw-transition replay error、而非
model score 或 outcome，筛选未使用 roots `1800--1863`；screen 的最大 position error 为
`2.5044e-6`，且绑定了两个 evaluator 脚本 hash。回执为
`results/sra_pointmaze_replay_mujoco220_identity_screen_20261024_cuda/report.json`。

在此合同下，P1 的 `1808--1815` panel 暂时偏向 spectral，而完全独立的 `1816--1823`
panel 偏向 immediate baseline；P2 的 `1800--1807` 与 `1824--1831` panels 都不呈现
正的 predicted-to-physical action-order relationship。它们只绑定了运行时脚本文件，**未**
把每次 endpoint response 与其 request/XML 收据绑定，因此保留为发现 provenance 漏洞的
历史证据，不再承担效果结论。

#### PointMaze endpoint-response receipt repair 与新独立 panels（2026-10-24）

第二轮代码审计发现，仅在 position contract 中记录脚本哈希不能排除 endpoint 输出被交换、
来自另一份 XML 或另一请求的可能。`evaluate_sra_pointmaze_candidate_endpoints.py` 现将
`request_sha256`、`xml_sha256`、endpoint wrapper 与 replay backend 的 SHA-256 一并写入
每个 response；`pointmaze_physical_evaluator.py` 在读取 endpoints 前逐项重算，任一不符即
fail closed。相应的回归与编译检查为 14 passed；该修复发生在学习式面板之前。

新的 host-CUDA-gated MuJoCo 2.2 raw-transition 合同在 roots `1832--1863` 的 576 个
转移上通过 position-only 阈值：最大 position L2 为 `2.504383e-6 < 3e-6`，而最大 state
L2 为 `1.102916e-5`，所以结论仍严格限于 position labels。合同与当前 response scripts 的
哈希写入 `results/sra_pointmaze_replay_mujoco220_receipt_contract_20261024_cuda/report.json`。
可复跑入口为
`le-wm-main/experiments/spectral_reachability/run_sra_pointmaze_receipt_contract_cuda.sh`；它以
模块方式调用、固定该 root panel，并通过 `run_with_torch_cudnn.sh` 先执行 host-CUDA gate，
不会因文件路径导入失败或 CPU fallback 产生表面成功。

在同一不可变合同下，P1 使用 root-disjoint panels `1832--1839`、`1848--1855`，每个 panel
均通过 CUDA、frozen hash、nonzero finite Jacobian 和 outcome-blind selection gates，但其
spectral-minus-immediate regret 分别为 `+0.062032`、`+0.214093`。P2 使用另外两个
root-disjoint panels `1840--1847`、`1856--1863`，其 global Spearman 分别为 `+0.058185`、
`-0.138406`。最终 reducer 同时拒绝 root overlap、混合 evaluator contracts 和任一实现门槛
失败；汇总回执为 `results/sra_pointmaze_receipt_contract_cross_panel_v2_20261024_cuda/report.json`，
其 P3 gate 为 `false`。

该轮结果关闭了当时已知的 endpoint provenance 替代解释，但未覆盖后续发现的 P1 可行域与
P2 目标指标错配；因此其数值保留为历史 receipt，而不是当前机制结论。

#### P1/P2 可行域与目标指标修复（2026-10-24）

对上述 receipt-bound 面板做进一步代码级对齐审计时，动作归一化往返误差为
`7.96e-8`，archive 第一 block 的 position replay error 为 `1.20e-7`，P1 的两次物理
replay 与直接十动作 replay 的最大差为 `2.78e-17`，P1 chart composition 与直接 two-step
unroll 完全相同；P1/P2 的 CUDA batch--serial latent 最大差分别仅为 `1.91e-6` 与
`9.54e-7`。这排除了 action coordinates、batch order、time axis 与 two-call evaluator
作为结果来源。回执为 `results/sra_pointmaze_planner_alignment_v2_20261024_cuda/report.json`。

但审计也揭示 P1 原始 analytic DLS 代理的实质性协议缺陷：48/48 chart 的连续解都触发了
action box，而真实 evaluator 评价的是固定的 17-arm 离散 continuation bank。P1 因此增加
`discrete_reachable_set` scorer：Jacobian/SVD 仍作为行动响应的工程门槛，但选择每个第一
block 时，冻结模型直接在**同一** 17-arm bank 上预测最优 continuation；该过程不读 physical
outcome。新的 raw-transition contract 覆盖 roots `1880--1895`（最大 position L2
`2.169257e-6`），两个 root-disjoint CUDA 面板的结果仍未给出正向证据。对应合同与 panels
在 `results/sra_pointmaze_replay_mujoco220_receipt_contract_v2_20261024_cuda/` 与
`results/sra_p1_pointmaze_discrete_reachable_set_panel_{a,b}_20261024_cuda/`。

P2 的 `±0.8` actions 没有离开 archive 支持范围（每一维约 20% archive actions 的绝对值
至少为 0.8），且无目标 latent-L2 对比与物理 position-L2 的量纲不一致。P2b 将其改为由
记录 RGB goal 编码出的 latent goal direction 上的 swap-order effect；physical goal 只在冻结
后作为同构标签使用。roots `1864--1879` 中的 `1877` 已在 raw replay screen 以预先固定的
阈值排除，余下 15 roots 的新合同最大 position L2 为 `1.107285e-6`。两套 P2b panels 仍无
正向相关，故不把该修复写成成功或再进行 post-hoc metric sweep。

上述汇总随后被 archive-to-encoder 输入修复取代，详见下一节；它保留为发现此前已知
action/evaluator 协议问题的历史回执，不能继续作为方法结论。

#### Archive-to-encoder 双重预处理修复与 P1/P2 重跑（2026-10-24）

在继续把 P1/P2 的负向结果归因于方法之前，审计覆盖了此前遗漏的**观测输入契约**。
`PointMazeDataset.get_frames()` 已将 RGB 转为 `[0,1]`，并已将 proprio z-score；而发布的
`EncPredWM.encode()` 规定输入为 raw `[0,255]` RGB 与 raw proprio，随后自行执行 `/255`、视觉
transform 和 proprio normalization。旧 SRA 的公共 `make_root_context()` 将 archive tensor
直接送入该接口，于是 RGB 被缩小两次、proprio 被标准化两次，而 action suffix 仍仅标准化一次。
这是实现错误，所有依赖旧 context 的 P0/P1/P2 数值均撤回为历史诊断，不能表示方法负结果。

修复将反变换集中在
`experiments/spectral_reachability/run_sra_p0_pointmaze_cuda.py::prepare_archive_observation_for_model`：
视觉值在进入 encoder 前恢复至 `[0,255]`，proprio 恢复原始坐标；已标准化 action packet 不变。
两个回归测试锁定此单次-preprocessing 合同。新的 host-CUDA target audit 以 recorded actions
核对 packet 和目标帧：一段（五个 raw actions）目标为 `t+5`，两段目标为 `t+10`；direct two-packet
和 sequential chart rollout 的最大差为 `0.0`。修复后，正确目标帧分别在 4/4 与 3/4 roots 成为
`±1` 帧比较中最近的 encoded RGB target；修复前对应仅为 1/4 与 0/4。回执为
`results/sra_pointmaze_rollout_target_alignment_fixinput_20261024_cuda/report.json`。

修复后的 P0 重跑在 roots `1896--1899` 上通过 CUDA、frozen hash、非零 Jacobian、中心有限差分
和 action-order gates（Jacobian Frobenius norms `2.171--7.328`；median FD relative error
`0.000558--0.001612`）：
`results/sra_p0_pointmaze_fixinput_20261024_cuda/report.json`。独立 planner/physical alignment
审计也全部通过，包含 action round-trip `9.28e-8`、chart composition `0.0`、P1 batch/serial
`1.91e-6` 和 P2 batch/serial `9.54e-7`：
`results/sra_pointmaze_planner_alignment_fixinput_20261024_cuda/report.json`。

在相同 receipt-bound physical contracts、同一 frozen checkpoint、同一 root splits 和无 outcome
selection 下，P1 的两个独立 panels 给出 spectral-minus-immediate physical regret `-0.095590` 与
`-0.053619`。P2 goal-projection 的两个独立 panels 的 global Spearman 为 `+0.368447` 与
`+0.184108`。reducer 汇总 P1 `-0.074604`（95% CI `[-0.352513,+0.221041]`）与 P2 global
Spearman `+0.288175`、root-mean `+0.318095`（95% CI `[+0.112381,+0.512381]`），并将 P3
mechanism gate 设为 true：
`results/sra_pointmaze_protocol_repair_fixinput_summary_20261024_cuda/report.json`。

这只允许 P3 的预注册 horizon-dependent expansion 开始。它不建立 RGB-only closed-loop success、
跨 checkpoint 泛化、谱理论普适性或完整状态 simulator equivalence；P1 的 pooled CI 仍跨零，故
P3 必须以独立、固定预算和真实执行结果检验收益，不能把正向 fixed-bank diagnostic 包装成最终效果。

#### P3 长时域 rollout 代码审计（2026-10-24）

在运行任何 P3 efficacy panel 前，target-alignment audit 已扩展到 H=1--4 个 action packet。
在主机 CUDA 上，四个 roots 均满足每个 packet 恰为五个 raw actions、目标 frame 为 `t+5H`、
参数 hash 不变，且直接 H-packet unroll 与逐 chart 递归 unroll 的最大 absolute latent 差为 `0`。
因此 P3 不存在 packet concatenation、递归 chart 或 target-index off-by-one 的已知代码缺陷。
H=3 的 `±1` target-frame 最近邻仅为 1/4（H=1/2/4 分别为 4/4、3/4、3/4），这记录为冻结模型的
长时域 temporal-fidelity 风险，而非实现失败；后续 P3 必须报告它，不能把任一 horizon 的负收益
解释成代码问题。回执为
`results/sra_pointmaze_rollout_target_alignment_p3_h4_20261024_cuda/report.json`。

P3 fixed-tree physical diagnostic 随后以 H=`2,3,4`、6 个首段候选与每段 5-arm continuation
tree 在 fresh roots `1832--1835` 上运行。首次 H4 将 750 条候选一次送入 attention，触发 CUDA
OOM；这是实现资源问题，不是方法证据。runner 改为确定性 64-candidate 分块后，tree-size、CUDA、
frozen-hash 与 outcome-blind selection gates 全部通过。相对 immediate 的真实 physical regret
差为 H2 `-0.264436`、H3 `-0.033042`、H4 `+0.052249`。因此 H4 的负向差异只可暂记为该固定
checkpoint/tree 下的长时域方法边界，不能归因于 OOM、候选截断或 time-axis bug；四 roots 的小面板
也不足以支持跨 horizon 的显著性主张。回执：
`results/sra_p3_horizon_expansion_v2_20261024_cuda/report.json`。

#### P3 批处理排错与 root-disjoint 确认（2026-10-24）

为检验 H4 的非正向结果是否由 OOM 修复后的 CUDA 分块造成，而不是由冻结模型/固定树本身造成，
同一面板以 `model_batch_size=64` 与 `8` 各执行一次。新的 fail-closed reducer 同时要求 checkpoint、
物理 label contract、roots、horizons 与所有 learned-model gates 相同，并逐 root/H 比对首段
选择 index、physical costs 和 regret；model-first cost 只允许 CUDA 浮点舍入级差异。原始面板的
最大 model-first-cost 差为 `3.2425e-5 < 1e-4`，所有选择和 outcome 完全一致。该收据为
`results/sra_p3_batch_invariance_20261024_cuda/report.json`。这排除了“64 候选分块改写 H4 选择”这一
已知实现解释，但它不是新效能结果。

随后在不与修复后 P1/P2、首个 P3 面板重叠的 roots `1800--1803` 上，以相同 checkpoint、
position-only physical contract、H=`2,3,4`、6×5^(H-1) 固定树和 outcome-blind selection
重跑。主机 RTX 5060 Ti / CUDA 13.0 preflight、frozen hash、tree size 与 CUDA-only gates 均通过。
其 spectral-minus-immediate regret 为 H2 `-0.363640`、H3 `+0.036304`、H4 `+0.009148`；同一
root panel 的 batch-64/8 最大 model-first-cost 差为 `2.1935e-5 < 1e-4`，选择和 physical
outcome 再次逐项相同（`results/sra_p3_batch_invariance_independent_1800_1803_20261024_cuda/report.json`）。

只对两份 batch-64 receipt 做 root-disjoint 汇总、没有改动作 bank、metric 或 roots：八个 root 的
H2 pooled mean 为 `-0.314038`（bootstrap 95% CI `[-0.521400,-0.131255]`）；H3 为 `+0.001631`
（`[-0.050716,+0.053979]`）；H4 为 `+0.030699`（`[-0.001580,+0.080670]`）。汇总的 contract/
root-disjoint/finite gates 都通过，回执为
`results/sra_p3_root_disjoint_confirmation_20261024_cuda/report.json`。因此 H2 的 fixed-tree
diagnostic 在这两个面板中得到确认；H3/H4 的区间仍跨零，不能声称长时域收益或稳定理论失败。既有的
输入、time-axis、候选树、CUDA OOM/分块与标签交换解释已被排除；剩余结论严格限于该 checkpoint 与
position-only fixed-tree protocol。P4 仍被阻塞，原因是缺少 matched-budget、带 latency accounting 的
RGB-only receding-horizon 环境执行，而不是把 H4 暂时不确定错误改写成理论否定。

### Reacher D2P-P1：margin-band 代码审计与 fail-closed outcome（2026-10-24）

在将 P1 的模型—观测次序失配解释为冻结模型边界之前，runner 审计发现并修复了两个
实现缺口：相对 checkpoint path 会被 compatibility loader 误解释为 cache identifier；以及
输出目录仅在长 CUDA 运行末尾创建，允许两个同名运行重复执行后争抢 receipt。runner 现于
CLI 边界解析显式本地路径，并在开始 GPU 工作前原子预留不可覆盖输出目录。

旧的 scalar margin guard 在新的 calibration roots 上仍保留了两个排序翻转，因而不能把
“更大 margin”当作可泛化置信度。新的 calibration-only margin-band selector 只枚举
compiler-side margin 的连续闭区间，在原 calibration/fidelity receipts 上选择
`[0.036398888, 0.055487633]`，并把 selector、源 receipt hashes、checkpoint、adapter、
proposal distribution、damping 与 action block 绑定到后续 calibration receipt。该选择
不读取 outcome qpos。

根不重叠的 binding calibration (`20266900--20266915`) 在 RTX 5060 Ti/CUDA 13.0 上
获得 11 个 model-eligible roots，11/11 均通过 observed-RGB successor ordering；独立
CUDA replay 同时通过 warmup/successor root identity、raw/normalized action inversion 与
finite-latent gates。它因此只授权一次新的 outcome panel，而不是 P1 efficacy 结论。

该 outcome 使用另 24 个 roots (`20267000--20267023`)；9 个 root 落入预先绑定的 band，
其中仅 7 个保持观察次序，故既未达到预注册的 12 个 eligible roots，也未通过所有
first-chunk mechanism gates。所有 9 个 action witnesses 的独立 CUDA replay 仍通过
root replay、动作反演和 finite-latent gates。这排除了当前已知的路径解释、并发输出、
GPU/CPU、动作归一化、记录动作重放和 root identity 解释；剩余失败是该冻结 checkpoint
在新根上的 action-conditioned successor-order calibration 不稳定，而不是 D2P 理论负例。
P2 保持封存；不得把 outcome 中 reconfiguration 对 zero-prefix 的点估计写成效应结论，
更不得忽略其相对 CEM 的不利区间。

关键 receipts 为
`le-wm-main/results/reacher_d2p_official_causal_repair_20261024_cuda_training_stats_v7/`
下的 `p1_margin_band_guard_selection_v1.json`、`p1_margin_band_calibration_v3/report.json`、
`p1_margin_band_first_chunk_fidelity_v3.json`、`p1_margin_band_outcome_v1/report.json` 与
`p1_margin_band_outcome_first_chunk_fidelity_v1.json`。

随后对 counterfactual repair branch 做 provenance audit 发现：其旧 simulator pool 的
seed range 为 `20267000--20267127`，与上述官方-checkpoint outcome 的 seeds 有重叠。
这不污染官方 checkpoint 的结果，因为它没有使用该 repair checkpoint；但它会污染任何未来
使用该 repair checkpoint 且复用这些 roots 的比较。P1 runner 现可接收 pool 或 repair
manifest，并在 calibration/outcome 启动前拒绝 simulator-training seed overlap。旧
counterfactual checkpoint 因而不能用于这些 outcome roots；这一点是 source-provenance
修复，不是 P1 失败或 D2P 理论结论。

### Reacher D2P-P0：代码级 admission（2026-10-24）

新建的 CUDA-only preflight
`le-wm-main/experiments/spectral_reachability/preflight_reacher_d2p_p0.py` 明确处理了
一个在通用 LeWM planning path 中会污染解释的 layout 风险：`LeWM.rollout()` 把
candidate tensor 的前 `history_len` rows 当作动作历史编码，却不读取
`info['action']`。如果向该接口传递的是 future-only CEM candidate，优化中的未来动作
会静默覆盖观测历史；这属于代码/协议问题，不能由旧 CEM 的成败来解释理论。

P0 adapter 将显式、不可优化的 normalized history prefix 与 future suffix 拼接，并且
只对 suffix 做 Jacobian。主机 RTX 5060 Ti/CUDA 13.0 上的四个 fresh reset roots
（`20261024--20261027`）全部通过：future-suffix Jacobian Frobenius norm
`10.158--14.803`，central-FD median relative error `9.45e-5--4.22e-4`，checkpoint
parameter hash 前后相同。回执为
`le-wm-main/results/d2p_reacher_p0_20261024_cuda/report.json`。该通过只准入后续的
matched live P0 evaluation；它不是 D2P success、physical prediction 或 online planning
收益结论。

### Reacher D2P-P0：fresh physical outcome（2026-10-24）

在上述 admission 后，`run_reacher_d2p_p0_goal_progress.py` 使用同根真实 Reacher
执行两段 action block，且把 qpos 只保留为 action 执行后的 evaluator label。运行器为
每个候选显式保存 action witness、预测 successor、Jacobian spectrum、forward rollout
数、VJP rows、GPU planner wall time 和 physical qpos outcome；模型输入始终只有 reset RGB、
rendered goal RGB 以及不可优化的 normalized zero-action history prefix。

先前的 generic CEM 不能用于这一判断，因为它没有绑定该 prefix。新实现依次关闭了三个
替代解释：D2P 与同配置 local Gauss--Newton 在每个 root 的 action 参数完全相同；五个
预先固定的 damping 候选只在 calibration roots (`20261040--20261041`) 中选择 `10.0`；
fresh outcome roots 与 calibration seeds 不重叠。最终 confirmation 使用 seeds
`20261202--20261213`，CEM 使用 192 forward rollout evaluations，对应 D2P 的 192
Jacobian VJP rows；实测 D2P wall time 为 `0.1780` s/root，CEM 为 `0.0341` s/root，
故结果不能归因于 CEM 获得更长的运行时间。

这是该 checkpoint 上的 **方法负结果**，不是代码负结果：D2P/GN mean final qpos distance
为 `0.8658`，CEM 为 `0.1480`；paired D2P-minus-CEM 为 `+0.7178`，95% bootstrap CI
`[+0.6222,+0.7898]`，12/12 outcome roots CEM 更好。D2P 的 physical progress 为
`0.0535`，CEM 为 `0.7712`。冻结 hash、CUDA、finite qpos outcomes 与 D2P=GN equality
均通过。完整 final receipt 为
`le-wm-main/results/d2p_reacher_p0_goal_progress_vjp192_cem192_fresh_20261024_cuda/report.json`；
damping calibration receipts 位于
`le-wm-main/results/d2p_reacher_p0_damping_calibration_20261024_cuda/`。

因此，D2P 的 reconfiguration/composition P1、adaptive-horizon P2 和跨任务 P3 均为
`blocked_upstream`：不能以一个没有超过强 CEM 的 local goal-progress primitive 为基础，
声称程序级 planning runtime。这个结论仅覆盖该 frozen Reacher checkpoint、reset-state
two-block protocol 与所报告的 resource accounting；它不否定其他 checkpoint、接触任务或
经由独立机制验证的 reconfiguration hypothesis。

资源记账修正后，用另一组完全未使用的 confirmation roots（`20261302--20261313`）
重跑同一 CEM-192/D2P-VJP-192 protocol，结论保持：paired D2P-minus-CEM 为 `+0.6288`，
95% CI `[+0.5042,+0.7302]`，CEM 仍在 12/12 roots 获胜。该 machine-readable receipt 的
`status` 是 `method_negative_against_cem`，且 `stage_gate.p1_unlocked=false`：
`le-wm-main/results/d2p_reacher_p0_goal_progress_vjp192_cem192_confirmation_20261024_cuda/report.json`。

### Reacher D2P-P0：history normalization 修复与重跑（2026-10-24）

对上述负结果的第二轮代码审计发现了一个实质性协议 defect，故不能直接沿用上述
P0 数值。`_normalized_block()` 已将 raw zero action 转为 checkpoint z-score，而
`_Preprocessor._prepare_info()` 又会对 `action` 调用 `ActionNormalizer.transform()`。
因此旧 runner/preflight 将**历史 prefix 双重标准化**，但优化的 future suffix 只有一次
标准化；当前 adapter 的“同一动作坐标系”合同不成立。实测该数据集的最大 prefix 偏差为
`0.32945`。这首先是代码问题，而不是 D2P 或世界模型的理论负证据。

修复后的 `preflight_reacher_d2p_p0.py` 新增 `_raw_block()`，把 raw zero action 交给
preprocessor 做唯一一次变换；`run_reacher_d2p_p0_goal_progress.py` 使用同一 helper。
新的回归测试锁定该前缀只能标准化一次（相关 suite `10 passed`）。旧的 Reacher D2P
preflight、damping calibration 与 outcome 回执因此仅保留为 defect 发现的历史记录，
不再作为方法效果证据。

修复后，在主机 RTX 5060 Ti / CUDA 上新跑的四个 preflight roots 仍全部通过 nonzero
future-suffix Jacobian、central-FD、candidate-prefix 和 frozen-hash gates：
`le-wm-main/results/d2p_reacher_p0_fixnorm_preflight_20261024_cuda/report.json`。随后用
未与 confirmation 重叠的 seeds `20261420--20261427` 在五个预注册 damping 值
`{0.001, 0.01, 0.1, 1, 10}` 中选择 D2P mean final qpos distance 最小的 `10`；所有
calibration receipts 位于
`le-wm-main/results/d2p_reacher_p0_fixnorm_damping_calibration_20261024_cuda/`。

固定该选择后，新的 outcome roots `20261502--20261513` 使用同一 history-bound RGB
protocol、CEM 192 forward evaluations 与 D2P 192 VJP rows，并记录 wall time。CEM 在
12/12 roots 获胜；D2P-minus-CEM final qpos distance 为 `+0.66017`，95% bootstrap CI
`[+0.61393,+0.70741]`。D2P=local Gauss--Newton、CUDA-only、finite physical outcomes
和 frozen parameter hash 全部通过。最终可机读回执为
`le-wm-main/results/d2p_reacher_p0_fixnorm_vjp192_cem192_confirmation_20261024_cuda/report.json`。

所以，修复前的负结果不能用于理论判断；但修复后的独立 CUDA panel 仍显示这个 frozen
checkpoint / reset-state two-block protocol 下的 goal-progress primitive 缺少相对 CEM 的
收益。P1 reconfiguration、P2 composition/adaptive horizon 和 P3 cross-task 仍为
`blocked_upstream`。这不否定其他 checkpoint、接触任务，或有独立机制证据的
reconfiguration hypothesis。

### Reacher D2P-P0：execution identity 与 resource-gate 审计（2026-10-24）

第三轮审计不把“192 VJP rows”与“192 CEM forward evaluations”错误地称为严格的
matched compute：两者是不同的操作类型。runner 因此改为逐 root 报告 forward、VJP
和实际 planner wall time，并把 P1 gate 定义为：CEM 在每个 outcome root 物理胜出、
paired CI 严格为正，且 CEM 的 measured planner wall time 不高于 D2P。另加入 reset
RGB 与 rendered goal RGB 的 SHA-256 identity contract；每次物理执行 reset 后都必须
与 planner 的 root 图像逐字节一致，否则 fail closed。该检查不把 qpos/target state
提供给 planner。

在新的、未使用 outcome roots `20261602--20261613` 上，所有 CUDA、frozen hash、
D2P=GN、finite outcome、planner/execution RGB identity 和 CEM wall-time-noninferior
gates 通过。D2P-minus-CEM final qpos distance 为 `+0.62639`，95% bootstrap CI
`[+0.52624,+0.69906]`，CEM 仍为 12/12 roots 获胜。此回执取代前两份 Reacher P0
effect receipt 作为当前 P0 gate 的权威证据：
`le-wm-main/results/d2p_reacher_p0_fixnorm_identity_wall_confirmation_20261024_cuda/report.json`。
因此，代码、执行 identity 和资源表述均已审计后，P1--P3 的 `blocked_upstream` 状态
仍成立；其含义仅限当前 checkpoint/protocol 的方法边界，不是理论或模型类别的否定。

### Reacher D2P-P0：物理 action-box 与 calibration binding 修复（2026-10-24）

对上述看似已闭合的负向结论继续做 code-level 审计时，发现 runner 仍使用 checkpoint
normalized-space 的对称 `[-3,3]` box。按保存的 Reacher action mean/std 反变换后，该 box
可产生约 `[-1.94,+1.97]` 的 raw controls，但环境声明的 box 是 `[-1,1]^2`。主机 CUDA
contract test 直接比较 `(+/-1.75)` 与 `(+/-1.0)` 的一步 MuJoCo state，最大 qpos 差为 `0`，
确认环境会静默 clip。因此模型曾给不可执行 action 打分、环境却执行另一个 action；此前所有
Reacher effect receipt（包括 fixnorm/identity/wall confirmation）均撤为发现该协议 defect 的历史
诊断，而非方法证据。

修复把 future parameter 的 `tanh` 直接映射到**环境 raw action box 经 checkpoint normalization
后的逐维区间**，故零参数等于物理零 action；raw action 超界会 fail closed，不再依赖环境裁剪。
preflight 与 outcome receipt 同时写入 raw/normalized boxes 及 action witness。另一个审计发现
outcome runner 先前只检查 calibration 文件存在，未验证选择的 damping、action parameterization 或
根集不重叠；现在它要求一个 passing selection receipt、匹配 damping、匹配 physical-box
parameterization，并把 receipt SHA-256 与 calibration seeds 写入结果。相应测试覆盖 action
round-trip、超界拒绝、damping binding 和 seed overlap。

修复后 CUDA preflight 在 seeds `20261700--20261703` 通过 nonzero future-suffix Jacobian、finite
difference、frozen hash 和 prefix gates（norm `4.573--6.187`，median FD relative error
`3.84e-4--5.67e-4`）：
`results/d2p_reacher_p0_physical_action_contract_20261024_cuda/report.json`。五个 damping 值仅在
calibration seeds `20261710--20261713` 上评价，机器可读 selector 依明确记录的最小 D2P mean final
qpos rule 选出 `0.1`：
`results/d2p_reacher_p0_physical_action_damping_selection_v2_20261024_cuda/report.json`。

绑定后的新 outcome seeds `20261740--20261751` 与 calibration seeds 无重叠。所有 CUDA、frozen
hash、D2P=local-Gauss--Newton、RGB execution identity、finite outcomes、raw-action-in-box 与
CEM wall-time-noninferior gates 均通过。CEM 在 12/12 roots 优于 D2P：D2P-minus-CEM final qpos
distance 为 `+0.737925`，95% bootstrap CI `[+0.552437,+0.911954]`；CEM 的实际规划时间也不高于
D2P。权威 receipt 为
`results/d2p_reacher_p0_physical_action_bound_confirmation_20261024_cuda/report.json`。因此，在
已排除 history normalization、candidate-prefix、raw-action clipping、calibration binding、root
identity 与资源表述后，该 frozen checkpoint / reset-state two-block goal-progress primitive 仍是
相对 CEM 的**方法边界**；它不构成对 D2P 总体、谱理论或其他环境的否定，Reacher 的 P1--P3
继续保持 `blocked_upstream`。

### Reacher D2P-P0：单步线性化接受规则修复（2026-10-24）

仍不能把上一段的负向结果直接固化为方法结论。代码级检查显示，原 DLS/Gauss--Newton
实现只在 zero-action 的局部线性模型中求一次更新，随后将该更新穿过 `tanh` 和 physical
action box 直接执行；它没有检查该**实际冻结模型**上的目标是否下降。因而参数空间的
大更新会被映射为交替、接近 `+/-1` 的饱和控制，且旧 receipt 中存在
`predicted_cost_after > predicted_cost_before`。这属于 planner globalization/acceptance 的实现
缺口，而不是 D2P、世界模型或物理 evaluator 的反例。

runner 已改为固定候选步长 `{1, 1/2, 1/4, 1/8, 1/16, 0}` 的 objective-backtracking：在相同
frozen model、相同 prefix 和相同 physical action box 下，逐一重算真实 terminal latent cost，
选择最低成本的候选；若最低值仍高于起点则 fail closed 到错误。每个 root 现在记录
`accepted_step_scale` 与 `actual_model_cost_nonincreasing`。这不是把结果调向 physical label：
选择中没有读取 qpos、target state 或 outcome，local Gauss--Newton 仍由完全相同的代码重算并
逐 root 验证参数相同。

新 runner hash 绑定的五个 CUDA-only calibration panels（seeds `20261820--20261823`）用同一
机器可读选择器，在 `{0.001, 0.01, 0.1, 1, 10}` 中选择 `damping=0.001`；选择回执为
`le-wm-main/results/d2p_reacher_p0_trust_region_bound_selection_20261024_cuda/report.json`。
随后 outcome 使用完全不重叠的 `20261840--20261851` roots。RTX 5060 Ti 上所有 CUDA、frozen
hash、D2P=GN、RGB execution identity、finite physical outcome、raw-action-in-box、以及
`actual_model_cost_nonincreasing` gates 都通过。D2P-minus-CEM final qpos distance 为 `+0.703100`
（95% bootstrap CI `[+0.467015,+0.897378]`）；CEM 在 11/12 roots 更小，唯一反向 root 是
CEM 自身未收敛的 root，而非 D2P 的正向证据。CEM 仍在每个 root 的实测 planner wall time
不高于 D2P。完整 receipt：
`le-wm-main/results/d2p_reacher_p0_trust_region_bound_confirmation_20261024_cuda/report.json`。

因此，接受规则缺口已被修复，旧 action-bound outcome 不能再承担最终方法结论；但修复后的
fresh panel 显示，**单次** local goal-progress compilation 仍未达到 CEM 的优化质量或速度。现有
预注册的 P1 kill rule 要求 CEM 在所有 root 胜出，故该 receipt 的 machine status 是
`completed_without_efficacy_verdict`，不能借 1 个 CEM 失败 root 宣称 D2P 通过或解锁 P1。
研究上更保守的状态仍是 `blocked_upstream`：下一条有效路线必须作为独立、预注册的多次
relinearization/closed-loop program protocol，重新做 calibration/outcome split，而不能事后把本
panel 的 qpos 结果用于调参。这是当前 checkpoint 和 two-block reset protocol 的优化边界，绝非
对谱几何或 D2P 在其他模型/环境上的理论否定。

补充的计算路径审计还排除了一个表面上可能改善 D2P 时间的实现解释。Reacher 的终点 latent
为 192 维、可优化 action 参数为 20 维，故曾尝试以 20 个 forward-mode JVP column 取代 192 个
reverse-mode VJP row。主机 CUDA 的 efficient SDP attention 不支持该 JVP 所需的高阶导数；强制
CUDA math-SDP 后虽可运行、也没有 CPU fallback，但实测约 `0.975 s/root`，反而比原 reverse-mode
实现约 `0.18 s/root` 更慢。该路径已撤回，不进入 planner；独立 smoke receipt 保留在
`le-wm-main/results/d2p_reacher_p0_jvp_math_calibration_20261024_cuda/damping_0.001/report.json`。
因此，现有的运行时间差不是把较小输入维度误写成 VJP 所致，而是当前 attention kernel 的
高阶自动微分约束与实际 GPU 执行成本；这同样不应被解释成谱几何的理论失败。

最后，针对“CEM 的优势可能只是 receipt 缓存、action block/time-axis 重排或记录错误”的
解释，新增 CUDA-only immutable-witness replay audit。它不优化动作、不访问 qpos、也不重做
physical evaluation；只用每个 receipt 已保存的 normalized action witness 重新构造不可变 history
prefix 和 frozen rollout。实现时首先发现审计器自身把已映射的 normalized witness 又传给了会
施加 `tanh` 的参数接口；该二次映射 bug 已修复为直接拼接 recorded suffix。最终 12-root audit
逐项重算的 terminal cost 与 receipt 完全一致（最大绝对误差 `0`），normalized-to-raw witness
最大误差也是 `0`，所有 RGB identity/action-box gate 通过，并且 CEM 在冻结模型目标上 12/12 roots
均低于 D2P。回执为
`le-wm-main/results/d2p_reacher_p0_trust_region_bound_witness_replay_audit_v2_20261024_cuda/report.json`。
这排除了 stale candidate、双重/遗漏 action transform、time/block reshape 和 receipt 写入错误作为
该 panel 负向差异的来源；它仍只限定当前 checkpoint 与协议，不能被扩大为谱几何的一般性否定。

### Reacher D2P-P0：多次 relinearization（2026-10-24）

在上述实现/receipt 路径闭合之后，剩下的可操作机制不是再解释旧数值，而是原 program 只在
zero-action 处做一次线性化。runner 因此增加显式 `relinearization_steps`：每一轮从上轮已接受的
bounded action parameters 重新计算 Jacobian、DLS proposal 与 frozen terminal cost，并只接受不增的
候选；每轮的 cost、step scale、Jacobian norm/rank 都写入 receipt。该字段进入 damping selector 的
contract fingerprint，outcome 必须匹配 calibration 的 iteration count、runner SHA 和 action box；
不会把前一 panel 的 qpos 用于选择迭代数或阻尼。

固定四次 relinearization 后，五个 CUDA-only damping calibration receipts（roots
`20262020--20262023`）选择 `damping=10`：
`le-wm-main/results/d2p_reacher_p0_relinearized4_selection_20261024_cuda/report.json`。全新的 outcome
roots `20262040--20262051` 上，D2P 的每个 root 都完成四轮、每轮 frozen model cost 都不增，且
CUDA/frozen hash/D2P=GN/RGB identity/raw box/finite outcome gates 全部通过。该修改确实能让一个 root
进入与 CEM 相近的低目标盆地并取得 `0.0425` 对 `0.0523` 的物理 qpos；但总体上 CEM 仍在 11/12
roots 更好，D2P-minus-CEM 为 `+0.610833`（95% bootstrap CI `[+0.423730,+0.781229]`），CEM 的实测
wall time 仍不高于 D2P。回执为
`le-wm-main/results/d2p_reacher_p0_relinearized4_confirmation_20261024_cuda/report.json`。

这把当前边界收紧为：即使修复单步 local update、并在同一 frozen objective 上做受接受规则保护的
四次重新线性化，当前 checkpoint/two-block protocol 的 DLS path 仍通常不能进入 CEM 找到的低成本
盆地。它是新的独立方法/优化边界，不是旧代码错误的重述，也不构成对其他模型、长时闭环或谱几何
理论的否定；P1 仍不应由这个 11/12 的负向 panel 解锁。

### Reacher D2P-P0：checkpoint action-token 合同与严格 P1 准入（2026-10-24）

在把上述差异写成方法边界之前，新增一次代码级 action-token 审计。风险是：D2P 和 CEM 虽然共享
同一个错误的模型输入布局，因而仍可能共同优化错误接口。Reacher runner 过去没有将请求的
`--action-block` 与 released checkpoint 的 `action_encoder.input_dim` 直接绑定。现在 runner 和
CUDA preflight 都 fail-closed 地要求 `input_dim = action_block × environment action_dim`，并要求
action statistics 宽度相同；同时新增非恒定 action 的 round-trip 回归测试，证明
raw `[time, action_dim]`、normalization、flattened token 与执行时 reshape 保持 time-major 顺序。

新的主机 CUDA preflight 在 roots `20262100--20262101` 上确认 checkpoint 为 `10 = 5 × 2`，所有
contract、frozen hash、nonzero future-suffix Jacobian 与中心差分门槛通过（两根的中位相对误差为
`0.000306` 和 `0.000556`）：
`le-wm-main/results/d2p_reacher_p0_action_protocol_preflight_20261024_cuda/report.json`。因此，当前
P0 的 CEM 差异不能归因于 action-block 长度、统计量宽度或 time/block flatten--unflatten 排列错误。

新 runner hash 绑定的五个 calibration receipts 仍只在 `20262020--20262023` 选择
`damping=10`，而 selector 的不可变 fingerprint 现在也包含 checkpoint action-block 与 layout：
`le-wm-main/results/d2p_reacher_p0_action_protocol_v2_relinearized4_selection_20261024_cuda/report.json`。
完全 root-disjoint 的 outcome roots `20262140--20262151` 上，CUDA、frozen hash、D2P=GN、RGB
execution identity、raw action box、finite physical outcome 与 CEM wall-time-noninferior gates 全部通过。
CEM 在 12/12 roots 更好；D2P-minus-CEM final qpos 的均值为 `+0.830692`，95% bootstrap CI 为
`[+0.666535,+0.956171]`。权威 receipt：
`le-wm-main/results/d2p_reacher_p0_action_protocol_v2_relinearized4_confirmation_20261024_cuda/report.json`。

同次修复还堵住了 stage-gate 逻辑漏洞：旧逻辑把“尚未满足 CEM 逐根全胜的 no-go”错误地当作可进入
P1；现在 P1 必须有 fresh paired 95% interval 实际偏向 D2P 才可解锁。当前 `p1_unlocked=false`。
这不是把负结果升级为理论否定：它只是在 action-token/input-contract 已排除后，把这个 frozen Reacher
checkpoint、two-block、四次 DLS relinearization 程序相对 CEM 的负向结果标记为局部方法/优化边界。

### Reacher D2P-P0：零起点局部盆地修复（2026-10-24）

继续排查后发现，action-token 合同正确仍不等于 D2P optimizer 已被公平实现。旧的 DLS 固定从 zero
action parameters 出发；在其最后一个 panel 中，D2P 的 frozen terminal cost 在多数 roots 仍为
`1.65--2.14`，而 CEM 为 `0.010--0.041`，并且 D2P 的 raw action 幅度约 `0.16--0.34`，CEM 接近
physical box 边界。相反，同一 receipt 的 restart-GD 已在多个 root 达到低 frozen cost。这是可复现的
local-basin/initialization 缺陷，不能继续归因于 D2P 理论或物理 evaluator。

runner 因而新增 `d2p_restarts`：restart 0 保留原 zero-action DLS，其他 starts 只由固定 CUDA seed 在
同一 unconstrained action coordinates 生成；每个 start 做同样的四次受接受规则保护的 DLS，唯一选择
信号为 frozen terminal cost。restart count 写入 calibration fingerprint，outcome 必须精确绑定它；不使用
qpos 选择。4-start 的独立 calibration roots `20262210--20262213` 在五个 damping 中选择
`damping=1`：
`le-wm-main/results/d2p_reacher_p0_multistart4_selection_20261024_cuda/report.json`。

全新 outcome roots `20262240--20262251` 上，所有 CUDA/frozen/RGB/action-box/D2P=GN gates 仍通过。
多起点修复将 D2P 相对 CEM 从 12/12 失败改为 5/12 胜、7/12 负；paired D2P-minus-CEM qpos mean 为
`+0.105018`，95% bootstrap CI 为 `[-0.021512,+0.306856]`。这已经排除“zero-start DLS 失败”作为旧的
最终结论，却仍不足以建立 D2P 优于 CEM 的 P1 admission，因此 `p1_unlocked=false`。权威 outcome：
`le-wm-main/results/d2p_reacher_p0_multistart4_confirmation_20261024_cuda/report.json`。

独立 CUDA witness replay 重放了该新 receipt 的全部 action witnesses、terminal costs、RGB identities 与
physical boxes；全部通过。CEM 仅在 4/12 roots 具有更低 frozen model cost，说明未通过的部分不再是
共同错误的 model interface 或 receipt 变换链；目前剩余的可检验限制是仅四个 seeded local basins 的覆盖
是否足够，以及其相对 CEM 的计算代价：
`le-wm-main/results/d2p_reacher_p0_multistart4_witness_replay_audit_20261024_cuda`。

### Reacher D2P-P0：未执行 history-prefix / reset-padding 修复（2026-10-24）

继续把不利结果优先视为协议缺陷后，发现此前的“history-bound”实现仍有更基础的因果错位。LeWM 的
`rollout` 将 candidate 的前 `history_len=3` 个 action token 与三帧 RGB 一同用于预测下一状态；每个
token 又对应 checkpoint 固定的 `action_block=5` 个环境步。旧 P0 却用重复的 reset RGB 和未执行的零
action prefix 进行模型评分，随后从 reset 直接执行优化出的 future suffix。换言之，模型评分的是
“先执行三个 token、再执行计划”的轨迹，物理评估执行的却是“立即执行计划”的轨迹。旧 panels 因而只能
作为发现该缺陷的诊断，不能作为方法或理论结论。

新的 `reacher_d2p_context.py` 将这个边界实现为 fail-closed 的 executed-warmup contract：每个 root
先按 root seed 生成与目标/方法无关、位于物理 action box 内的确定性 excitation，实际执行 3 个 action
token（15 个环境步），保存每个 token 之前的 RGB、time-major raw action blocks、warmup action hash 与
warmup 后的 planning-root RGB hash；所有方法从同一 warmup 后根执行。校准选择器、outcome binding、
preflight 与 witness replay 均显式绑定 `context_mode=executed_warmup`、
`warmup_policy=seeded_bounded_excitation_v1`、history length、warmup length 和 runner hash。因而历史
reset-padding receipt 不能被新协议复用。

新的两-root host-CUDA preflight（seeds `20262280--20262281`）通过了冻结参数、action-token、非零
future-suffix Jacobian 与中心差分 gates；Jacobian Frobenius norms 为 `5.9961`、`6.3962`，中位
relative FD errors 为 `4.98e-4`、`5.59e-4`：
`le-wm-main/results/d2p_reacher_p0_executed_context_preflight_20261024_cuda/report.json`。
五个仅 calibration 的 damping receipts（seeds `20262300--20262301`）以 D2P physical final-qpos
最小均值选择 `damping=0.01`，并冻结了 4 个 D2P restarts：
`le-wm-main/results/d2p_reacher_p0_executed_context_selection_20261024_cuda.json`。

随后 12 个完全不重叠的 host-CUDA outcome roots `20262320--20262331` 显示：修复不会支持 P1 admission。
D2P-minus-CEM final-qpos mean 为 `+0.306744`，paired bootstrap 95% CI 为
`[+0.107766,+0.524649]`，CEM 在 10/12 roots 有更小 final qpos。CUDA-only、frozen parameter hash、
D2P=local-Gauss--Newton、action box、physical finite outcome 与 warmup 后 execution-root RGB identity
全部通过，且 `p1_unlocked=false`：
`le-wm-main/results/d2p_reacher_p0_executed_context_confirmation_20261024_cuda/report.json`。
独立 CUDA witness replay 逐根重建 history、warmup 后根、normalized/raw witness 和 terminal cost；所有
replay gates 通过，且 CEM 在 frozen terminal objective 上 12/12 更低：
`le-wm-main/results/d2p_reacher_p0_executed_context_witness_replay_audit_20261024_cuda.json`。

因此，当前可辩护的结论不是“谱理论失败”，而是：在这个**已修复 temporal-context、冻结 Reacher
checkpoint、two-future-token、four-start local DLS**协议中，单一 goal-progress primitive 相对 CEM
没有建立物理收益。P1--P3 仍不可由 P0 解锁；未来若研究 reconfiguration 或 composition，必须作为
新的预注册机制和新的 calibration/outcome split，不能事后用本 outcome qpos 调整该设计。

### Reacher D2P-P0：重线性化默认值与局部盆地覆盖修复（2026-10-24）

对 executed-warmup runner 的后续代码审计发现一个独立的配置回归：已验证的多起点协议使用
`relinearization_steps=4`，但引入 executed-context CLI 时默认值意外回退为 `1`。这会把每个
seeded start 从四次受接受规则保护的局部更新缩减为一次，因而不能将单步 receipt 当作四次
relinearization 的方法检验。默认值已恢复为 `4`，并有回归测试锁定；这是一项代码修复，不是
对 physical outcome 的事后调参。

修复默认值后，新的 executed-context calibration roots `20262350--20262351` 在四-start、四次
relinearization 合同下选择 `damping=0.1`，选择回执为
`le-wm-main/results/d2p_reacher_p0_executed_context_r4_selection_20261024_cuda.json`。独立 outcome
roots `20262360--20262371` 的 CUDA、frozen hash、D2P=GN、warmup-root identity、raw box 和 finite
physical outcome 全部通过；然而 D2P-minus-CEM final-qpos mean 仍为 `+0.231237`，paired 95% CI
`[+0.018365,+0.510110]`，CEM 胜 `8/12`。对应 CUDA witness replay 全部复现 history、root、动作变换与
frozen cost，且 CEM 在 5/12 roots 的模型目标较低：
`le-wm-main/results/d2p_reacher_p0_executed_context_r4_confirmation_20261024_cuda/report.json` 和
`le-wm-main/results/d2p_reacher_p0_executed_context_r4_witness_replay_audit_20261024_cuda.json`。

这份 witness 的细分诊断很重要：CEM 模型目标更低的 roots 表明四个 local starts 的盆地覆盖不足；而另一些
roots 中 D2P 已取得较低的 frozen latent objective 却没有取得较低的 qpos distance，才是此 checkpoint
的 endpoint-objective calibration 限制。前者仍是 planner implementation/design 问题，后者不能被误报为
action layout、history root 或 action transform 的 bug。

因此 runner 新增独立的 `screened_gaussian_proposals` 初始化协议：用固定 CUDA seed 生成 32 个
unconstrained Gaussian candidate（候选 0 永远为 zero program），只按 frozen terminal latent cost 选出
4 个 elite，再分别执行四次 D2P/DLS refinement，并仅按同一 frozen cost 选 winner。proposal count、elite
count、初始化策略、damping、runner hash 和 executed-context contract 都进入 calibration fingerprint；该
protocol 不覆盖或改写任何已封存 receipt。

在全新 calibration roots `20262380--20262381` 上，`{0.01,0.1,1,10}` 的 CUDA-only search 选择
`damping=10`（mean final qpos `0.114643`），选择收据为
`le-wm-main/results/d2p_reacher_p0_screened_r4_selection_20261024_cuda.json`。完全隔离的 12-root
outcome panel `20262400--20262411` 显示这项修复消除了此前“CEM 显著优于 D2P”的证据：D2P-minus-CEM
mean 为 `+0.060148`，paired 95% CI `[-0.107874,+0.290743]`，胜负为 `6:6`；D2P 相对 random、single-start
action-GD 与 restart-GD 的区间均偏向 D2P。CUDA witness replay 通过全部 RGB root、normalized/raw
action、action-box 与 frozen terminal-cost gates，CEM 只在 1/12 roots 的冻结模型目标较低。权威产物为
`le-wm-main/results/d2p_reacher_p0_screened_r4_confirmation_20261024_cuda/report.json` 和
`le-wm-main/results/d2p_reacher_p0_screened_r4_witness_replay_audit_20261024_cuda.json`。

该 panel 的正确状态是 `completed_without_efficacy_verdict`，而不是 P1 通过：CEM comparison 的
95% interval 仍跨越零，故 `p1_unlocked=false`。当前可排除的负向解释包括 temporal-context reset
mismatch、单步默认值回归、action-token/layout/normalization、receipt witness 写入和局部盆地覆盖不足；
当前仍未解决的是冻结 latent endpoint objective 与 physical qpos consequence 的充分对齐，以及是否存在
在预注册、资源匹配的 RGB closed-loop protocol 中稳定超过 CEM 的 D2P mechanism。它们是后续研究门槛，
不是本次代码修复后的理论否定。

### Reacher D2P-P0：观测 root 锚定的 rollout 修复（2026-10-24）

在继续检查“不利 outcome 是否仍由代码造成”时，发现 executed-warmup 修复虽已让 physical evaluator
从 warmup 后的真实 root 执行，`_terminal_batch` 和 preflight 却仍把三个 warmup **之前**的 RGB frame
送入通用 `JEPA.rollout`。该 API 会先依据这三帧和 action prefix 预测一次 post-warmup root，之后才
预测 candidate suffix；所以 planner 对“模型虚构 root”打分，evaluator 却从已观测到的真实 root 执行。
这是时间索引/conditioning 的代码错误，不能被解释为 world-model 或 D2P 的负结果。

修复后的 adapter 使用 `[frame_{t-2}, frame_{t-1}, observed_root_t]` 作为 RGB context，保留连接前两帧到
root 的两个已执行 action token，并把第一个优化 action 作为最后一个因果 action token 直接预测
`t+1`；后续 future action 才自回归展开。该 observed-root contract 同时进入 preflight、calibration selector、
outcome binding 与 CUDA witness replay；旧 receipt 因 runner hash 和
`rollout_root_conditioning=observed_post_warmup_rgb_anchor` 不匹配而不可复用。相应的回归测试还显式拒绝
回退到先预测 root 的通用 rollout 路径。

新的主机 CUDA preflight（roots `20262420--20262421`）通过所有 gates；Jacobian norms 为 `6.6571`、
`5.9058`，中心差分中位相对误差为 `4.04e-4`、`4.81e-4`：
`le-wm-main/results/d2p_reacher_p0_root_anchor_preflight_20261024_cuda/report.json`。独立 calibration
roots `20262440--20262441` 在 `{0.01,0.1,1,10}` 中选择 `damping=0.01`（mean final qpos `0.062970`），
选择收据为 `le-wm-main/results/d2p_reacher_p0_root_anchor_r4_v2_selection_20261024_cuda.json`。

完全不重叠的 outcome roots `20262460--20262471` 上，所有 CUDA、frozen hash、D2P=GN、RGB root identity、
raw action box 和 physical finiteness gates 都通过。D2P-minus-CEM final-qpos mean 为 `+0.042812`，paired
95% CI `[-0.178206,+0.209411]`；D2P/CEM 为 `4/8`，不再支持 CEM 的显著优势，但也不支持 D2P 胜过 CEM。
D2P 相对 random、single-start action-GD 与 restart-GD 的区间均偏向 D2P。CUDA witness replay 逐根重建
observed root、normalized/raw actions 和 frozen terminal costs，所有 replay gates 通过；CEM 仅在 7/12 roots
取得更低的 frozen model cost。对应产物为
`le-wm-main/results/d2p_reacher_p0_root_anchor_r4_v2_confirmation_20261024_cuda/report.json` 与
`le-wm-main/results/d2p_reacher_p0_root_anchor_r4_v2_witness_replay_audit_20261024_cuda.json`。

因此，这次修复撤销了此前所有由“预测 root 替代观测 root”产生的负向解释，但独立新 panel 的状态仍是
`completed_without_efficacy_verdict`，且 `p1_unlocked=false`。现阶段唯一可辩护的边界是：在 root-anchored、
frozen Reacher、two-token D2P 与已校准 proposal-screening protocol 下，尚未建立优于 matched CEM 的物理
效应；这不是对理论的否定，也不能以旧的错位 receipt 反推该结论。

### Reacher D2P-P0：proposal 覆盖匹配修复（2026-10-24）

root-anchored witness 中 CEM 仍在 7/12 roots 取得更低 frozen cost。代码审计发现 screened D2P 只在
32 个 Gaussian proposal 中选 4 个 basin，而 CEM 用相同每轮 population 却经过 4 轮，实际评分 128 个
candidates。此前将这种 `32` 对 `128` 的全局覆盖差解释为 D2P 机制边界是不成立的。D2P 默认 proposal
budget 已改为 `128`，并有回归测试；budget 继续进入 immutable calibration contract，故旧的 32-proposal
receipt 不可被混合或覆盖。

全新 CUDA calibration roots `20262480--20262481` 在 `{0.01,0.1,1,10}` 中选择 `damping=0.01`
（mean final qpos `0.052564`），选择回执为
`le-wm-main/results/d2p_reacher_p0_coverage128_r4_selection_20261024_cuda.json`。独立 outcome roots
`20262500--20262511` 的 D2P-minus-CEM final-qpos mean 首次转为 `-0.055637`，胜负 `6:6`；但 paired
95% CI 仍为 `[-0.278127,+0.106066]`，尚未满足 P1 的确认标准。D2P 相对 random、action-GD 和
restart-GD 的区间均偏向 D2P。CUDA witness replay 通过全部 observed-root、action transform、action box
和 frozen terminal-cost gates；CEM 仅在 4/12 roots 的 frozen objective 更低：
`le-wm-main/results/d2p_reacher_p0_coverage128_r4_confirmation_20261024_cuda/report.json` 与
`le-wm-main/results/d2p_reacher_p0_coverage128_r4_witness_replay_audit_20261024_cuda.json`。

故该代码/协议解释已被实证性修复：不足的 proposal coverage 足以改变效应方向，旧的 CEM-negative panels
不能承担理论结论；但当前独立样本仍不足以断言 D2P 胜 CEM，`p1_unlocked=false` 保持不变。下一步若继续，
只能在预先冻结的 128-proposal protocol 上增加新的独立确认 panel，不能事后调参或启动 P1。

### Reacher D2P-P0：128-proposal 的第二独立确认（2026-10-24）

在未修改 runner、checkpoint、damping selection、proposal budget 或任何执行设置前，先重算了源码 SHA-256，
确认它逐字匹配首个 128-proposal receipt；该 receipt 也同时绑定 observed-root rollout、`128/4` proposal
screen、四次 relinearization、CUDA/frozen/action-box/RGB-root gates 及 calibration roots
`20262480--20262481`。因此没有发现能将首个正向方向归为 stale runner、配置漂移或 action/root contract
错误的代码解释。

第二个完全未使用的 outcome panel `20262520--20262531` 只复用上述冻结选择收据。D2P-minus-CEM final-qpos
mean 为 `-0.078150`，95% CI `[-0.298051,+0.057208]`，胜负 `5/7`；方向与第一个 panel 一致，但区间仍跨零。
其 CUDA witness replay 通过全部 root RGB、normalized/raw action、action box 与 frozen terminal-cost
checks，CEM 在 6/12 roots 的 frozen objective 较低：
`le-wm-main/results/d2p_reacher_p0_coverage128_r4_confirmation2_20261024_cuda/report.json` 和
`le-wm-main/results/d2p_reacher_p0_coverage128_r4_confirmation2_witness_replay_audit_20261024_cuda.json`。

这构成两个独立、同方向但各自未确认的效应估计；它们绝不在事后 pooled 后解锁 P1。当前 P1 gate 仍为
`false`，因为冻结协议要求一个 fresh panel 的 paired CI 自身偏向 D2P。可排除的解释已包括 warmup root
错位、预测 root 替代观测 root、单步重线性化回归、32-vs-128 proposal coverage、动作布局和 witness
变换；剩下的是需要预注册样本量的确认问题，而非将不确定性误写成理论反例。

### Reacher D2P-P0：第三个扩大确认 panel 的首次尝试失效（2026-10-24）

在两个 `n=12` 独立 outcome panel 都保持 D2P 相对 CEM 的负向点估计、但各自 paired 95% interval
仍跨零后，不再更改 runner、checkpoint、adapter、warmup、root anchor、damping、action bounds、
relinearization、initialization 或 proposal budget。为区分“单 root 物理效果方差导致的精度不足”与残余
实现问题，第三个 panel 原先固定使用连续 roots `20262540--20262587`（`n=48`），并复用 immutable
selection receipt `results/d2p_reacher_p0_coverage128_r4_selection_20261024_cuda.json`。

运行合同固定为：host CUDA；冻结 checkpoint 与 adapter；executed warmup；
`rollout_root_conditioning=observed_post_warmup_rgb_anchor`；`damping=0.01`；
`screened_gaussian_proposals`；`128` screened proposals / `4` elites；四次 relinearization；matched
CEM (`32` samples × `4` rounds)；每 root 成对物理执行。P1 仅在该**单一 fresh panel**的 D2P-minus-CEM
final-qpos paired bootstrap 95% interval 上端 `<= 0`、且 CUDA/frozen-hash/root-identity/action-box/finite
physical-outcome gates 与独立 witness replay 全部通过时开放；无论方向如何，都不得与前两个 panel 事后
合并来通过 gate。若 interval 仍跨零或偏向 CEM，P1 保持 blocked，并将结果解释为此冻结 local
goal-progress implementation 的精度不足或方法边界，而不是继续改变协议。

该尝试在 host CUDA 上完成前 25 个 root 的规划与部分物理执行后，于下一 root 的首次 candidate
execution 触发 `execution reset RGB differs from the planner root`，并以非零退出状态 fail closed；没有
生成 `report.json`，因此不形成 outcome receipt、不得合并、也不得用于 P1。直接的 48-root CUDA
determinism audit 显示“立即 reset + warmup replay”本身可重现，但该失败暴露出 runner 仍把**已经观测**的
planner root 的候选物理执行依赖于长 GPU planning 之后的第二次重建。这是 execution-identity 代码/协议
缺陷，不是 D2P 的负向证据；已从头读取或在日志中暴露的 `20262540--20262564` roots 永不再用作 outcome。

### Reacher D2P-P0：snapshot-root 修复后的替代确认 panel 预注册（2026-10-24）

修复后，context 在 executed warmup 的 observed RGB root 同步捕获 qpos、qvel 和 target-qpos 的
**evaluator-only** snapshot；planner 仍只读取 RGB、goal RGB 与 executed action history。每个 method
execution 先 reset episode bookkeeping，再恢复此 snapshot，随后必须重新渲染出与 planner root 完全相同的
root/goal RGB 才允许执行。snapshot 不进入 model input、frozen cost、proposal screening、damping selection
或任何 method comparison；receipt 和 CUDA witness replay 均独立验证其渲染 identity。

该变更使 runner hash 改变，旧 selection receipt 因此不兼容。先以同一、仅用于 calibration 的 roots
`20262480--20262481`，相同候选 damping `{0.01,0.1,1,10}`、相同 `128/4` proposal 与四次
relinearization，重新产生四个 calibration-only receipts 和新的 immutable selection receipt。若 selection
schema 需要补入新的 execution-root 合同字段，旧 JSON 不覆盖而另建版本
`results/d2p_reacher_p0_snapshot_r4_selection_v2_20261024_cuda.json`。随后唯一的
fresh outcome panel 固定为从未读取的 `20262600--20262647`（`n=48`）。运行合同保持 host CUDA、冻结
checkpoint/adapter、executed warmup、observed-root-anchored rollout、matched CEM (`32×4`)、`128/4`
screened D2P 和 root-paired physical execution；除 evaluator root-restoration 修复外不得再调整算法或资源。
P1 仍仅在这一个 fresh panel 的 D2P-minus-CEM final-qpos paired bootstrap 95% interval 上端 `<=0`，且
CUDA/frozen-hash/action-box/root-identity/finite-outcome gates 和独立 witness replay 全部通过时开放；不能与
任何先前或失效 panel 合并。

首次 snapshot implementation 在 calibration-only execution 中把 goal 误渲染为 `root + target_delta`；
而该 task 的真实 target 是 reset 时生成并在 warmup 后保持不变。goal-RGB identity gate 因此 fail closed，
没有写出任何 calibration receipt。修复为从保存的 `target_qpos` 直接渲染 goal 后，四个 host-CUDA
calibration receipts 全部通过，选择仍为 `damping=0.01`；selection-v2 同时绑定 runner SHA
`4d6210d75c4da5c4c05ad08a7eecb3056ed454c8764ccfb71e40bc9cc354d994` 与
`execution_root_reconstruction=evaluator_only_observed_root_snapshot_restore`。

在上述 immutable selection-v2 下，fresh roots `20262600--20262647` 的 `n=48` outcome panel 全部通过
CUDA、frozen parameter hash、D2P=Gauss--Newton、raw action box、finite physical outcome、root/goal RGB
identity 和 CEM wall-time noninferiority gates。D2P-minus-CEM final-qpos mean 为 `-0.175531`，paired 95%
bootstrap CI 为 `[-0.312659,-0.047403]`（D2P/CEM wins `27/21`）；这是首个单独满足预注册准入条件的
fresh panel，故 `p1_unlocked=true`。独立 host-CUDA witness replay 在 48/48 roots 上重放了 snapshot
root/goal RGB、normalized-to-raw action transform、action box 和 frozen terminal costs，全部通过。该结果仅
建立冻结 Reacher 上的短时 goal-progress P0 admission；它不证明 reconfiguration、program composition 或
一般 Planning Runtime，后续 P1 必须以此固定 runner/selection 为前提并使用新的 root-disjoint protocol。

### Reacher D2P-P1：先完成 reconfiguration plumbing 审计（2026-10-24）

P0 准入后，P1 的首要风险不是立即报告长程收益，而是把“预测 successor 上的局部几何”与
“物理执行后的 observed successor”混为一谈。新增
`preflight_reacher_d2p_p1_reconfiguration.py` 因而只实现并审计下列最小闭环：从 RGB/action
context 生成一个短 reconfiguration chunk；仅用该 chunk 后重新计算的 frozen local
`V_reach` 排名；要求候选相对于 no-reconfiguration **短期 frozen goal cost 更差**、但未来
`V_reach` 更低；从 evaluator-only snapshot 执行该 chunk，重新观察 RGB successor，更新 causal
action history 后再重新编译一个 P0 goal-progress suffix。qpos/qvel/target 只在执行后用于环境
outcome 和 root restoration，绝不进入候选或排名。

第一次两-root CUDA diagnostic 使用 4 个 proposal（`20262700--20262701`）只有一条 root 找到
合规 witness；16-proposal diagnostic（`20262720--20262721`）仍只找到一条。逐项审计显示失败
并非 zero/NaN Jacobian、action layout、reset identity 或 GPU 问题：失败 root 中较低 `V_reach`
候选恰好也是短期 goal-greedy，故被 non-greedy contract 正确拒绝。这个结果揭示的是 candidate
coverage 不足，不能写成 reconfiguration 的理论反例。

与已修复 P0 相同，将 model-only proposal coverage 固定为 128 后，新的主机 CUDA diagnostic
roots `20262740--20262741` 全部通过。各 root 分别选择 proposal 79 与 127；相对 no-reconfiguration，
未来 `V_reach` 分别下降 `12.6141` 与 `39.7682`，即时 frozen goal cost 分别增加 `0.07761`
与 `0.00861`。冻结 hash、RGB root/successor identity、raw action box，以及 observed-successor
二次 Jacobian 的中心差分门全部通过：
`le-wm-main/results/d2p_reacher_p1_reconfiguration_preflight_coverage128_20261024_cuda/report.json`。
为直接验证这不是 root 差异，先前 16-proposal 失败的同一 diagnostic root `20262721` 仅改变
coverage 为 128 后选中了原小 bank 中不存在的 proposal 43，future `V_reach` 下降 `34.4681`，
即时 frozen goal cost 增加 `0.18635`，全部 CUDA/root/successor-Jacobian gates 同样通过：
`le-wm-main/results/d2p_reacher_p1_reconfiguration_preflight_seed20262721_coverage128_20261024_cuda/report.json`。
proposal bank 现在明确保证 deterministic prefix stability 和 zero-program baseline，因此扩大覆盖只
追加候选、不会重写先前候选；小 bank 的失败被确定归类为 coverage defect，而非机制反例。
该 receipt 是 **plumbing admission**，不是 calibration 或 outcome panel；所有 `202627xx` roots
永不进入 P1 efficacy 分析。P1 后续只能在新鲜、root-disjoint calibration/outcome roots 上，固定
128-proposal、P0 snapshot selection-v2、总共三 action token 的相同物理预算，并与 no-reconfiguration
two-stage D2P 和 matched full-horizon CEM 比较真实 final/min qpos distance。若 physical advantage
不成立，结论只能是该已审计 reconfiguration mechanism 的边界，不能回写为 P0 或 world-model 理论失败。

### Reacher D2P-P1：物理非贪心机制审计与重启（2026-10-24）

第一个 root-disjoint P1 outcome panel（`20262860--20262871`，`n=12`）在 CUDA、冻结参数
hash、root/successor RGB identity、action box、有限 physical outcome 和三 token 等预算上均通过，
并且每个 root 都有模型侧的 non-greedy / future-reachability witness。然而它**不能**承担
reconfiguration mechanism 的负结论：模型的短期 latent cost 与 evaluator 的短期 qpos 距离并不一致。

新加入的只读 audit
`experiments/spectral_reachability/audit_reacher_d2p_p1_physical_nongreediness.py` 对不可变回执
`results/d2p_reacher_p1_reconfiguration_outcome_n12_20261024_cuda/report.json` 重新计算该机制条件：
12 个模型合格根中只有 3 个在第一个 action token 后物理上没有接近目标。其输出
`results/d2p_reacher_p1_reconfiguration_outcome_n12_physical_nongreediness_audit_20261024.json`
因此为 `fail`。这定位的是 model-to-physical short-horizon semantic alignment 的协议漏洞，
不是“reconfiguration theory 失败”；旧面板的 final-qpos 比较及其 `p2_unlocked=false` 不再用于
科学机制结论。

runner 现已把物理非贪心写为硬门，并将 frozen immediate-cost margin 参数化为
`--non-greedy-margin` 与可选的 `--max-non-greedy-margin`。新协议先仅在未使用的 calibration
roots 上，从仅在 calibration 中声明和比较的模型侧 immediate-cost 区间候选中寻找一个满足以下条件的合同：至少 4 个模型
合格根，且每个合格 root 的首 chunk 都物理非贪心。
只有该 calibration receipt 全部通过、runner SHA 和 margin 精确绑定，并与 outcome roots 严格
disjoint 时，才允许重开 12-root outcome panel。若没有任何预先声明的模型侧阈值能通过这项
calibration，P1 将停在“当前 frozen latent immediate score 无法预测物理 reconfiguration”的
模型校准边界，而绝不将其写成理论负结果。

该校准已在主机 CUDA 上完成。相同的新鲜 roots `20262900--20262905` 上，无上界
`[1e-6, +∞)`、中等区间 `[1e-6, 0.15]` 和窄区间 `[1e-6, 0.05]` 分别有
`6/6`、`6/6`、`3/4` 个 model-eligible roots 物理非贪心；三个 receipt 的其他
CUDA、frozen hash、identity、action-box、finite-outcome、future-reachability 与模型非贪心
gates 均通过。只读 reducer
`results/d2p_reacher_p1_physical_alignment_calibration_audit_20261024.json` 的状态为
`blocked_model_to_physical_alignment`，并绑定三个输入回执的 SHA-256。因此 P1 不能重开新的
efficacy outcome panel，P2 也不能解锁；这仍是**当前 frozen model 的短时 latent immediate
score 缺少物理语义校准**，不是 reconfiguration mechanism 或更一般 world-model theory 的负结论。

为防止将 evaluator 错误误诊为模型语义边界，另以
`audit_reacher_d2p_p1_first_chunk_replay.py` 对最窄区间 receipt 的六个 root 独立重放了 deterministic
warmup、snapshot restoration、记录的首 raw-action block、successor RGB 与周期环绕 qpos distance。回执
`results/d2p_reacher_p1_physical_alignment_calibration_band_000001_005_n6_first_chunk_replay_audit_20261024.json`
为 `pass`：所有初始和首段后距离的绝对误差均为 `0`，且 warmup/successor RGB identities 全部重现。
所以当前阻断不能归因于 action normalization、首段 action witness、snapshot、目标角度环绕或 qpos
记录代码；唯一保留的解释是 frozen RGB latent immediate objective 未提供足以预测该物理方向的信号。

### Reacher D2P-P1：zero-prefix 修复后的模型—物理对齐边界（2026-10-24）

继续 P1 前先排除了两个会制造假阴性（或假机制）的协议缺陷。第一，物理“非贪心”原先将候选首段后的
qpos 与执行前 root 比较；warmup 后 root 可能带速度，所以正确反事实应是**相同 snapshot、相同 action-block
长度的 zero-prefix**。第二，P1 原先没有真正的 zero-prefix two-stage control；现已要求 zero prefix 后也从
其观测 successor 用同一 D2P suffix compiler 重编译。独立 replay 对 warmup、snapshot、raw action、successor
RGB 和角度环绕 qpos distance 的逐值误差均为零，排除了 action transform、snapshot restoration、qpos 记录和
zero-control 实现作为原因。

在修复后，默认 model-ranked prefix 于 `20263100--20263105` 有 `5/6` 个物理非贪心。候选审计显示失败不是
“无可行候选”：同一 frozen top-k bank 中经常存在物理非贪心 chunk，但最低 `V_reach` 的模型排序没有稳定挑中它。
因此随后只尝试了两条不读取 qpos 的、显式写入 receipt 的模型侧规则，并都使用了新的、不重叠的 roots：

1. relative-`V_reach` 上界 `0.20`：preflight `20263200--20263201` 通过；calibration
   `20263210--20263215` 只有 `3/6` model-eligible roots 物理非贪心。
2. post-prefix spectral-balance 下界 `sigma_2 / sigma_1 >= 0.61`：preflight
   `20263240--20263241` 通过；calibration `20263250--20263255` 中只有 4 个模型合格 roots，且 **0/4**
   物理非贪心。

所有上述 preflight/calibration 都在主机 CUDA 上运行，且 CUDA、frozen parameter hash、RGB root/successor
identity、action box、finite physical outcome、future-reachability 和模型侧 non-greedy gates 均通过；唯一失败的是
physical-non-greediness gate。聚合 reducer
`le-wm-main/results/d2p_reacher_p1_physical_alignment_calibration_extended_v2_20261024.json` 状态为
`blocked_model_to_physical_alignment`。因此不打开 12-root efficacy outcome，不解锁 P2，也不把这个结果叙述为
D2P、谱理论或世界模型理论的反例；可辩护的边界只是：这个冻结 Reacher checkpoint 的短时 latent criteria 目前
不能稳定识别会在真实环境中改变未来可达几何、同时又不立即接近目标的 prefix。继续工作必须改善模型的物理语义
或另立新预注册机制，而不是继续事后搜索阈值。

### Reacher D2P-P1：候选测度与 preflight 接口修复后的复核（2026-10-24）

在把上述边界解释成模型问题之前，又完成了一次代码级审计。cross-environment calibration 的原始 Reacher actions
由环境 action Box 均匀采样；其十维 block 的标准差为 `0.54--0.60`，与均匀 `[-1,1]` 的 `0.577` 一致。
但旧 P1 bank 在**无界参数**中采样 `N(0,1)`，随后用 `tanh` 约束；它实际给出的物理动作是 `tanh(N)`，并不是
训练/校准所用的均匀 Box 分布。因此代码已改为先采样物理 `U[-1,1]`，再用 `atanh` 得到优化参数；历史
`tanh(N)` bank 仅保留为显式 legacy replay 选项。候选 bank 分布也写入所有新 receipt，不能与旧 receipt 混用。

同轮还修复了 P1 preflight 的路径契约：preflight 输出是含 `report.json` 的 immutable directory，后续 runner 和
candidate audit 现显式接受该目录（也兼容报告文件），并在 calibration/outcome binding 中检查 proposal distribution。
该问题曾使一次 calibration 在读 preflight 时立即失败；它是 orchestration bug，而非实验结果。

新的 host-CUDA preflight（roots `20263320--20263321`）在
`le-wm-main/results/d2p_reacher_p1_reconfiguration_preflight_physical_uniform_r2_20261024_cuda/` 完整通过：CUDA、frozen
hash、执行 RGB identity、action-box、nonzero FD Jacobian 和模型侧 witness 均通过。随后在未用于旧校准的 roots
`20263330--20263335` 上，以 `physical_uniform` 的 128-candidate bank 重跑了 six-root calibration；receipt
`le-wm-main/results/d2p_reacher_p1_physical_uniform_calibration_r2_n6_20261024_cuda/report.json` 的模型 eligibility 为
`6/6`，但物理 non-greediness 仍为 `3/6`。因此分布错配是必须修复的代码/协议风险，却**不足以解释或消除** P1
的阻断，不能把旧的负向观测简单称作 sampling bug。

为分辨“候选池没有物理非贪心动作”和“模型排序没有选对”，又在 root `20263330` 上固定模型 top-8 后才执行物理
诊断。`d2p_reacher_p1_physical_uniform_candidate_prefix_audit_seed20263330_20261024_cuda.json` 显示 top-8 中有 `3/8`
个相对 zero-prefix 的物理非贪心 candidate，但模型 rank 1--3 全为物理贪心；rank 4、6、8 才满足物理条件。这再次
指向 frozen latent `V_reach` 的**排序语义失准**，而不是 action-box、CUDA、snapshot、动作变换、候选支持不足或
CPU fallback。P1 继续保持 `blocked_model_to_physical_alignment`；下一种方法必须在新的、预先声明的物理语义学习
或可观测校准机制下建立，而不能以 qpos 事后调排名规则。

### Reacher D2P-P1：有界 suffix selector 的门控一致性修复与复核（2026-10-24）

上述 `physical_uniform` 复核又暴露出第三个、会造成假阴性的代码问题。selector 已改为按实际会经由 `tanh`
执行的 `bounded_dls_suffix_goal_cost` 排序，但 preflight 仍把旧的无界解析 surrogate
`analytic_v_reach` 当作通过门。两者可以排序相反：例如旧 preflight root `20263340` 的 bounded
improvement 为 `+0.053756`，但 analytic improvement 为 `-21.521729`。这不是方法未改善其**声明的
选择目标**，而是 eligibility/selection 两处使用了不同的目标函数。

代码现将“声明的 future-score improvement”抽为单一映射，并由 preflight 和 P1 runner 共同调用；新的门名也明确为
`every_selected_chunk_improves_declared_recompiled_future_score`。相关 Reacher-D2P 单元、上下文及 runner
测试共 `41 passed`，并通过 `py_compile` 和 `git diff --check`。修复后在主机 CUDA 上用新 roots
`20263342--20263343` 重跑预检，所有 CUDA、frozen hash、RGB identity、action-box、FD Jacobian、模型侧
non-greedy 和**声明的有界 score** gates 均通过：
`le-wm-main/results/d2p_reacher_p1_reconfiguration_preflight_bounded_dls_r2_20261024_cuda/report.json`。

随后以不重叠的 calibration roots `20263350--20263355` 执行 six-root host-CUDA run。新的
`bounded_dls_suffix_goal_cost` 规则使模型 eligibility 为 `6/6`，且所有非物理 gates 均通过，但仍只有 `3/6`
相对 same-snapshot、same-length zero-prefix 的首段在物理上非贪心。独立 reducer
`le-wm-main/results/d2p_reacher_p1_bounded_dls_calibration_audit_20261024_cuda.json` 的状态为
`blocked_model_to_physical_alignment`。因此这次不能打开 12-root outcome：已排除的三项分别是 proposal
分布错配、preflight 路径契约和 bounded selector/legacy gate 指标错配；剩下的是该冻结 checkpoint 的模型排序与
真实物理 reconfiguration 几何尚未可靠对齐，而不是 D2P/谱理论的负结论。

### Reacher D2P-P1：selector 与执行 suffix 的 solver 对齐审计（2026-10-24）

继续审计发现，前一版 P1 prefix 以 predicted successor 上的 `bounded_dls_suffix_goal_cost` 选择，但首段执行后，
实际 suffix 却切换成 screened 128-proposal、最多四次重线性化的 goal-progress solver。这使候选的排名目标与
实际执行的后续程序不同。代码现改为在 observed successor 上重新编译**同一** `bounded_single_step_dls` solver；
它仍只使用 RGB/action history，且 receipt 显式记录 `selection_equivalent_suffix_solver: true`。这是一项
selector/executor 一致性修复，不是用 qpos 改变候选排名。

新的 host-CUDA preflight（roots `20263360--20263361`）全部通过；随后以完全不重叠的 roots
`20263362--20263367` 做 six-root calibration。所有 6 个 roots 均模型 eligible，所有非物理 gates 通过，且每个
执行的 suffix 都确认是 `bounded_single_step_dls`；但首段物理 non-greediness 为 **0/6**，独立 reducer
`le-wm-main/results/d2p_reacher_p1_suffix_aligned_calibration_audit_20261024_cuda.json` 仍为
`blocked_model_to_physical_alignment`。这一结果还带来一个重要限定：solver mismatch 是必须修复的 P1 efficacy
协议缺陷，但物理 non-greediness gate 在 suffix 执行前就已测量，所以它不能解释旧 calibration 的首段失败。

至此，代码审计已排除候选分布、路径契约、selector/gate 指标、selector/executor solver、动作 bounds/normalization、
snapshot/RGB identity、zero-prefix control 与 CPU fallback；当前可辩护的诊断是 frozen checkpoint 的 RGB latent
目标与未来可达性分数无法稳定识别真实环境中的首段非贪心 reconfiguration。P1 保持关闭，不能依靠已打开 roots
再调阈值、beta 或 qpos 规则；任何后续机制必须预先声明、使用新的 roots，并明确构造新的 RGB 可观测物理语义。

### Reacher D2P-P1：物理标签与候选支持的独立复核（2026-10-24）

为避免把 calibration 的 `0/6` 误读为候选池没有 reconfiguration 动作，继续审计了目标和物理标签：goal RGB
由同一 wrapped `target_qpos` 渲染；qpos evaluator 只比较两个关节并处理角度环绕；candidate 与 zero-prefix
都从同一 execution snapshot 运行相同五步 block，且 root/goal RGB identity 是 fail-closed 的。这个标签路径没有
发现坐标、目标、时间步或 snapshot 错位。

随后在未用于任何 preflight/calibration/outcome 的 root `20263368` 运行 host-CUDA fixed-top-8 prefix diagnostic，
只在 frozen model 的 128-candidate 排名固定后读取 qpos。receipt
`le-wm-main/results/d2p_reacher_p1_suffix_aligned_candidate_prefix_audit_seed20263368_20261024_cuda.json` 显示
top-8 中 `7/8` 相对 zero-prefix 物理非贪心，且 model rank-1 也非贪心。这证明该 bank 可以包含、且模型有时能
排中所需动作；它不能挽回 six-root calibration 的 `0/6`，但将失败进一步定位为**跨 root 不稳定的模型排序/物理语义**，
而不是 candidate support 缺失或 evaluator bug。由于该诊断是后置机制定位，它不能重开 outcome 或选择新的阈值。

对 calibration 中失败的 root `20263362` 进行同样的 fixed-top-8 replay 后，原因进一步具体化：model rank-1
候选的 bounded score 为 `0.500621`，但相对 zero-prefix 的物理距离差为 `-0.010927`（即时更贪心）；rank 2--8
全部物理非贪心，差为 `+0.114361` 至 `+0.286195`。因此该 root 不是“没有满足 P1 物理机制的候选”，而是 frozen
bounded-DLS score 将物理贪心 candidate 排在了这些候选之前。该 qpos replay 只用于诊断，严禁把 rank-2 的标签、
差值或由此导出的权重写回 selector；现有 P1 outcome 仍关闭。

### Reacher D2P：full action-packet warmup support 修复（2026-10-24，重验进行中）

在把上述排序失准写成 checkpoint 边界前，进一步追踪了训练样本和在线 history token 的时间结构。训练 loader 对每个
`frameskip=5` token 保留五个逐环境步的 raw actions，并与该 packet 起点 RGB 对齐；checkpoint 的十维 action
输入因此不是“一个动作重复五次”。旧的 `seeded_bounded_excitation_v1` warmup 却把单个、仅覆盖物理 box 35% 的动作
复制五次，构成低秩且低幅度的 history-token 子流形；P1 candidate prefix 已经使用完整、逐时间步的物理 box。这是
模型对候选排序时所条件的 history 分布与其候选分布不一致的实现/协议风险，而不是可直接归因给 D2P 的负结果。

`reacher_d2p_context.py` 已将该 contract 改为 `seeded_full_packet_uniform_v2`：每个 warmup packet 的五个原始动作
独立且确定性地从环境 Box 采样，并按 checkpoint 的 time-major layout 记录、执行和哈希。P0/P1 preflight、P1 runner
与 candidate diagnostic 现在显式绑定该 policy；任何 `v1` receipt 都会 fail closed，不能被新 runner 或 outcome
绑定复用。新的 non-constant packet、policy binding 和既有 root/action contracts 共 `43` 个测试通过（这是逻辑
验证，非效力结论）；加入 immutable-output fail-closed contract 后为 `44` 个通过。此前 P0 runner/preflight 允许
`exist_ok=True`，会静默覆盖同名 receipt；该缺陷也已改为拒绝已存在的输出目录，第一次错误的 systemd shell
变量展开只生成了被覆盖的 `d_n4` 目录，未进入 selector 或任何科学报告。

新的主机 CUDA 重验证已完成。roots `20263460--20263461` 的 P0 preflight 通过 CUDA、frozen hash、root identity、
action-box、future-suffix Jacobian 和 central-FD gates。随后在 calibration roots `20263470--20263473` 上运行五个
独立 damping receipts（`0.001, 0.01, 0.1, 1, 10`）；所有 gate 通过，预注册 selector 仅按 calibration D2P
final-qpos 选择 `damping=0.01`。根不重叠 outcome `20263500--20263511` 同样全部通过 CUDA、冻结哈希、D2P=local-GN、
raw-box、有限 physical outcome 与 execution-root identity gates，但 D2P minus matched CEM 的 mean 为 `+0.299469`
（负值才有利于 D2P），paired bootstrap 95% CI 为 `[-0.052704,+0.656729]`，12 roots 中 D2P/CEM 分别获胜 `5/7`。
故 `d2p_reacher_p0_packet_support_v3_outcome_n12_20261024_cuda/report.json` 的 `p1_unlocked=false`；后续 P1
preflight 自动拒绝执行。这是 full-packet support 修复后该冻结 checkpoint 的**P0 效力未成立**，而不是 P1、D2P
整体或谱理论的反例。旧 P1 物理负向结果仍只能作为发现 support mismatch 的诊断；新 P1 不得绕过新的 P0 gate。

### Reacher D2P-P0：默认 proposal budget 未生效的配置回归（2026-10-24，重验完成）

对上述 full-packet P0 回执作第二次代码—回执交叉审计后，发现 runner 的默认值当时为
`d2p_initialization=gaussian_restarts` 且 `d2p_restarts=1`。该分支只有零动作的局部 DLS 起点；尽管协议和命令行
同时记录了 `d2p_proposal_samples=128`、`d2p_proposal_elites=4`，这些字段在该分支没有参与候选生成或排序。相反，
matched CEM 实际执行 `32 x 4 = 128` 次候选评估。因此这不是 qpos outcome 后的参数调整，而是一个默认配置与已声明
资源合同不一致的实现/协议回归：D2P 所声称的全局 proposal coverage 没有发生，旧的 full-packet outcome 不能作为
screened-D2P 相对 CEM 的效力结论。

runner 已改为默认 `screened_gaussian_proposals`：固定、仅模型内的 128 个 Gaussian action-packet proposals（其中
proposal 0 是零动作基线）先按冻结目标筛选 4 个 elite，再各自执行 4 次局部 DLS 重线性化；若显式选择
`gaussian_restarts`，回执会明确标出两个 proposal 字段被忽略，避免再次把未执行的预算写成证据。screened 模式同时
拒绝无意义的多 restart 设置。相关 P0/context/P1 测试共 44 项通过，但这只是实现验证。

这个新 runner 改变了 calibration contract hash，因此已以未使用根完成整条重验链：CUDA preflight
`20263600--20263601` 全部通过；五个共享但 outcome-disjoint 的 4-root calibration-only receipts
`20263610--20263613` 覆盖 damping `{0.001, 0.01, 0.1, 1, 10}`，全部通过 CUDA、冻结、raw-box、RGB root identity
和 D2P=local-GN gates。selector `results/d2p_reacher_p0_screened_default_fix_selection_20261024_cuda.json` 只依据
这些 calibration roots，冻结选择 `damping=10.0`。

该选择随后绑定到新的 12-root physical outcome `20263630--20263641`：
`results/d2p_reacher_p0_screened_default_fix_outcome_n12_20261024_cuda/report.json`。每一个 D2P row 都实际执行
`screened_gaussian_proposals`、128 个 proposals 和 4 个 elites（不是仅记录参数）；其 forward rollout 计数为 260，
matched CEM 为 128，并且 CUDA、冻结参数、raw action box、有限物理结果、执行 root RGB identity 和 D2P=local-GN
gates 全部通过。D2P-minus-CEM final-qpos mean 为 `-0.190292`，D2P/CEM root wins 为 `8/4`，方向与旧的
`+0.299469` 相反，直接证实默认 budget 回归足以改变效应方向。

但该 fresh panel 的 paired bootstrap 95% CI 仍为 `[-0.581613, +0.069129]`，跨越零；stage gate 因而明确写为
`p1_unlocked=false`。结论是：修复已消除一个实质性代码/协议替代解释，并给出有利 point estimate，但 P0 尚未达到预注册
的正向准入标准；P1 及后续需要 P1 准入的实验继续 fail closed。既不能把旧的 `+0.299469` 当作理论反例，也不能把新的
负均值当作 D2P 效力成立。

### Reacher D2P-P0：重放与模型—物理排序边界审计（2026-10-24）

为继续区分“仍有代码错误”与“冻结模型能力边界”，对上述 outcome 运行了独立 host-CUDA witness replay。第一次启动
暴露了审计脚本的真实入口缺陷：以文件路径执行时没有把项目根目录加入 `sys.path`，因而在导入 `experiments` 时失败；
这已在 `audit_reacher_d2p_p0_witness_replay.py` 修复，并由 direct-entrypoint 回归测试覆盖。首次可运行的审计还把
“至少一个 root 上 CEM 的冻结模型 cost 更低”错误地列为通过条件；当 D2P 在所有 root 的模型 cost 都更低时，这个方向性
假设会把有效回放误报为 fail。该 gate 已改为只要求所有重放 cost 有限并记录两种方向的计数。

修复后，host-CUDA receipt
`results/d2p_reacher_p0_screened_default_fix_witness_replay_audit_v2_20261024_cuda.json` 为 pass：12 个 roots 的
normalized-to-raw action、action box、frozen terminal cost、warmup/snapshot root RGB identity 均精确重放。它同时记录
模型端 D2P 的 terminal cost 在 `12/12` roots 低于 CEM。这排除了 action reshape、stale score、执行 root 漂移和
错误回执作为 P0 未准入的解释。

随后只读的 `audit_reacher_d2p_p0_model_physical_alignment.py` 将该回执与物理 outcome 逐 root 配对：
`results/d2p_reacher_p0_screened_default_fix_model_physical_alignment_20261024.json` 显示模型排序和 physical final-qpos
排序只在 `8/12` 个严格 roots 一致、在 `4/12` 个反转；D2P 模型 cost 优势为 `12/12`，真实 final-qpos 优势为 `8/12`。
留一 root 分析还显示移除 seed `20263640` 后 paired mean 由 `-0.190292` 变为 `-0.014219`。因此剩余的未通过原因是
冻结 latent terminal objective 的模型—物理校准边界及高异质性，而非已知代码路径错误。该结果是 post-outcome 诊断：
不得用四个反转 root 的 qpos 来改 proposal、damping 或 rank；任何后续新 P0 协议若要继续，必须在独立 calibration roots
上预先冻结模型—物理排序准入，再使用新的 root-disjoint outcome。

### Reacher D2P-P0：执行 witness 的 rollout-fidelity 审计（2026-10-24）

为排除“模型—物理排序反转其实来自审计器对渲染 RGB 的错误处理”，新增只读、CUDA-only 的
`audit_reacher_d2p_p0_rollout_fidelity.py`：它从同一 sealed receipt 重建每个 snapshot root，执行已经固定的
raw action witness，并逐 packet 比较 frozen rollout latent 与实际 successor RGB 的 encoder latent；全程不读取
qpos，也不改变 candidate、damping 或 rank。首次主机 CUDA 启动在该审计器本身暴露了一个独立实现缺陷：MuJoCo 的
`env.render()` 有时返回 channel-negative-stride 的 NumPy view，而 torchvision/PyTorch 拒绝该 buffer。审计器现将
RGB 显式 materialize 为 contiguous `uint8` HWC array，负-stride 回归测试加入后，相关 P0/context/fidelity 定向套件
为 `20 passed`。此修复只恢复审计的正确运行，不会修改已经 sealed 的 P0 planner 或 outcome。

修复后的 host-CUDA receipt
`le-wm-main/results/d2p_reacher_p0_screened_default_fix_rollout_fidelity_v2_20261024_cuda.json` 为 pass（RTX 5060 Ti / CUDA
13.0，12 个 sealed roots，所有误差有限，且 24 个重放 terminal cost 均在 `2e-5` 内逐条匹配 sealed receipt）。两个 action packet 的 mean latent L2 分别为 D2P `1.997712`、`3.164282`，
CEM `1.605477`、`3.072550`；即 D2P 的 executed witnesses 分别高 `0.392235`、`0.091733`。因此 action witness
serialization、snapshot reconstruction 与 successor image encoding 已再次排除；但 frozen rollout 对 D2P 轨迹的局部预测
较差，这与前述 `4/12` model--physical order reversals 相容。它是当前 checkpoint/protocol 的 rollout-calibration
boundary，而非 D2P、谱理论或 P1 的效力结论；P1 继续保持关闭，且不得用这些已打开 roots 调整方法。

### Reacher D2P-P0：模型—物理校准前置门与独立复验（2026-10-24）

既然 sealed outcome 已显示 rollout-calibration 异质性，后续不能再把模型 terminal-cost 直接当作足够的物理
代理。新建 `select_reacher_d2p_model_physical_admission.py`，预先规定一个 calibration-only 准入：在至少 8 个
fresh roots 上，D2P--CEM 的严格模型 cost 排序须至少 `7/8` 与 evaluator-only final-qpos 排序一致，并以精确单侧
sign test 要求 `p<=0.05`；否则不允许打开新的 P0 outcome roots。该 gate 不使用 qpos 选择 action、proposal 或 damping，
只判断冻结模型是否有资格支撑下一面板的模型排序。

在新 calibration roots `20263700--20263707` 上，所有 CUDA/frozen/action-box/RGB identity/D2P=GN gates 通过，
排序 agreement 为 `7/8`、`p=0.03515625`，故 admission pass：
`le-wm-main/results/d2p_reacher_p0_modelphysical_calibration_admission_20261024_cuda.json`。它与既有的四个 damping-only
roots 以不可变 binding 合并，形成 12 个 calibration roots；随后才开启完全 root-disjoint 的 outcome roots
`20263760--20263771`。

新 outcome receipt
`le-wm-main/results/d2p_reacher_p0_modelphysical_admitted_outcome_n12_20261024_cuda/report.json` 的全部实现 gates 通过，
且独立 CUDA witness replay 也通过 action transform、action box、snapshot RGB identity 与 terminal-cost replay：
`le-wm-main/results/d2p_reacher_p0_modelphysical_admitted_outcome_n12_witness_replay_20261024_cuda.json`。D2P-minus-CEM
final-qpos mean 为 `-0.243902`，wins 为 `7/5`，但 paired bootstrap 95% CI 为 `[-0.617956,+0.095139]`，仍未满足
预注册的 P1 准入条件。因此校准门修复了“未经验证的模型排序代理”这个协议缺口，却没有把不确定的 efficacy 结果伪装成
通过；当前边界是 effect precision/heterogeneity，P1 继续 fail closed。

### Reacher D2P-P0：扩大 confirmation panel（预注册，2026-10-24）

为区分真实但尚不精确的效应与剩余 root heterogeneity，下一面板固定使用未读 roots
`20263800--20263847`（`n=48`）。它复用 `modelphysical_admitted_selection` 的 damping `10.0`、同一冻结
checkpoint/adapter、executed full-packet warmup、observed-root snapshot、`128/4` screened D2P、四次
relinearization 与 matched `32×4` CEM；不得更改 action bounds、proposal distribution、horizon、预算或任何选择规则。
所有 12 个 calibration roots 与此前 outcome roots 都不得进入此 panel。P1 只在该**单一**新 panel 的
D2P-minus-CEM paired bootstrap 95% CI 上端 `<=0`、所有 CUDA/frozen/action-box/RGB identity/finite-outcome gates
通过、且独立 CUDA witness replay 通过时解锁；不得与前述 `n=12` outcome 做事后 pooled admission。否则 P1--P5
继续 fail closed，并将结论限于该 checkpoint/protocol 的效应精度或方法边界。

扩大 confirmation 已完成。48 个完全 fresh outcome roots 上全部 CUDA/frozen/action-box/RGB identity/finite outcome/
D2P=GN/CEM wall-time gates 通过；D2P-minus-CEM final-qpos mean 为 `-0.262609`，wins `34/14`，paired 95% CI 为
`[-0.427273,-0.112204]`。独立 host-CUDA witness replay 在 `48/48` roots 上重放了 action transform、action box、
snapshot RGB identity 与 frozen terminal cost，均通过，并记录 D2P 在 `48/48` roots 的 frozen terminal cost 低于 CEM：
`le-wm-main/results/d2p_reacher_p0_modelphysical_confirmation_n48_20261024_cuda/report.json` 与
`le-wm-main/results/d2p_reacher_p0_modelphysical_confirmation_n48_witness_replay_20261024_cuda.json`。因此该单一、
预注册 panel 的 `p1_unlocked=true`；它只准入 P1 的 calibration-only plumbing 和物理机制验证，不证明 reconfiguration、
program composition 或跨任务效力。

### Reacher D2P-P1：fresh full-packet 校准的双侧回放审计（2026-10-24）

在 P0 的独立 confirmation 准入后，P1 先在 fresh roots `20263900--20263903` 完成 CUDA plumbing
preflight；CUDA、冻结 hash、raw action box、root/successor RGB identity、模型侧 non-greedy witness、声明的 future-score
improvement 与 observed-successor FD Jacobian 均通过。随后用完全不同的 calibration roots
`20263920--20263923` 运行 full-packet warmup 的 P1 机制校准。四个 root 都先由冻结 RGB/action-history
模型判为 eligible，且不读 qpos 地固定候选；但仅 `3/4` 个首段相对同 snapshot、同五步 raw-zero prefix
在物理上非贪心。失败 root `20263923` 的候选减 zero-prefix qpos distance 为 `-0.280763`：它在模型侧被判为
立即更差、未来 score 更好，却在物理上反而更接近目标。

为排除这是记录或执行代码问题，`audit_reacher_d2p_p1_first_chunk_replay.py` 已升级为双侧回放：逐 root 重建
warmup/snapshot，同时重放 candidate 与 raw-zero control，校验两者的 successor RGB、qpos 距离和动作哈希。升级过程
暴露并修复了审计器自身的一个假阴性：它曾错误地以 `1e-8` qpos replay 容差判断 float32 反归一化后的 raw-zero
动作；实际最大舍入残差约 `4.5e-8`。该动作契约现独立使用 `1e-6` 容差，且加入回归测试，不能再把正常 float32
round trip 报成非零控制。

修复后的主机 CUDA receipt
`le-wm-main/results/d2p_reacher_p1_reconfiguration_calibration_warmupv2_n4_bisided_replay_v2_20261024_cuda.json`
为 `pass`（RTX 5060 Ti / CUDA 13.0）：4/4 warmup root、candidate successor、zero-prefix successor、初始/候选后/
zero 后 qpos distance 均精确重放，且 4/4 raw-zero controls 通过。相关 P1/context/preflight 定向测试为 `29 passed`。
因此这次 `3/4` 并非 action normalization、零 control、snapshot restoration、路径复用或 qpos 记录造成的负结果；它只能
作为该 frozen checkpoint 当前短时 latent 排序与物理即时方向不稳定对齐的 calibration boundary。P1-qpos outcome 与 P2--P5
继续 fail closed；不得以这四个已打开 root 的 qpos 调整 margin、rank 或候选选择规则。

随后，新的 host-CUDA `first_chunk_fidelity` 审计以相同四个**已封存、只作诊断**的候选/zero packets 比较模型预测 successor
latent 和真实 renderer successor 的 encoder latent。它在 4/4 roots 上复现 warmup、candidate/zero successor 和
normalized-to-raw packet；candidate-vs-zero 的模型 goal-cost 排序与真实观测 encoder goal-cost 排序也为 4/4 一致。候选的
mean latent-L2 为 `2.051507`，与已通过 P0 D2P 首 packet 的 `1.997712` 同量级。因此没有证据支持“首块动作被错误执行、重复
bounding、错误编码或预测/真实排序反转”这一代码解释；失败的精确原因是视觉可观测目标代价和 evaluator-only qpos 即时距离并不等价。
回执：`le-wm-main/results/d2p_reacher_p1_reconfiguration_calibration_warmupv2_n4_first_chunk_fidelity_20261024_cuda.json`。

为避免把这个 measurement mismatch 伪装成原机制通过，新增**独立的 P1-Observable 分支**：候选仍只由冻结 RGB/action-history
模型以原有 future score 固定；但机制 gate 改为 candidate 与 same-root raw-zero prefix 在**实际执行后渲染的 successor RGB**上，
经同一冻结 encoder 得到的目标代价是否非贪心。qpos 不进入候选、该 gate 或任何调参，只保留为独立的最终 qpos outcome。该分支的
receipt 显式记录 `mechanism_observability=observed_encoded_goal_cost`，与 P1-qpos 不可互换；必须用从未读过的新 calibration roots
重跑，成功后才可绑定一个 root-disjoint outcome panel。它改变的是可观测机制的定义，不是对已打开 qpos 结果的重标或事后通过。

该新分支现已在 fresh roots `20264020--20264025` 运行主机 CUDA calibration：6/6 roots 均有冻结模型 non-greedy
witness、声明的 future-score improvement、CUDA/parameter-hash/action-box/RGB identity/finite-outcome gates 都通过；但实际 successor
RGB 上的 observable mechanism gate 仅为 `5/6`。失败 root `20264024` 的模型 non-greedy margin 为 `+0.023397`，bounded-DLS
future-score improvement 为 `+0.895233`，但实际 encoded goal cost 的 candidate-minus-zero 为 `-0.009321`（`2.002418 - 2.011738`），
即模型预测的首块排序没有保留到真实观测。回执为
`le-wm-main/results/d2p_reacher_p1_observable_calibration_n6_20261024_cuda/report.json`。因此 P1-Observable 也不得绑定 outcome；
这排除了“qpos 只是选错机制测量坐标”的解释，并将当前 fail-closed 原因精确限定为该冻结 checkpoint/protocol 的短时
prediction-to-observation ranking calibration 边界。不得用这六个已打开 roots 调整 margin、proposal bank 或 selector。

最后的代码级时间对齐审计还将该结论固定为回归约束：训练目标以
`[o_{t-2},o_{t-1},o_t]` 和 `[a_{t-2},a_{t-1},a_t]` 预测 `o_{t+1}`；root-anchored P1 adapter
在丢弃 pre-context frame/action 后使用恰好同一因果 action window，而不是错误地保留被丢弃的首 action。新的 slot-sensitive
测试会在该 window 偏移时失败；P1/context/fidelity 定向套件现为 `48 passed`。因此 temporal off-by-one 也不是当前
calibration failure 的代码解释。

#### Reacher-D2P：训练期 successor-target 梯度合同诊断（2026-10-24，官方源复验进行中）

上述审计覆盖的是**已发布 checkpoint 的运行时**：action layout、raw/normalized
transform、warmup、snapshot、因果时间窗、RGB encoder 和实际 successor 排序。它们不能
证明发布 checkpoint 的训练目标本身没有缺陷。随后对 `train.py` 的 LeJEPA prediction
loss 做反向梯度审计，发现未来 observed-successor embedding 曾直接参与
`(pred_emb - tgt_emb)^2`，却未从该 prediction-loss 图中分离。这让 encoder/projector
可以通过移动未来 RGB target 来降低误差，而不是仅要求 action-conditioned predictor
解释 successor；对 P1 的“真 action chunk 应优于反事实 chunk”排序合同尤其不利。

该行现已改为 `tgt_emb = emb[:, n_preds:].detach()`，并有梯度级回归测试：仅出现在未来
target 的末帧不得接收 prediction-loss gradient。另设独立的诊断性 repair trainer：固定
encoder/projector，只优化 predictor、action encoder 与 prediction projection，使用
detached observed-successor latent 的 all-position MSE 加上 true-action 对 in-batch
action-swap 排序损失。训练/验证按 episode 而非 clip 切分，并固定使用在线 runtime 的
十维 action-packet normalization receipt，避免以本地 HDF5 重算统计量悄然改变执行合同。
该 repair branch 还显式复刻原入口的边界行为：HDF5 对 episode 末端缺失 action 的 NaN
必须先通过 z-score、再映射为 neutral normalized token `0`；若先把 raw NaN 置为 `0`，在
非零 action mean 下会伪造一个非零模型动作。该条件已有独立回归测试。

这不是对旧 P1 failure 的事后翻案，也不是 D2P 的最终方法：改变权重违反最终论文的
frozen-model 约束。它是一个可证伪的**训练代码诊断**。正式判定必须使用
`quentinll/lewm-reacher` 的官方训练 archive、host CUDA、全新的不可变输出目录和 fresh
root-disjoint P0/P1 gates：若 repaired checkpoint 仍不能通过 observed-successor
calibration，则该缺陷不足以解释负结果；若通过，也只能说明训练合同修复了 checkpoint
能力边界，之后仍须以该新冻结 checkpoint 重新完成 P0 admission 与 P1 outcome，才能支持
任何 D2P 机制结论。合成 Reacher pool 只可作为 smoke/diagnostic source，不能作为这条
判定的正式证据。

#### Reacher-D2P P1：margin-band 选择收据的运行时绑定缺口（2026-10-24）

对官方 checkpoint 的 margin-band P1 链作代码级复核后，发现 selector 收据虽记录了其来源 runner 的
SHA-256，却没有要求 fresh calibration 的 runner 与该来源相同；同时缺少对 reconfiguration/progress horizon、
proposal 数、target delta、warmup policy 及两条可选谱/relative-reachability guard 的绑定。这些量都会改变
可被 margin band 接纳的候选分布。因此，旧收据
`results/reacher_d2p_official_causal_repair_20261024_cuda_training_stats_v7/p1_margin_band_guard_selection_v1.json`
的来源 runner hash 与执行 fresh calibration/outcome 的 runner hash 不同，不能再支持对
`p1_margin_band_outcome_v1` 的科学归因：该 outcome 不是理论负结果，也不是可封存的 checkpoint 边界。

代码已 fail-closed：selector 现写入并绑定上述完整 planner geometry；P1 runner 会逐项验证，并拒绝不同
runner hash。用旧 selector 收据对新 runner 作只读验证会得到
`ValueError: P1 margin-band selection is incompatible on runner_sha256`；相关 P1/context/fidelity/repaired-training
测试共 `54 passed`。这只修复 provenance/协议合同，未更改候选排名、模型权重或任何已封存的物理结果。要重新判定 P1，
必须按**同一 runner**生成新的 calibration-only selector，再以全新、与 calibration root-disjoint 的 host-CUDA
calibration 和 outcome 运行；在此之前 P2--P5 保持关闭。

#### Reacher-D2P P1：fidelity 输入绑定修复与全新 root 的 v6 fail-closed（2026-10-24）

对上一项 runner-geometry 修复链继续审计时，发现独立 `first_chunk_fidelity` replay 仍有一个更窄、但会影响
selector 证据含义的实现缺口：它会从 CLI 重新加载 checkpoint、action adapter 与 `ReacherGoalImageWrapper` 的
target geometry，却没有逐项验证它们等于 source P1 receipt。默认 `target_delta` 恰好与当前协议相同，因而不是已
观察到的数值差异；但若之后调用者更换本地 checkpoint、adapter 或 target，该 audit 仍可能生成一个看似 completed 的
fidelity receipt。现在 replay 强制验证 weights/adapter SHA-256，从 source receipt 读取并验证二元 `target_delta`，并
将这些 binding 与 audit-runner hash 写入输出。margin-band selector 还要求 fidelity 的 source-report SHA 和全部
runtime binding 匹配 source calibration。相应的 P1 定向测试为 `53 passed`；这修复的是审计证据合同，不改变 frozen
planner、候选排序或 checkpoint。

旧 v4 的 outcome 因而仅保留为诊断；一次 v5 回归启动在 source calibration 后因运行中修改 launcher 的 shell
变量而 fail-closed 退出，未生成 fidelity、selector、bound calibration 或 outcome，不进入任何科学比较。随后启动器
允许显式传入 seed base，并从其确定性生成四段不交叠 roots。正式 v6 在 host CUDA（RTX 5060 Ti / CUDA 13.0）使用
全新 preflight `20270200--20270203`、source calibration `20270300--20270315`、bound calibration
`20270400--20270415` 与预留 outcome `20270500--20270523`；启动后未改动任何被该证据链读取的代码或脚本。

v6 preflight 通过 CUDA-only、冻结参数、raw action box、root/successor RGB identity、非零 FD Jacobian 与 P0
admission gates。source calibration 的 15 个 model-eligible roots 经过输入绑定的独立 CUDA replay 后，有 14 个保存
model-to-observation 首段排序；selector 仅从无翻转 source roots 冻结 margin band
`[0.0252213478, 0.3558989763]`。这一选择在 root-disjoint bound calibration 不能泛化：16 个 bound roots 中
10 个 model-eligible，9/10 在实际 successor RGB 的 frozen-encoder goal-cost 下仍为 non-greedy；seed `20270412`
则从模型侧 `+0.02783656` 翻转到观察侧 `-0.01326418`。独立 fidelity replay 对同一 root 得到
`+0.02783608 -> -0.01326418`，并通过 warmup/candidate/zero successor RGB replay、raw-action inversion、有限 latent
与 CUDA gates。其余实现 gates（frozen hash、动作边界、three-token budget、root identity、finite physical outcomes）
也全部通过；没有 simulator-training-root overlap。

因此该 v6 不是一次效力负结果：outcome 目录没有创建，预留的 `20270500--20270523` 根没有被打开，P2--P5 继续关闭。
在已审计的运行时路径上，失败不能由 checkpoint/adapter/target 漂移、action transform、warmup/snapshot replay、因果
action window、CPU fallback 或 selector/fidelity provenance 缺失解释。最窄的有效结论是：对这个冻结 checkpoint 和
该预注册的 one-token reconfiguration/observed-RGB mechanism contract，模型侧 margin band 不能稳定外推到新 roots 的
真实 successor 排序；这限制了当前 Planning Compiler 假设的 P1 可验证性，不是对 D2P、谱分析或世界模型规划整体的理论否定。

#### Reacher-D2P P1：task-critical qpos 校准的独立复核（2026-10-24）

为排除“RGB successor 排序不等于任务坐标”只是测量定义造成的假阴性，新增了一条独立、预先冻结的
task-critical 链。planner 的候选生成、future-score 排名和 RGB/action-history 输入完全不变；`qpos_distance`
只作为 evaluator-only 的校准判据，绝不进入模型、候选 bank 或排序。新的 selector
`select_reacher_d2p_p1_task_critical_margin_band.py` 仅在 source calibration 中，按**模型侧**
non-greedy margin 选择一个连续闭区间，并绑定 checkpoint、adapter、runner hash 和完整 planner geometry。
后续 runner 拒绝将该 qpos selector 用于 observed-RGB protocol，且拒绝 source report 被改写。

新的 host-CUDA 链使用 preflight `20270600--20270603`、source calibration
`20270700--20270715`、bound calibration `20270800--20270815` 与尚未打开的 outcome
`20270900--20270923`。source 的 15 个 model-eligible roots 中，连续区间
`[0.09582424, 0.23112917]` 含 5 个物理 non-greedy roots，故只授权 root-disjoint bound
calibration。bound 中只有 3 个 roots 落入该 interval（少于预注册的 4）；其中 2/3 保持
physical non-greediness，而 `20270814` 的 model margin `+0.14357758` 却对应 candidate-minus-zero
qpos distance `-0.23533100`，即真实首段反而更接近目标。所有其他 gate 均通过：CUDA、冻结 hash、
raw action box、三 token 预算、P0/preflight binding、root identity、finite outcomes、模型侧
non-greedy 与 future-score improvement。receipt：
`le-wm-main/results/reacher_d2p_official_causal_repair_20261024_cuda_training_stats_v7_task_critical_margin_revalidation_v1/p1_task_critical_bound_calibration/report.json`。

这次失败再次经过独立、无模型推理的 host-CUDA 双侧 replay；16/16 roots 的 warmup/root RGB、candidate
successor、raw-zero prefix、角度环绕 qpos distance 和记录动作都精确重现，replay receipt 为
`p1_task_critical_bound_first_chunk_replay.json`。因此它不是 qpos 环绕、snapshot restore、zero control、
动作记录、normalization、GPU 路径或 selector binding 的代码问题。由于 bound 未通过，脚本在创建 outcome
目录或执行 outcome root 前退出；P1-qpos 以及 P2--P5 保持关闭。该证据只建立一个更强的 checkpoint/protocol
边界：该冻结 RGB dynamics 不能以可泛化的 model-margin interval 稳定识别物理上非贪心的 reconfiguration prefix；
它不构成 D2P 或世界模型理论整体的负结论，也不授权事后改阈值继续开 outcome。

#### PushT-D2P 独立 P0：输入/API 合同排错后通过（2026-10-24）

PushT 不能拿既有的 privileged oracle、state heuristic、adapter 或 decoder 输出充当 D2P/P5
证据。因此新建了独立的 CUDA-only P0：
`experiments/spectral_reachability/preflight_pusht_d2p_p0.py`，其唯一目的为验证发布的
冻结 `pusht/lewm` 是否具有可微、有限的 future-action path；它不读取 simulator state 到
模型/score，不调用 oracle successor，不训练 adapter/repair predictor，并且明确保留而不打开
outcome roots。

首次 host-CUDA 调用在科学行生成前暴露两个实现合同缺口，均已修复并重跑：单 goal RGB
被错误扩成六维而不符合 ViT 的 `[B,T,C,H,W]` encoder 接口；以及对 `LeWM.rollout` 的
action API 误作 `[B,H,D]` 调用，遗漏其必需的因果 history prefix 与 future suffix。修复版
以三帧、零 live-action 的**真实因果** history（每帧均在一个真实 simulator step 前渲染）
建立 rollout context，未来两 token 才是可微编译变量；它们进入 checkpoint 的路径严格为
`live [-1,1] -> /100 -> public z-score -> time-major fivefold block`。两处失败均属于代码
问题，未产生可解释为理论负结果的 P0 receipt。

主机 RTX 5060 Ti / CUDA 13.0 的新 receipt
`results/pusht_d2p_p0_preflight_20261024_cuda/report.json` 通过 CUDA 真分配、冻结参数 hash、
action-block、RGB-only planner input、无特权/无修复器与未打开 outcome gates。四个独立
calibration roots 的 Jacobian Frobenius norm 为 `0.056526--0.096180`，中心差分中位相对
误差为 `0.002384--0.021278`（每根均小于预注册的 `0.10` 门）。这是 PushT-D2P 的 P0
工程准入，而非物理 fidelity、reconfiguration、闭环效力或跨任务 P5 结论。该独立分支的
下一步是 calibration-only P1：仅用冻结模型构造带动作见证的非贪心 reconfiguration chunks，
在真实执行后重新以 RGB 编译，并以 root-disjoint receipt 决定能否打开任何 outcome 或强基线
比较。

#### PushT-D2P P1：先修复代码合同，再作 root-disjoint 物理判定（2026-10-24）

PushT P0 的旧 receipt 随后发现一个会改变冻结 action encoder 输入的代码缺陷：
`repeat_interleave(5, dim=1)` 产生 `[ax,ax,ax,ax,ax,ay,ay,ay,ay,ay]`，而发布数据和
checkpoint 合同要求 time-major `[ax,ay]` 重复五次。该 receipt 已降级为诊断，修复后的
`preflight_pusht_d2p_p0.py` 改为 whole-pair repeat；新的 host-CUDA P0 receipt
`results/pusht_d2p_p0_preflight_packet_layout_repair_20261024_cuda/report.json` 在四个独立 roots
通过全部 action-Jacobian、中心差分（中位相对误差 `0.003524--0.006759`）、冻结 hash 与 RGB-only
gates。

在进入 P1 物理结论前又发现第二处形状正确但语义错误的实现：P1 `_advance()` 在
`LeWM.rollout()` 所需的 history-only bootstrap prediction 之前插入 candidate action，造成 action 与
transition 错一拍。修复版先使用完整已观察 action history 预测 bootstrap successor，再 append prefix；
P1 receipt 同时升级为 replay-bound schema，记录 root seed、候选 bank、horizon、damping 和 sketch
维度。该修复后的 calibration admission（4 roots）与 physical receipt 分别为
`results/pusht_d2p_p1_rollout_alignment_repair_20261024_cuda/report.json` 和
`results/pusht_d2p_p1_physical_calibration_rollout_alignment_repair_20261024_cuda/report.json`：四根均
通过 same-root raw-zero、observed-successor RGB recompile、非零 Jacobian、冻结 hash 和独立 replay；
selected prefix 在 block pose cost 上均不比 raw-zero 更贪心。

随后以 P0 预留 seed `202710281` 打开独立 24-root outcome cohort；它与 calibration roots 显式
hash-disjoint，并要求 calibration receipt 已通过。模型侧有 `23/24` 个 non-greedy witness，但物理
outcome receipt `results/pusht_d2p_p1_outcome_physical_20261024_cuda/report.json` fail-closed：所有 CUDA、
frozen hash、sealed-prefix reconstruction、same-root raw-zero、observed-successor recompile、独立 replay 与
root-disjoint gates 均通过，然而 23 个 model-eligible roots 中有 5 个 selected prefix 真实 pose cost
反而低于 raw-zero（最小 selected-minus-zero 为 `-7.0581`）。因此物理 non-greediness 没有由 calibration
泛化到 outcome；P2--P5 不得打开。

这条 PushT 负结论不能归因于此前的 action packet 或 rollout 时间对齐 bug：两者均在 outcome 前修复、单测覆盖并
以新的不可覆盖 CUDA receipts 重跑。当前精确边界是：在该冻结 `pusht/lewm` checkpoint、one-prefix /
observed-successor-recompile 合同下，模型内 reachability 排序尚不能稳定识别物理上非贪心的 contact prefix；它既不
否定 D2P/世界模型的一般理论，也不授权通过事后调阈值绕过 root-disjoint outcome gate。

#### TwoRoom-D2P P5：训练期 action-statistics 合同审计（2026-10-24）

TwoRoom 的历史 live-planning artifact
`results/live_cross_environment_prediction_20261015/tworoom_cuda/cross_environment_prediction.json`
为十维 packet 的五个位置分别估计 mean/std；这与发布训练入口不同。`train.py` 先对 HDF5
的二维 raw action 调用 `get_column_normalizer`，之后 `HDF5Dataset` 才把五个相邻 action
按 time-major 顺序打平为十维 token。因此，将该 artifact 接入 frozen action encoder 会改变模型
所见的坐标系，既不能成为 P5 的正向证据，也不能用于把旧负结果归因给机制本身。

新的 `build_tworoom_d2p_training_action_stats.py` 从
`/home/gbk/.stable_worldmodel/datasets/tworoom.h5` 建立不可覆盖 receipt；它绑定 HDF5 SHA-256
`129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623`，并使用 trainer 同一的
unbiased `torch.std`。得到的物理统计是 mean `[0.00309342, -0.05298254]`、std
`[0.86754781, 0.86863881]`，在五个 token slots 中严格重复。receipt 位于
`le-wm-main/results/tworoom_d2p_training_action_stats_20261024/tworoom_training_action_stats.json`；
新测试会拒绝 position-specific legacy stats。

这修复的是 P5 的输入坐标系前置条件，不是 TwoRoom 的效力实验。旧的 adapter、latent readout、teacher、
support gate 或 RGB reactive fallback 均不可包装成 P5。下一条可解释的 P5 路线必须由此 receipt 驱动一个
全新 CUDA-only frozen runtime：RGB/rendered-goal/action-history-only、fixed shared candidate bank、
matched model-call budget、calibration/outcome root-disjoint，并以真实环境距离/成功仅作 evaluator；在该
runtime 完成前，P5 仍未打开。

#### Reacher-D2P P1：selected-packet provenance 收紧后的独立复验（2026-10-25）

若先用一个 checkpoint 选择 P1 packet、再更新 predictor/action encoder 并用新 checkpoint
部署，阻尼相同也不能保证 deployed selector 仍来自同一 selected-packet 分布。新增
`audit_reacher_d2p_p1_selection_stability.py` 因此在封存 source roots 上以 host CUDA 重放完整
proposal bank；P1 runner 现在要求每个 selected-packet source 都有同一 deployed checkpoint 的
全根稳定 receipt，缺失、hash 不同或任意 selector 漂移均 fail closed。

训练后的 distribution-repair checkpoint 仅在 `4/60` source roots 保持原 selected proposal，
故不得用于 P1 科学结论。回到实际生成该 pool 的 frozen joint checkpoint 后，
`results/reacher_d2p_p1_selector_stability_joint_source_20261024_cuda/report.json` 在 `60/60`
source roots 获得完全稳定的 selector receipt。排除该 mismatch 后，新的 root-disjoint P1
preflight (`20274700--20274703`) 通过 CUDA、冻结 hash、RGB identity、action box、nonzero
Jacobian、模型侧 non-greedy 与 future-score gates；新的 calibration (`20274800--20274805`) 仍仅有
`4/6` model-eligible prefixes 在真实 successor RGB 的 frozen-encoder goal-cost 下保持 non-greedy。

独立 fidelity receipt
`results/reacher_d2p_frozen_source_validation_20261025_cuda/p1_fidelity.json` 重放了 warmup、candidate
与 raw-zero successor、动作反归一化和 CUDA encoder。两个排序反转根从模型 margin
`+0.00570202`、`+0.00506282` 变为观察 margin `-0.05088305`、`-0.01766348`；六根按模型
margin 排序的 observed signs 为 `+,+,-,-,+,+`，故不存在含预注册最少四根且零反转的连续
model-margin band，现有 selector 正确拒绝创建 receipt。

此 Reacher P1 继续为 `blocked_upstream`：已排除 selector 漂移、动作尺度、zero control、
warmup/snapshot、root reuse、CUDA 与 fidelity replay 的代码解释，但不构成对 D2P/谱理论的一般反例。
P2--P5 不得从该 Reacher chain 打开；独立可执行路径是 TwoRoom 的 fresh frozen RGB/action-history
P5 runtime，且不得复用 adapter、teacher、privileged oracle 或 reactive-controller 结果。

#### TwoRoom-D2P：fresh P0 runtime 已通过输入与微分合同（2026-10-25）

TwoRoom 的旧 live-ranking 路径同时使用了 legacy post-packet action statistics、重复单帧和全零
history；它不能检验这个 checkpoint 的 D2P。新的
`preflight_tworoom_d2p_p0.py` 改为只读取训练语义 receipt：每个 token 都是五个**实际执行**的 raw
`[x,y]` controls，按 time-major 打平并在 packetisation 后保持相同的二维训练 z-score。它在计划根前
执行三个完整 packet，模型只接收由这些因果 RGB/action pairs 构成的历史和 rendered goal；post-warmup
RGB root 仅作为 evaluator-side identity anchor，环境 state 从不进入模型。

主机 RTX 5060 Ti / CUDA 13.0 的不可覆盖 receipt
`le-wm-main/results/tworoom_d2p_p0_preflight_20261025_cuda/report.json` 在 seeds
`202610250--202610253` 全部通过：Jacobian Frobenius norms 为 `9.08495, 4.52302, 2.37676,
4.14428`，中心差分中位相对误差为 `0.000384--0.000851`，checkpoint 及所有 frozen parameters
hash 不变。这个结果只建立新的 TwoRoom D2P 微分 runtime 可用；它既不是闭环收益也不是 P5 成功。
它解除的是独立 TwoRoom P1 calibration 的**实现前置条件**，而不是绕过 Reacher P1 或改写其
checkpoint-specific reliability boundary。

#### TwoRoom-D2P P1：训练动作包支持修复与重新校准（2026-10-25）

在把第一轮 TwoRoom P1 的模型—物理失配归类为 checkpoint 边界前，新增了训练源的
**五步包支持**审计。此前 runtime 虽正确复用了训练期二维 mean/std，但 warmup 和
prefix proposal 在连续 action box 内独立均匀采样；这不能证明 ten-dimensional
time-major token 位于训练支持内。训练 HDF5 中约 `40.7%` 的有限 raw actions 精确位于
离散边界网格，故单步统计匹配不足以排除 action-sequence OOD。

新的 `tworoom_d2p_packet_support.py` 只从同一训练 HDF5 的**同 episode、连续五步**
raw action window 采样，不读取环境 outcome。它将 source HDF5 SHA、870,809 个有限包的
pool SHA、warmup source indices 和 prefix proposal source indices 写入 immutable receipt；
raw-zero 保留为明确的相同根对照。P0 的 source-supported v2 receipt
`results/tworoom_d2p_p0_preflight_support_v2_20261025_cuda/report.json` 在主机 CUDA 上重新
通过非零 Jacobian、有限差分、冻结 hash 和 source-supported warmup gates。其 P1 preflight
在新 roots `202610300--202610303` 上获得 `3/4` 模型侧 non-greedy witnesses，且没有执行
任何候选。

物理校准
`results/tworoom_d2p_p1_calibration_support_v2_20261025_cuda/report.json` 保持 same-root
raw-zero、sealed prefix、observed successor recompile 和独立 replay 的全部合同，但在三个
model-eligible roots 中仍有一个 selected packet 在真实距离上比 zero 更接近目标，因此
physical non-greediness gate 仍 fail closed。独立 fidelity receipt
`results/tworoom_d2p_p1_prefix_fidelity_support_v2_20261025_cuda/report.json` 又确认
P1 recurrence 与 checkpoint 原生 rollout 精确相等、每包五个低层动作及 packet-boundary
label 可复放、RGB successor 和 post-prefix causal context 均匹配。故第一轮连续均匀
packet 的结果降级为 `superseded_protocol`；当前 source-supported 失败不能归因于
action-sequence support、time axis、replay、CUDA 或 label timing。P2/P5 仍不得从这条
TwoRoom P1 链打开；它仅建立该 frozen checkpoint 在此训练支持及 RGB/action contract
下尚未通过 model-to-physical reconfiguration calibration，不是 D2P 或谱理论的一般反例。

为排除“16 个 support packets 覆盖太稀疏”的剩余实现解释，另行预声明了完全独立的
64-candidate、8-root source-supported calibration；它使用新 roots `202610320--202610327`，
不读取第一轮 calibration 的物理标签。preflight receipt
`results/tworoom_d2p_p1_preflight_support_coverage64_20261025_cuda/report.json` 在 8 根中
生成 7 个 sealed model-side witnesses。物理 calibration receipt
`results/tworoom_d2p_p1_calibration_support_coverage64_20261025_cuda/report.json` 显示
这 **7/7** 个 eligible selected packets 都比 raw-zero 更接近真实目标：
selected-minus-zero distance 为 `-3.23999` 至 `-24.44435`。因此扩大 source-supported
proposal coverage 不但没有消除翻转，也排除了候选稀疏或连续-uniform OOD 的解释。
最终 CUDA fidelity receipt
`results/tworoom_d2p_p1_prefix_fidelity_support_coverage64_20261025_cuda/report.json`
再次通过 native-rollout equivalence、five-action boundary label、RGB hash、post-prefix
context、frozen hash 及 exact replay gates。TwoRoom P1 现以该 8-root independent
calibration 作为权威的 `blocked_model_to_physical_alignment` 边界；它关闭这一个 frozen
checkpoint 的 P2/P5，而不外推为 D2P、SRA 或世界模型规划的一般理论反例。

#### TwoRoom-D2P P0/P1：observed-root 因果窗口修复与 fresh 重建（2026-10-25）

在继续将 TwoRoom 的 P1 物理排序失配归类为冻结 checkpoint 边界前，审计发布的
`LeWM.rollout()` 实现发现 P0 的一个因果窗口错误。旧 P0 将三个 warmup 前的 RGB frames
`[x_{t-3},x_{t-2},x_{t-1}]` 与三个已执行 packets
`[a_{t-3},a_{t-2},a_{t-1}]` 输入 native rollout；这首先预测的是已经实际观察到的
`x_t`，并非从 observed root 编译 future。尽管 `root_frame` 被记录为 receipt identity，
它没有进入模型的 pixel context。因此旧 P0 不能支持其声称的 observed-root action Jacobian，
属于实现合同错误而不是方法负结果。

`preflight_tworoom_d2p_p0.py` 现以 `[x_{t-2},x_{t-1},x_t]` 为模型 context，并使用
`[a_{t-2},a_{t-1},a_t]`，其中 `a_t` 是第一个可微 future packet；这与 checkpoint 的
three-token causal predictor 对齐。回归测试同时检查该左移不切断 future gradient。新的
source-supported host-CUDA P0 receipt
`results/tworoom_d2p_p0_preflight_support_v3_rootanchor_20261025_cuda/report.json`
在 `202610250--202610253` 上通过所有门：Jacobian Frobenius norms 为
`7.28018, 3.54570, 4.97365, 5.79947`，每根中心差分中位相对误差不超过 `0.000910`。

因此此前 v2/coverage64 P0 不再作为 root-anchored P1 的上游 admission。以 v3 receipt
为唯一上游、在全新 roots `202610340--202610347`、64 个 source-supported prefix candidates
上重建的 P1 preflight
`results/tworoom_d2p_p1_preflight_support_coverage64_rootanchor_20261025_cuda/report.json`
获得 `8/8` 个模型侧 non-greedy witness。其 root-paired physical calibration
`results/tworoom_d2p_p1_calibration_support_coverage64_rootanchor_20261025_cuda/report.json`
没有通过严格的全根物理 non-greediness 门：selected-minus-zero packet-boundary distance 为
`-7.90187, -7.22272, -6.13089, +3.38226, +4.17706, +6.59291, +9.12689, +25.65547`；
三根 selected prefix 在真实距离上仍比 raw-zero 更贪心。所有其他门均通过，包括 CUDA、
冻结 hash、sealed selection、same-root raw-zero、observed-successor recompile 与独立 replay。

独立 fidelity receipt
`results/tworoom_d2p_p1_prefix_fidelity_support_coverage64_rootanchor_20261025_cuda/report.json`
对 selected 和 raw-zero 的全部 16 branches 均通过：P1 recurrence 与 native `LeWM.rollout`
的 L-infinity 差为 `0`，五个低层 action 的 packet-boundary distance、successor RGB hashes
及 post-prefix three-frame/three-packet context 均精确重现。故新鲜 calibration 的失败不能
归因于 root-frame omission、action time axis、packet label、successor replay、CUDA 或
source packet support；当前严格结论仍限于这个冻结 TwoRoom checkpoint 的
`blocked_model_to_physical_alignment`。P2/P5 继续关闭，且该结论不外推为 D2P 或谱理论的一般否定。

#### TwoRoom-D2P P1：同可行域 greedy comparator 修复与全 comparator CUDA 审计（2026-10-25）

上述 root-anchored calibration 的正式 physical comparator 仍是 raw-zero；但 P1 的 selected
prefix 来自 training HDF5 的 source-supported packet bank，raw-zero 不属于该 bank。因此它会把
可行域不对称误写成 P1 的机制失败，和此前 Reacher P1 已修复的问题同构。P1 preflight schema
升级为 v2：每个 immutable receipt 除了 model-side selected prefix，还明确承诺正式物理对照为
同一个 sealed bank 内、按冻结 `immediate_goal_cost` 最小且以 `proposal_index` 稳定打破平局的
packet；raw-zero 仅保留为 OOD diagnostic，不能再关闭 mechanism gate。

全新、与 P0 roots 不重叠的 host-CUDA preflight
`results/tworoom_d2p_p1_preflight_support_coverage64_samebank_20261025_cuda/report.json` 在
`202610360--202610367` 以 64 个 source-supported candidates 得到 8/8 model-side witnesses。
相应 calibration receipt
`results/tworoom_d2p_p1_calibration_support_coverage64_samebank_20261025_cuda/report.json` 不读取
任何 physical label 来选 packet，并通过 CUDA、frozen hash、sealed roots、source support、successor
recompile 与 independent replay；但其 strict same-bank physical non-greediness gate 仍为 false：
selected-minus-same-bank-greedy 的 packet-boundary distance 在 8 根中为
`-19.4010, +35.9616, -19.1502, +23.8902, +11.7178, +7.7145, +31.9020, -8.3371`。也就是
5/8 符合“selected 较不贪心”，另 3/8 方向翻转；这不是通过替换阈值可以解决的问题。

为了排除新 comparator 自身的重建或执行 bug，v2 exact-fidelity audit
`results/tworoom_d2p_p1_prefix_fidelity_support_coverage64_samebank_20261025_cuda/report.json`
在 24 个 branches（selected、same-bank greedy、raw-zero 各 8）逐一验证了 source indices 到
raw packet bytes 的恢复、五个 low-level action 的 packet-boundary label、successor RGB、P1
recurrence 与 native `LeWM.rollout` 的严格相等、以及 post-prefix 的三帧/三 packet context；所有
gates 通过。额外代码复核确认 `TwoRoomGoalAliasWrapper` 使用环境 reset 时由同一
`target_position` 渲染的 `_target_img`，而 evaluator 也是同一环境的
`||agent_position-target_position||`，故不存在 stale/不同 target 的目标语义分叉。

因此旧 raw-zero 负结果已降级为 comparator/protocol diagnosis；新的同可行域 receipt 才是当前
冻结 TwoRoom checkpoint 的结论。它仍不授权 P2/P5 outcome roots，也不构成对 D2P、SRA 或世界
模型规划理论的否定；若要改变这一边界，需要新的 frozen checkpoint/训练合同，而非修改已封存
calibration 的 selector、阈值或 evaluator。

#### Reacher-D2P P1：预测 latent readout 的因果 action-slot 修复（2026-10-25）

在继续把 source-supported Reacher P1 的 selector 失配归为冻结 checkpoint 边界前，审计发现
原 qpos chart 只由**真实 RGB encoder latent**拟合，却直接用于 predictor rollout latent，并把
预测 terminal 与 RGB goal 强行置于同一线性 readout。训练损失只鼓励二者接近，并不保证这一
坐标复用成立。新实现将 `predicted_coordinate`、`observed_coordinate` 与 `goal_coordinate`
拆开；world model 保持冻结。旧的单 readout 接口仅作兼容，不能再作为新 P1 证据。

第一次 CUDA rollout-readout calibration 还暴露了脚本自身的 action-slot bug：它把待预测的
`a_t` 同时放进 history 与 future，实际计算了错位的
`[a_{t-1}, a_t, a_t]`，而部署的 root-anchored predictor 应使用
`[a_{t-2}, a_{t-1}, a_t]`。该 v2 artifact 因而降级为 `superseded_protocol`。修复后，
episode-disjoint 官方 HDF5 calibration
`results/reacher_d2p_frozen_rollout_qpos_readout_v3_causal_20261025_cuda/report.json`
在 host CUDA 上达到 held-out mean joint wrapped error `0.02420` rad、qpos-distance
RMSE `0.04662`，并通过预声明门；它只训练闭式 readout，未更新 encoder、predictor 或 action encoder。

同一 frozen checkpoint、相同的 8 个 root / 128 个 source-supported packet 的 P1 重跑
`results/reacher_d2p_source_supported_causaldualreadout_p1_v7_20261025_cuda`
仍有 7 个 model-eligible roots 中 4 个 physical selected-minus-zero qpos direction 反转，
故 fail closed。独立的 outcome-blind exhaustive successor audit
`results/reacher_d2p_qpos_successor_fidelity_v3_causaldual_p1matched_20261025_cuda/report.json`
显示 7/8 roots 的 Spearman 为 `0.992--0.998`，但 seed `2026102520` 为 `-0.2250`，且
zero-relative sign gate 不通过。这排除了被修复的 readout coordinate/action-slot 漂移作为
该 panel 负结果的解释；它仍只限制这个 checkpoint 的 selector-critical root，不能外推为
D2P 或理论的一般反例。P2--P5 不得由这一 Reacher P1 receipt 打开。

#### Reacher-D2P P1：完整 MuJoCo state 与 selector 合同复核（2026-10-25）

为继续排除执行层解释，`reacher_d2p_context.py` 的 evaluator snapshot 不再只恢复
`qpos/qvel`；若 dm-control 暴露 `physics.get_state()/set_state()`，它会同时保存、恢复并逐元素
核对完整的 MuJoCo state。这样可防止“RGB 与 qpos 相同、但未记录的 actuator state 不同”被误判为
world-model 的排序失配。兼容旧 receipt 的 qpos/qvel fallback 保留，但新的完整 state 路径还要求重放
state 与保存值精确相等。定向 context/readout/fidelity/P0 测试为 `23 passed`。

主机 CUDA 的新鲜、不可覆盖 fidelity receipt
`le-wm-main/results/reacher_d2p_qpos_successor_fidelity_v4_fullstate_20261025_cuda.json` 对同一
8-root、128-packet panel 仍给出 mean Spearman `0.843079`：seed `2026102520` 为 `-0.224999`，其余
七根为 `0.991703--0.998003`。因此完整 simulator-state 恢复是已修复的代码合同，但不是这个反向根的
充分解释；该结果不得被写成 D2P 或谱理论的反例。

随后检查 repair-to-deployment 合同发现：P1 selected-packet pair pool 的 selector 依赖生成它时的
predictor、action encoder 和 damping。旧启动链可在未提交 selector-stability receipt 时继续 P1，且有
历史 audit 在一次 repair 后仅保留 `4/60` source selections。该现象首先是 training/deployment
protocol defect，不能解释为 reconfiguration 理论失败。新的
`run_reacher_d2p_selector_matched_repair_cuda.sh` 因而执行：以部署 checkpoint/damping 重新生成独立
pair pool，低漂移 repair，随后对每一个 source root 要求 selector 完全一致；不满足则 fail closed，
不进入 P1 calibration。新的 64-root source pool、repair 与 stability audit 均绑定全 MuJoCo state
路径，且与任何 P0/P1 calibration/outcome root 隔离。

#### Reacher-D2P P1：训练源到部署物理转移合同复核（2026-10-25）

为继续排除“官方 HDF5 与部署 Gymnasium/dm-control 对同一 raw action 的解释不同”这一代码层假设，新增
`audit_reacher_d2p_source_runtime_transition.py`。该审计在主机 CUDA 上从 official HDF5 恢复 32 个独立的
`[qpos,qvel]` source root，执行记录的后续五个 raw controls，并逐根与 HDF5 的 `t+5` 状态比较；它不加载、训练或
选择任何世界模型。v1 用 `1e-8` 绝对阈值会把 HDF5 的 float64 action 与部署 PyTorch/Gymnasium 的 float32
执行舍入误报为失败，最大误差仅为 qpos `1.1217e-8`、qvel `1.0746e-7`。审计已据此将声明阈值固定为
`2e-7`，而非掩盖差异；新的不可覆盖 host-CUDA receipt
`le-wm-main/results/reacher_d2p_source_runtime_transition_v2_20261025_cuda/report.json` 为 `32/32` 通过。
因此 source action order、物理 step、qpos/qvel state layout 与部署 simulator 在 float32 execution contract 下
一致，不能解释 selector-critical 反转。

随后以完整 MuJoCo snapshot、因果 dual qpos readout 和同一 8-root/128-packet panel 再运行
`reacher_d2p_source_supported_p1_qpos_fullstate_v3_20261025_cuda/report.json`。7 个模型 eligible root 中，selected-minus-
zero 的真实 qpos directions 为 `-,+,-,-,+,+,-`，仍有 4 个不满足首段物理 non-greediness；实际 successor RGB
qpos-chart directions 也为 `+,+,-,-,+,+,+`，不能形成机制 gate。这个 receipt 只复核当前 selector 的代码/协议路径，
不是 sealed efficacy outcome，且不授权事后改 margin 或打开 P2--P5。它将剩余问题收窄为该冻结 checkpoint 在
selector-critical roots 的预测-to-physical 排序可靠性，而不是训练源/部署环境动作转移不一致或 CPU fallback。

继续检查 renderer 后，上述“可见 marker”的解释被 source RGB 反证推翻。官方 HDF5 的
`target_pos` 确实对应 MuJoCo geometry，但默认 variation 将 target 设为透明。host-CUDA
`reacher_d2p_source_rgb_contract_v2_20261025_cuda/report.json` 在 24 个 HDF5 roots 上将 geometry
在 source target 与 fingertip 间互换，所有 runtime RGB 的差均为零。因此
`reacher_d2p_goal_marker_contract_v1_20261025_cuda/report.json` 只验证 model-geometry/snapshot
的一致性，**不**验证一个会改变冻结模型输入的视觉修复；旧 P0/P1 receipts 不应因 marker 而被
废弃，也不能被叙述为 marker 修复后的因果收益。

名称含 `goalmarker` 的五个 damping calibration 和后续 P0 panels 仍是有效的 CUDA、冻结、RGB
identity、raw action-box 与 outcome receipts，但它们仅可支持其明确记录的 complete-state/root-anchor
合同与 P0 比较，不能作为 target-marker 像素修复的证据。独立 48-root confirmation
`d2p_reacher_p0_goalmarker_confirmation_n48_20261025_cuda/report.json` 的 P0 结果由其自身配对
receipt 支持；P1 仍必须由 root-disjoint fidelity/calibration 门独立通过，不能以 marker 解释绕过。

#### Reacher-D2P P1：同可行域贪心对照与 runtime-support 审计修复（2026-10-25）

P1 的 reconfiguration prefix 已改为从 state-conditioned、official-source packet bank 中选取；
此前的 continuous D2P/CEM baseline 则可在连续动作空间搜索。两者的 feasible set 不相同，
所以新增 `source_matched_greedy_two_stage_d2p`：它逐 root 复用**同一个**128-packet bank、
冻结 RGB/action context、动作 chart、一个 prefix token 与 observed-successor suffix compiler，
仅把首段规则改为该 bank 上 frozen immediate-goal cost 的 argmin。它是可行域公平性诊断，
不取代连续 D2P/CEM 这两个更强 baseline。

新鲜 host-CUDA calibration
`results/d2p_reacher_state_conditioned_matched_greedy_20261025_cuda/p1_calibration/report.json`
在 roots `2026104600--2026104605` 仍有 6/6 model witness，但实际 RGB qpos-chart 的首段
non-greedy gate 只过 4/6；因此没有 outcome 或 P2 结论。逐 receipt CUDA fidelity audit
`.../p1_calibration_exact_fidelity_audit` 对 6/6 roots 的 warmup、raw/normalized action、candidate/zero
successor RGB、recorded model cost 及 observed cost 全部重现，并确认模型-观测 order 仅 4/6 一致。
这份 receipt 排除了重放和记录路径的错误，但不应被表述为理论反例。

在追查该失配时发现 `audit_reacher_d2p_runtime_support.py` 自身存在诊断代码错误：它把
`p0._prepare()` 的第二返回值（goal tensor）当作实际 successor，故旧数值并非 model-to-successor
fidelity。修复为先用 `advance_executed_context()` 执行同一 raw-zero packet，再以 shifted RGB/action
history encode rendered successor；相应回归测试已加入。修复后的 8-root CUDA receipt
`.../runtime_support_fidelity_fixed_8.json` 给出 source-supported mean latent L2 `1.01874`，
uniform 为 `1.03527`，且并非每根更好。这是 source action-only warmup 对任意 live reset state 的
有限改善，而不是模型修复。

三个独立 source-contract 审计进一步缩小剩余实现空间：64 个 official clips 的 time-axis audit
`.../source_temporal_alignment_64.json` 中声明 shift=0 严格最好（MSE `0.00593`，相邻 shifts
`0.61959/0.74484`）；32-root source-to-runtime transition audit
`.../source_runtime_transition_32.json` 在 float32 contract 下全部匹配（max qpos
`1.59e-8`，qvel `1.15e-7`）；32-root RGB renderer audit
`.../source_rgb_contract_32.json` 的 marker swap 对 runtime RGB 始终为零，且 static source render
mean absolute pixel error 约 `0.102`。因此下一项可检验的系统修复不应是改阈值，而是把
“source action packet”支持提升为**source-state / source-history**支持，并在新 preflight、
calibration 与 root-disjoint outcome 中重新验证；旧 action-only P1 outcome 不能挪用为该新协议结论。

进一步的 deployed-root source temporal audit
`.../source_root_anchored_temporal_alignment_64.json` 也确认 root-anchored action slots 没有错位：
`[x_{t+5},x_{t+10},x_{t+15}]` 与 `[a_{t+5},a_{t+10},a_{t+15}]` 预测 `x_{t+20}` 时，shift=0
MSE `0.00608`，相邻 shifts `0.59996/0.75419`。相反，source-state/history root 在 raw-zero
future packet 上仍有约 `1.017` latent L2；这表明 zero action 是 source-packet support 之外的 control，
不能再作为 source-supported P1 的唯一机制 comparator。

于是 P1 保留 zero-prefix 为显式 OOD 诊断和同 token-budget control，但将正式机制 comparator
改为同一个 state-conditioned 128-packet bank 内 immediate-goal greedy packet。新的 root-disjoint
calibration `.../p1_source_matched_calibration/report.json` 在 roots `2026104800--2026104805` 通过全部
门；这不是阈值放宽，而是让 future reconfiguration 与 immediate greedy 使用相同动作可行域。
新的 12-root outcome `.../p1_source_matched_outcome/report.json` 也通过 CUDA、frozen hash、root identity、
three-token budget、source-matched mechanism 与模型 future-score gates，且 reconfiguration 对 zero-prefix
为 `-0.12165`（95% CI `[-0.17356,-0.07430]`）。但相对于同 bank greedy，效应只有 `-0.00405`
（95% CI `[-0.07911,0.05302]`），并且相对 continuous D2P/CEM 均显著较差。因此该协议修复后，
P1 的机制合同已重新打开并可复现，但 efficacy 仍不成立，P2 不得打开。最终 CUDA exact-fidelity receipt
`.../p1_source_matched_outcome_exact_fidelity_audit` 对全部 12 个 eligible roots 的 warmup、actions、
successor RGB、model/observed costs 全部通过；它记录 model-to-observation order 11/12 一致，
所以这个“无同-bank 优势”的结论不是 receipt replay 或 comparator wiring 错误。
审计随后扩展到正式 comparator 本身：
`.../p1_source_matched_outcome_all_comparators_exact_fidelity_audit` 对 reconfiguration、raw-zero 与
same-bank greedy 三种首段的 action inversion、successor RGB、recorded model/observed costs 均通过，
避免将只审计了 method 而未审计新 baseline 的结果误写为完整比较。

#### Reacher-D2P P1：source-state root 与 goal 绑定缺陷修复（2026-10-25）

为落实上述“source-state / source-history 支持”而进行的诊断又发现一个实际的任务合同错误：
`audit_reacher_d2p_runtime_support.py` 曾在从 official HDF5 恢复 source qpos/qvel/history 后，保留
`env.reset()` 时另一个随机 root 所生成的 goal RGB。于是模型输入是 source-state RGB/history，目标却是
reset-state 加 `target_delta` 的图像；这不是同一个 qpos-match 任务，不能据此判定 source-state 支持无效。
`ReacherGoalImageWrapper.set_goal_from_state()` 现以恢复的 source root 原子地重建 evaluator
`target_qpos`、可见目标几何和 goal RGB，再捕获执行 snapshot；回归测试明确拒绝回退到 reset goal。

修复后的不可覆盖 host-CUDA receipt
`results/reacher_d2p_runtime_support_source_root_goal_v2_20261025_cuda/report.json` 使用 8 个 roots
确认该路径实际按 `official_source_state_history_v2_source_root_goal` 运行。更关键的根配对候选审计
`results/reacher_d2p_qpos_successor_fidelity_v5_source_native_goal_20261025_cuda/report.json` 在同一冻结
checkpoint、因果 dual readout、128 个 source packet 下得到 8/8 正 Spearman（mean `0.976131`，6/8 physical
argmin 一致）；旧 action-only/reset-root panel 的 seed `2026102520` 曾为负相关。故“该 checkpoint 连真实
successor 排序都不能做”的旧表述过强，至少部分来自 root/goal 实例错配，必须撤回。

source-native P1 仅是 debugging diagnostic，不能替代 root-disjoint efficacy：同一 8 根中 6 根得到
reconfiguration witness，且相对 raw-zero 的真实首段 qpos 均非贪心。raw-zero 本身是 OOD 被动控制，因而又在
同一 128-packet bank 上加入 frozen immediate-greedy comparator。receipt
`results/reacher_d2p_source_native_p1_samebank_v5_20261025_cuda/report.json` 显示这 6 根的模型与物理首段
direction 全部一致、且均为非贪心；selected-minus-greedy qpos 为
`+0.49736,+0.37967,+0.39847,+0.67460,+0.38362,+0.58191`。然而其两段 final-qpos 仍全面劣于直接 greedy，说明
这些 source-native roots 不具有 P1 所需的“贪心局部停滞、重构后才能获益”的可控性缺口。后续 P1 必须在**预先声明**、
episode-disjoint 的 controllability-gap root family 上做新的 calibration/outcome；不得把这份 source-derived
diagnostic 当作 efficacy，也不得用它的 qpos 反调 selector、margin 或 damping。

#### Cube-D2P P0：从 action-only warmup 改为 official source-native RGB/goal 合同（2026-10-25）

Cube 的 P0 在启动前发现与 Reacher 修复前同构的协议错误：旧实现虽从官方 HDF5 抽取连续五步 action
packets，却把它们执行在一个新的 live reset root，再把该 reset 的 RGB history 和 goal image 输入冻结模型。它只
匹配 action marginal，既不保证 RGB prefix 来自训练 episode，也不保证 goal obeys 训练期间的 same-episode
future-image 分布；因此任何由该路径得到的 P0/P1 阴性都不能归因于模型或理论。

新的 `cube_d2p_training_source.py::sample_source_native_contexts` 只接受 SHA 绑定、完整物理 extent 的官方
HDF5。每个 context 严格取同一 episode 的 `[x_t,x_{t+5},x_{t+10}]`、连续的三枚 time-major 五步 actions、
observed root `x_{t+15}`，以及严格未来的 `x_{t+20}` 作为 goal RGB；`qpos/qvel` 和 target block/pose/task
仅写入 evaluator provenance，绝不进入模型。若 target 在 context 内变化、任何字段缺失、episode 边界不足或
source SHA 不匹配，代码立即拒绝。相应 P0 schema 升级为 v2，旧的 action-only output 名称不再被启动器使用。

下载完成后的 host-CUDA waiter 已改为 `lewm-cube-source-native-p0-waiter.service`，它先验证 full-extent
HDF5，再生成 source receipt，最后执行新的 CUDA-only P0。定向 source-contract、P0 与 Reacher 回归测试
为 `49 passed`；在该 P0 receipt 真实写入前，这一项是**已修复的代码合同**，不是 Cube efficacy 或 P5 结果。

### Reacher 原始冻结权重：原始 RGB successor 排序复核（2026-10-25）

新增同根、同 packet、无 qpos readout 的主机 CUDA 审计，在修正的 source-native
root/history/goal 合同下，原始发布权重的实际 RGB argmin 一致为 8/8，mean Spearman
为 0.971824，min 为 0.858863。因而旧的无条件 checkpoint 排序能力负结论应撤回。
这仍是 source-episode diagnostic，不能当作 episode-held-out P1 efficacy，也不能自动
打开 P2。完整合同、原始与 repair 权重的配对结果及仍缺证据，见
[Reacher 原始 RGB 审计](REACHER_D2P_SOURCE_NATIVE_RGB_AUDIT_20261025.md)。

2026-10-25 后续：固定的新 32 根、每根 128 个实际 successors 的主机 CUDA 复核已完整完成，
原始权重 argmin 一致 28/32，mean rho=0.972377，min rho=0.882811，模型最优相对 zero 的方向
一致 32/32。先前多次中断来自将“同一物理根”误要求为“反复 rasterization 必须逐字节相等”：
现场只有单个颜色分量差 1，完整 integration/scene 状态相同，随后 4 次 render 又恢复完全相同。
新诊断显式采用 sealed root observation + exact integration/scene proof，所有 successor 仍为真实
新鲜 RGB；与旧严格运行已完成的 8 根逐 root 比较，所有动作、图像 hashes、predicted/observed
costs 均精确相同。旧中断不追认为成功；该 diagnostic 不等同于正式 P1。

另修复 selector 的 weak-reference 缺口：旧 selected prefix 的模型 future cost 已比同库 immediate
greedy 差，却能因只胜 zero-prefix 被称为 witness。正式新 preflight/P1 默认使用显式绑定的
`same_bank_greedy` future reference；同一初始 8 根修复后正确 abstain（0/8 witness），不是 P1 收益通过。
不得为开启后续阶段而放宽门槛、复用旧 comparator 回执或使用本次 RGB diagnostic 替代 P1 outcome。

### Reacher-D2P 后缀求解器与预算审计（2026-10-25）

继续审计发现两项真实实现错误，故此前“尚存代码错误已排除”的全局措辞不再有效。
旧后缀生成器无条件接受一次 full-step DLS，即使非线性模型目标上升；预算回执又将
20 个奇异值误记作 192 维输出 Jacobian 的 VJP 行数，并遗漏多个前向预测。

默认求解器已改为 `bounded_single_step_dls_backtracking_v2`：固定比较
`{1,1/2,1/4,1/8,1/16,0}`，返回并执行真实模型目标最小的缩步；零更新回退保证不升高。
同 8 根、128 候选、原始冻结模型、四个预声明 damping 的新主机 CUDA 复跑表明：

| 检查 | 实际结果 | 允许的结论 |
|---|---|---|
| full-step 因果复现 | 与旧网格逐根、逐候选、逐 damping 成本精确一致 | 不是换根/换分数导致差异 |
| 旧更新目标上升 | damping `.01/.1/1/10` 分别 `412/301/5/0` 次，每格 1024 候选 | 已确认优化器逻辑错误；不能据此解释 damping=10 的历史排序反转 |
| 新接受更新 | 全部 4096 格无目标上升，181 格回退零 | 接受规则修复通过，不等于物理模型预测正确 |
| 强模型 witness | `.01` 从 2/8 变 1/8，其它仍为 0/8 | 不能凭模型候选打开 P1 |
| 主机 CUDA 实际调用插桩 | 128 候选编译 2048 预测样本；单个两步后缀 14 样本，均与回执精确相等 | 计数已校正，不等于已与强基线匹配资源 |

preflight、proposal-training、margin selection、calibration/outcome 均绑定新 solver；
正式 outcome 额外校验 selector SHA，拒绝旧 full-step 校准回执。新结果与旧失败现场分开
保存，不追认或覆盖旧实验。120 项集中逻辑回归通过；与真实 CUDA 证据分开记账。

核心回执为 `le-wm-main/results/reacher_d2p_suffix_optimizer_repair_comparison8_20261025.json`
及 `reacher_d2p_compute_accounting_full128_20261025_cuda.json`。完整实验路径、数字与边界见
[Reacher 原始 RGB 审计](REACHER_D2P_SOURCE_NATIVE_RGB_AUDIT_20261025.md)。
另修复 `mj_forward` 在 snapshot 恢复后改写 warm-start 的执行错误；同一失败根恢复前后
的主机 CUDA 审计已证明从 4/4 不一致变为 4/4 完全一致。全 32-root 物理确认现已完成：
6 根 model witness，5/6 终态优于两对照；但真实 qpos 对 zero 的首段非贪心只有 1/6，
全根原 neutral-fallback 策略对 greedy 为 5 胜 27 负、平均差 `+0.465292`。这明确把
执行代码中断、局部机制正例、目标度量失配和回退策略负后果分开，而不是一概否定理论。
173 项集中逻辑回归通过；正式链还新增 context-module SHA 绑定，旧校准不得挪用。

正式 P1 仍缺 controllability-gap 任务前提、独立 calibration/outcome 和强基线等预算收益；
不把这些缺项称为 D2P 理论失败，也不通过调 margin 或换训练权重绕过。

### Cube-D2P live-runtime 修复：2026-10-25 最新审计入口

后续 v4/v5/v6/v7 的诊断、seed-overlap 纠正、同根 snapshot 修复、CUDA singleton 数值对齐
以及完整闭环剩余门槛，见 [Cube runtime-repair 审计](CUBE_D2P_RUNTIME_REPAIR_AUDIT_20261025.md)。
特别注意：v5 原始 P1 seeds 与训练池种子重叠，不能作为新根结论；其真正 disjoint 重测
与 v6 仍未通过 top-1 校准。v7 的首个 8-root panel 通过（top-1 7/8，mean rho=0.765），
但相同权重的独立 32-root 确认失败（22/32，mean rho=0.709，min rho=-0.118），不能算作解决。
所有这些 predictor 修复属于非冻结诊断分支，不能替代本方案的 training-free D2P 主张。

## 论文主张的可证伪形式

若后续 P1--P5 全部通过，主张应被限定为：冻结视觉世界模型的有限时域 action
geometry 可以构造任务条件的局部搜索坐标、解析短时域 value 和 chart transition，
并在匹配资源的 RGB 闭环中改善长时域任务。若 P0 通过而 P1/P2/P4 失败，正确结论
是“已验证动作微分路径不足以带来该方法收益”，而不是“谱理论被推翻”或“模型没有
动作响应”。

### 2026-10-26：完整前缀能力地图及度量合同定位

Reacher-D2P已用固定双渲染、保留第二次的显式观察策略，在主机CUDA完成
八根/三任务/17程序共408单元、22576条真实未来、47192预测样本，13项完整性
门全部通过；原52单元逐位一致，原始模型参数/buffer不变。137184次真实draw
及完整物理状态检查单独核账，旧v1/v2失败不追认，P1未因此开放。

独立全库4896拟合与1632个响应—残差组合拟合，定位到不能由简单后继/残差
纠正解释的度量失配：全驱动/仅肩/仅腕分别有3/20/7个物理诊断候选，全部
被旧common-gain拒绝，且真实RGB增益也不支持这些关节能力见证。不能把
“库中没有候选”或“都只是梯度算错”作为当前解释，也不能把它写成理论失败。
下一步必须显式处理模型可用RGB能力代理与物理任务度量的关系，之后重新作
独立校准/outcome及完整强基线，而非使用oracle程序、放宽阈值或筛选有利根。
完整证据见[D2P前缀能力审计](D2P_PREFIX_CAPABILITY_AUDIT_20261026.md)与
[研究闭环缺项](D2P_RESEARCH_CLOSURE_AUDIT_20261025.md)；C1--C3、实验1--3
及P0--P5的原范围不变。

### 2026-10-26：独立目标度量评分与完整编译器分开记账

独立八根202654000–007评分验证已经完成403920个主机CUDA预测样本、456
物理分支及23项完整性检查，独立复算全部QP和选择。两种固定模式都保留，
不因仅肩改善而忽略仅腕损害；原动作与后缀未重新优化，不能称完整编译器。
另一批202655000–007现已用统一目标Q生成谱动作、优化后缀并计算own-state
reach代价，现已完成805536主机CUDA预测样本、912物理分支、18项门及
1632独立QP/完整动作见证审计。模型重配置见证三任务均0/8，选择23/24次
等同自身greedy；对原固定greedy关节终距均差均为正，未修复P1。
全部408程序的覆盖归因仍找到16/29/21个物理首段/双终态合取候选，不能
简单归为库中没有动作，也不能越过未测真实能力的缺口。原模型封存四角色
能力探针现已完成58单元、3116真实未来、6522主机CUDA预测样本、12门与
1044独立拟合，不按事后候选挑程序。common规则的部分RGB能力收益没有
形成同根完整关节机制见证；不能称完整Q一致化已修复P1。真实能力改变、P1独立校准、
强基线等预算与反馈重新编译仍须补齐。源码绑定与CEM返回值接线错误在
物理执行前被捕获并修复，保留失败归档，不归为理论负结果。
详见[目标度量新根验证](D2P_GOAL_METRIC_PROSPECTIVE_CONFIRMATION_20261026.md)。
