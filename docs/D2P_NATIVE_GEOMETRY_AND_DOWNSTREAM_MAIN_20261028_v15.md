# D2P：原生几何复算与后续阶段并行推进

承接 v14。以下区分 MAIN 实际执行、已交付实现与尚待执行；不缩减原 C1–C3、E1–E3 或多环境范围。完整研究未闭环。

## P5：原生观察器已在主机执行并严格消费

MAIN 审读观察器源码后，实际 prepare、verify、主机 run、完整 consume 均完成。
固定四词，先执行原 constructor/seed/reset/prefix10，44 个 prefix 帧五文件逐 byte 一致；
再执行事前固定的 xy suffix11–13，不因结果选择重跑。共 434 次 Space.step、43 次 control、
47 帧、86 次原生几何调用；errors=[]，wall 1.795076 秒。

- prepare：`le-wm-main/results/agent_p5_native_geometry_observer_20261028_v1_prepare_MAIN01.json`，SHA `eee4c5d84adea1bb38873efe6d207daa88fa7b66fcd56ee4fe92ab489da4beec`。
- 实际报告：`le-wm-main/results/agent_p5_native_geometry_observer_20261028_v1_diag_MAIN01/report.json`，SHA `e717c0eeb688d9b9b621b70a3caaa7496e6225f6733865b0357d13f3a30811fa`。
- 完整消费：`le-wm-main/results/agent_p5_native_geometry_observer_20261028_v1_consume_MAIN01.json`，SHA `430a933cb8e4376ea86825ff92501537472bec1796e6e57c9bee913b166a00db`，状态 native_observation_verified。

观察器捕获实际原生 get_vertices/local_to_world 返回、shape/body 身份与返回 GEOS WKB，
没有用三角公式重建变换，也没有替换 renderer 或物理函数。用本次实际世界坐标仅交换目标 polygon
顺序，可精确得到 coverage 的两个值 0.35743524812972455 / 0.35743524812972466，
对应 reward 0.3762476296102364 / 0.3762476296102365；block 两种顺序都保留。
本次实际顺序恰好命中前一个值，严格消费者逐 bit 复算通过。

因此，目标 polygon 迭代顺序足以解释这里观察到的末位数值差异；这是原生几何计算/重放协议问题，
不是模型科学负结果。但历史实际 world vertices/缓存/loaded bytes 缺失，仍不签历史等价：
historical_membership=null，historical_transform_certification=null，renderer_release=false。
旧 replay renderer 的失败报告保持原样，不把本次诊断变成其成功回执。

独立 live raster intervention 已完成 MAIN完整审读、25项测试、实际prepare/verify和主机执行，
605次Space.step/60控制步/85render，errors=[]，wall14.110754秒。
报告 `le-wm-main/results/agent_p5_live_raster_intervention_20261028_v1_render_MAIN01/report.json`，
SHA `0e78ceb18b7748aebd511ed90ce4fd6f93dc639773ec517181f59d495c0b0487`。
新协议保留完整coverage/reward差异，不 hydrate、不修改旧门。

实际 saved consume 发现时间接线错误：consumer把早于goal场景constructor的baseline全局RNG，
与后于constructor的颜色anchor比较。MAIN独立逐字段复算，四word仅差`/global_rng/numpy/2`，
public/topology/color完全相同；goal行全相同。原生constructor调用seed(None)消耗全局np RNG，
不是颜色render改变了物理状态。新saved consumer保持封存run/失败和全部RNG记录，
分开核baseline自身render不变、当前anchor到每个color/restoration全状态不变；尚待验收。
MAIN进一步从四word各自保存的完整NumPy RNG状态独立复算原constructor的一次
`randint(0,25536)`：均得到10740，游标4→6，完整624word状态与flags逐项命中当前anchor，
不是只看游标或忽略随机状态。
新十图CUDA encoder代码/30项测试已完成，但无合法新strict receipt前不执行。

## P1：封存模型选择的真实同位评价已产生，严格复核通过

MAIN 完整审读新评价器，实际 analyze 完成：
`le-wm-main/results/agent_p1_model_library_selector_evaluation_20261028_v1_MAIN01.json`，
SHA `d8b3bedd38e9f9a56d0156f2452d5a20f363acbdad42ea5e3ee4d91f9626ff67`。
随后 MAIN 实际 verify 完整重建比较 exit0，不是仅核输出文件哈希。
先重新验证只读模型五词选择，再读物理结果；40 槽全部核原 plan/job/context、prefix/suffix raw bits、
实际20步返回与 teacher15 历史/图像。index12 的 actual latent 仍 null，分母不删。

8 个 context 中4个选择与原 diagonal 不同。4项 allXY 误差的改善量（正为改善）为
+2.520365、−8.628791、−22.946182、−3.368894；其他4项相同。
不能把 native7/reward 的速度项改善误称位置控制改善。全部来自两个已知根的历史真实执行，
不是新策略实际派发、新鲜确认、因果非贪心能力证据或 P1/C2 通过。
MAIN 联合新评价器与 Hi-LeWM bridge 的84项测试通过；测试不是新实验结果。

## 强基线：Hi-LeWM 已完成静态清单与批处理接线复现

有界解码后的 checkpoint SHA 为 `b87805747d40037841877ce7b99b7dda3ebe7a52202c0ba46bf0006ab5d6f008`。
MAIN 实际静态扫描完成：431 ZIP 成员、data.pkl 170471 bytes、66872 opcodes、49声明符号，
没有执行 pickle reducer、torch.load 或外来模型代码。报告
`le-wm-main/results/agent_hilewm_pickle_static_inventory_20261028_v1_MAIN01.json`，
SHA `7c2658ea69caae3117a376d9804357b851d44b4e7babbed9fa3f91fe0fdde5c2`。
这不是可信 loader 或全部 storage CRC 认证；下一步为不执行对象的符号 IR/参数-storage 映射。

MAIN 审读 batch bridge source/doc 并实际运行 B=5、chunk=2、samples=3 的数据接线复现。
官方 policy 的 high_info 首值为字符串 high；当前安装 CEM 用首值长度推断 batch，得到4而非5，
并把 metadata 切成 hi/gh。仅重排键仍会切 metadata。新 bridge 只让 tensor 进入 solver，
在原 get_cost 边界注回完整 high，保留2/2/1尾批和全部5行；返回原 cost，不改 CEM 数学。
这证实本地版本接口不匹配，不证明官方论文实现失败；NumPy seam 不是完整 CEM 或官方模型复现。
后续真实 torch/CEM CUDA 接线、可信 checkpoint 加载、embedded-low、预处理/normalizer 配对仍待补齐。

上述真实CEM接线随后已完成 MAIN完整审读、36项测试和主机CUDA执行：
固定B5/B2、chunk2、3迭代，完整12次analytic cost/168candidate rows，5.331391秒、errors=[]。
报告 `le-wm-main/results/agent_hilewm_solver_cuda_seam_20261028_v1_cuda_MAIN01/report.json`，
SHA `709a9b7c194f426b6f4465da1da78d705985904051b62a374026a516273c60f1`；
MAIN实际consume严格复算exit0，expand/metadata、elites、mean/sample-std、链和输出均通过。
这是非神经解析代价的安装CEM完整CUDA接线验证，不是Hi-LeWM模型或任务结果。

MAIN 续审 inert IR source/doc 后，实际48项测试捕获2项工程失败：新实现复用 decoder 的
small metadata reader，容量小于实际约6.4MB静态清单，真实 saved binding/cold-plan 因此拒绝。
随后定向修复新实现的独立有界 reader，封存 decoder/静态报告不改；MAIN审读改动、53项回归通过。
MAIN 实际 IR build exit0，0.244578秒，报告6,103,904 bytes：
`le-wm-main/results/agent_hilewm_inert_pickle_ir_20261028_v1_MAIN01.json`，
SHA `709b31e28bee8039d577668cbd9613347be1d070a7619e3db3ca61e3674cabbe`。
IR 28012节点、66872 opcodes，无未知reducer；字面量nn state遍历闭合，531模块、491参数/buffer项、
425 storage声明，errors=[]。这是不执行对象的结构解析，不是加载模型或tensor值认证。
trusted_loading_allowed=false、deserialized=false；storage CRC/权重字节导出继续独占实现。
上述首轮失败属于代码读取容量接线问题，不是权重、理论或方法负结果。

下一安全storage导出也已完成 MAIN完整489行源码/doc审读、56项测试、实际export与verify。
报告 `le-wm-main/results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json`，
SHA `3cd7dd6783129893f397ddef63bd253e447fbef93582dd5f4dc8d7144310efd3`。
全部425storage实际CRC/大小/byteorder与原ZIP核验；425storage NPY+425非None entry NPY共850数组，
MAIN saved verify从原storage/view逐字节重建全部通过，66个显式None不填0。
导出不执行pickle或foreign模型；shape/stride/offset/alias映射仍保留，暂不签runtime state_dict、
模型等价或low/high配对。可信自建架构、严格key/shape/dtype加载与实际模型CUDA仍待补。

## 后续阶段的实际分工与前置

P2 已有 v14 四分支实际物理/严格验收，不重复执行。继续实现真正多程序组合/去组合对照的
动作与起点历史接口，不把 suffix-only selector 当完整10步计划或物理放行。

P3 成对实际配额 runtime 已经 MAIN完整四源码/doc审读、55项回归通过，实际prepare/verify完成，
prepare SHA `46e191695e6e121324daf691bafe1b2862a36349e11232822957d8ca6785e6d8`，
后续两个新增capacity/journal故障测试亦通过；源闭包绑定最终tests/doc SHA。
主机CUDA已完成468.782秒、errors=[]，原模型及buffers前后完全相同。报告SHA
`c8030ebc8c48f74688f4c2ab732c0b2adb184b49a30a4e5ab4bdce5f0bbfbf83`，结果目录
`le-wm-main/results/agent_p3_paired_quota_runtime_20261028_v1_cuda_MAIN01`。
MAIN完整strict消费也exit0，73,057,807bytes回执
`le-wm-main/results/agent_p3_paired_quota_runtime_20261028_v1_consume_MAIN01.json`，
SHA `3b62bc4cbf669ecbea10f5ecbf9863a1d96d2432ce029b75059c4edbc2838e6f`。
实际36compiled/10paired paid refusal/2not_reached，分母48不变；fixed/adaptive各634
predict API/sample。完整资源向量仍不同（actual_vectors_equal=false），不称等FLOPs或C3收益。
不是追加预算 manifest：两侧先各付8次信号，
再以24次搜索预测配额加1次选中前缀，H1/H3 分别12/4候选；原请求不适配则成对拒绝，
不能暗缩请求、删除付费失败或把 API 次数匹配称 FLOPs 相等。实际新 CUDA 未完成前不记通过。

P4 的原 reset ledger、loaded-byte/session身份仍缺；不得制造 freshness 或开放未认证保留根。
P1/C2、完整P2组合增量、P3/C3收益、E1/E2/E3强基线与多环境独立确认仍未闭环。
本轮无训练、无CPU神经回退、无删除、无GitHub发布。

## P2 新组合对照已实际CUDA完成，saved协议错误修复中

MAIN完整审读571行源码/doc、41项测试通过，固定两known根×off/on四槽，joint80/no-composition
与conditional40/own-imagined-context重编suffix。每臂standalone168predict（共享base4单列），
全部480候选/8计划，非等FLOPs，也不是旧equal144域或observed teacher15接口。
实际prepare/verify通过；主机CUDA51.103793秒、errors=[]，报告
`le-wm-main/results/agent_p2_contextual_composition_runtime_20261028_v1_cuda_MAIN01/report.json`，
SHA `498b45ed9f4c580064d8f2a5dfd00ac8542b7ce9d0b86e6587d9ab6a9639d49c`。

实际consume发现producer canonical本身带newline、Store.event又追加newline，consumer把空分隔
当JSON记录而拒绝。MAIN独立解码核177,846,232bytes、38376 splitlines，其中19188空行，
严格record/empty交替；不是科学负结果。新saved-only adapter将严格核双分隔及全部19188事件，
不忽略非空坏行、不改封存producer/report、不重跑CUDA挑结果。strict与物理交接尚未完成。

P5 v2saved严格消费已实际exit0，回执SHA
`f9919c23924b41d118d3da038e7cc6f0e3f88b291d36f83df6fccaeb910671c8`。
MAIN新encoder接线曾混用两种canonical hash域，prepare拒绝；已定向修复并33项测试通过，
actual prepare/verify通过，prepare SHA `6c0fa9420bfed269880279ff6ce53aef479364870f12278796fe6016825627a1`。
十图主机CUDA已实际完成19.949038秒，10次官方encoder、0 predictor、0 proprio、无AD/physics，
errors/finalization_errors均为空。先封存10个完整256×384 float32 patch数组，再作16项交叉评分。
实际报告 `le-wm-main/results/main_p5_live_raster_encoder_20261028_v2_cuda_MAIN01/report.json`，
SHA `a11c0c5fb5b4b040027e50cac851e94eb0ff16bc5f48d8d1174bc8b61dd4a226`。
MAIN完整strict消费exit0，状态 all10_saved_raster_encodes_and16_cross_scores_verified；回执
`le-wm-main/results/main_p5_live_raster_encoder_20261028_v2_consume_MAIN01.json`，
SHA `594ae118088c9c02f7de671d456f396dd8dd0d79765297fc4d889f42fed8776a`。
固定指标分析尚待；新raster frame15与旧coverage frame20不可当作同一任务标签直接比较。
旧physics/renderer失败保持，不伪造旧receipt，也不以编码完整性称P5/C2通过。

P3 新物理交接源码、测试与文档已由MAIN完整审读，40项回归通过；实际prepare正在重新执行
成对quota完整数组消费者与历史原生物理消费者。尚未产出prepare前，不预报复用/新执行数量。

上述prepare现已实际完成：7,577,358bytes，SHA
`f1d6cb2a6ab61e39134ee37a49f0643d58164e6007e8b3b80f566c42f65fa118`；
严格路由为28 reused_saved、8 pending_new_physics、10 paid_refused、2 not_reached。
8个新分支主机run已提交，入口先完整verify；终态待核。新执行上限200 public/2032 physics，
包含恢复成本，不把28条历史轨迹算作新执行或新独立根。

## P5 完整颜色敏感性分析与P1理论线

MAIN完整审读analysis v3三件套，36项回归通过，实际prepare、analyze与完整consume均exit0。
分析报告 `le-wm-main/results/agent_p5_live_raster_analysis_20261028_v3_saved_MAIN01.json`，
SHA `6b167d6a24c2756f938a5840a02ee29211612558f8e780d1a6c905581731a928`；
完整复算回执SHA `28785a84f552e7276a462e852dfab74c33170d8aec1dfff7d4853fd229125547`。
全部16评分、4词精确有理数factorial、48有序比较和72双向条件比较保留，不选词/颜色。
36无序词对×条件对描述性比较中4项严格反转、6项含平局；不是36独立样本。
目标visible/white feature MSE为0，故目标颜色没有实际表征干预，不能用零效应证明goal不敏感。
terminal同物理状态的visible/white feature MSE为0.451698–0.619694；
yx与zero_zero的相对评分方向随terminal颜色改变。四词均值terminal主效应为−0.0253024，
不能用均值掩盖逐词异向变化，也不能从latent敏感性推出任务改善或C2反证。
新frame15与历史coverage20仍分栏，native_vs_raster_rank_agreement=null。

用户要求新增P1理论worker已实际启动：分析局部谱可达性到非贪心优势的条件、
预测误差/margin到真实排序的保证，以及当前gate与假设的对应；独占新理论文档，不修改gate。

P2双LF修复后实际saved消费继续捕获第二项接口错误：冻结event-order checker访问
native_event.event.kind，但实际事件缺此字段；新版本需先按真实事件ABI闭合验证。
原报告/日志和失败保留，不称P2通过、不重新CUDA挑结果。

Hi-LeWM实际runtime属性提取/完整verify亦完成，MAIN完整审读源码/doc并36项回归通过。
报告 `le-wm-main/results/agent_hilewm_runtime_attributes_20261028_v1_MAIN01.json`，
7,262,848bytes，SHA `5415f2a2e97e9bdeb282200e4420b9270c607190bf15a1a482963755b850b3c8`。
low/high保存context分别3/4，三projector实际BatchNorm而非本地默认LayerNorm；
所有派生构造参数逐来源保留，没有用默认值覆盖saved配置。属性认证不等于forward版本等价。
原生源码中StandardScaler.fit依赖外部dataset，冻结normalizer身份仍未恢复。

P2上述两项接口错误的新saved consumer v2现已MAIN完整源码/doc/tests审读、35项测试及
实际全量analyze通过。480候选/8plans/19188事件完整对账，严格保留双LF原bytes；
无kind的480候选终结字典先与原attempts逐canonical相等后，才进入typed helper视图，
不是忽略未知事件。原全部Gate、模型、数组、入口/返回、tensor/replay校验均保留。
回执 `le-wm-main/results/agent_p2_contextual_composition_saved_consume_20261028_v2_MAIN01.json`，
SHA `578c5bd93394ae9c3863cb420afe4cfd0c59633fbbea503b4012c029211e063f`；
status strict_saved_verified_science_blocked。MAIN实际完整verify已提交，物理交接待审。
这两次失败属于代码ABI问题；不能因此推断组合收益已成立，也未重新CUDA挑结果。

该MAIN完整verify现已exit0，独立重建全部8plans及完整receipt相等，SHA保持不变。
P1理论worker Pasteur（`01a18157-f5e4-7562-b34f-41675f1d5bdd`）已实际派发：
目标文档 `docs/AGENT_P1_THEORY_GATE_ANALYSIS_20261028_v1.md` 尚待交付，不能将任务派发计为理论证明。
要求同时给出exact-model条件下的正例、谱优势不蕴含任务优势的反例、误差/margin条件、
Reacher与PushT各层gate对应，以及可执行区分代码/协议/条件失败的事前判据。
不修改gate、冻结模型或科学门槛；主线程继续承担代码审计与主机实验。

## P1理论交付已由MAIN复核，不是P1科学放行

理论文档现已交付，MAIN完整审读全部公式、正反例与T0–T6判据：
[P1理论与gate分析](AGENT_P1_THEORY_GATE_ANALYSIS_20261028_v1.md)，
SHA `94ef31f06f49e11195ede31cec8fe9dfe2ce807c0b37bf02c9d5de1836222246`。
MAIN另用标准库Fraction独立复算exact-model正例的ridge值4/41、合法后缀40/41、
stage+reach获选阈值123/37及一维排序误差恒等式，全部断言通过；0神经/物理调用。

复核结论：机制存在性可成立，但局部SVD/ridge不蕴含当前强gate必过。主要缺桥是
任务相关方向和合法动作域、selector实际选中非贪心见证、预测到真实编码的成对误差/margin、
编码目标到物理任务的关系，以及同根真实机制/独立强对照。
完美模型也可能因stage罚项拒绝存在的有益非贪心前缀；这是明确可算的选择条件，
不是建议按真实结果调整beta。thin-SVD的null投影诊断与正确ridge值分开，未据此判现有V错误。

下一归因优先T3与T4：同见证/同时间逐pair拆开预测→实际编码与实际编码→物理任务；
T2若没有非贪心selected则先记录任务/候选/选择前提不足，不能以终态收益替代机制。
理论文件既不认证freshness，也不打开任何科学gate；原12/76反转、缺失和有害选择全部保留。

## 同动作/同时间归因继续推进：first-stage复现与真实运行终态

MAIN完整审读新P5窄诊断源码/tests/doc，9项纯数组测试通过，独立saved运行
`agent_p5_raster_first_stage_diagnosis_20261028_v1_MAIN01.json` 与worker结果逐字节一致，
SHA `f7d89438ab63a23e6badc9c29daddfb3d7411ea4b59f003155a3291107f8100b`。
goal两种颜色的实际RGB像素差为0，不能解释为encoder颜色不变性；实际物体与goal同pose，
原生绘制先goal后物体。xy/x_minus_x在frame15是共同动作前缀，frame16才分歧，
故frame15编码不能与旧frame20 coverage拼成同时间结论。未发现这一现象对应dispatch错误。
新frame20全四词同时间验证准备已独占派发，旧16score与所有null结果保持。

P3新8条真实物理已完成，报告SHA
`1cd5d5808b615fbf6519191a6862f966fb6b3a934c1b9c1a3a1091d6db1dac70`，
200 public controls / 2032 physics / 216 renders，errors=[]，8条native_success均false。
完整strict仍在消费原生闭包并合并28条精确复用；终态未出前不计算完整48槽策略结论。

Hi-LeWM可信loader真实CUDA已实际复制425个tensor并完成构造，两项direct forward逐位一致；
第三项B2/S2/T5 rollout不一致（maxabs `1.4901161193847656e-7`），cost未执行。
失败报告SHA `3ebe0f5932aa80aff19f052b9d1da42a3a926b6397a6ea2a38a8b675d1583077`。
当前只确认验证失败，不据小数值断定逻辑错或模型边界；owned逐时stack与reference切片reshape的
布局差异是待证假设，已派独立新版本诊断。禁止容差放宽，旧失败保留，仍非完整基线复现。

P1固定40槽/80对的T3/T4误差分解实现继续进行；未交付/未运行不算结果。
所有神经调用仍限主机CUDA，不训练模型、不改gate、不按结果挑根。

## P1同时间分解已验收，研究执行继续

正式40槽/80对saved分解与第二次全闭包verify均已完成，SHA
`f723c5eb73ecebd0615a2ded0aa4b741c6f132ca5f6063c673f12d5e98539143`。
12/76模型→真实编码反转保留，精确恒等式残差全0、无float64符号分歧。
MAIN原生距离轴分解确认36个latent→distance反转中34个涉及速度项翻转pose排序；
但object XY/角度仍22/33反转，不能用整体XY的4反转掩盖。完整归因、结果SHA与边界见
[P1同动作同时间研究记录](D2P_P1_SAME_TIME_MARGIN_MAIN_20261028_v1.md)。

P3上述完整strict现已完成，receipt SHA
`e138a33d175e1de850264c41f1df459354d77b1cedc7d3a3b0b50ed09c1e327c`；
28原精确复用+8新执行合并、10paid-refusal与2not-reached保持48分母。
两策略各634predict，但metric/replay调用不同，不称等FLOPs或C3收益。

P2新8臂full10物理handoff已MAIN全文source/tests/doc审读、75联合测试通过；
实际prepare SHA `978725abf11efa70b8eaef77abbdfe3ca2a88516fe5046444d3d4e517376eb53`。
主机run已提交，未终态前不称实际任务结果；后续同条件checkpoint0/5/10 RGB编码桥准备并行。

P2该run现已终态exit0：46.89054秒，8条full10，160public/1632physics/528PNG，errors=[]，
原生成功0/8。报告SHA `c9e8c37984926a68bf2e4d3a92fe47737c03e719ed51917da58b72c888eb038d`。
完整saved消费进行中；不将0/8改写为理论不可行，不按结果删除任一分支。

P2完整saved消费现已exit0，receipt SHA
`cc2de96eb6569319228644b62afe45218d0e4df1261c2bdabf6c0a9b052ad818`，
8臂raw动作、完整native journal、528PNG、所有checkpoint及真实计量验收通过。
composition相对joint的四维XY终距变化按root0/off,on,root1/off,on为
`+21.93597,-46.83959,+23.81956,-59.74539`（负为改善）；只有两个已见根，
显示条件依赖、不是独立统计优势。object XY/角度各轴单列，禁止用整体XY替代全任务成功。
0/5/10同条件RGB因果history编码桥已交独占实现，真实CUDA尚未发生。
