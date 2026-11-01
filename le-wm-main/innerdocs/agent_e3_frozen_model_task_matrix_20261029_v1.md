# E3：单冻结checkpoint内的任务交叉协议

日期：2026-10-29。状态：**事前协议草案，供MAIN审批；不是runner、运行授权或E3通过回执。** 本轮仅新增本文；不新批根、不训练、不执行NN/physics/生产prepare/fullstrict，不修改旧任务、源码、结果或canonical。本文冻结只冻结设计字节，不自动批准下述新增16轨迹。documentation-and-adrs技能用于记录选择理由、来源与不可执行边界，不创建另一套状态工具。

## 1. 原要求、已有件与本件的唯一增量

原附件 `/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt` 的 `Experiment 3: One Frozen Model, Many Tasks` 位于第659–665行：同一冻结checkpoint切换任务目标、规划horizon、动作限制，比较成功率、规划耗时、model/backward调用及长时任务完成；建议PointMaze绕行、Reacher连续控制、PushT接触配置。附件没有规定本文的样本量、32步优化或统计门。

项目canonical映射：外层 `docs/D2P_RESEARCH_REQUIREMENT_EVIDENCE_MATRIX_20261026.md` 第41/49/59/60行，分别是C3b、E3、P4、P5；原总计划 `docs/SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md` 第410–413行。每环境使用该环境自己的固定checkpoint，**不是要求一个权重跨所有环境**。单checkpoint内多任务与独立架构复制分栏；正式Full D2P E3依赖相应P1及采用的P2/P3/P4，不审计或代办主线程P1。

定向查阅了 `agent_p4_p5_prerequisite_connections_20261029_v1.md` §3D/4.3、`p5_closure_readiness_20261027_v1.md` 的same_model_multitask槽、`P4_RECEDING_RUNTIME_20261027.md`、P3 shared-panel、actual-cp5 builder与feedback-GD交付；指定docs/innerdocs文件名中未发现完整同checkpoint交叉协议。这是有界核对，不声称全仓绝无类似设计。上述件已有接口/前置/单目标多策略结果，不重复生成fresh resolver、资源框架或资产清单。

本轮原附件全文byte SHA为 `6cd1ebb67192485f19fa8debb1d50ee6586a5099efa5816099bd19c3511a9aae`，需求矩阵为 `b8fcbb9b27a4d0b41e25376cf49f6b6f2f14cb10b1067276000a16532d97f388`；绑定的是当前文档字节，不将canonical后来追加误报成封存实验源码漂移。

本件新增的是：一个明确的**known-root描述性16任务格**，及其与正式E3尚欠内容的接点。关键纠正：shared-panel的 `endpoints/midpoints` 是候选bank，不是两个任务goal；fixedH1/adaptive是策略，不是不同任务。旧8episode不能改名为本任务矩阵。

## 2. 固定来源与模型权限

所有结果路径以下均为canonical绝对路径，使用前必须精确核字节SHA；不按mtime/latest发现来源，不递归历史、不重新执行原fullstrict。

| 已有权威 | 实际path | 实际SHA256 |
| --- | --- | --- |
| shared prepare | `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json` | `7ca85050eeab4e47ed35cf84c284ed7e08fc64bce8064d7db9450074045b643a` |
| shared report | `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_MAIN01/report.json` | `668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a` |
| shared complete receipt | `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_consume_MAIN01.json` | `d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd` |
| same-action native receipt | `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_physical_20261029_v1_consume_MAIN01.json` | `48a1a5ecadf7043e3162ca104d8f43b0d59baf88655a076f675c9f96d697b9f7` |

本文实际只读取上述prepare、两个receipt的小型字段及文档；report SHA继承已完成MAIN receipt/既有builder交付，未再读取132MiB report，更没有重建其native/source闭包。前三权威的接收接口直接复用冻结 `agent_p3_actual_cp5_task_builder_20261029_v1.resolve_actual_cp5_tasks(authorities)`，返回known2任务与provenance；不改builder，不把其原goal改写进旧task。原任务压缩export路径/SHA由prepare.layers给出，未来resolver按已有ABI有界读取，不能从打印的normalized token反推原raw10。

可读源固定于 `/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/`，本轮实际hash如下；只复用其公开原语/数值规则，不把新任务声明为旧consumer已验收：

| 文件 | SHA256 | 对接内容 |
| --- | --- | --- |
| `agent_p3_actual_cp5_task_builder_20261029_v1.py` | `703f5f552c1911159b2144f72711c542ca48c408ed4e306b1fc17b019df889f1` | 原2任务/4分支一致性与actual cp5输入 |
| `agent_p3_shared_panel_20261029_v1.py` | `8cc8f796f313d4cc3f209ce231ebcaec4808f096acc02781cd6df734800dc464` | 原native时序与逐事件计量，尤其第673–680行 |
| `p1_constructive_gradient_mpc_cuda_20261027_v1.py` | `2c874aa56cc0b7727dbc20fbcbe60c3c0265bd368a6f5e1d050301f095be5be6` | 原32步递推、33候选、三次独立replay，第195–214行 |
| `agent_p3_feedback_gradient_mpc20_20261028_v1.py` | `081a18d7a43fb48a6ea3ebd77420db19cfcbcf487d03b90fa69ab3ecd1b1a878` | 原真实反馈、scalar codec/runtime守卫与H2硬shape边界 |

唯一模型：`/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm`；config SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`，weights SHA `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`，原tensor identity `4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`。本轮不读取/加载权重；未来run须重新hash完整source/installed-class/weights、逐native IO与before/after buffers，不以本文引用代替runtime认证。

Q固定原identity192 terminal mean-square；不拟合readout、改变Q、goal图渲染或normalizer。原raw action mean/std以prepare实际float32字面量为准（约`[-.0087,.0068]`、`[.2019,.2002]`）；raw世界动作经原scale100一次、模型归一化一次，不能重复除100。RGB-only，planner没有native7、物体pose、reward/success、未来checkpoint或任何新proprio权限。环境off、原camera/resolution/render_action及MATH/TF32-off/autocast-off合同继承原源。

## 3. 事前16格：两个来源根 × 两目标 × 两horizon × 两动作限制

固定root顺序0、1，只是原两个known roots；每根8格，总16任务轨迹，不是16独立根。固定枚举 `root → goal[G0,G1] → horizon[H1,H3] → domain[Dxy,Dx]`；cell_id=`r{r}.g{g}.h{h}.d{d}`。不重试、扩池、替补、去重或按Q/原生成功筛格。两个goal碰巧相等、已成功或受限域不可达也原位保留。

| 因素 | literal定义 | 必须保留的区别 |
| --- | --- | --- |
| 起点 | 该root的actual control5，原prefix5已真实执行 | 所有16格均真实独立reset/place/warm10/prefix5；不能hydrate或从另一臂poststate继续 |
| G0 | prepare.tasks[r].goal及原来源配套native goal/goal RGB | 原冻结derived目标，不能换成本次哪个臂更优的终态 |
| G1 | prepare.tasks[r].root_context.latent_history[-1]，即原control0/absolute10实际RGB的既有编码；native goal取同源root_state/cp0 | “回到已观察根”任务，goal早于起点，不用未来physics/新winner endpoint/随机latent；不是新任意可达性证明 |
| H1 | 每边界预测5raw，即一个packet | 不是把执行总时长也缩为5 |
| H3 | 每边界预测15raw，即三个packet | 晚边界允许预测超出control20，单列forecast绝对终点；不截断以凑同终点 |
| Dxy | 每未来packet logit半径 `.75*sqrt(5)`，原两维chart/合法raw box | 沿原逐packet域，不借全H球扩大支持 |
| Dx | Dxy内的坐标子空间，所有新未来raw_y为**正零float32**，raw_x仍原域 | 限制的是raw，不是normalized-y=0；warm10/prefix5历史不mask、不投影、不重做 |

G1没有新encoder调用：已有root embedding、原control0 RGB及native rootstate必须有完整同源join。按root检查所有原分支root/context/raw10/goal/source一致，之后仅以固定索引序列化，不按outcome挑代表。不能把仅像素近似相似当编码免费复用；缺原goal图、rootstate或原latent身份则该格`blocked_input_join`，不新生成替代goal。G1对目标RGB/native predicate的绑定产生**新own task identity**，保留original_task完整provenance；不能保留旧task_id假称同一任务。

时间：原history绝对0/5/10对应control−10/−5/0；actual-cp5三帧为绝对5/10/15，对应control−5/0/5。新增规划边界control5/10/15，实际提交到10/15/20；实际任务执行窗口固定control5→20共15步，原prefix不计作该任务优化行动。H1预测终点10/15/20，H3预测终点20/25/30；只有真实已到达且时间相同的位置能报告预测实现误差，预测25/30在本协议中无真实标签，必须null。

### 3.1 生成器与对照范围（不是现成可运行adapter）

为使矩阵费用可计算，提出固定的**GD-MPC描述性probe**：复用已冻结feedback-GD的数值规则，推广packet数K=1/3；每节点全零logits初始化、32次scalar gradient更新、eta1、每packet原投影，保留zero＋全部32iterate，稳定`(terminal meanQ,iteration)`选优，再独立三次full-H重放。AD只优化动作，fresh raw leaf为`[5K,2]`，scalar原float32 meanQ，解析`tanh`链一次；部署原NumPy `tanh(float32 logits)` chart。Dx只在新参数/梯度上固定y坐标为正零，投影不放开该坐标；不是先优化Dxy再裁切winner，也不mask原历史。无warm start、线搜索、调beta或按任务调整次数。

这是一份**待批准的新数值/调度合同**，不是称封存H2 driver已支持K=1/3。现有GD的leaf/codec/choice/journal consumer硬绑定H2，必须未来NEW adapter显式实现新shape、mask及完整saved重建；禁止globals patch、冒充旧receipt。每格只执行已seal赢家FIRST5；返回ACK及本臂实际RGB后才能下一节点。原真实cp5输入可继承，但必须先重新物理完整重建并exact gate；cp10/15使用各臂自身新图，不能从旧轨迹借标签。

只用该probe的16格可以描述任务敏感性，**不能支持Full D2P复用/调度/组合优越性**。后续正式Full与强基线矩阵须MAIN另外锁定算法实现和适用组件；本协议不偷偷把缺失Full臂删掉并称完成E3。不存在一个当前可供生产的16格CLI。

## 4. 固定分母、预算与物理停止

每格最多3个节点、每节点33候选/32scalar reverse/3winner replay。每个候选或AD-forward完整执行K个原生packet预测；三次winner重放是额外成本，不复用候选输出。

| 资源轴（无提前停上限） | H1每格 | H3每格 | 全16格（各8格） |
| --- | ---: | ---: | ---: |
| episode / node / 候选评分 | 1 / 3 / 99 | 1 / 3 / 99 | 16 / 48 / 1584 |
| 非AD预测packet：`(33+3)*3*K` | 108 | 324 | 3456 |
| AD primal预测packet：`32*3*K` | 96 | 288 | 3072 |
| predict API / batch1 samples | 204 | 612 | **6528 / 6528** |
| predictor上下文frame-samples（每call3） | 612 | 1836 | 19584 |
| scalar reverse / scalar grad API | 96 | 96 | **1536 / 1536** |
| 新encoder / projector（cp10/15） | 2 / 2 | 2 / 2 | **32 / 32** |
| 本协议新Jacobian / SVD / solve | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| warm10 / prefix5 / 新任务动作15 | 10 / 5 / 15 | 10 / 5 / 15 | 160 / 80 / 240 |
| 全部public动作 | 30 | 30 | **480** |

原生AE/predictor/pred_proj各6528次与predict是嵌套不同资源单位，不加成26112“独立forward”。1536 scalar reverse不是192方向J-VJP；backward engine实际内部次数未知，不由1536推填。独立FLOPs、kernel时间、显存/训练实耗未知保持unknown。若早停，已attempt失败不退款，保留16/48/1584 roster分母，未执行候选另列not_reached，不能报实际全返。

32新编码只在原cp5精确RGBA/预处理/latent继承证明闭合后成立；任务G1目标和初始history编码为继承成本，另列其实际原authority，不称未认证zero-cost。若未来实现要求额外encoder、预测或replay，必须先另版修改预算，不能在run偷偷加调用。最终cp20只保存RGB/native评价，本协议不新编码其latent；若要终态latent Q须另版预算，不隐含16次免费编码。

物理计量按实际source而非只抄文档：`agent_p3_shared_panel_20261029_v1.py` 第673–680行的完整路径为constructor1/reset2/set_state4/goal_render2/goal_setter3/goal_bind1/public30/physics304/explicit_render32/close1；physics公式为`2*reset+10*public`。全16完整路径为16/32/64/32/48/16/480/**4864**/512/16。旧shared文档还列reset3/physics306的预留上限，**不能把它写成实际完整路径**。未来driver应同时保存完整路径预期、原硬caps与实际attempt/return；失败/cleanup也扣账，不能把unused reserve当实耗。新增G1绑定不准额外reset/render，若现有原语不能在上述完整路径内表达则blocked并先交MAIN，不静默扩cap。cp20无需新encoder不等于省掉原PNG/render/native账。

每轨迹独立job沿GD原工程上限：wall1800s、输出512MiB含64MiB final reserve、disk floor200MiB、CUDA allocated3GiB/reserved4GiB；全16固定总输出上限8GiB、job wall上限和28800s，**串行总inclusive wall必须另量**，不将节点/父子inclusive时间相加为真实总耗时。wall从verify前起，直到report fsync/hash及独立completion边界；超限仅partial，不能签all16 complete。尚未做新H3运输shape/NPY/gzip每event flush/重复metadata整数预算证明，故这些cap不是“已证明能装下”；未来实现先RAM算完整单job及failure reserve，超限先报告，不能扩cap/删IO。本文没有生产输出。

每格先原reset/place/warm10/prefix5，首个任务intent前持久化exact native7/body snapshot/history RGB/raw bits guard。原早停/终止/truncation立即生效，不强推至20；checkpoint不足、标签缺失为null，不padding或拿early terminal当same-time20。prefix期间若已terminated，任务轨迹不派发，保留`not_started_native_terminated`；不能因G1目标重绑定绕过已终止环境。

任务原生成功使用原source evaluator的完整goal7及同一success predicate，不用模型Q、object-only或新阈值替代。沿原PushT联合agent/object XY阈值20及原角项`min(abs(dtheta),2*pi-abs(dtheta)) < pi/9`；wrapped-angle独立诊断，不能悄换原角实现。目标G1仅更换预先source-bound goal参数，不改物理state/renderer。首任务边界已成功必须记录`initial_success`及时间5，不按效果删格；native实现是否允许零action完成须future adapter固定并审读，未能忠实表达则blocked，不伪造一次step或完成。

## 5. 评价、资格及当前能做什么

主描述量：每格native success、首达控制时间/任务elapsed、实际任务步数、jointXY/objectXY/angle/velocity、terminated/truncated/error/guard拒绝；分别保存model预测终点与真实同时间终点。规划inclusive wall、模型/编码/AD各轴、真实public/physics/render及IO字节全报。跨horizon204对612predict不是等总算力；相同32步、同上限也不是equal-FLOPs。Dx支持更窄且可不可达，不把不同任务成功差写成算法优劣。

同root内各goal/horizon/domain配对；root为独立性单位，只有known2，不给独立确认显著性、bootstrap伪N16或事后优选格。long-horizon E3仅有15步任务窗口的工程probe，不声称已满足长期任务完成；后续更长deadline必须新协议/预算，不延长失败格。跨goal比较只为任务描述，不跨不同目标直接用latent Q差论任务增益。

| 范围 | 当前可做 | 当前不可执行/缺项 |
| --- | --- | --- |
| 原known2、原goal、原H1/H3与bank | 只读既有shared及H3同动作/编码产物；可描述原已付结果 | 不能把bank endpoint/midpoint当goal轴；不能重跑已完成桥或称E3完整矩阵 |
| 本16格，单PushT LeWM | MAIN可审批本文；未来可做全16输入解析、H1/H3 mask/GD核及fixture | 当前没有支持G1＋Dx＋新leafshape的launcher/strict；不提供假可运行命令，不借旧H2/H3 schema impersonation |
| 正式fresh E3/P4 | 仅接收原来源附件及既有资格接口 | certified fresh=0；panel history_request/exposure_inventory/actual_roots/panel_approval未齐；不得开reserved outcome或新批根 |
| PointMaze/Reacher同checkpoint任务矩阵 | 可另读已固定任务配置/权威，分别提出独立固定矩阵 | 本文不批各环境新goal/root/预算；旧SRA受限P4不能移用到D2P，不能把已有calibration当fresh；evaluator state不进planner |
| JEPA/Hi等其它模型 | 已完成原生诊断保留，源码/权重/许可/输入权限可只读对接 | JEPA RGB＋proprio不是本RGB-only合同，Hi有自身head/layout/目标域；不能直接复用本goal192/normalizer/mask，更不能拿多个环境权重算多架构 |

fresh依据复用已存在的 `agent_p5_fresh_confirmation_preconditions_20261029_v1_MAIN01.json` SHA `d453e4cf70518839844916566fa274959c515d05310ae090fb5699e589cf3133`、P4 certificate audit SHA `15e3ece87e36eba469a0cc2188efbb7edaa94fa00f1562f9e6adc2052e4e4ae0`、panel manifest SHA `29c919fe1365e8f7422951026d6e8f8c90b293061981086f175096ab647c848d`。不重复全盘扫描；这些是既有source范围，不是所有模型训练ancestry的证明。远端weight revision、训练dataset/split/normalizer来源、预训练成本与许可接受仍分别披露unknown，不因本地freeze而抹除。

## 6. 下一可执行接点与封存顺序

1. MAIN先审批/修改**本文新协议**（尤其G1 return-to-root、Dx、GD probe和预算）；审批前无新优化/神经/物理。旧missing/negative/failure及本轮全部已封存件不改。
2. 获后续NEW实现授权后，输入侧调用原actual-cp5 builder全部known2 join；另外只读上述native receipt的原root/cp0来认证G1。输出新own task identity，provenance保留original task、所有path/SHA、绝对/控制时序。发生null/mismatch应给16槽blocked精确缺项，不自行新编码/换goal。
3. 再实现新K=1/3/Dx numerical core与完整runtime/saved consumer（不是改旧GD）。先通过RAM32步递推、scalar transport、mask raw正零/normalized非零、全候选/三replay、真实Store→reader链、IO预算与earlystop/null测试。接 `agent_downstream_resource_contract_20261029_v1.validate_shared_panel`；不同task/goal/domain明确分组，不能为满足旧同task接口谎填相同hash。
4. 新source/tests/doc/预算与实际input pins冻结后，MAIN审读决定是否另行授权known-root HOSTCUDA＋native生产；每job完整strict、saved同时间分析由MAIN执行，科学门仍false。未来正式E3须另有独立根资格、Full/强基线与适用P阶段准入，不由这16格替代。

**本次验证范围**：仅原文/需求矩阵与指定可读源合同、三份保存JSON小字段的定向读取及整数资源公式核验；没有运行builder、optimizer、模型、物理或任何fullstrict，没有生成生产manifest/results。本文仅文档，不新增测试文件/代码工具；最终SHA与文字/公式核验结果在交付回复列出。
