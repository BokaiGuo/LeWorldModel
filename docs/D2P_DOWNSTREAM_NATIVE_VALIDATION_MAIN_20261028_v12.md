# 后续 P：真实历史、RGB 跟踪与物体响应验证

承接 v11；原 C1–C3、三组实验与 P0–P5 范围不缩小，整体仍未闭环。
神经执行仅主机 CUDA；不训练、不拟合 Q、不放宽门槛、不按结果选根。
以下区别实际执行、严格验收、代码测试和科学机制，保留旧结果与全部未知槽。

## P2：缺失历史已真实重建，完整历史接口仍待验证

原 collector 固定 seed202753010/oversample4；核全部四根身份，只补 root1/2/3，
root0 使用已有历史。prepare SHA `376adbeebb4a9f749269448121ed477d16705392fc3bb9fe212b57f3b21a9a8c`。
主机真实执行 11.409302 秒，报告 SHA `89d545d538a75f8ae7778d3d4761651ff5b7866ad2041712b26915b82e303c1f`。
constructor/close 各1、collector reset32、root reset3、warmup step45、render61全部返回，
errors=[]。这是公开 API 计量，未挂钩的内部 Space.step/setter/reset 工作为 unknown，不能写成0。

完整验收 `le-wm-main/results/agent_p2_warmup_history_reconstruct_20261028_v1_consume_MAIN01.json`
SHA `151e9ec6a36a0bd189476298f32381efd32d21c98781c0bedac06ac660579330`，
status=history_reconstructed；四根八臂均保留。补出12张真实 PNG（每根3历史+goal），
核原 goal、native7、raw history 与身份。stageB_adapter_ready=false、freshness=false，
不是独立新根或 P2 组合收益。下一项为完整三帧 public-rollout/manual ABI 对照，不能用重复帧替代。

## E1：共同控制器八端点主机 CUDA 与完整独立验收完成

实际 prepare SHA `b6396d35d2d1b146867d123747392ee4b9573e46088e23d86c59350d93c5c5ad`。
八张封存端点 encoder/projector 各8次真实返回，0 predict/AD/新物理，模型前后不变。
报告 SHA `3f64f2b3f74be98fdd0f0eda36fb0c309990ae42d046a71ce4ca8d7289d654c9`。
MAIN 完整审读 adapter 并调用独立 saved consumer，重建全部模块输入/输出、原像素、
192维均方计算、完整 seal 与资源账；完整回执：
`le-wm-main/results/agent_experiment1_common_controller_rgb_20261028_v1_consume_MAIN01.json`，
SHA `4d5d8e9b5014487e7f4a932fe19a92a5d3a011c4db874cb9cd5ca14b47f04dc7`。
薄持久化 wrapper 另4项测试通过；P1 RGB40/E1 RGB adapter 联合113项测试通过，测试不算科学样本。

| 已知根 | 任意 latent 跟踪代价 | 编译程序跟踪代价 | 任意 latent 终态真实 Q | 编译程序终态真实 Q |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.71470283 | 0.36715827 | 0.43138775 | 0.47633620 |
| 1 | 1.02675545 | 0.21538976 | 0.84865898 | 0.28699516 |

代价严格为0.5实际中点到各自目标的MSE+0.5实际终点到原task goal的Q；
不是物理距离或成功率。两个根跟踪代价均下降，但root0终态Q及native7距离均更差。
原native成功仍0/4，六科学槽中learned两槽仍unknown；任意latent是有限弱搜索，
compiled继承proposal成本不同，不能称强基线或等总算力胜出，更不能宣称P1/C2成立。

## P5：四词物体响应严格分析完成，非任务收益证明

实际分析 `le-wm-main/results/agent_p5_contact_word_response_analysis_20261028_v1_MAIN01.json`
SHA `f38b62550c347b66e5533885694a53db22ae1b58798d18de3dc11f6f1c3edf2c`，
analyze及完整verify通过。全部4baseline/40suffix/20pair保留，重新消费原物理与模型strict。
MAIN另从40个原生double body JSON逐SHA独立复算：

| 固定配对 | 最大物体XY差（像素） | 终态物体XY差（像素） | 终态主值角差（弧度） |
| --- | ---: | ---: | ---: |
| XY−YX | 21.65204254 | 15.18354488 | 0.04263766 |
| X−X−zero | 20.97339335 | 20.97339335 | 0.12876319 |

确有物体pose响应，不只是agent或整图变化；但只有一个构造OOD源、四词不是四独立根。
不据此证明接触因果隔离、非贪心任务收益或C2，task_success仍null。
后续独占任务为固定原生任务指标及前瞻收益验证前置，不事后选胜词。

### 原生任务指标也已实际重算并完整verify

MAIN全读155行实现及文档，21项测试通过；实际prepare与deterministic verify均exit0。
`le-wm-main/results/agent_p5_task_benefit_prerequisite_20261028_v1_MAIN01.json`，
SHA `81334398431194b33e98a4212246688fb56ba58f80bec9f47ec1ba54ac713ba0`。
所有四词原prefix coverage相同0.32396671，终态coverage：XY0.36832320、YX0.34322680、
X−X0.37083639、zero0.32396671。固定XY−YX差0.02509640，X−X−zero差0.04686968；
全部十帧与四词保留，不输出best word。原生reward为clip(coverage/.95,0,1)、done=False，
不新造任务成功阈值。此合同在既有结果后实现，只能回顾性校准。
前瞻任务仍缺合法冻结goal-scoring、预封存public-only决策和比较策略；已交新独占实现补具体接口。

## P3：原生子节点执行推进中

MAIN全读566行物理桥与协议、53项独立测试通过；源 SHA
`7f1d0565d961d9b1e33dc6c601ca4f4bd62ea1969dc2411126beed7b99704955`。
prepare已完整重算compiler strict及原P4消费，补查全部native snapshot/PNG/goal来源，
实际 prepare SHA `e2a8752f351d47d780a49774c07b9bead1a33c75690ca286ef9bff4727681f95`。
主机run已exit0，91.17753971秒，报告
`le-wm-main/results/agent_p3_child_native_physical_20261028_v1_MAIN01/report.json`，
SHA `56c3e97208597bdc8424dcfb263c8840683c6b3413abeb79d01fa46762dc5df6`。
41可执行实际完成37条returned5、4条native_stopped；5预算拒绝/2未到达不执行，完整48槽不删。
993 public/10094内部physics/1075PNG全部attempt=return，41construct/close、82reset、164set_state、
82goal_render、123goal_setter、41goal_bind，errors=[]、0新神经。
报告四次原native成功终止，但含相同轨迹（e01 I/R b05fixed），不是四独立成功或P3准入。
两known根不能称41独立样本；保留原生早停，不补动作来凑1000步预算。

完整consume原v1确实被上述计数假设拒绝；MAIN全读原factory/metered_class/setup并核真实账，确认是消费端代码错误。
新saved-only consumer仅改两处精确census（reset2/set4/goalrender2/setter3/setup physics4），其余原验证逐字保留；独立42项测试通过，实际完整strict已落盘：
`le-wm-main/results/agent_p3_child_native_physical_saved_20261028_v2_consume_MAIN01.json`，SHA `93785928ebaf03c1711610cba58ac9995e16b47d0fa2d741c6394867471b4f08`。
完整48槽通过，37 returned5+4 native_stopped、5 blocked+2 not_reached，198 own、993 public、10094 physics、1075 PNG，0新神经；不重跑物理、不改旧journal/prepare。修复确认该负结果曾含代码层面验收错误，但仍不构成独立根、C3或P1科学通过。

### P2 Stage B cache 与 compare 均已完成

三件套源码/doc/tests冻结，独立42 tests通过；实际cache prepare SHA `8814d514f0f0f925bb437a8f85917a202727d4e2d1ccbfa4a11f3da78cfc8542`。
主机CUDA全8根臂真实三帧编码完成，report SHA `90682bed11e5ecf4bc2fd65564bbc733827bcbedcf8105f44ab661df577865d2`，8 encode/encoder/projector、24 samples/frames；strict回执 `d533bf48190a117a275a941b9103a5624f97f106cc8c3dc213cb7326a6872c9e`。
新三帧与旧teacher差异仅约1e-6，作为实际接线诊断保留，不调容差、不替换旧值。compare prepare SHA `1e28ab8c4c784794caa053bc01e25065145f294eb347f178852409346470ca18`。
主机权限恢复后16路线已实际完成：报告 SHA `70bc87cdc68fc701cb9a2cf5175d569b452ccee21a5651cd9817c7adaea05985`，strict 回执
`le-wm-main/results/agent_p2_stageb_triplet_cuda_20261028_v1_compare_consume_MAIN01.json` SHA `9d7e429a2c52d3c6f03adcfdd38ab0c7b07ed68248009663936bf62a732d374b`。
public/manual 各8路线的所有 native buffers 均逐项相等，16路全返回、无错误；这排除了当前 wrapper 接线差异，但不等于真实任务收益或P2组合机制成立。
fixed391/adaptive634实际predict不同，不称等预算C3。

三条已有本地基线 GRASP、matched sampling、moving gradient 已完成 saved verify/consume；结果状态均为 verified/commands_verified_not_admitted，仍不冒充论文原协议复现。

## 新发现的代码接线失败

P1 teacher RGB40 v1 的真实prepare在加载模型/创建输出前失败：roster读取
`counts.physics_step`，真实物理报告字段是 `counts.internal_physics_step`，触发KeyError。
虽然该adapter单测通过，mock字段没有覆盖真实schema；这是代码问题，不是科学负结果。
原v1保持冻结，新v2修复及真实saved接线回归已交实现线；不得改旧报告字段来迎合adapter。
原index12 final-on缺失仍保留unknown；修字段不能使full40或P1通过。

### P1接线修复后已实际完成主机CUDA及完整验收

MAIN审读v2完整差分与文档，v2/基线桥175项联合回归通过；真实prepare SHA
`499f1d310b1844bb05ba78c88d3eecc4285c7041a6482fd7c68fc2fa39ce8676`。
主机CUDA19.57198318秒，39encoder/projector全部返回，其余预测/AD/物理为0，errors=[]、模型前后未变。
报告 `le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v2_cuda_MAIN01/report.json`
SHA `cb08d580b3ee2e8d15d2cc83065a8dc68a74672df0e2b602479a836469cb15f3`。
原生预处理逐元素一致、257token/CLS/projector/192维latent完整trace和原goal/Q逐项严格消费通过。
完整40槽回执 `le-wm-main/results/agent_p1_teacher_library40_encode_20261028_v2_consume_MAIN01.json`
SHA `9dfb1081711505cef281f1ff547b1dad46b59cb0ea207e985497ecbf9cecf0a9`。
39编码、index12缺失null、full40/P1保持false。新独占线继续做全槽预测/真实Q排序归因，不事后挑库winner。

### P1 全槽排序归因已完成（仍不准入）

独立 saved-only 归因已通过 14 项 RAM 测试，并用固定40槽重建验证通过：
`le-wm-main/results/agent_p1_teacher_library40_rank_error_20261028_v1_analysis_MAIN01.json`
（SHA `e23e60588fa9766b690e605022e089656c4c80ad185853773e23a942b5e0228f`）。
它保留40个固定槽、39个可用终态、index12 unknown；同词 Z/S 配对为19个可比、1个unknown，
未观察到反转；同角色库内为76个可比、4个unknown，其中12个预测/真实排序反转，
观察反转率为 `12/76=0.1578947`。这些是固定库的误差归因，不是 winner 选择，
也不是完整40槽率；P1 仍为 false，因为只有2个已知根且缺独立 fresh root、真实闭环收益和完整槽。

## 强基线协议交接

MAIN全读236行桥与协议，实际verify通过manifest
`le-wm-main/results/agent_baseline_protocol_bridge_20261028_v1_prepared_JASON01.json`，
SHA `e2c27d974f39191f7b6e07e6f1980abfc86553c0d84f5308c069377dcbf6c61c`。
GRASP/LeWM三个既存工作流核同两task身份、原预算轴和已有strict入口；
作者值保持独立引用栏，Hi-LeWM可信完整对象与PlanWAM官方驾驶资产/evaluator仍缺，
不把本地PushT变体冒充论文原协议，也不把metadata verify冒充实验结果。

### baseline saved consume 后续回执

此前在运行中的三个 saved-only consumer 已完成回收，均保留为“已核验但未准入”
而非比较胜出：kinodynamic 的状态为
`fixed_H2_commands_verified_not_admitted`，no-composition 与 sampling 均报告
`P1_admission=false`；三者都没有官方复现或等总计算量成立的证据。
kinodynamic 明确报告 errors=[]，其余回执缺此字段，不能补写为空。
kinodynamic 回执为
`le-wm-main/results/p1_moving_kinodynamic_baseline_cuda_20261027_v1_consume_main01.json`，
no-composition 回执为
`le-wm-main/results/p1_moving_no_composition_baseline_cuda_20261027_v1_consume_main01.json`，
sampling 回执为
`le-wm-main/results/p1_moving_sampling_baselines_cuda_20261027_v1_consume_main01.json`。
其中 no-composition 的 176 次 action/predict 与 384 次 VJP、sampling 的 412 次
action/predict 都是资源账，不等于公平预算；sampling 明确 `new_physics=0`。
因此本轮只关闭了“saved 结果未消费”的工程缺口，没有关闭强基线比较或 P1/P2。

### 本轮后续 P 线的严格回归

P2 Stage-C saved consumer 已对 `p2_composition_paired_physical_20261027_v2_main01`
完成 8/8 分支、116 个保存文件和三份外部报告的 SHA/PNG/journal 绑定；持久回执为
`le-wm-main/results/agent_p2_stagec_composition_saved_receipt_20261028_v1_MAIN01.json`
（SHA `7297bb9b5788bf379a21dd791ee61cbe9e21c36026b625a0c24027f50284d7a4`）。
回执明确 `P1_admission=false`、`P2_admission=false`、`composition_benefit=null`，
新增 neural/physics/predict 均为0；这是 saved 合同闭合，不是组合收益。

P3 child budget authorizer v2 已修复预算向量篡改、失败 ticket 退款和 fixed/adaptive
重入问题；6 项回归通过，仍未授权 CUDA，因尚无 reviewed native adapter。P4 freshness
audit v2 的9项回归通过并保持 blocked：现有 construction NPZ 不能升级成 freshness/history
证书。P5 terminal attribution v2 的15项回归通过，但实际 CLI 审计发现 authority 顺序
失配；v3 又发现内部 translation point 仍未交换 numeric/physical，故当前不启动 CUDA，
等待带真实九文件路径测试的独占修正版。

## 仍需完成

### P5 terminal attribution v5 已实际完成（覆盖前文待修快照）

主线程发现并逐一拒绝 v2/v3/v4 的 authority 顺序、Path/dict API、bindings 方向和
全局 monkeypatch 递归问题；v5 改为独立 runner，不修改旧封存结果。项目指定解释器
`.venv/bin/python`、`OPENBLAS_NUM_THREADS=1` 下26项测试与真实九文件 prepare→verify通过。
系统解释器曾在底层 contrasts 严格算术复算中报错，项目解释器通过；尚未判定精确差异来源，
不将其认作神经负结果，也未改容差来消除差异。

实际 prepare SHA `79c3c2f1aa1b8f1b9223e8d87f22986f5c3437144e10bbe78a6cc556093beafd`。
主机 CUDA 7.290963秒，四词四终态完整 official_encoder 4 API/4 samples/4 frames，
predictor/proprio_encoder/新物理均0，模型前后/expected相等，errors/finalization_errors=[]。
报告 `le-wm-main/results/agent_p5_terminal_goal_attribution_20261028_v5_cuda_MAIN01/report.json`
SHA `9505f332fc6c1cbb670d9a77f46290f8d2822486b00d89006ea038359898bdc5`；
严格消费回执 `le-wm-main/results/agent_p5_terminal_goal_attribution_20261028_v5_consume_MAIN01.json`
SHA `33f3f24735131fd8b569c1d0a87068d1b3f4e26dab31618a4afb90005cf8c927`。
MAIN另直接读取四个实际token、不调用producer/consumer，float64重算三组MSE及model equality通过。

| 固定词 | 预测目标MSE | 真实RGB特征目标MSE | 预测—真实特征MSE | 原生coverage |
| --- | ---: | ---: | ---: | ---: |
| xy | 1.82019808 | 1.87302287 | 0.08352148 | 0.36832320 |
| yx | 1.90322721 | 1.94143139 | 0.13638054 | 0.34322680 |
| x_minus_x | 1.63789701 | 1.89871782 | 0.11282939 | 0.37083639 |
| zero_zero | 1.53321063 | 1.14708618 | 0.75619494 | 0.32396671 |

真实编码度量本身仍偏好 zero，但coverage改善并不一致；这提示预测误差与任务度量失配需分轴处理。
单个已知构造OOD根、无新物理、未隔离agent/goal raster混杂，不能将此归为理论失败或C2/P5通过。

P3 child budget native wiring 真实 saved prepare→verify 已完成：
`le-wm-main/results/agent_p3_child_budget_native_20261028_v1_prepare_MAIN01.json`，
SHA `22f162fb493b6b83b30bddd66ec1b7f900f4bf36ea73c6249a5a97258002e50d`。
状态 `strict_saved_wiring_verified_not_authorized`；完整48槽保留，不将历史资源当作新付费。
下一缺口为新HOST CUDA lifecycle/完整交错fullarray consumer；当前未运行P3新神经。
本轮八组新旧代码在项目解释器、固定数值环境下联合89项测试通过，`git diff --check`通过；
测试数量不是科学样本数量。

P1历史定位新消费者MAIN完整verify通过：
`le-wm-main/results/agent_p1_teacher_library40_history_localization_20261028_v1_analysis_NASH01.json`
SHA `7d171fea5ad553541dfd6393e4bf2530992ef916c92fcc39295590de7337d3a2`。
完整40槽/800实际动作返回/80次evaluate-replay接线一致；八组反转数为0,1,1,2,1,3,2,2。
这排除这批保存输入的历史/动作/目标绑定漂移，不识别反转因果来源。P1全40槽official/cached
同输入CUDA API比较已续派实现，不能以P2八臂ABI结果替代。

强基线v3修正了将算法family/name差异误作不可比前提的逻辑错误，实验控制字段未知仍拒绝。
实际inventory SHA `0683c5d90afb00680f2bcfe45d0fc6bf8309d8c82a4c9448b26960069ebe289f`，
状态 `prepared_contract_refuses_run_until_controls_complete`，不授权执行；独占resolver继续从真实
prepare/sourcepack解析缺失的控制证据，而不是把未知字段补0。新P1定位38项及baseline48项联合86测试通过。

P1非贪心能力改变与实际收益；P2组合优于强对照（完整历史public/manual ABI已完成）；P3完整树与等实际预算；
P4原始历史认证（缺失ledger不能用metadata修复凭空补出）；P5前瞻任务收益规则和对照；
强基线、learned-prior合法资产、独立新根及多环境/跨模型三实验闭环。
现有五实现线独占新文件，MAIN亲自审计和执行；不把实现交付或正在运行算成实验完成。

### 后续 P 的继续推进：全空间归因与任务合同隔离

MAIN 已执行 P5 全空间 saved analyze 和独立 verify，并重新运行17项测试，全部通过。
报告 `le-wm-main/results/agent_p5_spatial_patch_attribution_20261028_v1_MAIN01.json`
SHA `c9e747a3d028cba8d79a47402b0282663e2e2ac7b0a765f6b320dfeba91f2f09`；
全词可视化 `le-wm-main/results/agent_p5_spatial_patch_attribution_20261028_v1_MAIN01.svg`
SHA `b8bf0eb89e0e17cc19951a295204efff36891226d8256defe7a7fec2ed64a1c5`。
四词各256 patch、全部16个固定tile，逐patch恒等式残差不超过1.87e-15；未挑ROI、词或胜者。
这是对既有主机CUDA结果的保存数组归因，新增神经调用与物理步均0，不是新实验样本。

原始RGB目标MSE依次为 xy=307.97365939、yx=313.48054183、x_minus_x=328.37610278、
zero_zero=342.65931920，与真实feature目标MSE偏好zero的方向不同。
这进一步说明不能用单一latent分数替代物理目标；全局attention下patch位置也不是因果物体归因。

MAIN源码核查发现三种合同必须分开：

- 当前goal_spec明确agent只是scene nuisance，物体目标为[256,256,pi/4]；原生coverage只衡量物体。
- 官方PushTWrapper.eval_state对agent和block四个位置坐标联合求范数，要求小于20，再检查角度；不能冒称与当前object-only任务相同。
- RGB goal保留agent=[64,448]。四个真实终态agent距该位置为208.5143、206.9553、212.2326、211.6247；
  因此若错误套用该wrapper判据，四词即使忽略物体误差也均失败。但这不能证明agent就是feature排序失配的因果来源。

原生renderer每次都画goal polygon；with_target仅在_setup中决定goal_color，事后只改with_target布尔值
不会隐藏goal raster。下一诊断须固定完整物理状态，使用经审计的原生绘图开关隔离agent/target因子，
不step、不涂像素、不更换已封存planner分数。该实现已续派P5 worker，尚未运行，因果结论未成立。

P3 worker继续host lifecycle与48槽交错fullarray consumer；P4 worker新增明确缺失authority的
certificate请求合同；baseline worker继续精确控制证据解析。P1仍由MAIN审计，并等待全40槽同输入
public/cached接口实现以执行主机CUDA。上述任务均不等于对应科学P阶段通过。

### MAIN 实际完成 P1 双接口，P3 进入主机执行

P1固定40槽/80路线已实际主机CUDA完成，wall=26.77823577秒，errors=[]；rollout40次，
AE/predict/predictor/pred_proj各80次，encode/encoder/projector/AD/physics均0，模型前后hash一致。
prepare SHA `157603d6cbf62457fc961eb9bbd4f22fe4e45df200a257ef7e0f6a4fca1b7d71`，
report SHA `38ad7134ba7f13b7244a9bc9ad311d476ccd67a0df53daf9fa7f314e7876ae3e`。
完整saved consumer已通过：
`le-wm-main/results/agent_p1_teacher_library40_api_equivalence_20261028_v1_consume_MAIN01.json`
SHA `fc6f7022efc1df77b3eb85ff8262a87d0cbb5cf6bd1f65d6d77f512ae5651d68`。
40槽所有同语义原生buffer精确相同；两路仍为76已知组内pairs/12反转/4未知、19已知同词ZS全一致/1未知。
这是显式使用同一emb缓存的public API验证，不是RGB重新编码或独立新根验证；index12实际终态继续缺失。

MAIN同时捕获consumer诊断字段的语义错误：`vs_original_prediction`将三行native输出
与旧的shifted context三行比较，前两行不是同语义，40个false不能据此解释为旧预测漂移。
MAIN不导入producer/consumer，直接读取实际npy/gzip，确认旧context前两行=原history后两行，
且40个对应终态与新cached终态全部精确相同，最大绝对差0。冻结v1不覆盖，另派独占新saved语义消费者修正解释。

P3 MAIN已完成完整实际prepare→verify（5527项source closure），新prepare：
`le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_prepare_MAIN01.json`
SHA `92f2edacae8e0f519e09d573ed1c78c61247e78b7b99042e27e29f39868146c8`。
已提交主机CUDA执行；此处尚无终态report/strict receipt，不能称P3实验已完成。

P4实际request→consume已完成，request SHA
`4af2cd29179f58957979da837031f693ccf1af037809c668339ea68bd525f2b5`，文件
`le-wm-main/results/agent_p4_certificate_requirements_20261028_v1_request_MAIN01.json`。
仍blocked，明确原运行ledger/load、历史scope/cutoff、exclusion approval及未来pre-witness缺项；
保留8来源/16引用/10dispatch/975 unresolved occurrences，不混成独立样本，不打开保留根。

baseline resolver saved02由MAIN完整重建verify通过，SHA
`baa16c6bb9661be94c897077a48824278148c66efc01e5981e69af824b198685`：
kinodynamic/no-composition各两行9项control恢复；sampling仅root_ids，仍缺显式task路径链；
完整scoped budget与原加载闭包缺项不填充。后续比较消费者继续实现，未宣称matched或公平胜出。
MAIN新联合host/resolver54项、P1 API44项、P4 requirements25项测试通过；P5空间17项通过。

### P3 新预算原生执行与完整 strict 消费均已完成

MAIN主机CUDA实际run已完成334.89383817秒，errors=[]、模型前后hash相同，
CUDA allocated/reserved峰值142634496/146800640 bytes，48槽全部保留。
报告 `le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_cuda_MAIN01/report.json`
SHA `066b2f7f6e37fc9c4ed2795a07387c473da7e2e4418414e05ce0e6080ddc2978`。
41个budget_complete_native_unverified、5个original_whole_node_search_budget拒绝、2个not_reached。
actual fixed=391 predict、adaptive=634 predict；各自46 Jacobian、8832 scalar VJP、23 schedule，
fixed/adaptive各23/18 compile。相同cap不代表actual相同，不能宣称等总FLOPs或C3收益。
完整交错事件/48槽原生数组strict consume现已实际exit0，回执
`le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_consume_MAIN01.json`
SHA `14c87cd5c69562a53cacc7d3082a4938fcb776b814c9fc2b00dacd99df20bd0d`，
68,933,400 bytes；状态 `interleaved_native_fullarrays_budget_verified_not_admitted`。
独立saved复算92次signal-array SVD/41次compiler-array SVD，不计为新增CUDA信号成本。
actual_vectors_equal=false，P1/P3 admission均false；主线程另直接读取回执确认完整41/5/2分母。
新增物理执行仍0，后续physical handoff独占实现中，旧物理结果不可无条件换绑到新报告。

P1新语义消费者 `agent_p1_api_endpoint_semantics_20261028_v1_analysis_NASH01.json`
SHA `5e5aa57064574a09f2328ab9ff54267532020f0f5038b21b615ec79f6914a013`
由MAIN实际完整verify通过：两路各40端点逐bit相同、原Q差0，全部3行native与原native也一致；
原跨语义误比字段保留但deprecated，不拿它解释模型误差。仍不识别12反转因果来源。
baseline逐字段比较saved01 SHA
`1c299b6b8eb03037b3b82787cc1b314984bd741c129e08673490fb28a091c1cc`
亦由MAIN实际verify通过：kinodynamic/no-composition九项声明metadata匹配、resource未知，
与sampling的两对十项均unknown。两模块新增33+15=48项MAIN联合测试通过。
