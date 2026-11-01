# P4前置证据：可恢复字段、未记录字段与历史范围

2026-10-29；仅新增本文，保持所有source/tests/doc及结果冻结。不生成新null草案、response、approval或certificate，不修改gate。不运行模型、物理、网络或重采样；不读取planner/native outcome值。本分类使用 documentation-and-adrs 技能区分合同来源与证据能力，不增加准入条件。

## 1. 结论：不能恢复字段，不等于不能限定研究范围

现有原件足够恢复8来源/16引用的文件身份、52条selected身份记录、10次已发现dispatch的命令/版本候选/部分进程状态；这些已恢复，不应重复索取或再生产null草案。52行对应44个不同的**声明root-goal hash**，不是44个物理独立根。21份byte-verified来源候选也不是21份原运行加载证明。

原8报告的selected记录**没有序列化**state7、snapshot、episode/reset-attempt映射、逐帧RGB或完整尝试ledger；不能从hash、root_id或root_seed反推。对这些指定原件，可以明确说“当时此报告没有记录，无法靠再次投影追补”。对是否另有原始日志、原进程加载清单或其他资产，只能说本有界证据未取得，不能升级成“全世界记录都不存在”。

原协议硬前置是：对声明研究范围内截至prepare的既有校准/探索/预留来源作可信排除，固定四槽且不按outcome筛选，并在witness前核实际物理身份。它**没有要求证明所有机器、所有项目、无限时间内从未出现同一状态**。原文“全局未用”应结合其第3节的项目/任务范围解释，不能脱离该范围新增无限证明义务。无法消除的历史局限可以披露，但披露本身也不自动满足原合同的缺失身份/覆盖要求。

## 2. 哪些是原硬前置，哪些是后续接口或披露项

权威顺序：原协议/预注册定义本轮设计；后来的工具定义其自身工程接收合同，不反向改写原设计。

|要求|原证据定位|性质与当前处置|
|---|---|---|
|固定4槽、各一次proposal/attempt、无替补，无outcome/cost/difficulty筛选|原协议§2；预注册 `/candidate_order`、`/sampling_denominator`、`/proposal_per_slot`、`/attempt_per_slot`、`/replacement_or_retry`、`/selection`|原设计硬约束，不因某槽失败补根；本文不打开新槽|
|历史排除覆盖截至prepare的PushT校准/探索/预留及失败、未选、派生observed根|原协议§3第一段；预注册 `/exclude/5`、`/cutoff`|原设计硬前置；必须明确cutoff及相关后续delta，不能只列最后成功的report，也不能要求无限外域历史|
|历史清单及范围由MAIN外部SHA绑定，不自签|原协议§3 `approval{authority:"MAIN",coverage_scope,approved:true}`；预注册 `/approval`|原硬前置。字节校验、独立测试或本分类文档均不是批准；没有新增审批人|
|实际新根state/pose/snapshot/history排重，最迟witness前检查；换goal/velocity/2pi不能逃逸|原协议§3的三项身份比较及末段；预注册 `/exclude/2`、`/exclude/3`、`/freshness`|原硬前置。hash不同只能说未命中该hash，不能证明物理根不同|
|每次原reset完整ledger **或** 原加载闭包+可证明有限包络|后续authority请求的“最小证据的两条可替代路线”；工具 `agent_freshroot_authority_20261028_v1.py:assertion`|恢复原历史覆盖的两个工程证据路线，不是要求两套材料同时具备；原协议本身没有规定必须使用这种文件格式|
|`source_episode`必须有原文件 `/episode_id` pointer；`panel_scope:*` token|后加panel工具 `normalize`、`required_coverage_scope`；冻结panel文档“输入合同”|该v1工具的严格接收要求；原协议要求source/observed lineage，但**未规定历史资产必须有名为episode_id的字段**。旧材料缺此字段是工具适配/证据表达问题，不可冒充原科学合同新增定理|
|全部机器/worktree/时间/unsaved-failure枚举及永不缺失的访问日志|后续authority response `/historical_scope`可以表达有界机器/worktree/时间；原协议无无限枚举条款|范围声明可追溯、相关探索不遗漏是必要的；不存在额外“证明任意机器永远未见”的义务。无完整访问记录时，强表述“从未被任何人打开”仍unknown，不因此否定已授权known-root工程研究|

冻结工具若因上述后加schema缺字段而blocked，其输出保持原样。本文既不override它，也不宣称新替代表达已获采用；若MAIN选择有限、证据充分的另一表达，需要另行明确合同/实现并保留旧unknown，不能修改旧结果或仅写approved=true。

## 3. 字段级恢复分类（以原件为边界）

分类R=原字节已存在、可直接再取；D=可从原源码候选导出或校验，但不是原现场观测；N=指定原件未保存、不能从这些字节追补；U=是否有独立原资产尚未知。N与U必须分开。

|字段/事实|分类与可做的最小动作|原证据pointer/源码位置|不能据此声称什么|
|---|---|---|---|
|报告path/SHA、两parent继承引用|R，已恢复，不重扫|A–H各原report；旧resolution `/history_file_requests/0`…`/7`；原两prepare解压payload `/exclusions/unknown_sources/0/source`…`/7/source`及`/exclusions/files/<原绝对report路径>`（此处路径作JSON键，JSON pointer须转义`/`为`~1`）|16条引用不是16次独立运行，也未消除reset阻塞|
|selected root_id/root-goal/history/goal hash|R，已按全部52行保留|A–C `/calibration_roots/i/root_id`、`root_sha256`、`causal_history_action_sha256`、`causal_history_rgb_sha256`、`goal_rgb_sha256`；D–H `/rows/i/root_id`、`root_sha256`、`history_rgb_sha256`、`goal_rgb_sha256`|root_id是selected ordinal，不是原episode；whole-stack RGB hash不是逐帧hash；root-goal hash不是state-only hash|
|root/reset候选seed族、root count、oversample等预声明|R，读取原config即可|A–C `/protocol/calibration_root_seed`、`/protocol/calibration_roots`；D–H `/protocol/root_seed`、`root_count`、`root_oversample_factor`；A–C `/protocol/outcome`仅既存预留config|seed设置/预留不证明已执行或未执行；不能写`reset_seed=base+root_id`|
|checkpoint身份、P0祖先path/SHA|R，只核字符串/原文件身份，不打开模型|A–C `/inputs/checkpoint_weights`、`checkpoint_weights_sha256`；D–H `/upstream_p0/path`、`sha256`|输入祖先绑定不是原进程加载闭包，也不是物理同根证明|
|原dispatch命令、call_id、已恢复进程状态|R，复用旧catalog/next disposition|coverage `/recovery/dispatches/0`…`/9`；next解压payload `/disposition/sources/i/dispatch_status/j`；新恢复报告 `/dispatch_load_closure/i/saved_process_disposition`|B前两次已有traceback；A/E/H已有exit0。不再索取这些已恢复状态，但exit0不证明reset次数或来源加载|
|producer/P0/factory候选字节、静态import/reset/collector公式|R（保存字节）+D（静态推导），无需执行|E postrun三文件；H downstream seal `/inputs/0`…`/6`；later archive `/files/0`…`/6`；coverage `/historical_chain/0`…`/3`的固定Git对象。对应新报告 `/byte_verified_source_candidates/0`…`/20`|postrun、后续prerun、Git家族候选均不能倒签为A–H原加载版本；静态import图也不是完整动态closure|
|P0零history命令数组的**源码含义**|D，已用24个零字节核算float32[3,2]的SHA，A–C全部12行action hash匹配；未重跑env|E postrun P0源码99–125行：`history_actions=np.zeros((history_len,2),dtype=float32)`；338–341行写history hash；A–C上述action hash pointer|这能核算source请求数组，不是实际每步latest_action/返回记录，也不是新moving合同的raw10资产；不能混同三token、RGB3和实际public步数|
|state7、goal数值、snapshot、原RGB像素/逐帧hash|N（A–H selected原报告未保存），若有另存原数组则属U且可定向接收|缺失证据是**存在的父对象**A–C `/calibration_roots/i`与D–H `/rows/i`的完整key集合，见§4；P0 row构造338–344行、P1 row构造219行|SHA不可逆；换新环境重建不是旧现场记录。不能从后继endpoint或root_label倒推root，也不能把新重放结果绑定成旧数组|
|每个selected对应哪个原reset index/episode；全部未选、失败、重启尝试及count|N（原selected报告不保存）；是否有原日志/可信有限包络属U|相同父对象无attempt/episode映射；固定factory collector 84–123行在筛选后只返回state/goal副本；末层factory 505–518行使用候选index并含reset|selected数不是尝试数；源码有限for可支持有条件包络，不证明原执行及失败位置；无法记录的现场不能由新运行“修复”|
|原installed env/wrappers及完整run-bound加载证据|U，只有收到明确原manifest/环境快照附件才做path/SHA接收|新报告 `/dispatch_load_closure/i/roles/j/original_loaded_sha256=null`；H原seal七项无env/wrappers条目；旧report行无加载manifest|“当前未取得”不是“当时从未记录”；当前包版本/mtime不能替代原加载收据|
|有界历史范围完整性、cutoff后实际相关delta|U，MAIN/原运行责任方提供明确范围依据与metadata资产；无需扫描无限外域|原协议§3；预注册 `/cutoff`；coverage `/scope`及next原差集绑定只能证明其快照范围|618或1033文件数量不证明全历史；也不能无限递归追索无关机器来拖住所有研究|

表中源码行号针对§5所列**指定快照字节**，不是当前工作树同名文件。D项只作条件性算法解释，不伪装观测。对N项仍不声称已排除其他副本：未来如交付明确合法原附件，可按其原身份补证；没有附件时继续处理同一hash不能增加信息。

## 4. 原记录缺字段的可核对依据

本轮lexical复核了全部52个父对象的key集合，只解码身份值，不解码`jacobian`或`compilation`。A–C的每行键恰为 `{root_id,root_sha256,causal_history_action_sha256,causal_history_rgb_sha256,goal_rgb_sha256,jacobian}`；D–H每行恰为 `{root_id,root_sha256,history_rgb_sha256,goal_rgb_sha256,compilation}`。因此可以确认**这些持久报告没有保存**state/episode/attempt字段，不需要为不存在的字段伪造pointer。所有原pointer以其各自report SHA为前提：

|ID|原report（项目根`/home/gbk/code/LeWorldModel/le-wm-main`下）|SHA256|存在的父pointer范围|
|---|---|---|---|
|A|results/pusht_d2p_p0_packet_contract_v2_20261026_cuda/report.json|2f29a195f291d040a666cc3db6673f9be12c315c96162b126ddaef9bcdb5e135|`/calibration_roots/0`…`/3`|
|B|results/pusht_d2p_p0_preflight_20261024_cuda/report.json|b1b0fbc29ceb050939ae1403c7644d0ca704b133e47a2411c079dbd626ee0eaf|`/calibration_roots/0`…`/3`|
|C|results/pusht_d2p_p0_preflight_packet_layout_repair_20261024_cuda/report.json|c499ececab67135a2f4d4d698359e6f62b557be66f9132480ef95b14a0dce080|`/calibration_roots/0`…`/3`|
|D|results/pusht_d2p_p1_outcome_admission_20261024_cuda/report.json|a9266fb3cd3e68ee197ec187e8b7916cf3ca8c11966778df3d4c92083b7d6b54|`/rows/0`…`/23`|
|E|results/pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda/report.json|a0c9c53bd0161946ec4466878de49795abdf6db102dbb97b174035ed4028a006|`/rows/0`…`/3`|
|F|results/pusht_d2p_p1_reconfiguration_replay_bound_20261024_cuda/report.json|4911ae7e945abfb042b332f71d7cda0163e14c9727ee8131ff7881bc67257a14|`/rows/0`…`/3`|
|G|results/pusht_d2p_p1_rollout_alignment_repair_20261024_cuda/report.json|1c5fdd414a5104b6487c2c6e5cb38d05943f902a9d7a695b891ae73202ef7abb|`/rows/0`…`/3`|
|H|results/pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda/report.json|99cbb6b6dc4653eb629427353f6f6d05022c219333f3f7362cf4ca43a3076bc2|`/rows/0`…`/3`|

缺失字段与`jacobian`/`compilation`内部科学数值无关，不需要读取后者来分类。若未来提交其内部或其他原文件中的真实source身份证据，应按精确pointer重新检查；本轮未把未知子树宣称为已搜尽。

## 5. 证据字节与最小下一步

合同：外层 `docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md` SHA `42f1bcbf517dfd46b8ba05525663d82efac32dbb5fb24f87a8ca572bef507a2f`；`docs/chat_freshroots_20261027_preregistration.json` SHA `79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e`。后续路线请求 `docs/D2P_FRESHROOT_AUTHORITY_REQUEST_20261028_JASON.md` SHA `46e9df59cb1f1d9bb344132a9a39a996c85cc26c5b79ef45c7b5d4dc4b3af39b`，不将其格式反称原科学协议。

恢复事实：`results/agent_p4_historical_source_identity_20261029_v1_LOCAL01.json` SHA `577580e536598d850cd449fda2a02c314197267bdc41bff9d85692432165e724`；coverage SHA `00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb`；resolution SHA `032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00`；next压缩envelope SHA `5ef07ab7815fff0c392feef59309c0116596e95d541eb95413d1d966a8b2a29b`，其pointer均指解压semantic payload。

源码定位：`results/pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda/postrun_implementation/preflight_pusht_d2p_p0.py` SHA `5215f0b49b772d8def14997996774bfd49db614b52b9f615c164271591f7b83b`；同目录P1 `preflight_pusht_d2p_p1_reconfiguration.py` SHA `80b9674063360a2f51d2e146c830c29a5399bb53b9086b177f3b3f2dc3b0d5e4`。factory固定Git object为 `d26e4bff84f31138adf5d1bda809bccafd9be10a:le-wm-main/experiments/` 下 `audit_pusht_reachability_horizon.py` SHA `73871055f10b09aea15f03acf87b04e79e30d546668d44bb8160c7be3c439a67`、`run_causal_visual_pusht_world_model.py` SHA `261f609e91da9110f8ba5254033529527314d6bcf38e058c3a237c920d3c4190`。

H downstream seal `results/pusht_d2p_p1_root_anchored_v3_physical_calibration_20261026_cuda/prerun_seal.json` SHA `4e024c0ded185e0bb58c6ceaeb33a0ab1d784165a962995c28f6216d439ef81e`；later archive `results/pusht_d2p_sealed_program_execution4_v1_20261026_cuda/prerun_source_seal/manifest.json` SHA `0f0f5b8b8a525990dcffd2393ada1a49270dd50a3d4865800343cfd9b8a39aef`。二者只能证明各自后续封存关系。

最小实质下一步不是再造manifest，而是MAIN/原来源方对**固定十次dispatch**定向说明：是否有真实原attempt ledger；若无，是否有原加载依据可使已有候选闭包的有限包络成立；是否另存足够的root/state/observed lineage资产。提供明确文件/pointer/SHA或明确遗失/未记录，不能用新重放替代。只请求这份有界清单及相关delta，不默认索取任意机器全历史。

若这些证据确实未留存，原严格fresh确认前置可能不能闭合，应报告其不可补证项。MAIN仍可另行预声明与历史局限相称的工程/已知根校准研究，但不能把它标成已满足原fresh合同；本文不替MAIN选择新合同、不解除P1或签发科学准入。未记录的格式字段不应成为无限新增wrapper的理由，真实身份无法排除也不应被一个scope声明掩盖。

本次实际检查：8原report SHA、52个完整父key集合、16原ref pointer、12个源码零数组hash匹配均通过；旧恢复三件套和saved报告SHA未变。既有恢复模块36 RAM tests及完整verify复用，不新增测试实现，不把测试当fresh证据。原合同/指针复核不读取科学outcome。

MAIN可在inner项目根直接复用以下只读回归；不会生成新草案或重跑producer：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_historical_source_identity_20261029_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p4_historical_source_identity_20261029_v1 verify --report results/agent_p4_historical_source_identity_20261029_v1_LOCAL01.json --report-sha256 577580e536598d850cd449fda2a02c314197267bdc41bff9d85692432165e724
```
