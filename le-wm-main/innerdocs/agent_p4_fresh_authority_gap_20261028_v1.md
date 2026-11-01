# P4 fresh authority：必要来源与可修复性证据表

日期2026-10-28；本轮仅新增本文，不另造source/validator。现有resolver、manifest及
agent_freshroot_authority已有完整metadata/request接口，复制实现不能恢复历史事实。
本轮只读指定历史metadata/catalog和实现，不打开四预留根的内容、动作、outcome，
不读取catalog指向的原报告/NPZ/checkpoint/巨大日志，不运行采集、模型、物理或网络。
v1/v2 producer/consumer、MAIN文件和共享总账均不改。以下是限定快照缺项，不是全历史新认证。

## 结论与最小阻断集合

**当前没有可供fresh_candidate使用的完整历史证书。** 缺项分为：

1. A–H原运行的reset尝试覆盖及来源真实性：8份来源被两prepare各引用一次，共16条阻断；
   已发现10次调用，其中B三次，不能只交最终一次。原run provenance不能从当前文件SHA倒推。
2. 所有应排除历史根的物理身份与slot/episode对应：现有metadata关联不是物理等价证明。
3. 完整历史scope/cutoff及之后delta的处置，包含失败、未选、预留及未保存/重启；MAIN批准
   必须绑定精确content SHA与cutoff，不能由family齐全或本报告生成代替。

以上全部满足之后，仍只得到历史source前置，不是四新根fresh认证。未来实际根的
pre-witness历史/批内排重是另一个必要步骤；本任务绝不打开保留槽来完成它。
不要求凭空存在的全局访问日志来阻止工程：缺项可以由既存原始ledger或可追溯加载闭包/
穷尽包络、明确范围依据分别满足；拿不到的历史事实保持unknown，不事后造日志。

## 已封存metadata能修什么、不能修什么

|必要来源/接口|本轮读到的既存证据|可离线推进与精确边界|
|---|---|---|
|618来源inventory与resolution capture|618 captured、344 metadata、199 direct identity；原960 refs完整保留|可复用path/SHA/pointer与来源分母，不需再读约285MB原文件。614 refs仍缺直接身份；与16 reset refs不是互斥计数，不能相加当新总数|
|原reachable/moving实际身份桥|capture `physical_identity_bridges`、`bridge_payload`；mapped_original_records=6，physical_mapping_errors=[]|可将已封存bridge字段显式接入新request，绑定该capture SHA并交MAIN准入；4根映射6记录不是6个独立根，也不自动消除继承unknown|
|conditional源slots/全部尝试|inventory `supplemental_sourcepack_metadata`中conditional项；8次attempt，reset 202781010..017及两个half-open spans|可修旧adapter只读rows造成的`no_identity_records_in_container`，保全部8尝试而非selected2；不能据state/root-goal代填缺snapshot/RGB3/raw10|
|observed_history自身根身份|同inventory的observed项、`identity_join_candidates[0:2]`；各自actual reset 202781011/014|可修`unsupported projected schema`路由；自身state7须来自旧checkpoint，full snapshot/rawhistory及MAIN maps_to尚缺。不能把conditional父state复制成observed root。本轮不打开checkpoint|
|旧预留池|inventory中procedural calibration request，8旧slot reset spans覆盖[2028102700,2028102764)|可从原metadata保全部预留排除，不以缺sourcepack推“未执行”；这些不是本轮四个保留确认槽|
|metadata有向身份关联|closure含370双绑定边/278 associations/28 alias unknown|可沿path+SHA+pointer提供父来源定位；仍199 direct、614缺直接refs。其中120关联、494未关联；关联不是实际根映射，28 alias不能直接合并|
|original_four family|resolution该family inventory_paths=[]；inventory另有17项hint=`original_four_candidate_scope_unverified`|可提交17项准确source_id供MAIN分类，不能直接重命名为original_four。例authority_ab4aa05cc7ec9c70、authority_3bbcc22df2c61c6e均为p3 prepare候选，名称本身不证明四根身份|
|overlay/feedback authority|resolution分别列5份overlay prepare和1份feedback prepare，status=inventory_received_requires_MAIN_scope_approval|可从已有明确path/SHA继续metadata schema接线；prepare存在不证明实际observed根或失败/重启全覆盖，不能把本轮已知prepare数当全部历史|

conditional source_id=`authority_c7da6545e91eee53`，observed source_id=`authority_780368c13c8ca5a8`。
两条候选连接的原state pointer分别为`/slots/0/proposals/0/attempts/1/state`、
`/slots/1/proposals/0/attempts/0/state`；目标observed自身state pointer仍null。
这两个schema失败在inventory8 manifest中逐项存在，不是找不到源文件。

### 不应重复实现或宣称已修复的事项

resolver已把480个旧path实际捕获为618来源快照；closure也已生成显式关联，不必重扫。
旧manifest v1没有“approved disposition override”接口：即使新的metadata修好schema，
也不能在旧结果中删掉960继承unknown。要将历史证据裁定接入可用certificate，需MAIN明确
接受原证据并另封存新合同/新实现；不能改sealed manifest，或只写approved=true绕过unknown。
618数量超过原manifest512文件cap；新resolver512MiB capture合同已处理全部来源，
但不等于原manifest能直接吞入618。工程新版本可读紧凑已验证投影，必须保全来源/unknown分母，
不得拆batch后把局部通过拼成全局通过。本轮不再新增此类框架。

## A–H：必须向原运行方补什么

下表path相对于 `/home/gbk/code/LeWorldModel/le-wm-main/results/`；全部report原SHA
及来源方request ID已封存在coverage `.sources`和现有authority `request`输出。
每行对应`<ID>/reachable_prepare`、`<ID>/moving_prepare`，同一真实补证可供两refs使用。

|ID|原report路径|已发现call_id（不是穷尽重启声明）|
|---|---|---|
|A|pusht_d2p_p0_packet_contract_v2_20261026_cuda/report.json|call_981f76dba2b241d79588468b36e979d7|
|B|pusht_d2p_p0_preflight_20261024_cuda/report.json|call_a41c0e8d73624b16959e29b9f05fd4ed；call_3d96838206a84aba9b97fe0ce82bf8e4；call_4b779dfb4ad045b696db7409d861c35f|
|C|pusht_d2p_p0_preflight_packet_layout_repair_20261024_cuda/report.json|call_b0f211dffc4e451fab05b0827fdfa415|
|D|pusht_d2p_p1_outcome_admission_20261024_cuda/report.json|call_871afe66f51744c3a8964fe49fab6b08|
|E|pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda/report.json|call_7c4638e67c234a65ad8b02be380375cb|
|F|pusht_d2p_p1_reconfiguration_replay_bound_20261024_cuda/report.json|call_e0f5ab66af7848a89e59e21a975c4165|
|G|pusht_d2p_p1_rollout_alignment_repair_20261024_cuda/report.json|call_a0928b48326a4f43b88a25a223834196|
|H|pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda/report.json|call_947bf4c5b1ce4004a5ab5c4e2ef3d0e5|

两条合法补证路线沿用现有validator，不能用新执行代替旧历史：

- L：每个call_id穷尽reset ledger，覆盖random/explicit_state/goal_render/replay及
  selected/unselected/rejected/failed；失败未观测identity允许null但不造state。还需原来源依据
  及restart/attempt范围。0次也必须有依据，不能由report没有记录推0。
- C：原run实际加载的producer、factory_1..4、environment、wrappers及有限穷尽包络；
  D–H加p0_dependency；B的bash入口加shell_launcher。字节可用历史blob/patch恢复候选，
  但必须有原加载provenance；当前源码SHA/后补seal不够。无可证明上界的循环需回到L。

精确缺的外部权限是：取得原运行方已有ledger/load manifest/环境快照或可追溯metadata导出，
以及MAIN对机器/工作树/时间区间/未保存失败/重启范围与证据真实性的裁定。
不新增外部审批层；MAIN coverage approval本来就是原协议必需项。原记录若遗失，
应逐call记不可恢复/unknown，不从未发现推出不存在。B前两次失败退出状态已恢复，
不必重复索取其是否退出，但完整reset尝试仍未认证。

## 快照差异：不可混用分母

inventory618 capture：960 refs、614缺直接refs、16 reset refs；closure保留上述数值。
inventory8 manifest实际975 unresolved = 960 inherited + 6 observed identity missing +
2 schema/projection failure + 3 family missing + 4明确declared scope问题。
这是一个8文件旧request，不是完整618覆盖。其approval=false、source_run_preconditions_met=false，
四槽seed checks无冲突但independence_certified=false。**975不是975独立run。**
coverage固定request实际8来源/16refs/10dispatch，original_execution_bounds_certified=0。
后续1033路径delta在原MAIN authority请求文档中已有记录，本轮未重读/重算其原文件；
不能把Oct27快照说成Oct28 P2/P4/P5全部后续接触史已覆盖。MAIN应显式说明cutoff后的
delta纳入或协议适用边界，不应让本worker为此递归扫results。

## 本轮真实输入绑定与复现

以下均在inner项目根；只读这些指定metadata本体，不follow其引用：

|文件|本轮重核SHA256|
|---|---|
|results/agent_freshroots_resume_20261027_coverage.json|00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb|
|results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json|e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591|
|results/moving_authority_identity_link_closure_20261027_v1_MAIN01.json|7c7e1792c5a17986aaf3a38539a280ec46d5910eb4d9351babbe9bf2a70b3576|
|results/moving_root_exclusion_manifest_20261027_v1_inventory8_main01.json|a17bc732c9a021a4ca81a7e4da573c752ad1f66eafe08b1a3dad2b64464d7060|
|innerdocs/moving_historical_authority_inventory_20261027_v1.json|c425052a9a43e362eff77befc21e37c6a72fdc3fa455a6f5ac7c53b1c0487341|
|experiments/spectral_reachability/root_exclusion_authority_resolution_20261027_v1.py|41474b63621545c04dc9ece6e81fc8bf9dd1c712e5ced5fee992c85d25450c04|
|experiments/spectral_reachability/moving_root_exclusion_manifest_20261027_v1.py|e6521ae535bdf26b6544868bf3c9018761332ee3b339936475ea54707c0db79c|
|experiments/spectral_reachability/agent_freshroot_authority_20261028_v1.py|f7a30fcad18483e1848170acdd1fe4bd96e7d08b38804b18f1cf2cf1087870e7|

实际执行现有metadata request返回8/16/10、coverage_complete=false、认证0；
现有authority RAM回归38 passed in0.13s。未新增软件，不需要重复写同义tests。
工程测试不是fresh认证；没有生成response/approval/certificate或新根结果。

MAIN下一条安全入口（stdout，只读固定catalog；不触碰预留根）：

```bash
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_freshroot_authority_20261028_v1 request
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_freshroot_authority_20261028_v1.py
```

仅收到原运行方真实response及原证据后，才可用同模块`validate --response /ABS/REAL.json
--response-sha256 REAL_SHA`做字节/schema接收；该工具即使exit0也固定origin_review_required，
reset_blockers_cleared=0，不签证。最小实质下一步是向原运行方取得B三次及A/C–H的L或C路线材料，
而不是打开四保留槽试运行或继续重建metadata绿灯层。
