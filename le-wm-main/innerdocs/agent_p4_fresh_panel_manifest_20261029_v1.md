# P4独立确认panel根清单：准备与metadata认证接口

本三件套只补缺少的panel handoff，不采集根，不跑模型/物理/网络，不审计P1代码。不改旧工具、封存结果或共享文档。固定原四槽，不新挑seed、几何、结果或难度。

## 复用与真正缺项

已核查并复用：

- root_exclusion_authority_resolution_20261027_v1：只解码白名单身份，跳过outcome；原source/seed/input-lineage并非全历史证明。
- moving_root_exclusion_manifest_20261027_v1.build：原历史inventory、全部family、seed/span、identity缺失及外部MAIN exact-content批准。没有另造一套history批准。
- chat_freshroots_20261027_consumer_v2.assess：原四槽、物理pose/state/snapshot/RGB3历史及批内all-pairs排重。沿用XY1e-9、wrapped angle1e-12，不改阈值。
- 已有P4 fresh source v2及saved consume v2：原生采集和前witness检查已有实现，不在本任务重写/执行；当前source consumer本来不签freshness。

旧assess不比较source episode，旧618快照不覆盖后续接触史。本模块补source episode、当前明确exposure inventory、身份的原文件SHA/JSON pointer证明、固定panel完整消费和MAIN exact-content metadata认证入口。新脚本/manifest不是新根证据。

原协议预声明四槽root_seed203027101..104、reset_seed203027201..204、sign +−+−，每槽一次proposal/attempt，无替补。注册文件SHA79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e固定；新seed不等新根。

## 可立即执行的真实草案

目录 /home/gbk/code/LeWorldModel/le-wm-main：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_fresh_panel_manifest_20261029_v1.py

.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_panel_manifest_20261029_v1 draft \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_fresh_panel_manifest_20261029_v1_request_MAIN01.json

# PANEL_REQUEST_SHA取上一命令真实返回；不是未来默认hash。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_panel_manifest_20261029_v1 build \
  --request /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_fresh_panel_manifest_20261029_v1_request_MAIN01.json \
  --request-sha "$PANEL_REQUEST_SHA" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_fresh_panel_manifest_20261029_v1_manifest_MAIN01.json

# PANEL_MANIFEST_SHA取实际build返回；verify重建全部内容与绑定，不仅比status。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_panel_manifest_20261029_v1 verify \
  --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_fresh_panel_manifest_20261029_v1_manifest_MAIN01.json \
  --manifest-sha "$PANEL_MANIFEST_SHA"
```

draft实际引用已有root_manifest SHA3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838和consumer_manifest_v2 SHA2b80412eb2efc7b721d6edb754ce5d9d5d513e17744706aa1e5bab56c785f7e7。其余history_request、exposure_inventory、actual_roots、panel_approval均null。build可立即输出blocked planning artifact，四槽actual_root_identity=null、certified_fresh_roots=[]。不打开保留根/旧outcome来填它们。

当前已知历史缺口仍包括原8来源/16 reset refs、失败/未选/重启尝试、实际加载边界、未映射身份、cutoff后delta。工具保留这些unknown，不把生成草案当清除它们。缺项可能由原已有ledger/加载闭包及MAIN范围裁定解决；不凭空要求全局不可取得的日志。

## 输入合同

所有descriptor严格为 {path,sha256}；绝对canonical项目路径、无symlink/通配/..，必须实际匹配字节SHA。不得使用futureSHA、reported值或脚本名充作身份。请求只含源码template()的八个字段，不接收评分、成功、物理难度、排名等输入。

history_request：原moving_root_exclusion_manifest_20261027_v1.request文件descriptor。调用原build完整重建；原MAIN批准必须覆盖所有历史family、全部缺项和当前exposure文件SHA，不能把旧512文件cap分批局部通过拼成全局通过。若实际完整scope无法用原adapter表达，保留blocked，须由MAIN另定原历史兼容方案，不能本脚本override unresolved。没有history_request也可规划，不能认证。

原coverage_scope还须显式含required_coverage_scope(request)返回的两条旧manifest/consumer SHA token，以及failed/unselected/reserved/restarted/all_seen_outcomes/all_source_episodes六条panel_scope token；否则旧unresolved完整保留。只有family标签齐全的另一个小inventory，不能冒充已处置既存历史缺口。沿用原MAIN批准接口，不自签这些scope；MAIN对原材料真实性及可恢复范围负责。

exposure_inventory：复用原identity_metadata schema，严格字段：
`schema,cutoff_utc,project_root,includes,complete,unknowns,records,reserved_reset_spans`。
schema为moving_root_exclusion_manifest_20261027_v1.identity_metadata；
includes严格是 `[failed,unselected,reserved,restarted,all_seen_outcomes,all_source_episodes]`。
complete只能是外部责任方提交的范围声明，不由工具从扫描/文件数推断。unknowns保留未恢复来源；complete=false或unknowns非空阻断认证。cutoff必须UTC，且与重建的原history authority完全一致；之后的新活动须更新delta并重审，不宣称未来历史已覆盖。

每个records条目严格含：
`record_id,root_seed,reset_seed,root_goal_sha256,root_state,root_snapshot_sha256,history_pixels_sha256,raw_history_sha256,source_episode,evidence`。
state为真实state7，RGB3/SHA保留原口径。任何身份缺失填null并保留分母/阻断，不填0，不用父state代替自身state。evidence的key恰好对应所有非null的上述七种身份字段；每值为 `{file:{path,sha256},pointer:'/rows/0/root_state'}`。只用原lexical projector读取该身份；不能指向success/cost/outcome。数组索引用十进制JSON pointer，支持root_state/state、root_goal_sha256/root_sha256、history_pixels_sha256/history_frame_pixel_sha256显式别名；不猜其它schema。

source_episode严格为 `{source:{path,sha256},episode_id,pointer}`；pointer末字段必须episode_id，指向原source真实episode ID。不是当前analysis文件名、task编号或后来重放新episode伪装的起源。原材料没有此身份则null/阻断，MAIN需从实际source lineage提供，不能随手造ep0。相同source path+episode_id，或相同source bytes SHA+episode_id均排除；后者覆盖原文件不变的拷贝。跨重写/异路径异hash的原source别名仍需MAIN inventory给出规范原source；哈希差异不能证明独立。

reserved_reset_spans条目严格为 `{span:[lo,hi],evidence:{file:{path,sha256},pointer}}`，half-open语义沿用原协议；pointer末字段reset_seed_span或span。跨root/reset两个namespace检查所有已见及保留seed/span，不把seed0共享placement细节当所有物理根相等。

actual_roots：descriptor指向 `{schema:'agent_p4_fresh_panel_manifest_20261029_v1.actual_roots',slots:[record|null,record|null,record|null,record|null]}`。完整四槽顺序原样，record与上面同schema，每槽root/reset seed必须等于该原预声明slot。未来由获授权原生采集产生的实际source-bound identity才能填；本模块不生成它。记录的是pre-witness source identity，不读取planner/native suffix outcome决定保留哪槽。

排重包括：已知root-goal/state/pose/snapshot/RGB3、所有seed/span、原source episode、批内同类冲突。批内重复双方均保留且排除，绝不first-wins替补。仅raw_history相同不视为物理同根。保留hash-only旧记录，缺state不意味着未使用。

## metadata认证，不是freshness或科学批准

没有panel_approval时即便所有metadata无冲突，也只输出待外部批准。MAIN可对实际build的panel_content_sha256作独立裁定，再提供全新文件descriptor：

`{schema:'agent_p4_fresh_panel_manifest_20261029_v1.MAIN_panel_approval',authority:'MAIN',approved:true,panel_content_sha256:实际内容hash,cutoff_utc:实际cutoff,outcome_blind_selection:true}`。

此字段不在content hash内，避免循环；其余request、全部评估、旧authority和source/tests/doc字节均在内。完整原history authority通过、exposure被其批准精确绑定、全部四槽有source-bound身份且无排重/缺项时，metadata_panel_certified才为true。外部批准不能消除缺项或重复。不提供伪造批准生成命令。hash校验不是签名身份认证；MAIN负责真实性/范围/披露责任，RAM正例不是实际批准。

**所有输出永远 freshness_certified=false、source_run_authorized=false、scientific_admission/P1/P4=false、new_root_evidence=false、certified_fresh_roots=[]。**
metadata_panel_certified仅证明给定外部authority/文件范围下的清单合同闭合，不证明实际新的独立根、未打开blind、真实native重放、P1或P4收益。未来仍复用原native producer的before-witness排重、source strict与独立披露/准入，不能以本清单跳过。没有新增模型/物理run入口。

## 预算、失败与测试边界

只读显式文件，不glob、不递归仓库/会话扫描、不自动follow bindings。metadata每文件8MiB、身份proof原文件128MiB、unique累计512MiB；调用旧history工具仍保留其64/256MiB限制。输出8MiB、disk floor200MiB，xb+fsync排他写，不创建目录、不重试/覆盖/删除partial。sha冲突/不明字段/尾随JSON/非有限值/重复键/cap越界拒绝；正常缺认证返回blocked草案。verify完整重建，输入最终字节再核。

RAM测试覆盖原阈值/2pi与velocity逃逸、episode重复双方、历史episode、seed跨namespace与span、全四null/partial、原MAIN历史批准、当前inventory绑定、精确panel批准及篡改、cold import无torch/env、实际已有三份metadata草案读取。未运行新NN、physics、源采集或科学确认；测试不作科学证据。独占三件套冻结后修复只能新版本。

