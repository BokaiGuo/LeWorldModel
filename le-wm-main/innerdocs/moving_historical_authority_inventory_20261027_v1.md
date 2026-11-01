# Historical authority inventory v1：实际文件资产，不是完整排除批准

配套实际数据：本目录 `moving_historical_authority_inventory_20261027_v1.json`。
只新增此doc、大catalog JSON及小explicit request JSON；未改 Beauvoir parser、旧 source/prepare/output 或 live driver，未运行神经/物理。
MAIN 负责审读、桥声明准入与最终排重；本交付不设审计器，不自动审批新根。

## 优先交接：4,150B explicit request，MAIN可直接执行

本目录 `moving_historical_authority_inventory_20261027_v1.explicit_request.json` 是严格的
Beauvoir v1 request，不是改名catalog。SHA：
`480fa660c74a174f1e343eadc8208b686f8b391104eca1f0a32dbd94acf26c37`。
固定8项：两个reachable/moving prepare、四份下述sourcepack、两个原moving task gzip。
task00/01分别绑定SHA `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`、
`101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`，本次原字节核对一致。
approval=null；缺family、继承unknown、未批准桥均明确声明，不靠缩表宣布全历史完成。

MAIN从Python根只做saved resolver，不启动实验；stdout仅显示小摘要，不dump大表：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_root_exclusion_manifest_20261027_v1 --request /home/gbk/code/LeWorldModel/le-wm-main/innerdocs/moving_historical_authority_inventory_20261027_v1.explicit_request.json --request-sha256 480fa660c74a174f1e343eadc8208b686f8b391104eca1f0a32dbd94acf26c37 | jq '{schema,status,source_run_preconditions_met,freshness_certified,unknown_count:(.unresolved_sources|length)}'
```

本侧只验证request字段及8项文件SHA，未生成新的exclusion manifest。v1没有旧unknown的显式
证据处置/override接口，故此输入不能凭空消除原970；可用新增源的真实身份定位具体adapter/桥缺口。
小request仅解析8文件，大catalog只供MAIN按需查询，避免继续无限扩表。

## 本次确切交付

cutoff：`2026-10-27T11:11:41.650764+00:00`。静态输入逐文件原字节 SHA，读取后复核；
这是扫描期间记录，不是并发工作树的原子快照。

|文件分母|本次实际数|意义|
|---|---:|---|
|旧reachable/moving prepare继承unknown引用|960|每个prepare480，原引用指针与reason均保留|
|上述引用去重后的文件路径|480|全部存在且取得实际SHA；定位文件不等于消除旧reason|
|额外sourcepack/prepare/request入口|138|134 prepare/request +4 sourcepack；不按执行结果取舍|
|完整catalog|618|每项有绝对path、SHA、bytes、role、metadata pointer与unresolved|
|读取旧API投影的原文件字节|9,293,092|另有补充jq白名单投影，不解码模型成本/目标/证书|
|一般投影有身份字段的authority|28|无字段/不支持schema的项仍在完整分母内|

960条旧unknown不是960个独立来源。目录扫描结果也不是618个根；同一物理根可能被多次引用。
原20个root-seed值、2个显式reset-seed值及50个root-goal hashes保留于summary；
补充8次calibration尝试与observed实际reset另列，不能把前述基线summary误称所有已访问根。

发现规则固定为：两个原prepare `/exclusions/unknown_sources/*/source` 的全部480路径，
并上 results 全目录的 `sourcepack.json/prepare.json/plan.json/source_prepared.json` 及
results顶层 basename含prepare或request的JSON。无success/cost/selected过滤。
原文件哈希覆盖全部字节；metadata是限定字段投影，并非把原JSON完整解码作为规划输入。
未读NPZ/证书/witness/未来goal数组；status只用于描述来源，没有用于文件/候选取舍。

## 可立即给 Beauvoir 的真实 sourcepack authority

全部路径前缀为 `/home/gbk/code/LeWorldModel/le-wm-main/results/`；JSON内均是完整绝对路径。

|实际文件|SHA256|本次保留的身份|
|---|---|---|
|`p1_conditional_calibration2_main_20261027_v1/sourcepack.json`|`728f14cb83c4fcd0ea4f5348bad2bff5b5dd372ab0185b6e6b3a26dbfda58057`|两slot及全部8次proposal/reset尝试，包含未选者的state7/root hash/seed|
|`p1_observed_history_source_20261027_v1_main01/sourcepack.json`|`0b89d77494acc6cf03bb0110e87231c1c240a253e844326acc668b7e4f93d546`|两行actual reset、parent/root hash、6个checkpoint文件/SHA指针|
|`p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json`|`9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9`|两slot/reset/root hash/blind payload身份；source自报unknown calibration|
|`p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json`|`c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`|两slot/reset/root hash、RGB3 pixel/raw history hash；原recipe复用、不是fresh|

这些sourcepack的原status文本也保留于 `supplemental_sourcepack_metadata`；没有根据status删行。
上述四份本身没有本交付声称的失败row；历史失败/中断/未被选文件仍在480项及预留request分母内，
它们执行状态未从未来结果反推。不能把“状态未投影”标为已通过或独立；统一保not_certified。

### 被旧metadata投影漏掉的实际reset/未选根

conditional两slot的parent root seed为 `202761010,202761011`；完整half-open reset spans
为 `[202781010,202781014)` 与 `[202781014,202781018)`。
JSON补充区保存**全部8次**原记录的state7/root hash，不止选中2次。
observed两行actual random reset为 `202781011,202781014`，placement reset均0；
父source/root seed/spans及checkpoint身份保留。共享placement seed0不是新增随机根。

`identity_join_candidates`提供两个可复核的**候选lineage连接**，不是已批准identity_metadata：

|conditional源记录|observed记录|共同metadata hash前缀|源state7精确pointer|
|---|---|---|---|
|`/slots/0`|`/rows/0`|`02a139165f6e…`|`/slots/0/proposals/0/attempts/1/state`|
|`/slots/1`|`/rows/1`|`05844ca2c3ab…`|`/slots/1/proposals/0/attempts/0/state`|

JSON内每个连接含两端完整source_id/path/SHA/pointer与原64位hash。不得把源state直接复制给
observed current root来伪造桥：目标自身state7尚须来自原checkpoint，full snapshot/rawhistory身份
及MAIN批准maps_to也缺失。checkpoint仅文件名/declared SHA，未打开或假称独立核验数组。
给Beauvoir实现的接口补缺是明确的：旧adapter只看sourcepack.rows而conditional源是slots，
observed又用metadata.parent_source/random_reset_seed/checkpoints，不应静默当无历史身份。

### 实际prepare / 预留权威

同样使用上述绝对results前缀；每项及全部其他prepare已在JSON，不需要根据文档重猜路径。

|文件|SHA256|确切scope|
|---|---|---|
|`p1_constructive_reachable_source_20261027_v1_prepare01.json`|`98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086`|root202798101/102，reset202798201/202798202，旧unknown480完整继承|
|`p1_constructive_moving_history_source_20261027_v1_prepare01.json`|`c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9`|同原recipe seed；不同root hash不自动代表独立；旧unknown480仍保留|
|`p1_conditional_calibration2_main_20261027_v1_request.json`|`a53a6671e67eb6a79d0ccb6614bf6d5dfb4b6417ec3eb3b548ab63a41fb67d68`|两个slot和完整reset spans，不能仅保被选reset|
|`p1_conditional_procedural_source_20261027_v1_calibration_request.json`|`2f997c4ef8c1aab724d67531127e6fd41883769b695be57e3dd78b3f0d6aa8f6`|8个预留slot，执行与否未知仍排除预留池|
|`p1_observed_history_source_20261027_v1_prepare01.json`|`ecc5f4a488290b6d0eade4d3ff2034d256729de94c4d3a14b12d6eebad783385`|旧投影不支持selections；保文件，不虚填新slot|
|`p1_moving_target_overlay_source_20261027_v1_prepare_main01.json`|`88957b9d251595e5e35ed46ba4a9aa003eb937bbfff0f58ae12540418dcb39a1`|实际envelope存在；Beauvoir当前prepare-envelope BASES不接受此schema，明确unknown|

8个旧预留root seeds为2028102700/3700/4700/5700/6700/7700/8700/9700；
连续8个reset spans覆盖 `[2028102700,2028102764)`，逐slot端点保在JSON原pointer。
本交付不从缺sourcepack推断这些预留未执行，也不按任务性能释放它们。

## JSON合同与MAIN使用方式

schema为 `moving_historical_authority_inventory_20261027_v1.inventory`，**不是** Beauvoir request、
identity_metadata或MAIN approval，不允许把它直接作为批准覆盖的排除manifest。

主要键：

- `evidence_files`：618个唯一path/source_id，原SHA/bytes/role/schema/format/adapter，字段级metadata，
  inherited_by原prepareSHA与原unknown pointer，具体unresolved；adapter=null的legacy报告只是原权威引用。
- `supplemental_sourcepack_metadata`：四份文件的外部SHA与白名单jq投影，全部尝试/行保留。
- `identity_join_candidates`：上述两个明确pointer连接及缺项，不作state/snapshot自动归并。
- `input_budget_diagnosis`、`integration`、`unresolved_scope`：实际解析预算与准入边界。
- `family_hint`只是文件名路由提示，不能据此证明legacy/original_four/overlay/feedback范围全覆盖。

MAIN可先只读取这些**已绑定数据**，不启动实验：

```bash
jq '.summary | {files,inherited_occurrences,inherited_unique_paths,roles,projection_errors}' innerdocs/moving_historical_authority_inventory_20261027_v1.json
jq '.evidence_files[] | {source_id,path,sha256,role,schema,unresolved}' innerdocs/moving_historical_authority_inventory_20261027_v1.json
jq '.identity_join_candidates' innerdocs/moving_historical_authority_inventory_20261027_v1.json
jq '.supplemental_sourcepack_metadata' innerdocs/moving_historical_authority_inventory_20261027_v1.json
```

不要自动把全部evidence_files传给v1：目前**618>512**，原文件合计**282,406,682B>268,435,456B**。
这同时超过文件数和原字节总预算；不能裁去106个文件或把两个batch的“各自正常”冒充全局完整。
需MAIN明确不适用来源的有证据scope处置，或另行批准有全局汇总语义的新bounded合同；
本任务只交付资产，不改解析实现、不擅自提高cap。

## 确切剩余缺口

1. 两个prepare的480相同unknown路径全部已找到并hash，但reset attempt bound、旧metadata
   解析失败等原reason未获证据处置；**原970不能因本catalog就减成0**。
2. 当前metadata接口不识别的observed/selections、conditional.slots、overlay envelope已有具体文件、
   SHA和字段pointer，但仍需MAIN/Beauvoir在新合同中接纳并证明原账对应。
3. sourcepack/root-goal hash不能独自给出每个observed根的state/snapshot/RGB3/raw10完整身份；
   继承seed相同也不能证明同物理根或独立。未读取新live结果来补未知。
4. 此catalog按明确文件发现规则覆盖，不声称全历史任意命名目录/外部symlink/后来新增文件已覆盖。
   截止后新prepare、新observed roots及其他实验family须MAIN另增显式authority。
5. 无MAIN coverage approval，无已准入metadata bridge，无新根实际freshness或P1通过声明。

文档组织使用 documentation-and-adrs 技能；仅记录实际资产/接口边界，没有按通用模板扩展实验。
