# Historical reset quarantine v1：纯saved清单与seed-span检查

本次独占source/tests/本文及一个NEW小型metadata结果。只使用stdlib，不import numpy/torch/gym、旧resolver、producer或certificate，不执行源码、神经/物理/训练、不下载、不改任何sealed来源或共享docs。使用experiment-code技能固定输入、预算、RAM测试和错误保留；不采用其训练/调参/画图要求。个人goal只覆盖这个实现交付，不代表历史coverage或P1完成。

## Authority和证据强度

固定入口 `results/root_exclusion_authority_resolution_20261027_v1_MAIN01.json` SHA256 `032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00`，过滤exact original_reason=`historical_reset_attempt_bound_not_attested`。严格8个已知path/原SHA、每项两条原引用（reachable_prepare/moving_prepare）、总16refs，不按outcome筛选或按目录扩展来源。

绑定自己的 `innerdocs/moving_historical_reset_bound_disposition_20261027_v1.md` SHA256 `462508d664b2064544e2c246a885f1175026215de4bce5f77e4a0e3be256a5d3`；读取原报告的protocol与selected-root SHA，不读取收益字段作选择。实际构造输出source自身SHA。

另外只读 `git show` 固定commit/blob（不checkout、不执行其中Python）、E的postrun源码、H后续physical prerun seal及其7份snapshot、后续program execution archive及其7份source、D后续physical报告。全部按原disposition中的固定SHA核验。后续seal/快照通过也不倒灌成原8run源码认证。Git命令固定白名单、timeout10s，不可用时记录evidence_unavailable，不改用当前dirty代码代替。

各行包含：

* `bound_facts`：原report字节绑定的protocol值及JSON pointer，classification=`report_bytes_bound_fact_not_execution_source`。selected root SHA不冒充全attempt ledger。
* `conditional_envelopes`：标明公式、n/oversample、protocol pointer、四模块Git blob SHA/函数名、适用假设、unknown，`certified=false`。
* `explicit_state_reset`：seed0，显式state/goal重建，不是seed0随机任务身份，仍为条件事实。
* `historical_version_disposition`：A/H等逐项的版本证据性质；F明确exact P1v2未恢复。`historical_evidence_index`回指证据列表；该列表的scope区分Git、postrun、later-run与文字处置，均`proves_original_eight_run_source_binding=false`。
* 原`inherited_refs`不变，`historical_execution_source_bound=false`、`actual_attempt_bound_certified=false`、unknown完整保留。

## 包络合同

固定旧代码族：collector先生成4n个候选；boundary默认无finite-angle加倍；terminal_pose分支、support_gap_max=0、下层for-loop cap为2×候选数。采用更保守的闭区间 `[base,base+8*n-1]`，不是宣称真的尝试了8n次。默认无拒绝成功路径为4n，本文不以它缩窄隔离。

|ID|候选随机reset条件包络|
|---|---|
|A|202753000–202753031|
|B|202610241–202610272|
|C|202710241–202710272|
|D|202710281–202710472|
|E/H|202753010–202753041，各保留一行|
|F/G|202710251–202710282，各保留一行|

A/B/C分别另保留未打开的reserved包络202753100–202753291、202610281–202610472、202710281–202710472，kind=`reserved_not_observed_reset_envelope`。候选与预留不混称actual。所有8行单独保留seed0重建项；state/history排重仍缺。

假设明确包括原run用了该代码族、oversample4、infinite angle gate、terminal_pose非radial_contact、support_gap_max0、无未记录替代producer/失败进程或env reset。缺少其中任何历史认证，不得将这些有限区间当全部历史的穷尽证明。+70000/+10000是代码中其他RNG种子，不误记为env.reset。

## 新根检查schema与语义

可选NEW JSON（文件SHA须外部提供，不能只传文件）：

```json
{"schema":"moving_historical_reset_quarantine_20261027_v1.new_seed_spans","roots":[{"new_root_id":"MAIN-predeclared-id","random_reset_spans":[{"start":300000000,"end":300000031}],"explicit_state_reset_seeds":[0],"reset_coverage_declared_complete":true}]}
```

这是格式示例，不是已批准的新根/seed选择。最多32根、每根16个闭区间/16个显式重建seed；非负int64，拒绝bool/float/负数、反向span、重复root id、未知字段、重复JSON key、NaN。输入不能包含outcome/candidatecost/物理标签。coverage=true仅是调用方声明，代码不把它当实际执行认证。

用max/min做区间相交，不枚举seed（巨型span也不导致巨型循环）。候选和reserved每个hit回指source ID/SHA、kind、request span index与交集；同一新根与多条历史重合全部保留，不合并抹去16引用。`conditional_span_overlap`不是实际同根证据。

new explicit seed含0或新随机span含0时，单列`explicit_state_seed_label_overlap_not_physical_identity_proof`；这是seed标签重合，不将所有seed0状态误认为相同物理根。反之无已知区间相交仅`no_known_span_intersection=true`，仍缺历史完整coverage/执行绑定和physical state/goal/history排重。metadata缺项时`comparison_complete_for_listed_envelopes=false`。

**任何路径都保持status=`freshness_not_certified`、freshness_certified=false、historical_coverage_complete=false、old_freshness_blockers_cleared=false、certified_source_count=0。** 本版本没有签发/override开关。不能把无交集或metadata_status=assembled解读为排除旧阻断。未来MAIN有新authority也要另封版本，不能改本v1追认。

## API/CLI与输出

纯函数 `build(read=local_read)` 产出quarantine；`check_new(quarantine,request)`产出seed-span检查。RAM tests用内存read与mock字节SHA，无磁盘fixture。CLI每次重新核原saved字节，不加载任意用户构造的清单冒充authority。

内层根 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_historical_reset_quarantine_20261027_v1 \
  --output results/moving_historical_reset_quarantine_20261027_v1_saved01.json

# 新根检查可选；必须NEW输出，不覆盖saved01：
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_historical_reset_quarantine_20261027_v1 \
  --new-seeds results/MAIN_new_seed_spans.json --new-seeds-sha ACTUAL_FILE_SHA256 \
  --output results/moving_historical_reset_quarantine_20261027_v1_checked01.json
```

**CLI正常生成metadata仍exit2**，表示freshness未认证，而非程序运行崩溃或科学负结果。stdout含path/SHA/metadata_status。缺/坏report逐行blocked，仍保留8/16；缺Git/aux逐项记录失败；无法核固定authority/分母时顶层拒绝，不发布伪8项正常结果。

输出排他`xb`、flush/fsync、不覆盖/删除，<=2MiB；预写free至少200MiB+2MiB。单输入<=8MiB，Git限固定小型blob，无大产物复制。raw source/结果只读。磁盘不足拒绝，不放宽floor。

## 验证和实际边界

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
tests/test_moving_historical_reset_quarantine_20261027_v1.py
```

覆盖8/16分母、8n/seed0/reserved、闭区间边界、多来源重合、无相交仍未认证、巨大span常数算术、缺证据/坏SHA保留阻断、历史版本scope、strict schema、磁盘floor/输出cap/独占创建，以及help/CLI内存端到端。测试只验证实现合同，真实metadata只验证保存证据字节，不证明历史执行已完整绑定。

本次最终RAM测试实际 **27 passed in 0.14s**。实际saved构造仅使用原证据，没有提供/挑选新根request；NEW产物路径为上例saved01.json，new_seed_check.status=not_requested。其status始终freshness_not_certified，非模拟或CUDA结果。

剩余前置与原处置文档一致：8run的确切producer执行绑定、完整root factory/env闭包或等价全reset ledger、失败/未选/预留覆盖authority以及未来新根实际physical identity排重。本产物不消除旧resolver/certificate任一缺口，P1仍false。
