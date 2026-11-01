# Moving历史排除身份资产生成器 v1

独占本stem source/tests/innerdoc；不改sealed suffix/GRASP/历史结果或共享协议。MAIN审源码、范围及实际证据准入，不设专门审计agent。本交付只进行RAM tests，不生成研究manifest/approval，不运行模型、环境、物理、训练、拟合、网络或删除。

## authority与边界

直接依据`/home/gbk/code/LeWorldModel/docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md`，必须由输入提供该精确path及外部SHA；不将文档的四对seed当作已独立新根。协议要求排除截至cutoff全部已接触/探索/预留PushT来源，包含失败和未选候选，不止原moving两根。

最低family范围：`legacy_zero/original_four/constructive_zero/moving/overlay/feedback/reserved_prepares`。这是范围下限而非已证明完整历史。MAIN须用`expected_source_families`扩充其他已接触源（包括有需要的archive episode、B8、任务/episode分支等），并在`declared_unresolved_sources`明确尚未定位的来源。文件名未列出、seed未搜到或family名字齐全均不能证明完整。生成器**不扫描目录**、不glob、不自动跟随report的bindings/certificate/NPZ/未来图路径。显式白名单外的文件不会自动读取。

authority层级：

|来源|可贡献的身份|不能据此声称|
|原protocol外部SHA|固定候选seed、排除规则、MAIN权限|8个seed已独立/已排除|
|旧resolver已有authority JSON|旧root-goal hashes、seed/span及unknown基础索引|全历史覆盖；其unknown不能被新生成器抹掉|
|已封存prepare（含moving/reachable envelope）|全部slots的root/reset seed或half-open reset span，包括预留与未执行|已存在实际observed root|
|已封存sourcepack|全部rows的slot/seed/root-goal/task/history身份，不看status、success、cost|缺失state/snapshot时可自认为已映射|
|已封存task export|task_id/root-goal身份（如存在）|latent本身等于物理root身份|
|显式metadata bridge|MAIN待审的state/snapshot/history与task/episode映射声明，文件SHA保留|生成器已独立证明声明与原物理ledger相符|
|外部MAIN approval JSON|批准**本次精确content SHA与cutoff范围**|未来实际新根freshness、统计独立或P1通过|

已知可显式列入的原authority路径（均以Python根`/home/gbk/code/LeWorldModel/le-wm-main`为前缀；实际SHA须MAIN取得并封存）：

```text
results/p1_constructive_reachable_source_20261027_v1_prepare01.json
results/p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json
results/p1_constructive_moving_history_source_20261027_v1_prepare01.json
results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json
results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz
results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_01.json.gz
```

这些只是已知authority，不是完整白名单。legacy四根、全部旧zero/calibration、overlay/feedback新observed根、失败采样与历史reset尝试/预留跨度仍须MAIN明确文件及身份桥。不能仅取上述文件就签完整清单。旧resolver源码只复用lexical streaming skip，使用**本文件自有patterns**；不调用`resolve_history/as_exclusions`，不改旧PATTERNS/global。root-goal hash保留原authority的算法与数值，不重新定义、不解码未来derived goal来重算。

## 最小request与CLI

```json
{
 "schema":"moving_root_exclusion_manifest_20261027_v1.request",
 "cutoff_utc":"2026-10-27T00:00:00Z",
 "project_root":"/home/gbk/code/LeWorldModel/le-wm-main",
 "protocol":{"path":"/home/gbk/code/LeWorldModel/docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md","sha256":"实际64位SHA"},
 "evidence_files":[{"source_id":"唯一稳定来源ID","family":"moving","path":"精确绝对项目内文件路径","sha256":"实际64位SHA","format":"prepare_envelope","adapter":"prepare"}],
 "expected_source_families":["legacy_zero","original_four","constructive_zero","moving","overlay","feedback","reserved_prepares"],
 "declared_unresolved_sources":["尚待MAIN定位的具体历史来源"],
 "approval_file":null
}
```

上述是schema示意而非可运行假证据文件。`format=json/gzip_json/prepare_envelope`；`adapter=prepare/sourcepack/task/metadata/legacy_index`。path逐项绝对、项目内、无glob/`..`、无symlink，source_id和path必须唯一。未知adapter直接拒绝request；支持adapter但未知schema/损坏/hash不符时保留该evidence并列unknown，不悄悄略过。JSON/gzip采用流式身份投影：只解码白名单元数据，future/success/witness/modelcost等value语法跳过；哈希仍覆盖全部原字节。读取/hash不等于将outcome用于筛选。gzip和envelope不导入原producer，不调用verify/episode。envelope验证原canonical JSON+换行对应semantic SHA与单gzip完整性。

MAIN从Python根执行（本交付未执行CLI生成研究输出）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_root_exclusion_manifest_20261027_v1 --request /绝对路径/MAIN_inventory_request.json --request-sha256 实际SHA
```

默认stdout输出完整manifest，便于RAM/只读检查。显式增加`--output /绝对路径/NEW_manifest.json`才会排他durable写新文件，禁止覆盖；写入保持200MiB reserve。没有全盘发现命令，也没有默认results递归扫描入口。

## metadata bridge格式（任务/episode不能映射时保持unknown）

```text
schema = moving_root_exclusion_manifest_20261027_v1.identity_metadata
records = [
 {record_id, record_type: observed|derived|reservation,
  root_seed, reset_seed 或 reset_seed_span:[lo,hi),
  root_goal_sha256（可缺失）, task_id（可缺失）, episode_id（可缺失）,
  blind_sha256（可缺失；保留原source/task的payload身份，不读其数组）,
  root_state:float64[7], root_state_sha256（可省略，由数组重算）,
  root_snapshot_sha256, history_pixels_sha256:[三个原pixels SHA],
  raw_history_sha256, parent_refs:[source_id#record_id],
  maps_to:[被映射的source_id#record_id]}
]
```

raw/sourcepack记录ref为`source_id#/rows/0`，prepare为`source_id#/slots/0`，task为`source_id#task`；metadata用literal record_id。桥必须指向同一次显式inventory已有记录，且至少匹配一个原root-goal/task/episode/blind payload身份，已知seed/身份/hash不允许冲突；缺目标、无共同身份、冲突或多桥歧义都列unknown。source calibration_task_id和encoder task_id属不同命名合同，不能直接冒认相同。它是待MAIN核原账的**元数据声明**，不是代码偷偷读取证书或反推state。未提供映射时task/episode的缺失原样保留。parent_refs只记录lineage，不凭同seed、名字相似或parent提及自动复制root；moving warmup后的根可能与旧zero根不同，不能强并。

observed/derived记录需state7、snapshot SHA、RGB3 SHA、raw10 SHA及seed pair；任一缺失均unknown。reservation只需root seed与reset seed或完整span，不假造observed根。旧源的共享placement reset0是显式recipe实现细节，不凭seed0制造新随机根；MAIN仍须在全reset ledger范围中说明其例外。旧resolver reset尝试覆盖不明时unknown永久保留，不能以新metadata侧写消除。

## 确定性排除与输出

输出`schema=moving_fresh_source_exclusions.v1`，含协议要求的cutoff/project/evidence/families、root/reset seeds、half-open spans、root-goal hashes、observed roots及unresolved；额外保留逐来源`seed_authority`、全部identity_records、stable_roots、aliases、bridge与外部approval绑定。所有列表确定性排序；同inventory不同输入顺序生成同coverage content SHA。stable_root_id对固定inventory可重建；新增数值别名证据可能改变组代表，不能把其当跨任意新inventory的永恒随机ID。

state7 hash严格规范little-endian float64，保留signed-zero bits。physical pose由前四XY和angle的NumPy remainder到`[0,2pi)`组成，零规范+0；保存NumPy版本。每维XY差≤1e-9pixel且wrapped角差≤1e-12rad保守视为同pose；不根据outcome调阈值。同root-goal、state、pose、snapshot或history RGB3身份也保守归组，原记录/seed/来源不删除。**raw history相同本身不合并物理根**。同root-goal/snapshot hash却物理pose不一致另标unknown，不能用哈希声明掩盖冲突。换goal、velocity、signed zero、加2pi、renderer或task label不能使旧pose变新根。哈希不同仍不是独立性证明。

四对proposed seeds固定来自协议：root203027101..104、reset203027201..204。每个seed同时与root/reset已用集合及所有half-open reset span比较，不自动换号、不追加候选。输出四rows和每项conflict；`independence_certified=false`。失败/未选source row不会因status/cost消失。源文件metadata失败仍在evidence census，unknown阻断准入。

外部approval另为小JSON，字段恰为：

```text
schema=moving_root_exclusion_manifest_20261027_v1.MAIN_coverage_approval
authority=MAIN, approved=true, coverage_scope=[MAIN批准的完整family范围]
cutoff_utc=原cutoff, manifest_content_sha256=第一次输出coverage_content_sha256
```

MAIN可先获取无approval的content SHA，再独立封存该批准文件，重建request仅加其path/SHA；content SHA不含approval，避免循环。request完整SHA另存。`approval.approved`仅表示外部文件精确匹配，**有任何unknown/审批错误/seed冲突或缺批准，`source_run_preconditions_met=false`**。即使全部历史元数据前置成立，`freshness_certified=false/P1_admission=false`仍固定；实际新根尚须在未来producer以自身state/pose/snapshot/history做历史及批内排除。批准也不能替代MAIN对桥声明、全历史范围、原始证据的独立准入。

## 资源、失败与验证

最多512显式源文件、每文件64MiB（gzip expanded亦64MiB）、累计原文件及metadata展开各256MiB、4096 identity records、输出2MiB；每个prepare envelope≤256KiB、展开≤4MiB。预算耗尽后其余显式文件仍留census并标not_opened unknown，不继续读取。仅小代码/测试/doc写集，不复制旧protocol/证书或大数据。损坏/未知/缺失项进入unresolved，不用placeholder补根。无模型/物理预算；只写明确NEW output，fsync失败直接抛错、保留已写partial，不自动删/重试。

RAM命令（不创建pytest cache/tmp，不跑神经）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_root_exclusion_manifest_20261027_v1.py
```

本次48项RAM测试通过，覆盖显式白名单、不读opaque heldout值、不跟随bindings、sourcepack失败/未选分母、三种编码、原envelope semantic SHA、跨root/reset与half-open冲突、scope缺失/旧unknown继承、任务桥/歧义、signedzero/2pi/velocity/换goal、unknown即便外部批准仍阻断、approval hash/scope篡改、全局metadata预算/naive时间拒绝、RAM模拟排他durable写入/floor/fsync失败，以及无目录扫描/模型导入。RAM fixture只验证实现，不是历史覆盖或fresh根证据。experiment-code技能用于固定资产合同和失败透明；未使用其通用训练/调参/画图模板。

## 本次实际saved材料RAM smoke（不输出研究manifest）

仅显式上文6个path，全部原字节SHA核验并metadata投影成功；不follow其引用文件，也未读取证书/NPZ/heldout outcomes。已验证的外部SHA如下，顺序与上文path表相同：

```text
reachable prepare: 98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086
reachable source:  9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9
moving prepare:    c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9
moving source:     c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6
moving task00:     a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b
moving task01:     101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74
```

实得10条identity/reservation记录（4预留、6未完整映射的source/task记录），50个原历史root-goal hash、18个half-open reset span、20个root seed与2个reset seed；后两reset为202798201/202798202。这不是10/50/20个独立根的证明。

970条unknown原样保留：两个prepare各继承480条历史unknown引用，共960（不是960个独立来源）；4个sourcepack根和2个task缺失物理state/snapshot/history或seed映射6条；legacy_zero/original_four/overlay/feedback四类不在此显式inventory4条。旧unknown包含历史reset尝试无界、旧parser不能映射的metadata、及可能需MAIN澄清是否属于PushT范围的引用；本版不自作主张删去、重分类或用approval清零。实际old row不含足够state/snapshot，故即便已取得sourcepack SHA也不能假填pose。MAIN需要补窄范围、原字节绑定的身份/ledger/范围证据；若要消解旧unknown，需另行MAIN审读的显式证据处置合同，本版没有静默override开关。

四对proposed seed在**这份不完整inventory**中无已知seed/span冲突；但unknown存在且无外部MAIN approval，故status blocked、source_run_preconditions_met=false、freshness_certified=false。没有生成批准文件或可供fresh source准入的研究manifest，没有读取heldout outcome来筛根。
