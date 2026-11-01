# 历史排除元数据补全与显式request v1

本任务是saved-only资产/元数据实现，不是专用审计智能体；MAIN负责源码审查、历史范围裁定及新根准入。遵循`docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md`，不修改原manifest v1、任何封存source/结果或共享文档。不调用模型、环境、producer.verify、episode、训练或拟合，不根据outcome选择来源，不签发freshness或P1通过。

## 输入及协作路线

新source/tests/innerdoc同stem为`root_exclusion_authority_resolution_20261027_v1`。使用experiment-code技能固定来源、预算和失败计账；用户限定本任务无训练/图表/神经实验。

入口固定六份旧外部SHA（源码`PINNED`列出全部精确path及SHA）：两个reachable/moving prepare、两个sourcepack、moving task00/01。逐份核验原字节SHA；prepare核验单gzip、原semantic SHA，只投影身份字段。然后**仅**展开这两份prepare的`exclusions.unknown_sources[*].source`所列480个实际绝对路径；不是目录发现或自动follow bindings。保留960条原引用和原reason，不将两份相同列表当960个独立来源。

Arendt负责截至cutoff的历史inventory文件清单，MAIN中转。当前工具无跨线程消息接口，本实现不另派代理，也不冒认已收到inventory。可合并Arendt/MAIN提供的明确文件，格式：

```json
{"schema":"root_exclusion_authority_resolution_20261027_v1.inventory","files":[{"path":"项目内绝对文件路径","sha256":"外部核定64位SHA","family":"legacy_zero或original_four或overlay或feedback等"}]}
```

上例只是schema，非占位证据或可用authority。每个path必须实际存在、canonical、无symlink/`..`、位于Python根；SHA冲突或读取失败逐项留阻断。`family`是外部inventory声明，不是本生成器批准历史全覆盖。默认只解析JSON来源；额外gzip/envelope来源如非固定六文件，仍须另明确格式支持，不能让一个family标签掩盖不支持的metadata schema。

## 三类原缺口的实际进展

原970 unknown = 960继承引用 + 6物理映射 + 4 authority。

|原类|实际已取得|仍不能推导|
|960继承引用|480/480唯一文件实际可读且有当前SHA；原944条parse失败引用对应472文件已可语法投影；全部原reason保留|可读/可解析不等于每个文件都有根身份，更不是reset覆盖|
|6物理映射|4个source根与2个moving task别名全部映射，hashchain核验无错误|不是6个独立样本，不是新任务|
|4 authority|逐类生成显式request；等待Arendt inventory|legacy_zero/original_four/overlay/feedback全历史范围仍未获MAIN批准|

实际capture01：480文件、236文件有白名单metadata，173文件有直接root/seed/task字段；73文件有指向另一已捕获文件的显式输入绑定（可重叠），472/960原引用取得metadata，346/960有直接身份字段，614/960尚无直接身份投影。244文件无当前白名单metadata，另63文件仅schema/binding等非直接身份。不要把614和16历史reset缺口相加为互斥总数：16条reset bound未证实引用属于原960集合，身份投影也不能清除它们。

8个文件有旧prepare记录的原SHA，472个是`first_capture_not_historical_attestation`。后者即使当前SHA成功封存，也不是“旧prepare当时已验证这些字节”。旧unparseable不等于缺文件；本交付已把可读性、元数据可得性和历史覆盖authority分开。

## 身份投影与物理桥

每个request项保存绝对path、当前fileSHA、原引用JSON pointer、原reason、旧SHA或null、metadata白名单字段pointer、top-level keys、明确阻断和已捕获输入绑定边。绑定边仅在指向本次已捕获path/SHA时记录，**不自动读取**目标，不凭lineage认定物理state相同。

`stable_identity_projections`按原root-goal SHA归组，ID为`root_goal:<原hash>`，每条回指原文件SHA和pointer；root/reset seed另列，不按目录名推断seed，不把root_seed当所有reset尝试，也不按root_count猜seed分配。root-goal组明确`physical_root_equivalence_proved=false`：换goal仍可能是同一个物理根。

物理桥只增加固定两份source同目录`journal.jsonl`这两个明确路径。每个event先只解码kind/index/previous SHA/phase/step/resource/slot；所有行在原canonical字节上移除一个event_sha256成员并复算SHA，检查连续index、previous链和sourcepack绑定tail。未来/结果值只参与原字节hash，不解码/打分。仅对replay0的step0/5/10解码state、exported snapshot、PNG文件/pixel SHA；仅对前10步public_step读取raw_action及attempt/completed。完整10步缺失/未完成即拒绝桥。reachable由该真实ledger得zero raw10，不凭假定填zero；moving取得真实非零raw10，并与sourcepack的raw及RGB3 hash核对。

桥保存实际step10 state7、规范little-endian float64 state SHA、wrapped物理pose key、exported snapshot hash、RGB3 pixel SHA、raw10 little-endian float32 SHA、原root/reset seed、原root-goal及blind payload。snapshot hash使用原风格`sha256(json.dumps(sort_keys=True,allow_nan=False).encode())`，不是求解器隐藏缓存。PNG SHA是hashchain ledger声明，本程序不再次解码PNG像素；不能冒称独立像素审计。task仅在blind payload SHA和source_slot一致时连接，不能把source calibration_task_id当encoder task_id。

4桥的maps_to合计6原记录；bridge_payload使用冻结v1的identity_metadata schema。但整个resolution_request不是旧v1 manifest request，不能直接喂旧CLI冒称消解960项。旧v1继承unknown和其当时历史结论保持不变。后续MAIN若另封存桥资产并生成原v1 request，可补齐6条物理映射；其余历史覆盖仍须MAIN明确处置合同，不存在隐式override。

## 运行、预算与失败

Python根`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_root_exclusion_authority_resolution_20261027_v1.py tests/test_moving_root_exclusion_manifest_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.root_exclusion_authority_resolution_20261027_v1 --output /home/gbk/code/LeWorldModel/le-wm-main/results/root_exclusion_authority_resolution_20261027_v1_MAIN01.json
```

收到真实inventory后在第二条命令增加`--inventory <绝对path> --inventory-sha256 <真实SHA>`并另用NEW输出名。省略output只打印stdout，不创建资产。不覆盖既有capture01/02，不生成假approval。读取失败、SHA冲突、metadata无法映射逐项保留；物理桥失败保留对应source/error，不补假state。输出排他创建、flush/fsync；写失败不删除partial、不自动重试。

固定读取上限128MiB/单文件、512MiB累计唯一文件，输出8MiB，200MiB floor；原480文件总273113590 bytes，其中两个约100MiB文件超旧v1的64MiB单文件cap，因此新合同显式提高纯读取预算，不偷偷减分母。实际输入含六固定文件及两journal共275936057 bytes。无模型/物理/网络预算。RAM tests不写pytest cache/tmp。

## 当前阻断及下一项精确输入

1. Arendt补四类及任何其他已接触历史的完整path/SHA inventory，含失败/未选/预留来源；不能以旧moving两根替代范围。
2. 无直接身份的614条引用（307文件）需要有原字节绑定的父源/episode映射；其中文件top-level keys已保存，避免再次只报unknown。可读但纯物理执行结果不因文件名带root编号而被自动归根。
3. 原16条引用/8来源仍缺历史reset attempt完整bound/ledger，root-seed字段不足以替代。新捕获其余来源也没有自行证明reset完整性。
4. MAIN需裁定非PushT引用的scope，不能仅按Reacher目录名删除；需明确环境元数据/源recipe依据。
5. 全历史元数据覆盖获得MAIN批准之后，未来新source仍须用自身实际state/pose/history做旧根与批内排重；四对seed不等于八个独立根，P1仍false。

本交付把“找不到/旧parser失败”推进到可读原字节、显式身份投影和六记录映射；没有为缺authority签发freshness。

## 最终冻结验证

新32项RAM tests与原v1的48项联合80 passed。实际完整`generate → stable_projections → write_new`入口已运行，不存在仅mock绕过实际入口的NameError。最终新输出：

`/home/gbk/code/LeWorldModel/le-wm-main/results/root_exclusion_authority_resolution_20261027_v1_capture02.json`

SHA256：`032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00`。capture01保持原样；capture02绑定最终source及其完整本地元数据依赖闭包，不绑定后续doc文字。最终实际计数仍为480文件、236有metadata、173有直接身份、73有显式输入边、6原记录物理映射、614引用缺直接身份、16引用缺历史reset上界、4类authority未获批准。

MAIN另行产出的inventory8原manifest与其blocked结论保留，不由本实现覆写；Arendt后续负责historical_reset_bound_disposition实际上界证据，本实现不重复制作该证据。Arendt的618文件inventory累计282406682 bytes超过旧256MiB，须使用新512MiB合同；未收到其精确path/SHA文件前不计入本次480文件覆盖。新generator允许显式inventory合并，但不允许借合并family标签自签批准。
