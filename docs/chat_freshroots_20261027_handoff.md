# 独立新根：排重 metadata 闭包与待确认清单

本个人有界任务已完成软件消费者、实际 saved metadata 复算及未决证据请求交付；**没有完成 freshness / 历史 coverage 认证，也没有授权新 source run**。未运行 P1 模型/控制实验，未打开新 outcome；不训练、不使用子智能体、不删除、不推送。全部写入仅为本前缀的新文件，旧 resolver/catalog/source/结果未改。

## 交付与摘要

机器 root manifest：`/home/gbk/code/LeWorldModel/le-wm-main/results/chat_freshroots_20261027_root_manifest.json`。
紧凑 coverage 证据：`/home/gbk/code/LeWorldModel/le-wm-main/results/chat_freshroots_20261027_coverage.json`。

|现场复核项|结果及边界|
|---|---|
|inventory / metadata / 直接身份|618 / 344 / 199 文件；不是 coverage 认证|
|来源读取|reader 唯一原字节 285077447 B、622 文件；622 与 618 census 不是相同口径；另有历史源码/aux 只读验证|
|显式图|398 原边中 370 path+SHA 双绑定有效、28 内容别名 unknown、0 conflict|
|身份关联|120 来源、278 有向 witness 链全部另行复核；不迁移物理身份|
|继承缺口|960 原引用保留；614 缺直接身份，其中120有上游关联、494无上游关联；关联不清零缺口|
|真实物理桥|4 历史桥映射6原记录；其中 moving 两根仍为已知历史，不是 fresh|
|reset|8 来源 / 16 阻断保留；0/8获完整历史执行上界认证|
|seed/span|102条 seed 证据（不是102个独立seed）；46条继承span证据对应18个unique区间；另11条条件包络；全部57条证据共26个unique区间|
|测试|新增27项、联合163项全部通过；只验证软件合同|
|独立保存后检查|不导入消费者，重hash622文件及消费者源码、核278链、8/16分母、四槽及false flags：PASS|

解析为实际重跑：先核精确 source SHA，再调用只读 resolver.generate，重读618 inventory及两prepare中的历史引用、验证两journal桥；不是照抄saved summary。再重建来源图、逐链校验、实际重跑历史 Git/blob/postrun/later-seal 证据消费者。未解出的身份保留到逐来源 `required_evidence` / `blockers` / `unknown`，没有将新捕获字节当旧执行绑定。旧未知引用清除数为0。

全部历史记录都有原path+SHA+pointer或精确Git commit/blob+SHA；历史reset推导来自恢复的Git代码族，**不执行该源码，不拿当前dirty producer倒推旧reset**。postrun与后续prerun只证明各自范围，不倒灌原8run。条件区间用`[lo,hi)`统一表示，原quarantine的闭区间end明确加1；原式/假设/证据保留在 `historical_reset_evidence`。seed0仅显式state placement，不被当作一个随机根或所有state的等价证明。

## 可交主线程确认的固定候选（不是已认证root）

|slot|root_seed|reset_seed|witness sign|当前状态|
|---|---:|---:|---:|---|
|0|203027101|203027201|+1|待coverage与实际身份|
|1|203027102|203027202|-1|待coverage与实际身份|
|2|203027103|203027203|+1|待coverage与实际身份|
|3|203027104|203027204|-1|待coverage与实际身份|

四对seed对**已列**历史seed/span无已知冲突，不代表完整历史无冲突。抽样/候选顺序直接继承既有协议，先封存 `chat_freshroots_20261027_preregistration.json` 再执行本轮闭包；分母始终4，每槽1proposal/1attempt，无补根、重试、oversample或收益排名。duplicate/guard/degenerate/partial/not_started均保留。没有实际新root state，四行 `actual_root_identity=null`；认证fresh根清单为空，不能直接启动source。

新实际根仍须在witness dispatch前做历史及批内state7/pose/snapshot/RGB3排重：XY每维1e-9pixel、wrapped angle 1e-12rad的原固定容差；换goal/velocity/renderer/signed-zero/加2pi不能绕过。raw10相同单独不作物理合并。此处只交metadata消费者，没有伪称执行未来producer身份检查。

## 未决依赖与精确请求

1. A/B/C/D/G：补每个原report SHA到实际执行producer blob、collector三下游及当时env/wrapper的绑定；或与report绑定的完整reset ledger，涵盖失败、未选、重启与预留。selected root/count或相邻Git提交不能替代。
2. E：postrun源码已核，缺它与E执行版本的事前绑定和collector闭包。F：确切P1 v2 producer仍未恢复，不能用v3替换。H：后续physical seal已核，缺H自身生成run的source/attempt闭包。八项精确report路径、SHA、原16refs及原版处置均在manifest `historical_reset_evidence.rows`。
3. 对 `evidence_census` 的缺直接身份项：提供原字节绑定的source/episode→state7/snapshot/history映射。path+SHA上游关联不能证明消费者自身是哪个物理根；28条内容别名要精确path绑定，不能静默补边。
4. MAIN裁定 original_four 完整范围（不能把 `original_four_candidate_scope_unverified` 改标签冒认证），并覆盖legacy_zero、moving、overlay、feedback、预留及其他探索/失败来源；提供外部精确content SHA+scope批准，不自签approved。
5. **618是封存快照，不是截至当前墙钟的全部历史。** 主线程仍在推进，需显式delta path/SHA inventory及新cutoff；本任务不扫新目录假称补齐。
6. MAIN批准历史范围后，仍需固定四槽实际身份、历史/批内排重收据及单独P1条件任务准入。本文不替主线程做P1或其代码审计，不以source可达/模型编码成功替代科学门槛。

coverage_certified、freshness_certified、source_run_authorized、P1_admission均false；科学门槛未放宽。神经/物理调用均0，未用CPU代替CUDA。

## 文件 SHA256

以下路径的外层根为 `/home/gbk/code/LeWorldModel/`。

|文件|SHA256|
|---|---|
|docs/chat_freshroots_20261027_preregistration.json|79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e|
|le-wm-main/experiments/spectral_reachability/chat_freshroots_20261027_closure.py|d408e2dbab9a56d1daa1024910895eeb485dabab35ed2dac95ead328cac37b4a|
|le-wm-main/tests/chat_freshroots_20261027_test_closure.py|5b0d5f4c9c094a90af518aa95c9f4706cd1c316b33de5f0323ddf63d8b4a3ec7|
|le-wm-main/results/chat_freshroots_20261027_root_manifest.json|3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838|
|le-wm-main/results/chat_freshroots_20261027_coverage.json|86ada7fac842daf349951c59bb858c3086ed5566ad1465e9ef290afff37c0d74|

## 复核入口

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；不写cache/bytecode。消费者没有输出文件写入接口，只有stdout；本交付JSON由apply_patch新增，未覆盖旧结果。CLI为有界stdout分块，每次完整复算后返回full_sha256/full_bytes/index/chunk；本次5块全文SHA一致后拼接，磁盘文件SHA再次独立相符。下例可直接向stdout重算完整canonical内容，不生成文件：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -c 'import sys; from experiments.spectral_reachability.chat_freshroots_20261027_closure import run,canonical; sys.stdout.buffer.write(canonical(run()))'
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/chat_freshroots_20261027_test_closure.py tests/test_root_exclusion_authority_resolution_20261027_v1.py tests/test_moving_authority_identity_link_closure_20261027_v1.py tests/test_moving_historical_reset_quarantine_20261027_v1.py tests/test_moving_root_exclusion_manifest_20261027_v1.py
```

固定输入/依赖字节不符则拒绝；原resolver的unknown不会被override。读取预算沿用其128MiB/单文件、512MiB/累计唯一文件；本消费者输出上限8MiB。磁盘现场60GiB可用。本轮使用experiment-code技能的固定输入、失败计账及复算要求，不采用其训练、按效果选优或画图模板。
