# Compact C3 durable transport v1（实现中，非 HOST-ready）

仅新三件套，不修改 codec/routing/native_catalog、原12槽算法或任何封存产物。
4GiB总量含512MiB reserve、whole7200s/slot900s、GPU allocated3GiB/reserved4GiB、floor200MiB不变。
worker只跑注入RAM回调及临时磁盘测试；无神经、CUDA、physics、生产prepare/fullstrict、子代理。

## 已实现接口与真实顺序

`OnlineStore`直接实现原`original.execute()`所需`event/capsule/result`，固定known2×K1/K17×三policy=12槽、原144/槽=1728identity，原27轴与算法不变。
不是先收全槽再搬盘：每一事件先完成header/owner登记，再编码、CAS pack/index fsync，最后append104byte原序journal并fsync。
`Transport`与`GuardedCAS`共用live预算，不能各自占满4GiB。

Native调用一次只保留当前call buffer。返回前调用原`audit_native`，从真实nominal boundary/replay result建立完整context目录；task只允许原root与已认证parent context的`parent_task`。
完整返回写journal后才能建native/capsule W view并发布semantic pointer。失败prefix可保存，不能冒充完整source。
virtual capsule沿用冻结native_catalog显式命名空间，不声称物理JSON文件存在。

在线aux派发：两mapping按即将发生的queue ordinal登记；queue的4/8request先按原144预算分配存储owner，拒绝queue不改变job状态；progress bank、bank_event、0+8 R row、R_return的bank/group均在其event首次append前显式路由。
同一queue只允许其原decision对应的progress或R bank之一，因此parent桶使用`max(BANK,PROGRESS)`而非加和。signal_refusal不冒占第18个已调度parent。
整槽结束仅补result DAG及完整ledger W view，不重排、补洞或替换既有事件。

`import_settled`单独标为**离线fixture导入**，保留用于回归，绝非在线runner。`run_fixture`显式`permit_ram=True`，不能假装HOST执行。

## Header整数census（无NN、无平均压缩率）

所有bound使用冻结routing的逐字段`bound()`，包含每个容器/标量/索引，ref只在目录逐字节认证后才是0新分配。

- invocation≤4884。每call保守相加TASK_HEADER、SIGNAL_REQUEST、REPLAY_REQUEST、RESPONSE、≤9context的result list和method string，合计6781bytes/call，共33,118,404bytes。signal/replay互斥仍相加；不假设不同call数值相同。
- 完整context≤6912，每项显式layout/context_length/latent_dim/action_dim固定静态ref、3×192/2×10 literal float64。连原已计数数组payload/header再保守重复收费，共36,488,448bytes。
- scope≤12×(3×17+2)=636；只允许signal/schedule/program的parent-context variant或whole/segmented verification variant。总737,124bytes。
- delta只接受原`core.delta()`精确模板；固定native batch1/full3形状，VJP每次1方向，scalar无shape/directions；最多18种，总45,144bytes，不为每resource另建27轴dict。
- inclusive map≤12×18原operation名，20,268bytes。
- 顶层最多12initial+12return+12settled行，每行canonical≤4096bytes，按128×canonical bytes涵盖generic literal/type/container/index（每节点至少一个JSON语法byte；单个primitive+index≤57+字符串bytes，dict每field34、container+index≤53）。总18,874,368bytes。字面int/float/bool、-0均不转换。
- signal_refusal另按每槽NODE+2 AXES+256的完整逐字段bound收费；不借未生成parent免费空间。
- fixture execution envelope另64KiB；完整12最终行仍保存在CAS，blocked行不能只存在调用方RAM。
- static.json另重复计32MiB，manifest16MiB，completion64KiB。manifest root≤6988、每canonical entry≤256bytes；capsule≤4884、每entry≤1024bytes；其余固定部分≤64KiB，合计6,855,680bytes≤16MiB。W indices只存pack一次，不另写JSON数组。

`python -m ... proof`给出精确整数：原base2,930,867,648bytes；新增header/运输139,890,740bytes；live3,070,758,388bytes；加完整512MiB reserve共3,607,629,300bytes，距4GiB尚余687,337,996bytes。
这是明确的schema约束，不覆盖任意foreign metadata。生产静态seal与HOST before/after闭包未接完，`complete_production_fit_proved/production_ready`始终false。
完整错误文本不截断；任意无限错误或I/O故障不能保证完整receipt。传输失败会poison实例、保留已落盘prefix、不退款、不删除。`preserve_failure`从独立512MiB reserve写全12行字面状态和完整错误，按包括未索引suffix在内的实际物理文件长度核4GiB/floor；其own incomplete schema不签完整成功，Reader拒绝将其当manifest消费。

## 只读消费和完成边界

实际目录固定`cas/{objects.pack,objects.idx}`、`journal.bin`、`static.json`、`manifest.json`、`completion.json`。
`Reader`核全目录census/regular无symlink、完整file SHA/size、连续唯一CAS index/domainSHA、每104byte原序链、全部W family/实际位置、无环引用及无隐藏/未用对象。
manifest同时绑定旧冻结依赖和本三件套的实际source/test/doc SHA。对每个virtual capsule使用从原journal wrapper构建的`(slot,call)`索引，核all-and-only contiguous indices；不每次全扫无关events。
typed原dtype/shape/有限float由冻结decoder核验。virtual pointer的完整语义SHA/length在`capsule()`完整展开时核验，不混成material文件SHA。
`Reader.fixture_inputs()`只返回保存的完整prepare/12rows；测试再从读回事件复用原`validate_rows/validate_tree`全重建，不声称独立数学实现。
CLI的`check-transport`只检查运输，不替代原saved算法重建或HOST完整strict。

manifest在CAS/journal/static均持久化后写入并fsync目录；**之后**取独立clock写completion，明确排除completion自身write/fsync和shutdown。
completion也在最终file census内；额外运行态检查确保它自身未越whole7200s，但不伪称能在有限文件记录自己的fsync之后时刻。

## 当前可运行命令

在`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_p3_c3_compact_durable_transport_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_compact_durable_transport_20261029_v1 proof
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_compact_durable_transport_20261029_v1 check-transport --directory /absolute/exclusive/fixture --manifest-sha256 ACTUAL_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_compact_durable_transport_20261029_v1 fixture-reconstruct --directory /absolute/exclusive/fixture --manifest-sha256 ACTUAL_SHA
```

仅支持`-m`，不支持直接文件路径执行。临时磁盘整链含真实原12槽RAM控制流、1728分母、27轴、正常/cap/no_witness与首调用失败后11槽blocked；不是fixture×12空间外推。
完整回归尚未终态时不得称冻版或通过。最终实际测试结果与三SHA另交MAIN。

## 距生产仍缺什么

目前无prepare/run/consume生产CLI：还缺MAIN-only原source/task authority build、prepare重建及完整HOST model/class/precision before/after包装；生产reader还需对这些HOST元数据做逐项闭包核验。
不能直接调用原blocked的run，也不能patch其fit gate；新HOST包装必须真实使用本OnlineStore并保持原科学合同。
不得在这些缺口关闭前启动神经。全C3/P1/R原等价/physics release均false；本切片不作任何科学收益结论。
