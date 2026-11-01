# C3 compact：逐字段结构routing、静态seal和磁盘CAS组件

NEW source：`experiments/spectral_reachability/agent_p3_c3_compact_routing_20261029_v1.py`；NEW tests：`tests/test_agent_p3_c3_compact_routing_20261029_v1.py`。冻结codec及两个原协议均不改。本件是用户批准顺序中的第一个可验组件，**不是可启动生产的12槽runner**。没有生产prepare/run/consume入口，未执行NN、physics、生产fullstrict或子代。

## 已完成的实际结构路由

`SCHEMAS`是有限的逐字段schema，未知字段、未知variant、缺字段及整数/bool/float混淆均拒绝；不是generic JSON兜底或将大对象扔入event blanket。`Routing.record()`直接写冻结codec支持的M/L/V/primitive records，`Routing.tree()`逐项处理原结果全部字段并核完整canonical bytes一致。内部保留原tree schema，是无损保存其真实值，不是冒签原producer的新report。

|结构|归属桶|完整字段/引用规则|
|---|---|---|
|每个job|candidate(slot, original job.slot)|完整request、proposal、node、error、status；144个slot原序不缺项|
|request|同candidate|完整原decision各字段、binding、local response、所有prefix edges/raw；native nominal/J仅指向已catalogued原值|
|proposal/node|同candidate|完整raw/path/edges、真实postcontext、原cost、R suffix/solve/nominal pointer和资格；pre-eligibility eligible=null版本另付|
|mapping两份、signal、queue|parent(slot, **queue ordinal**)|完整root/parent双身份、原H1/H3、两response/spectra；request list精确引用candidate request|
|R bank或progress bank|同parent|所有SVD矩阵、方向、原数学输出、source/evidence/authority、完整ledger事件与manifest；不删除metadata|
|R control/group|同parent|zero独立row；八candidate精确引用既有proposal；失败prefix仍保留八项并绑定原已派发request|
|tree容器/receipt/call headers|slot(slot,0)|完整144/17/34等有界spines；native transcript、全ledger/context是明确catalog ref，不复制payload|

parent不能仅按context SHA分桶：不同depth/path即使实际context逐位相同，也保留不同queue ordinal。最多17个ordinal；mapping按原执行顺序每两份归同ordinal，并核该queue的真实parent context。

同一queue的原runner在`if extension_type == reconfiguration ... else ...`只派发一种bank。本router**也实际强制**progress bank只能传入queued/progress ordinal；R group只对应queued/reconfiguration ordinal；一ordinal重复bank拒绝。因此parent费用可取`max(BANK,PROGRESS)`，而不是凭“常见情况不同时发生”取max。`progress_banks`输入为显式`(queue_ordinal, bank)`，不能按相近context/文本猜关联。

`Catalog.bind(kind,value,key)`必须先将目标实际展开，核canonical bytes完全一致；lookup再次检查。任意同名SHA、shape相同或伪造未知ref都不自动授权。类型分栏包括context/native_literal/task/transcript/ledger/scope/static和已路由的request/node/job/bank等。

## 非平均/非重复率的最大shape证明

`bound(spec)`对**每一个节点都独立收费**：M含3byte+34byte/field+48byte index；L含5+32/element+48；primitive含tag/payload/index；f64数组按8×全部元素+128header/index。即使所有primitive/矩阵每次不同，也成立。只有显式catalog外部引用为0新payload，该payload仍须由原native/static预算负责，不能变成免费科学费用。

parent完整并集为`2*MAPPING+SIGNAL+QUEUE+max(BANK,PROGRESS)+RROW(control)+GROUP`，上界 **242,544 bytes ≤524,288**。

candidate为`JOB+preeligibility RROW`，上界 **28,200 ≤65,536**。JOB本身是`opt(REQUEST/PROPOSAL/NODE)`的完整内嵌schema，不是ref；单独record这些root会得到同一逐位对象ID，不另造容器。不能误读为只收JOB spine。

实际progress/cap fixture发现原queue在cap拒绝前已生成4/8个requests，但未赋值jobs；router现将其完整保留并归到剩余原144个预算identity，不修改原not_requested job值。该情况下JOB为null request/payload/node，新增REQUEST替代其bound内已有的REQUEST项，未增加候选预算或改变派发身份。queued request必须仍匹配原已赋值job；无剩余identity则明确拒绝，不能免费丢掉refused requests。

slot完整TREE（含最大639 native call headers、144 job refs、34 mappings、17 signal/queue/group refs）的上界 **215,424 ≤1,048,576**。不将639个transcript重新物化为spine列表。

RAM测试对全部14种schema构造逐字面类型正确、最大shape、不同float的值，实测raw+48byte index增量≤各独立整数bound；另在一个parent owner实际合并两mapping/signal/queue/最大bank/zero/group测其总占用。测试还用既有NumpyNative真实192维R tree回调产物逐字段编码/精确展开，owner census从实际新增object重新累加，不以fixture压缩比例推全量。

一次已完成43项联合回归的实际R fixture测得：static JSON310,076 bytes、typed object/index335,079、field/metadata315,801，总static650,880；结构562个object共148,760 bytes，最大parent38,549/candidate5,259/slot17,317。此为**RAM任务fixture实测**，非未来生产全source seal或全12输出实测；其source_entries为空。另一个独立测试实际读取本source原路径并核SHA/文件长度。所有实测仅验证实现，没有用上述小数值乘次数证明总体；总体结构证明用前述独立最大schema。

后补的actual progress/K1 cap fixture已通过；fixture原本误断言只生成一个bank，实际原cap允许两个，已改为核所有原队列一一对应，不改cap。partial-R测试通过第4个真实RAM native invocation内predictor故障触发，保留真实control postcontext/失败、全部八null rows和144 jobs。最初继承fail标志被NonlinearRAM覆盖的transition忽略，已在原资源/IO包装内部明确注入异常；这是测试注入器修复，不是模型负结果。

## 静态source seal

`seal_static(tasks[2], sources[path→expectedSHA], extra)`只读取调用方显式给出的固定source文件，拒绝symlink/非正规路径，读取前检查size≤32MiB，并流式核实际SHA。记录原path/SHA/实际file length；不扫目录、不读weights、不自行寻找任务权威。原task两个完整Q/goal/normalizer/context及字面额外metadata一起封存，未拟合或更改任何值。

field-code表由固定schema、显式literal metadata及`extra.fields`事前产生；`install_static()`重新核整个field table，不能在seal后偷偷追加。它实际构造typed pack并量出**完整static对象+48byte索引+field table/静态metadata**总量，要求≤32MiB，不仅检查JSON长度。所存source条目是原文件长度/SHA而非文件内容复制；source本体仍在原只读路径，未来prepare/strict必须核完整来源闭包。RAM测试用显式实际本source文件核路径/长度/SHA，另测超32MiB、源SHA变化及field seal变化拒绝。

`read_sources=False`仅为明确标记的literal fixture开关；其source_bytes_checked=false不能签生产授权。本组件始终production_ready=false。当前未运行真实生产build/static discovery，所以RAM static实测值不是整份未来生产prepare的实测值。

## 磁盘CAS：已实现，不冒称12槽接通

`DiskCAS`是显式传给`Codec.objects`的append-only Mapping，不猴补冻结codec/旧os/global。原bytes以同domain SHA寻址、原typed dtype/shape不变；index48bytes存offset/length，内存仅持有offset表。重复ID需逐bytes一致；不同对象追加。append前核真实pack+index上限和200MiB floor，超限不退款、不删旧prefix。

新目录必须排他不存在，只有objects.pack/objects.idx；不会写展开capsule。`checkpoint()`同时flush/fsync两文件及directory，记录实际返回边界，不在每callback重新全文件hash造成O(n²)；`finish()`仅最终流式核完整文件SHA。saved reopen核own descriptor、完整两文件census、总长度/SHA、连续唯一index、每条对象内容SHA、无unindexed suffix。只读映射不允许append/delete。

`attach_disk()`只迁移已封存的有界RAM前缀，所有object ID、raw bytes及nodes_bytes完全不变。测试将最大H3 bank实际在磁盘编码，关闭后只读重开并完整展开，还覆盖pack/index/census/length/boundary篡改及超限保留。没有将磁盘CAS descriptor伪装为production receipt。

## 仍未交付的生产闭包（不能启动）

1. 本阶段的真实R结构测试将外部native侧另作RAM catalog fixture；它**不是**完整生产native事件预算证明。未来需将typed journal的native/capsule/ledger views接到Catalog，并逐call用原saved validator证明actual来源及27轴，不可把fixture的generic transcript列表当生产view。
2. 12slot/1728控制流还未接到磁盘journal/在线字段routing；完整source/class/model前后闭包、failure reserve、completion及scientific saved reconstruction尚未实现。不会给虚构HOST CLI。
3. 当前ROW表示涵盖完整、初始、attempted和postcontext失败阶段，但任意无限长异常文本不是有限shape证明；完整失败收据/512MiB reserve接线仍由下一runner负责，不能丢弃文本或宣称已证明任意异常都能封存。

冻结codec的总预算仍为live2,930,867,648，含512MiB reserve总3,467,738,560 bytes；本件实证了其中结构桶及封存static的编码边界，没有扩大4GiB或减少12/1728/27轴。其余native/schema预算不是本件冒签的完整生产证书。工程wall/GPU cap也未改变。

## MAIN验收CLI（必须模块入口）

目录：`/home/gbk/code/LeWorldModel/le-wm-main`。不支持直接执行文件路径；请用`-m`保证`experiments`包可解析。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -s tests/test_agent_p3_c3_compact_routing_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_compact_routing_20261029_v1 proof
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_compact_routing_20261029_v1 --help
```

第一条stdout给`REAL_DIMENSIONAL_RAM_ONLY`实际static/object/owner census，且RAM disk文件只在pytest临时目录。没有生产prepare、NN/physics/fullstrict。接口设计技能用于逐字段边界、显示引用与实体身份分离；不产生科学admission或改变policy/gate。
