# C3 common-cap compact v1：紧致序列化协议，待MAIN批准实施

2026-10-29。仅此NEW协议及文末整数RAM公式；没有新runner、生产prepare、NN/CUDA、physics或fullstrict。保留`agent_p3_c3_common_cap_cuda_20261029_v1`三件套及所有冻结依赖/结果。紧致表示不是新算法，不改变2known roots×K1/K17×三policy的12槽、每槽144identity、27轴cap、source/model/precision、progress/spectral8、original exactgate、所有eligible beam、no_witness/cap/failure语义。P1/full_C3/physics release仍false。

原科学协议SHA `0ea4114c0bc88f780f2a788644ddc24851e7ba32e129029350cc5c98bad2255d`；工程仍4GiB总量含512MiB finalreserve，whole7200s/slot900s，GPU allocated3GiB/reserved4GiB、floor200MiB。PB级旧包络不说明实际空间不足，本协议不沿用它，也不扩大cap。

## 1. 唯一表示方案：无压缩typed pack + 内容寻址DAG

拟新增`agent_p3_c3_common_cap_compact_20261029_v1.py/tests/doc`，本文件是该stem的事前协议，尚无实现。物理文件固定为`binding.json`、`objects.pack`、`objects.idx`、`journal.bin`、`manifest.json`、`report.json`、`completion.json`。不每tensor建立文件，不创建任何展开版JSON、NPY、旧capsule/shard副本，不依赖压缩或稀疏值；不在临时目录先写原始大JSON。

1. tensor内容ID=`SHA256(domain || dtype_exact || rank || dimensions_LE64 || original_contiguous_bytes)`；原native blob先严格base64/SHA/length验证，再保存其**原dtype/shape/bytes**。同ID只append一次；碰撞须逐header/bytes相等，否则拒绝。不将f64改f32、不归一signedzero、不换endianness、不做模型容差。
2. pack tensor record含原始bytes；每个header+索引费用合计按128bytes上界。dtype仅原`<f4/<f8`、rank≤3、最大dimension192（rawchain维45也在支持内），记录精确shape。选择raw pack而非NPY，因此**NPYheader=0，落盘base64=0**；展开原blob时重建相同base64/shape/dtype/rawSHA。128bytes包含domain/type/rank/shape/payload length/contentSHA/offset/length和对齐，不另漏算tensor index。
3. Python字面float列表不是自动当native f32张量：采用独立`literal_f64_list` tag，保存binary64值位、嵌套list形状与类型；展开回Pythonfloat/list后原canonicalJSON必须逐字节相同。标量bool/int/float/null/string也分type；不把bool当int，不把int当float。int/timestamp/index按原有非负范围存u64；source静态大整数/字符串放封存字面表。不支持的类型/未知字段failclosed，不调用pickle、foreign、动态eval。
4. dict/list/record使用固定schema typed编码；key用prepare封存的field-code表，值是精确inline primitive或**32byte二进制SHA引用**。JSON摘要使用64hex，但pack不为每次引用花64ASCII。引用的目标内容不可变；不把任意同名字符串当引用。容器有type/arity/order，不排序原list。重复的task、context、request、ledger entry、native transcript、bank、路径和source capsule只保存对象/子对象一次；各出现位置保留引用和次序，不能丢重复事件。
5. journal每项104bytes：index u64、previous SHA32、payload SHA32、record SHA32；payload可再次引用原io wrapper/entry。原始payload的canonical digest另作逻辑身份，外层compact hashchain独立domain；不能伪造旧journal字节SHA。journal/native_calls/shard/capsule只是同一对象的不同可达路径，依然可逐项完整展开。
6. 每普通event/object索引条目48bytes（contentSHA32+offset8+length8），typed event row额外费用见下表。tensor index已含于tensor128header，不能重复记为普通event index。所有其它容器index/引用/标量字段已计入parent/candidate/slot/static表对应上界。pack自身header/footer、表版本/sourceSHA列入32MiB静态项。

这是**schema-specific codec**，不是声称generic JSON任意对象都能塞进某常数。实现须逐kind列固定字段/type/row length上界、decode→canonical equality；每次新字段或未分类对象立即拒绝，不能悄悄丢字段以满足row上限。后面的512/192等均是明确编码结构的可实现上限，需要RAM最大形状测试逐项证实，不是fixture平均或压缩率。

## 2. 从科学call caps推导全12计数

K1/K17每种各6槽。记N=all predict、J=all signal_compile/J、V=all VJP/grad/rawback；原cap给出：

|轴|K1一槽|K17一槽|全12上界|
|---|---:|---:|---:|
|N|80|1072|6912|
|J|11|187|1188|
|V=192J|2112|35904|228096|
|SVD|12|204|1296|
|solve|9|153|972|
|program compile|10|170|1080|
|replay|20|308|1968|
|metric|18|306|1944|
|schedule|17|17|204|

其它scalar总计B=7464。原`Ledger.call`真正条目数L≤4N+3V+2J+B=721776：四个forward资源，三种反传资源，jacobian+signal_compile，六类其余scalar。inclusive嵌套仍逐项计入，不将resource event当额外NN费用。每条成功2个resource事件；失败以failure替return；每槽允许另一个predispatch blocked，因此resource records≤2L+12=1443564。prefix失败只保留实际事件，不padding到该上界。

原`NativeSignal.observe/call/signal`、`NativeTree.replay`、原`Session`及`Ledger.call`给出下列分kind上界；字段见其真实源，非小fixture计数外推：

|kind/family|逐轴次数上界|全12数量上界|typed row最大bytes，含引用|
|---|---|---:|---:|
|resource_attempt/return/failure/blocked合计|2L+12|1443564|192|
|io_input|4N+V|255744|512|
|io_return|4N+V|255744|512|
|packet/program_packet合计|N|6912|512|
|boundary/program_boundary合计|N|6912|512|
|origin|2J|2376|512|
|nominal / AD_terminal / jacobian，各|J|各1188|512|
|raw_backward|V|228096|512|
|direction|V|228096|64|
|native_call_attempt/return/failure合计|2M|9768|128|
|其他已知noncall/top/bank事件合计|12×4096|49152|256|

array family合计A=10N+3V+5J=759348。native invocation M≤J+replay+12×144=4884；这里progress candidate replay按1728完整identity另外上计，R candidate replay已在replay cap计，不偷偷抵扣。已知noncall包括`signal_map,queue,progress_bank,bank_event,R_attempt,R_item_attempt,R_item_return,R_return,signal_refusal`及顶层denominator/arm/source/model/precision；其它kind不允许。每槽最多17parents：maps34、queue17、progressbank17；bank内部attempt/return等≤4×17、R组attempt/return34、9项attempt/return306、refusal≤17；加顶层少于1024，取4096含固定容器引用记录余量。**大bank/row/result不塞256byte行**，该行存已计费用的parent/candidate结构SHA引用。unknown kind不可用此余量冒认。

由此journal记录E≤2489928；逐index104bytes。event/object index≤E×48（另有的bank/request/context对象index在其分配桶中计费，不混为免费）。resource192bytes足以inline index/status/两u64时间及resource/shape/方向code、scope/error/delta引用；array512bytes足以kind/phase/scope/step/operation-index以及最多7个(fieldcode,tensorSHA)对；native-call128bytes按attempt/return/failure不同variant分别引用task/request或result/error，显式callerID/outerindex，不要求一行同时容纳四个SHA。transcript由callID和journal区间定位。wrapper的slot/call归属在journal关联typed header中编码，不新复制整payload；decoder必须恢复原`io(call,payload)`与顶层`arm_event(slot,payload)`，额外pointer/header纳入各row上限，不是省掉这些字段。

为避免只算event自身却遗漏原列表/容器spine，另独立收取**E×256bytes=637421568bytes**：每个原事件最多8个32byte次级位置/容器引用及其定长边界开销。typed transcript采用(callID,起止index)精确view；receipt ledger按entryID精确view；完整capsule/native_calls不重复append全事件SHA列表。每种view/container的header、计数及对象索引也从这个spine桶计费；实现要按原kind的可达路径逐项证明≤8，不允许无界generic list-of-lists。root/parent/candidate的短路径与结构引用已单列，这个额外桶不替代其费用。无需依赖事件值碰巧重复才能达到上界。

## 3. Tensor shape/字节逐轴census

native原arrays一律f32，不能在费用证明中误当JSON float；原数学/字面列表以f64另列。所有blob先验证原SHA再寻址。即使每次模型输出都不同，下列也成立；不靠全零/相同basis/重复J行得到节省。

|族|shape/内容|保存次数上界|未去重raw bytes上界|
|---|---|---|---|
|一次predict链的完整native IO|AE 1×3×10→1×3×192；predict/predictor双输入1×3×192→1×3×192；proj3×192→3×192；packet最多7数组、boundary2数组|≤19N个tensor，合计≤8192N个f32元素|4×8192N|
|一次VJP|grad input terminal192/raw≤30/basis192，grad return≤30，rawback≤30|5V个tensor、≤474V f32元素|4×474V|
|signal额外|2origin各(latent1×3×192/history1×2×10/raw≤30/mean2/std2)，nominal192、AD192、J192×≤30|≤13J个tensor；保守每J额外≤8192 f32元素|4×8192J|
|native return/response字面值|nominal192与J192×≤30，从tolist保留原Pythonfloat64|每J≤5952 f64元素|8×5952J|
|native path/context字面值|每步latent3×192+history2×10|每N≤596 f64元素|8×596N|

signal额外8192包含origin的2×(576+20+30+2+2)=1260、nominal+AD384、J5760，共7404≤8192；不能沿用旧草稿5952而漏掉origin。tensor record数T≤19N+5V+13J=1287252；字面数组记录另≤2J+2N，header/index同样128bytes（其payload费用已单列）。对fulltree/context引用不重复计所有prefix路径的bytes：每path保留全部packet引用顺序，每个新context只在产生时保存一次。

SVD/progress/ridge/bank的f64数学输出及新结构另按**每parent512KiB、每candidate64KiB、每slot1MiB**显式结构桶上计。不是将未知任意metadata塞进桶：parent最多4个192×30 f64密矩阵（raw/projected/left及保守一额外），2个30×30 right矩阵、两组30 singular、goal/residual/nominal等≤16×192向量、8×30 params/raw、固定mapping/rule/source引用，数值payload <256KiB；其余typed容器/index/引用≤256KiB。candidate含最多9context/3edge引用、raw≤90字面值、suffixraw10、postcontext引用、ridge仅params10/singular10+六scalar、原requests/eligible/node/nominalpointer等结构，numeric <4KiB，其余refs/indices <60KiB。引用的post-J/nominal/path tensor已列上表，不能再复制。

source原路径、module/class清单、原task非动态字段、projection/normalizer/goal/Q、恒定字符串/field表等**在prepare从固定原权威实测一次canonical bytes长度及各项SHA，封存exact bytes/len**，总静态预算32MiB（含packmanifest/header、固定名字/版本和已存在sourceclosure）。两个root只换动态context和binding.contextSHA，不为每次native-call复制Q或class表。prepare检查静态总长≤32MiB、可编码schema/field表，超则prepare明确拒绝；不是NN后发现才截断metadata。含自身新三件套source长度，consumer必须逐字节恢复原source对象，不能借另一cohort默认值。

## 4. 总bytes证明与边界

以上所有数量取各cap最大值独立相加，是保守上界，不要求它们同时可达；同一semantic对象内容寻址只省确定的结构重复，不假定两个不同predict数值相同。无压缩率、无浮点降精度、无删IO。文末纯RAM公式给出**2,930,867,648 bytes live（约2.730GiB）**，小于**3,758,096,384 bytes live（3.5GiB）**，余量827,228,736bytes。加完整512MiB收尾为**3,467,738,560 bytes**，小于4GiB。剩余不是新科学额度；512MiB收尾仍不用于追加NN。所有file/index/header/refs包括在相应项，pack容器的固定头尾在static32MiB项。

此为**拟实现codec的条件式事前证明**：source长度seal及逐kind结构上限必须在RAM实现中成立，不能在无实现时称production ready。schema字段不符、codec最大形状超桶、真实静态source>32MiB均应在prepare/RAM拒绝，先报MAIN；不得用截断/省字段/提升预算修补。wall/CUDA峰值不由bytes证明推断，仍按原cap守卫并保留partial。

对预期normal/cap/no_witness/unsupported四类settled，全部12槽完整表示在上述界内。异常文本/外部I/O故障不是有限shape张量：错误完整bytes走512MiB收尾保留，不裁剪假称完整；若异常本身大于reserve或磁盘故障，保持已持久化prefix、失败且无完整receipt，不能证明任意坏设备/任意长度外部错误也有完整归档。这不是模型早停/候选筛选，也不为成功批遗漏任何native证据。

## 5. source/capsule、strict与完成边界

compact有own `.prepare/.report/.consume/.packmanifest`；不伪造旧journal/report文件SHA或旧producer身份。准备时绑定所有冻结源SHA与本codec三件套。tensor/domain与container/node的物理SHA、原expanded-canonical语义SHA明确分栏。原bank/source中的完整native_call仍在逻辑capsule中；source pointer用本输出命名空间的**显式virtual artifact index**解析，条目标记`materialized=false`、expanded canonical SHA/length、pack object ID及实际pack/indexSHA。原bank只检查显式证据且`source_artifact_bytes_read=false`，不能把virtual路径宣称实际JSON文件存在或任意替换字节hash。未来实现若发现冻结校验必须读物理capsule，需报告真实阻断；不能伪造receipt/schema或临时解包旧大文件蒙混。

saved reader先验证目录精确census、manifest/hash/index/length、所有refs存在且无环/无重复key/无alias碰撞、tensor原dtype/shape/rawSHA；不容许未用对象藏额外结果。按journal逐事件精确展开，原native records按索引收集为相同Python结构（可按槽惰性加载），逐call `old.audit_native`，然后复用原共同cap `validate_tree/validate_rows`的完整saved reconstruction：choose、bank、parent双身份、全raw/history/J、所有eligible beam、所有27轴、whole/segmented、原1728denominator、不发布partial。复用原数学/重建，不冒称第二独立NN或第二独立数学实现。完整验证包含每事件、每原blobSHA、每语义capsuleSHA、sourcebeforeafter、模型精度/class闭包；消费者不做NN。

completion必须在pack/index/journal/manifest/report的文件fsync及目录fsync返回后独立取clock，记录其完整SHA/size census；自身完成标记的持久化明确排除在该saved边界外，落盘后再做运行态总wall检查，不能把report-finalfsync漏出已声明边界。新compact strict把completion纳入完整文件census并绑定receipt实际SHA，不沿用旧report早取wall。

## 6. 批准后最小实现及RAM验收

先codec+fullshape proof tests，再接现有12槽算法控制流，不修改封存globals/源码；无需再次发明算法contract。RAM逐kind最大field/type/rowbytes检查、全dtype/shape/tensorbytes包括-0/f64极端值、字面float64 canonical roundtrip、同内容复用但保留重复事件、完整12槽normal/cap/R/no_witness/failure与full reconstruction。错dtype/shape/data/ref/index/hash/unknownfield/unusedobject/循环/整数溢出/错误语义SHA都拒绝；source长度超seal拒绝。最大shape生成不运行NN，fixture不当数值分布估计。

目标CLI仍prepare/verify/run/consume/proof，只有MAIN审读并明确HOST授权后run才import torch；worker不跑production prepare/fullstrict/NN/physics。此时未实施，不给虚构runner SHA/生产ready承诺。

## 7. 纯整数RAM公式

```python
N,J,V = 6*(80+1072), 6*(11+187), 6*(2112+35904)
B = 6*((12+204)+(9+153)+(10+170)+(20+308)+(18+306)+(17+17))
L = 4*N+3*V+2*J+B
A = 10*N+3*V+5*J
C = 2*(J+6*(20+308)+12*144)
U = 12*4096
R = 2*L+12
E = R+A+V+C+U
T = 19*N+5*V+13*J
parts = dict(
    native_tensor_bytes=4*(8192*N+474*V+8192*J),
    literal_f64_bytes=8*(5952*J+596*N),
    tensor_headers_and_indices=128*(T+2*J+2*N),
    resource_rows=R*192, array_rows=A*512, direction_rows=V*64,
    native_call_rows=C*128, auxiliary_rows=U*256,
    journal=E*104, event_object_indices=E*48,
    container_spines_and_secondary_views=E*256,
    parent_structures=12*17*524288, candidates=12*144*65536,
    slot_structures=12*1048576, sealed_static_metadata=32*1048576)
assert (N,J,V,L,A,C,U,R,E,T) == (6912,1188,228096,721776,759348,9768,49152,1443564,2489928,1287252)
live = sum(parts.values())
assert live <= 4*1024**3-512*1024**2
assert live == 2930867648
assert live+512*1024**2 == 3467738560
print(parts, live, live+512*1024**2)
```
