# Secant FIRST5：独立原生物理adapter

实现stem `agent_p3_secant_commit_physical_20261029_v1`；本文件与source/tests为独占新文件。
协议为 `innerdocs/agent_p3_secant_commit_physical_protocol_20261029_v1.md`，
SHA `9c49b00afb37008c0a33b248d31e0258d368b54ec459d81ad8db98b345e3b049`。
worker只RAM/NumPy/临时文件fixtures；无NN/physics/生产prepare/生产fullstrict。
使用experiment-code完成实现与回归，不执行其训练、择优调参或绘图建议。

## 已绑定的真实authority与完整重建边界

全部相对路径根 `/home/gbk/code/LeWorldModel/le-wm-main/`。
source.DEFAULT_ARGUMENTS逐一列明以下实际材料，`prepare --fixed-authorities`使用这个固定集合，
不按文件名/latest发现或按结果重选；显式参数模式必须提供全部10项，不能与preset混用。

|材料|实际路径|SHA256|
|---|---|---|
|delivery prepare|results/agent_p3_secant_delivery_20261029_v1_prepare_MAIN01.json|ad289da249cb21190ee4b59748c2944c0866cea8c24be0be0a9af37e1ec7a15b|
|delivery report|results/agent_p3_secant_delivery_20261029_v1_cuda_MAIN01/report.json|7630a2751129fd7c28a9e708494c26ebe160b1cf2656bc480084838fd7f5d9fc|
|delivery completion|上述output/completion.json|b346de1fca8777fc8734dcfce68ef24d92156309b0ff062a636229c166a3ad2b|
|delivery完整receipt|results/agent_p3_secant_delivery_20261029_v1_consume_MAIN01.json|6f583507a9e5ce6d0f859032344fc303527c250b5633fa8d3482ded3615e4fb6|
|delivery source|experiments/spectral_reachability/agent_p3_secant_delivery_20261029_v1.py|7b3449b27ec68d7f923ac0af93f186c651ccb97231894152cbfa9a01649f57d2|
|物理history bridge|results/agent_p3_physical_commit_authority_bridge_20261029_v1_MAIN01.json|9c47743431166ddb2564bf5dea5d5335e794d98ec0873a8b90417e632eefc97d|
|bridge source|experiments/spectral_reachability/agent_p3_physical_commit_authority_bridge_20261029_v1.py|fd136726fc67eac13e91b3b21514eb8a2049d398b2e86fee0dadfdca588b6647|

material先核所有外部SHA，import精确producer并调用真实完整delivery.consume，包含外部
completion SHA；rebuild_delivery要求重建结果与持久receipt整份canonical相等，不只核status。
完整delivery consume自身递归核原secant及其来源。继而bridge.build原inputs整份重建，
native.source_material原source完整重建；所有model/source/history/PNG/config绑定并入新prepare。
verify重建整份prepare；run前及finally核输入闭包；consume同样执行verify。
这些生产saved重建由MAIN执行，worker未执行。RAM只mock明确的authority回调边界，
另有纯saved actual-shape assemble，不把它当完整strict或物理运行证据。

## 四组、动作与guard

固定 `(root0,H1),(root0,H3),(root1,H1),(root1,H3)`，4组/16原候选、known2/fresh0。
实际认证delivery只有root0H1；另外三行no_witness保留commands/model_delivery/native labels null，
零constructor/reset/dispatch，不退款、不zero fallback。资源继承完整accounting另列。
原 `(immediate,index)` selected不变；逐项核candidate_index、eligible_order、source_row pointer/SHA，
以及compiled.groups对应raw actions。失败不换项，不用物理结果反选候选。

只执行原prefix5 + 原delivery.commit_raw5；全edge/commit/lookahead little-endian f32 hash和
raw[-1,1]全核，禁止clip/tanh/roundtrip变换。H3余10只lookahead，不实际执行。
每可派发组新env从原reset recipe重放warm10/prefix5，非直接hydrate cp5。
原renderer224、off控制、原seed/goal/native流程保留contacts/cache自然演化。

control6 intent之前，Meter先调用guard_proof并durable emit accepted guard，写失败不派发：
完整51个off/on/off prefix RGB/file/pixel/native snapshot exact；实际warm/prefix bytes；
incoming raw10=(warm[-5:],prefix)与provenance相等；原mean/std一次normalize与原history tokens
逐位相等；认证abs5/10/15三实际RGB/native/latent context绑定。prepare也先核一次tokens一致。
latent只引用原认证编码，不冒称本物理runner现场重编码；新NN/encoder/AD/training均0。
guard失败保partial/差异，不设容差；成功/earlystop/工程失败/no_witness区分，不补步。
首个工程错误后不派发后续可执行组；剩余no_witness仍可关闭为null行，不发生新物理调用。

## 低层复用与计量的边界

沿已审旧R物理v2的执行和saved验证结构，复制到新namespace并独立更换真实secant
producer/schema/protocol/selection/完整consume接线；没有调用旧R assemble/run/consume，
没有global monkeypatch或修改冻结文件。
复用primitives.rows_for/prefix_match/audit_all_labels及native.episode/native_hooks/Meter、
原place/dispatch/observe、constructor_contract等低层接口，source闭包逐文件核hash。
这些保留原生joint agent+object XY和角度success、native7 reward/velocity，不换成object-only或RGB Q。

完整一行：constructor=close=1、reset=2、set_state=4、goal_render=2、goal_bind=1、goal_setter=3、
public_step=20、physics_step=204、explicit_render=PNG=66。
全四行上限分别乘4；当前实际最多派发一行，是否完成/早停未知，不冒认已经付费。
physics=2×reset+10×public，goal_render含原生goal bodies工作但不是66张观测PNG。
getobs是原episode/step内部行为，没有独立getobs hook或计数证书，不能把这个未单列轴写成0；
不改变原API以凑计数。intent/entry/return/failed及nested resources沿旧审定meter范围保留。

## 全四行预算：不是任意副本乘数

shape_census纯整数，不调用模型/物理，且writer/consumer硬核同一约束：

|项|上界bytes/数量|
|---|---:|
|RGB8 scanlines：224×(224×3+1)|150752 B/图|
|zlib compressBound +1024B PNG header/chunks|151834 B/图|
|全4行PNG：4×3×(11+11)，不是每行264|264图|
|PNG总编码上界|40084176 B|
|全部metadata硬额度|17825792 B（17MiB）|
|working上界|57909968 B|
|独立final reserve|8388608 B（8MiB）|
|总上界|66298576 B|
|固定cap|67108864 B（64MiB）|

working低于cap-reserve=58720256 B，总计剩810288 B，不扩大cap。
原PNG encoder只有RGB8/PIL默认压缩，无额外文本；151834B allowance覆盖原IDAT分块/CRC。
此外实际每张PNG写入及consume均严格核<=151834B，非假定压缩比。
噪声RGB fixture验证不可压缩数据；全四组真实临时Store保存264PNG并完整读回消费。

metadata复制路径明确：fresh_observation含pre/post/receipt仅写journal一次；其中prefix51
又在branch_closed.receipt与report.rows各一份（这两份不是额外三份整个journal）；完整row也
这两条持久路径各一份。operation receipt在journal intent/terminal各一份、report
actual_operation_receipts再一份，共3份，不是把整个report任意乘3。
binding包含planner authority/resource向量一次；run_started/finished与transition/own_response/
guard事件、row内部trajectory/checkpoints/plan复制，以及completion的artifact哈希清单均计metadata。
输出无NPY，也不复制上游NN IO；只保hash/pointer及完整实际task/plan必要数据。

17MiB是writer在每次真实write前强制的所有非PNG文件aggregate额度，不宣称每条record
天然满足某个未经证明的均值；上述所有重复都按实际序列化字节计入。超额立即failclosed保partial，
不能降低精度/少写行。consumer独立全文件重新计数；scope不是GPU内存或不受控Python堆的证明。
全四合成生命周期实际metadata/PNG/operation/row数量由RAM_CENSUS输出核验，不能冒充真实物理计量。
本版真实临时Store实测：264 PNG共182424 B（这是合成图，不能代替上面的最坏编码界）；
metadata=12511944 B，低于17825792 B；1216 operation receipts、2432 intent/terminal事件、
264 observation事件、4 branch_closed rows、4 report rows，完整consumer全PNG读回通过。
固定floor200MiB/wall900s、prepare128MiB；不改变科学预算。
900s沿原native执行器从execute的Meter建立开始，含finally输入SHA检查及report/completion出版；
run前完整saved verify是单独前置读取，不冒称该900s也覆盖其耗时。单次阻塞native callback
不能被协作wall强制中断，最终超时不得获完整strict认证。

## publication与saved consumer

execute先durable report，再hash全部已写文件，才生成completion，包含external prepare/report SHA、
inclusive wall及全文件bytes/hash census；消费者强制外部completion SHA。
completion明确不含自身最后fsync；其后另做最终wall guard，超限durable completion_failure且拒绝。
输出发生出版失败则保partial，不覆盖重写；不能把prepublication report.wall当最终wall证书。
consumer核全journal exactly-one-newline canonical/hashchain/操作栈、全四行guard/native labels、
每张PNG原file/pixel/state身份与完整文件census、所有source前后SHA。失败partial可保存诊断，
并非科学负结果或成功部署认证。未知/孤儿文件拒绝；不导入模型或physics进行验证。

## MAIN命令：仅MAIN授权运行

所有输出需新建；新prepare/report/completion SHA均取真实命令返回/文件hash，不提前填值。
最终本版RAM回归：`113 passed in 61.70s`；cold CLI `--help` exit0。
包含完整全四组Store→journal/hashchain→PNG消费、单派发+三null、guard拒绝、早停、
source/selection/flags篡改、完整上游回调参数及持久receipt深层差异拒绝、publication超时保留。
未执行生产prepare/verify/fullstrict/NN/physics；真实native结果及实际guard能否通过均未认证。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_p3_secant_commit_physical_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_secant_commit_physical_20261029_v1 prepare --fixed-authorities --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_commit_physical_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_secant_commit_physical_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_commit_physical_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREP_SHA"
.venv/bin/python -m experiments.spectral_reachability.agent_p3_secant_commit_physical_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_commit_physical_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREP_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_commit_physical_20261029_v1_physics_MAIN01 --authorization MAIN_P3_SECANT_COMMIT5_PHYSICS_ONLY
sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_commit_physical_20261029_v1_physics_MAIN01/report.json /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_commit_physical_20261029_v1_physics_MAIN01/completion.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_secant_commit_physical_20261029_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_commit_physical_20261029_v1_physics_MAIN01 --report-sha "$REPORT_SHA" --completion-sha "$COMPLETION_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_commit_physical_20261029_v1_consume_MAIN01.json
```

## 尚未认证、后续另立协议的缺口

当前仅candidate动作见证，不能独自证明非贪心收益、真实能力改变或P1/P3通过。
所有scientific/physical_release/freshness/independent_confirmation/full_C3等门保持false。
未来同时间zero/原progress固定对照、postprefix能力probe需要独立事前合同/完整cohort/预算；
不得塞入当前三条no_witness，不能以cp5当前状态冒充cp10 native baseline。
共同起点需原warm10+prefix5 exact，比较同absolute20/control10的真实状态；若测终点能力，
在各自实际cp10观测后新付认证编码/J/probe并固定相同probe配置与负担，不借cp5旧J冒充。
本实现不新增这些实验、不据此次物理结果择对照；需MAIN另行审定后实施。
