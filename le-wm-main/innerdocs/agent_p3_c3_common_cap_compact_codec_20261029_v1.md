# C3 compact codec v1：可运行RAM切片，不是生产runner

实现文件`experiments/spectral_reachability/agent_p3_c3_common_cap_compact_20261029_v1.py`；测试`tests/test_agent_p3_c3_common_cap_compact_20261029_v1.py`。原批准协议`innerdocs/agent_p3_c3_common_cap_compact_20261029_v1.md`保持不动，SHA `30dda31424ce2e244314cfeb2c5c573d886ca05ff95bc3f656c29a5fe5d2c36e`。本NEW实现说明使用不同文件名，未覆盖已封存协议doc。

## 交付及未闭合范围

已实现纯标准库Codec、原dtype张量内容寻址、Python字面量精确编码、所有已知event的typed variant、SHA journal、原transcript/ledger/slot/capsule显式view、virtual reader、排他RAM pack写入和saved fixture读取/全可达闭包校验。冷import不含numpy/torch/pickle/gym；没有prepare/run/consume生产入口，没有CPU神经、训练、physics或网络。`production_ready/full_C3_demonstrated/physical_release`始终false，fixture manifest不是生产receipt。

尚未实现：把12槽算法控制流接到磁盘CAS、生产prepare真实source静态长度seal、parent/candidate/slot结构桶逐项routing及全部producer installed-class/model闭包、生产completion及saved scientific reconstruction总入口。因而`proof()`是**条件式整数schema预算**而非已认证完整生产bytes证书；不能只看2.730GiB就启动NN。当前所有真实callback是既有NumPy RAM fixture，不是模型重算或生产fullstrict。

## 表示和literal精度

`objects.pack`串接原始typed record；`objects.idx`每项`<32sQQ`=48bytes：domain SHA、offset、length。SHA domain包含本stem，不冒认旧JSON文件SHA。每blob先检查原base64长度、规范base64、raw SHA、dtype、shape及finite，再存`B`+dtype-size/rank/LE32 dimensions+原bytes。dtype严格`<f4/<f8`、rank1..3、dimension1..192；native event进一步要求原f32与精确shape。最大tensor header含index是63bytes，低于批准128。无NPY、无压缩、落盘base64为0；展开回原blob包含相同base64和rawSHA。

字面`None/bool/int/float/str/list/dict`不同tag。int范围[-2^63,2^64)，bool不当int，float按IEEE binary64原位（含-0和subnormal）存；rectangular纯float列表用独立`V`tag和f64 bytes，混合int/bool/float列表不强制同dtype。未知类型、tuple、非finite或循环拒绝。原JSON语义hash用原canonical（sort_keys/compact separators/allow_nan=false）独立计算。字段名表在Codec构造时封存，未知key拒绝；`keys_of`只用于RAM/static schema发现，不是生产授权。

当前Codec的普通`put`是精确DAG原语，不自动声称其所有generic对象都已落入parent/candidate预算；生产routing仍须完成。`put_with_refs(original, overrides)`只接受显式字段替换，先验证ref展开与整个原字段canonical逐字节一致，再保存ref，不偷偷改schema。

## 真实variant与row费用

所有row都含wrapper flags/slot/call，恢复完整`arm_event(slot,io(call,payload))`，未丢dispatch归属。数组row精确字段为kind/scope/phase/step/operation_index/arrays；packet/program_packet/origin/boundary必须完整原array field-set；io_input有AE、predict、predictor、proj、grad五种精确组合，io_return每种仅原output；不能只凭shape接受任意名字。

row上限：arrays512、resource192、direction64、native call128、aux256 bytes。实际写前测`len(raw)`，超过即拒绝。resource精确10字段entry、完整27轴、原operation/status、shape、direction、u64 index/timestamps及未截断error；attempt/return/failure/blocked与status必须配对。native attempt/return/failure三个variant分别编码原method/task/request或result/error，不把四个SHA同时塞入128。AUX仅批准的原signal_map、queue、progress_bank、bank_event、R_*、signal_refusal和顶层denominator/arm/source/model/precision，顶层fields严格、嵌套field-code须封存；这里并不代替科学validator对嵌套bank/资格内容的复算。

`stats()`返回每kind实际row_max、原tensor shape计数、按tag物理bytes、journal104/event、真实view数和bytes；不将小fixture均值当总体预算。原H1/H3 NumPy native callback生成完整192VJP，所有事件roundtrip后原`old.audit_native`通过；全部27轴receipt事件通过ledger view按entry.index重建（不是按完成顺序），原`audit_ledger`通过。

## 容器spine实际证明（不只观察fixture）

只允许slot/native/capsule/ledger四种view；同family对同event最多一次，重复同一个view直接复用ID，部分重叠拒绝。journal为第一个membership，故任意event最多1+4=5，严格小于8。native_calls、bank.evidence和source pointer复用view ID，不新增逐event列表；未知view family拒绝。

view wire=`W`1+family1+count4+8×n事件index，含自身index48后为54+8n bytes。slot和ledger n≥1，故≤62n；native/capsule需同一个显式slot/call连续区间且不同attempt/disposition边界，n≥2，故≤35n。每family不重叠，四类总上界≤(62+62+35+35)E=**194E**，低于已批256E。不依赖事件数巨大来摊销，也不靠小fixture平均。`spine_proof()`给出此逐式证明，测试涵盖最小n到全部事件上限。当前实际view bytes也必须≤256×实际event数。

slot view不得跨slot；native/capsule不得跨call、拼非连续事件或缺边界；ledger仅最终return/failure/blocked、同slot每index一项，按原创建index排序恢复。decoder同样检查，不能只在writer检查后信任保存文件。

## source capsule不是material文件

`virtual_capsule(slot,call,original)`仅从**实际已记录的显式slot/call**取得整段transcript，不按邻近文本/文件名猜关联。输出own `.virtual_capsule` descriptor：slot/call/path/objectSHA/expanded canonical SHA/length及`materialized=false`。path必在明确`__virtual__`命名空间；不会创建或读取这个JSON文件。

`VirtualCapsuleReader.callback(slot,call,original)`是显式依赖注入reader，验证展开完全相等后返回冻结`source_evidence`回调所需path/sha256；此sha256明确是**展开canonical身份**，不声称物理JSON文件SHA。`provenance()`另外给出descriptor与实际pack/index/journal/fixture SHA绑定，两个身份不混列。没有替换旧模块globals、os读函数或伪造旧receipt。

只读冻结spectral bank的`_evidence`：它检查传入完整native_call、原source pointer字段、原producer/normalizer/actual arrays，未打开artifact_path，并返回`source_artifact_bytes_read=false`。RAM测试调用原`audit_native`认证展开IO，不把这一步当生产source闭包。未来12槽consumer必须以此明确reader适配完整capsule；若碰到别的validator必须打开material文件，则报告接缝缺口，不先写临时旧capsule或冒认materialized。

## 整数预算（仍需生产routing验收）

沿原协议12槽：N=6912，J=1188，V=228096；L=721776，arrays=759348，native delimiters=9768，resource events≤1443564，全部E≤2489928。张量形状次数：forward19N、VJP5V、signal额外13J；元素上界8192N、474V、8192J。最大shape RAM按真实variant算出6460/474/7404元素，分别不超过上界；signal额外已包含两origin，不再漏计。

`proof()`完整返回逐项parts；live总**2,930,867,648 bytes**≤3,758,096,384；含512MiB收尾总**3,467,738,560 bytes**≤4GiB。此公式不减12/1728，不改27轴、不退款、不用压缩率/FLOPs/physics假设。真实source静态32MiB长度seal、parent512KiB/candidate64KiB/slot1MiB桶还须生产接线中实测完整字段和引用证明；当前不能用generic DAG的实测小体积替代该工作。wall/GPU峰值仍是原运行保护，不声称先验效果保证。

## RAM持久化与篡改拒绝

`write_fixture`只排他写新目录四文件：objects.pack/objects.idx/journal.bin/fixture.json。schema `.RAM_fixture`绑定本codec源SHA、冻结协议SHA、封存field/static表、完整artifact hashes、roots及virtual capsules，非production report。saved reader先核全部文件census、symlink、总bytes、metadata cap、真实外部fixtureSHA、每文件SHA；随后核连续唯一index、全部record内容SHA、journal index/prev/ref/tail、所有ref和view、逐event重新编码一致性；整张图必须全部可达，unused tensor/object拒绝。capsule descriptor和pack物理绑定分别保存。

首轮fixture只封存event字段却又追加完整receipt，导致2项RAM测试在未知字段处拒绝；已把receipt字段在构造Codec之前显式纳入fixture seal，未放宽未知字段拒绝。这是fixture封存错误，不是科学失败。后续view边界加严使旧测试的错误message断言失配，已修预期，不降低边界验证。

## MAIN可运行命令（只有RAM，无CUDA入口）

项目根`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_p3_c3_common_cap_compact_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_common_cap_compact_20261029_v1 proof
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_common_cap_compact_20261029_v1 --help
```

排他RAM实物demo（MAIN自行选择尚不存在的新路径；非生产）：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_common_cap_compact_20261029_v1 roundtrip-demo --output /tmp/c3_compact_codec_MAIN01
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_common_cap_compact_20261029_v1 verify-fixture --input /tmp/c3_compact_codec_MAIN01 --fixture-sha "$ACTUAL_FIXTURE_SHA"
```

ACTUAL_FIXTURE_SHA必须取上一条实际stdout，不填未来SHA。worker只运行RAM pytest/CLI帮助/整数proof，未生产prepare/NN/physics/fullstrict。接口设计技能用于literal/physical身份隔离及显式partial边界；没有改变冻结policy或gate。下一独立slice才是完整12槽runner/strict接线，不能把本codec验收签成生产ready。
