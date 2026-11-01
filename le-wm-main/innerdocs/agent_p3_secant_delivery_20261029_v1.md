# Secant实际cp5模型delivery：实现交接

新stem源码/tests/本文，旧source/protocol/results不改。采用experiment-code skill组织可运行链和
fixture，不采用训练、择优修参或绘图建议。本交付未执行生产prepare/verify/consume/fullstrict、
NN、physics、网络或子代；只有RAM/NumPy回调、临时真实文件和冷CLI。

批准协议 `innerdocs/agent_p3_secant_delivery_protocol_20261029_v1.md`，SHA
`500bfc08e0b542f365c6912dfa1ec6333276035a33ac596ec3f21bf3f1b3d065`。
所有相对路径根为 `/home/gbk/code/LeWorldModel/le-wm-main/`。

## 完整authority，不只信status

固定上游 `agent_p3_reach_secant_actual_cp5_cuda_20261029_v1` 的：

|实际文件|SHA256|
|---|---|
| results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_prepare_MAIN01.json | b3cc19fe7361e6bcc03ebf533a9a5acc115438cb6efda2f3c5655e7f680799e6 |
| results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_cuda_MAIN01/report.json | 191684824c95b39fcaf4cd2683ad3f3f7cd071009cdaaa076014bd9f67df91e7 |
| results/agent_p3_reach_secant_actual_cp5_cuda_20261029_v1_consume_MAIN01.json | d96f97dd076894901edea43f6223cf8cb32da4dc630968fea8fae7834b143151 |

`load_source()`实际调用冻结 `upstream.consume(report_folder,report_SHA,prepare_path,prepare_SHA)`，
与持久receipt整份canonical相等才提取。该上游consume本身verify/build并完整消费它的原上游，
不是只读取成功字符串、摘要、同名schema或局部arm。每次new prepare/verify会完整重建；
run初始verify、finally verify和new consume也会重建，可能有明显saved CPU耗时，无新增模型调用。
worker测试只在这条明确I/O边界注入合成完整receipt，未运行真实上游fullstrict。

随后新 `source_material` 再核四组/16候选、三向authority、全model/classes/source、
原tasks/provenance/zero、精确gate、每行path/action/原rowhash及资源。
原所有候选与继承32+4zero仍由原完整authority认证，不删失败或其他三组。
输入group保存全部四候选的必要动作/path/评分字段和完整原row SHA/JSON pointer；
不把来源巨大native transcripts复制为新job成本。原字节继续由上游完整consume认证。

## 固定选择与新边界

顺序root0/H1、root0/H3、root1/H1、root1/H3；实际资格数必须4/0/0/0。
每组精确I>原zero.I且R<原zero.R，然后按 `(immediate,index)` 升序，唯一selected为首项。
没有候选明确no_witness，两verification槽`not_applicable_no_witness`，无NN调用/zero fallback。
selected失败后不尝试下一项，停止后续派发，保留四组及全部已付前缀，不重试。
这是已知结果后的探索性模型交付，不是盲确认/fresh；没有新增原科学阈值。

actualcp5完整history始终abs5/10/15、control−5/0/5，原goal/Q/normalizer和base/local身份不变。
prepare先封存选择和原raw/path；run实际重读prepare字节后使用其原raw，绝不重tanh/clip/inverse。
raw严格finite、可精确f32表示、[-1,1]、(5h,2)，little-endian f32 SHA逐项绑定。
两独立原生replay均从同完整history初始化，不借另一layout或源终点填下一输入：

- whole：H1[5]；H3[15]。
- segmented：H1仍独立第二次[5]，非缓存引用；H3精确[5,10]，仅first5后从自己的完整boundary重建。

两路结果和完整IO都durable后写`both_predictions_sealed`，再比较两路与原selected.path。
新 `bit_audit` 在原audit_native上强化所有19个packet数组的dtype/shape/bytes：raw、mean/std、
incoming2、一次normalize token、outgoing3、AE/full3 predictor/proj及完整boundary因果链。
原audit的numeric equality可接受signed zero；新层明确拒绝字节不同，不增加allclose门。
整path canonical精确相等，不只末帧；误差可保留为失败，不按ULP设新容差。

new delivery含完整task/base_task/provenance、selected/order/source row引用、raw_edge、
commit_raw5、lookahead_raw及各f32 SHA、path、两独立verification；suffix_in_plan=false。
control_start5、commit6..10、预测终点control5+5h/abs15+5h；H3余10不是执行授权。
own schema为NAME.prepare/report/group/delivery/consume；没有伪造旧R producer schema。

## 低层复用边界

冻结源精确pin：secant runner `a7ecbb3fba3957af2382aecf363fbf6dd996204f8b39aedae1a486279a0b3c08`；
secant kernel `8fd9e2210d3d9ed6814cf07b2e9f85501156eb9733ab29ec2bfac908fbca2468`；
`agent_p3_actual_cp5_r_delivery_20261029_v1.py` 为
`09af91981549547c5f770ad3d0ffa42ad92b48e37daefd612191d18ec9755a26`。

只复用其input_pair、verification_cost、NativeDelivery.replay及已封存Session/Ledger/Store/
hashchain/模型loader/精度审计、audit_native、replay_clock。所有传递闭包继承到新sources且前后核SHA。
原backend使用instance-local observe暂时包裹实际forward，finally恢复；不改旧文件、模块global、
数学函数或guard。new run还检查全部instance forward恢复及模型/classes前后不变。
新source_material/group/selection/barrier/bit_audit/make_delivery/consumer均自有，不调用旧group/delivery
来冒认新候选认证。没有重新J/SVD/求解、评分或编码。MATH为原观测guard，不自称kernel证书；
加载后tensorhash+source/文件SHA也不是训练lineage或任意历史loaded-buffer证明。

## 实际计量与完整shape-byte census

当前固定资格新付最多2predict、2AE、2predictor、2proj、2replay，样本各2、context/proj rows各6。
全四组预留最大16predict/16AE/16predictor/16proj、8replay、各48tokens/rows；不padding。
新J/signal/VJP/grad/backward/solve/SVD/metric/program_compile/encoder/imageprojector/physics/training全0。
27轴attempted/completed/new_caps与all_group_max分开，继承实际276predict/56J原全向量。
若当前唯一selected通过，累计278predict/56J；FLOPs/backward_engine未知保null，不宣称等算力。

`shape_census()`仅整数枚举四组都可交付的16packet上界，不只当前2packet：

|项|值|
|---|---:|
|每packet原生blob / 全部原生f32 payload |19 / 413440B|
|base64及每blob256B header上界 |629248B|
|精确持久blob副本数|3|
|全4组最大持久blob / NPY文件数量|912 / 0|
|metadata上界 |92274688B|
|working上界 |94162432B|
|收尾reserve |536870912B|
|总上界 |631033344B|
|output cap |2147483648B|

build先核每task192维且<=256KiB、每group元数据<=2MiB、source/provenance/cp5元数据<=16MiB，
连同64records×64KiB/packet的metadata额度。这是writer强制的全文件aggregate metadata限额，
不是假设每条record自然小于64KiB；actual Store逐次计算序列化字节减base64数据，派发写入前守卫。
blob计数/data字节也逐写入硬核912及三份预算；超过任一子限额fail closed，不缩分母/精度。
consumer独立遍历全部四个文件复算这些实际计数和metadata额度。header<=256B也实际检查。

三份的逐路径理由：①每个原生IO只作为一个group_event写入events.jsonl；②call transcript
位于group_return.row.result.native_calls，写入journal一次；③相同native_calls位于report.rows，
写入report一次。verification_return、model_delivery和所有path/state是普通JSON数值列表，
已计metadata，不含blob第四份；binding/completion无blob。内存deepcopy不算磁盘重复。
没有任何NPY输出；不能用“NPY未计”隐瞒额外文件，consumer精确文件清单拒绝孤儿。
全4合成RAM真实Store实测912blob=3×16×19、payload=3×413440B，和上述推导完全一致。
census不是单凭副本因子宣称证明：有限原生shape+精确序列化路径+writer强制metadata额度共同构成上界。
这只是磁盘上界，不冒称GPU峰值RAM认证。
工程cap沿协议：whole3600s/group1800s、floor200MiB、2GiB含512MiB reserve、GPU3/4GiB。
协作wall/逐写入guard，不是可强杀单个kernel的硬超时。失败保留原partial，不退款/删除。

## saved strict与测试入口

consumer完整重建上游及自己的全部group/native calls/时序/资源/path/barrier，不重新跑NN。
核全file census与SHA、canonical恰一换行journal、全hashchain、所有IO/事件/布局/四行。
成功目录固定binding.json/events.jsonl/report.json/completion.json四件，无NPY。
report原wall是publication前检查点，**不是最终inclusive wall证书**。
新增`publish_report`先真实write/fsync report及父目录，再SHA核全三已发布文件，才生成completion。
completion以外部`--completion-sha`绑定，内含report/prepare SHA、全三文件SHA与字节census、
inclusive_wall_seconds；consumer要求其>=report.wall且<=3600s、状态/来源/flags精确一致。
该持久时间scope明确覆盖report完整publication/hash，不虚称包含receipt自己的最后一次fsync。
run在completion写入/fsync/hash后还做最终wall守卫；若该阶段超时，新增durable
`completion_failure.json`并抛错，consumer全census拒绝这个五文件目录，不能忽略失败文件发证。
私有`_fixture=True`只供RAM测试，输出独立`.RAM_FIXTURE.consume`和
`RAM_FIXTURE_ONLY_NOT_PRODUCTION`，CLI不暴露；默认生产入口严格拒绝RAM backend。
临时测试元数据里的HOST host字段只是fixture，不能据此获得生产receipt。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_p3_secant_delivery_20261029_v1.py
```

测试涵盖完整source receipt与明确全consume调用边界、4/0/0/0、tie排序、H1/H3独立双replay、
same-context、5+10/reset索引、全模块逐位因果链、signed zero、失败不重选、双seal后拒绝、
真实Store→journal→完整own fixture consume、codec/hashchain/orphan/计量/source/gate篡改、
全4组存储上界、cold import无torch/pymunk。fixture通过不等于真实delivery或科学通过。

## MAIN完整命令（worker未执行）

所有输出须新建不存在；先审source/tests/本文。PREP_SHA/REPORT_SHA/COMPLETION_SHA取MAIN实际值，不能猜。
run返回report_sha256和completion_sha256；后者是新consume强制参数，不使用未来常量。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m experiments.spectral_reachability.agent_p3_secant_delivery_20261029_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_delivery_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_secant_delivery_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_delivery_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREP_SHA"
.venv/bin/python -m experiments.spectral_reachability.agent_p3_secant_delivery_20261029_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_delivery_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREP_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_delivery_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_SECANT_DELIVERY_HOST_CUDA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_secant_delivery_20261029_v1 consume --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_delivery_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREP_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_delivery_20261029_v1_cuda_MAIN01 --report-sha "$REPORT_SHA" --completion-sha "$COMPLETION_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_secant_delivery_20261029_v1_consume_MAIN01.json
```

当前未决：MAIN真实上游fullstrict耗时/新prepare、实际HOST CUDA双重放逐位相等/峰值、完整新strict。
模型delivery若通过，仅工程模型动作witness；P1/P3/scientific/freshness/physical_release等仍false。
原物理v2写死旧R身份，不能接新对象；若另获授权需新secant adapter、原reset/warm10/prefix5、
actualcp5 full51/RGB/state/history exact guard且在control6 intent之前durable，再只commit原first5。
本文件无物理入口，不宣称P1passed、真实任务成功或完整C3。

相对MAIN预读source1..390的必要增量：shape_census以精确三份及NPY0替代12倍；新增Store/ blob_stats
aggregate序列化守卫；新增纯saved `publish_report`及durable completion/failure；run最终publication接线；
consumer四文件全census、post-report wall及completion外部SHA、实际blob/metadata独立复算；
CLI新增--completion-sha。候选排序/资格/两路context/replay数学/2predict当前上限未变。
