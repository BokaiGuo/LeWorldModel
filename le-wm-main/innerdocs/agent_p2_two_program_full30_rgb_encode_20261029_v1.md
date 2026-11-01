# P2 full30：28槽原生RGB桥交接

## 范围与固定输入

实现批准协议 `agent_p2_two_program_full30_rgb_protocol_20261029_v1.md`，SHA
`c14a4cf207dc4c19eea6502ba198e5ce3da35d1c79974cc146c604ce66da8608`。
仅新stem source/tests/本文；旧planner/physical/协议/六槽RGB/MAIN标量分析文件与结果不修改。
experiment-code技能仅用于组织实现与夹具，用户限定优先，没有训练、科学绘图、神经或物理生产。

严格ACTUAL默认且只接受下列实际权威（路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`）：

|输入|实际SHA256|
|---|---|
|`results/agent_p2_two_program_full30_physical_20261029_v1_prepare_MAIN01.json`|`04d4a887e09381cc9869d76b259af41159d6678269c48e5a3cf0caa42df66100`|
|`results/agent_p2_two_program_full30_physical_20261029_v1_MAIN01/report.json`|`39dae04ac59d9fd1b1e3e65dfe675a4627c13ada00d8c065e655053018b35c90`|
|`results/agent_p2_two_program_full30_physical_20261029_v1_consume_MAIN01.json`|`aabc5cce3855021eba0fe5751c0ae87809cfefb4c2ed54b0d2899048fd6b3cd8`|
|physical源码|`df65ccb93a800dea8b8e810b488b4818a70f058ee721b685dd49ef10d680a4a9`|

模型prepare/report/receipt沿physical.PINS严格保持da180381…/99f2b5ed…/af2890ae…三个实际SHA。
`full_upstream_rebuilt_here=False`：继承MAIN已完成的完整model/native receipts，不调用任何上游consume/fullstrict。
独立重hash strict.bindings、physical prepare.bindings、三权威、model权威、全部source/model/weights和新三件套；
物理目录完整inventory也必须在bindings内，不能只指针声称旧artifact已被认证。
source/schema/plan/raw/context/time/RGB新接缝独立重建，不假称再次神经重算旧predictions。
prepare/verify/consume不构造env、不加载模型；run仅MAIN显式HOSTCUDA授权后加载原模型。

## 公开接口与身份

`fixed_rows(physical_prepare, physical_receipt, model_prepare, physical_output)`纯数组/metadata解析28位置；
`join_actions(physical_prepare, physical_report, events)`核全native journal链、所有warm/plan返回实际rawbits/time、
native_history_guard与首个plan intent顺序。未知事件不filter，原journal按单LF严格解析。
`build(**ACTUAL)`封存真实来源与全部bindings；`verify(path,SHA)`重建自身prepare，不重做上游fullstrict。
`process(p,rows,store,encode,check)`可注入RAM回调；`validate/consume`自有完整28槽/IO/双ledger消费。

四槽固定root0 transfer/joint、root1 transfer/joint，7cp=[0,5,10,15,20,25,30]。
新schema `agent_p2_two_program_full30_rgb_encode_20261029_v1.{prepare,contract,report,completion,consume}`。
cp0是原root192 anchor；cp5..30按序对应planner_plan.contexts[0..5]最后latent，4anchor+24forecast不混分母。
transfer A15→B15原raw连接、control15不reset；joint two-round动作不变，root1两臂相同也不dedup。
cp c为absolute10+c；causal raw10=concat(warm10,commands[:c])[-10:]，float32bits保存。
normalizer原subtract/divide恰好一次，不额外tanh/scale/clip、不重编译、不产生新的predict。

物理checkpoint只有step/png/hash/state/snapshot，解析器显式join同row plan/off/overlay0 trajectory再拼绝对路径。
cp0无response合法，核native root身份；cp>0必须同step response/state/native reward/goal成功门精确一致。
缺cp固定source/actual192/Q/error=null，不以terminal/邻帧/预测填真实标签；工程失败不伪装完整输入。
实际只读ABI检查已得到28positions/4anchors/24forecasts/28available，无新model调用或fullstrict。

## 原生IO、封存及资源

policy-free复用原bridge.Trace、encode traced_scope/visual_bridge、原preprocess_pixels和native forbidden-API guard；
不调用旧16位置H3或24位置contextual桥的fixed_rows/process/run/consume，不修改其globals。
Ledger(7)对应原生4n=28 image cap；Trace expected encoder/projector=n，其他资源0。
实际物理输入当前n=28；早停夹具可令n变小但28位置分母仍在。

每位置7份数组：pixels uint8[224,224,3]；preprocessed f32[1,1,3,224,224]；
encoder input[1,3,224,224]；完整encoder hidden[1,257,192]；CLS/projector input[1,192]；
projector output[1,192]；native latent[192]。不删重复input/output，不用只Q代替向量。
存NPY1+单gzip member、fileSHA+arraySHA+dtype/shape；消费者先核header与精确payload长度再分配。
scalar、197 tokens、尾随gzip member、未知shape、非有限值、未知数组/metadata/event全部拒绝。

仅原native model.encode，原CLS→projector；不增加latent归一化。冻结原policy/Q/goal/normalizer，eval/no-grad。
HOSTCUDA、TF32 off/autocast off/MATH、allocated3GiB/reserved4GiB含加载峰值，完整模块class-source闭包。
before/after weights/tensor hash及class inventory相同；0predict/AD/J/VJP/backward/physics/训练。
尝试/完成分别计账；callback中断保留partial tensors/ledger，hooks与predict观察器finally恢复，失败不退款。
先封存native_seal和所有原生输出，数值分析只在saved consume阶段执行，不能据输出选择配置。

硬cap：总64MiB；arrays48MiB；普通events/metadata8MiB；最终seal/report/completion保留8MiB；floor200MiB。
三组独立写前守卫，不互借、不扩容、不缩图、不少跑槽。增量字节账，不逐event扫描目录。
wall600s从verify前起；verify耗尽时不建输出/加载模型。report flush/fsync/hash后completion绑定实际report SHA，
再检查completion落盘deadline，超时写terminal_failure并拒签完整结果。final预算路径只用于保全partial。
写文件在with内部flush/fsync，之后目录fsync；真实Store/array/event→磁盘reader消费有测试。

## 完整预算证明（RAM/临时磁盘，不是模型输出证据）

固定196份数组、280个image/module/guard事件；同时计算initial_roster、image_return中重复row、
native_seal与report重复rows/modules/guards、binding、completion，未复制物理PNG目录或原journal。

|证明轴|bytes|
|---|---:|
|28位置全部原array payload|43,524,096|
|实际np.save生成完整NPY（含196个header）|43,549,184|
|保守gzip上界（不假设内容可压缩）|43,568,644|
|arrays硬cap|50,331,648|

逐array采用zlib compressBound式保守额外量n+(n>>12)+(n>>14)+(n>>25)+32；形状/dtype均固定。
完整真实Trace/guard合成回调+Store磁盘fixture实测：arrays65,632、events/metadata633,914、
final740,094，总1,439,640 bytes。常量假数据压缩好，不能据此估实际模型压缩率。
另作固定seed20261029熵压力运输夹具：全196份数组用uint8随机/float32随机数填充、保持相同shape，
完整保留所有事件/重复metadata，实测arrays40,650,839、events/metadata633,914、final740,094，
总42,024,847 bytes。它仅证明运输预算，不是有效native IO数值结果或模型证据；不交给科学分析消费。
实际rows或class inventory更长时仍强制组cap，不能从fixture小metadata冒称任意metadata必然能装下。

复现：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_two_program_full30_rgb_encode_20261029_v1.py
.venv/bin/python -m pytest -q -s tests/test_agent_p2_two_program_full30_rgb_encode_20261029_v1.py -k all28_complete_io_event_report_budget_actual_stress
```

## MAIN执行CLI

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；以下均交MAIN，不由worker执行。
prepare默认使用上述严格实际ACTUAL；也接受全部显式--physical-*参数，但值须完全一致。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_two_program_full30_rgb_encode_20261029_v1 prepare \
  --output results/agent_p2_two_program_full30_rgb_encode_20261029_v1_prepare_MAIN01.json
```

从stdout取实际sha256，不能填未来hash：

```bash
PREP_SHA=MAIN实际prepare返回SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p2_two_program_full30_rgb_encode_20261029_v1 verify \
  --prepared results/agent_p2_two_program_full30_rgb_encode_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREP_SHA"
.venv/bin/python -m experiments.spectral_reachability.agent_p2_two_program_full30_rgb_encode_20261029_v1 run \
  --prepared results/agent_p2_two_program_full30_rgb_encode_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREP_SHA" \
  --output results/agent_p2_two_program_full30_rgb_encode_20261029_v1_cuda_MAIN01 --authorize-host-cuda
```

从实际CUDA返回取report_sha256：

```bash
REPORT_SHA=MAIN实际CUDA报告SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p2_two_program_full30_rgb_encode_20261029_v1 consume \
  --output results/agent_p2_two_program_full30_rgb_encode_20261029_v1_cuda_MAIN01 --report-sha "$REPORT_SHA" \
  --receipt results/agent_p2_two_program_full30_rgb_encode_20261029_v1_consume_MAIN01.json
```

全部输出exclusive，不覆盖任何旧结果。三件套冻结后MAIN再prepare，避免source hash漂移。
consume输出`all28_full30_RGB_verified_NOT_admitted`及28rows、完整actual192、Q/MSE、4条真实history派生链、
counts、byte_census、模型/物理来源及继承费用；4anchor另核bitexact，不混24forecast。
observed context链只是保存数据派生，没有实际反馈predict/控制干预，不声称能力或效能改善。
不重复MAIN 28cp/14pair物理标量分析，joint更大支持/更多计算的notmatched边界不变。
P1/P2/C3/scientific/physical gate均false。尚待MAIN真实prepare/verify/CUDA/consume，无未来产物SHA。
