# Compact C3 HOST wrapper v1 — 待MAIN审读，未生产

独占NEW三件套，冻结durable/codec/routing/native_catalog及原algorithm不改。使用接口设计skill区分HOST与RAM schema，禁止将fixture来源重贴生产标签。worker仅RAM、fakeCUDA及临时磁盘，MAIN审读后才可HOST CUDA；无physics/CPU神经/训练/子代理。

## 真实接线与源封存

`build()`调用冻结`original.build()`原始authority/task/source发现；新own-schema原样继承tasks/policy/expected_model_hash/authority/rules/profile，并绑定整个原source closure与自身三件套。没有调用旧blocked `run/consume/require_output_fit`，没有替换其proof或global。

`run()`从verify前计whole7200s，显式MAIN授权、GPU allocation探针、冻结loader、eval/no parameter grad、MATH/noTF32/noautocast。preflight数据以字节相同的canonical JSON边界变为字面类型：例如PyTorch版本的`str`子类变为普通`str`，不转换float或接受非JSON对象。每次原Session dispatch/return和原slot guard均查whole wall、峰值与precision；原900s/slot不变。真实模型所有named module class source、state_dict/allbuffers hash先实测，再seal；after必须逐字相同，所有instance forward恢复。原`original.execute`直接使用冻结`OnlineStore.event/capsule/result`，固定known2×K1/K17×3policy十二槽、1728请求身份、27轴、原progress/R8及gate/beam均保留；不接secant。

重要实际ABI：原source map包含69MiB `weights.pt`。32MiB是输出静态metadata/typed pack/table预算，不能误作输入权重文件上限。新reader adapter逐一stream SHA所有原输入；≤32MiB的源交冻结`seal_static(read_sources=True)`，更大的原输入以path/size/SHA显式放入`large_input_assets`且同样stream验证，绝不复制权重或drop原sources。完整`prepared.sources`留在static；manifest只留冻结wrapper闭包及完整sources digest，避免将数千原source条目硬塞64KiB manifest固定部分。

`make_seal`只从真实prepare、当前实测model/class/precision/host、冻结bank contract及原numeric environment构造static。field table来自固定routing schema/固定native IO名/实际authority元数据，不扫描未来events，不跑fixture推断生产字段。首个event/native之前实际install并核完整typed pack+index+JSON table≤32MiB。若真实元数据超此cap，完整拒绝；不能以fixture census替代生产测量。

## 整数预算（必须先审）

冻结运输全shape live上界3,070,758,388bytes已包含：全12槽native payload、journal104bytes/event、W/index、parent/candidate/slot所有typed结构、finite native headers、static typed32MiB与static.json32MiB、manifest16MiB、completion64KiB、最后12行及64KiB execution envelope。新`host_execution`中prepared是既有static ref，rows用已计费第三组12行，不能再存展开native transcripts。

HOST新增物理分配：external prepare16MiB + host report8MiB + external receipt16MiB =41,943,040bytes。即live=3,112,701,428；加不动的failure reserve536,870,912，总=3,649,572,340；距4GiB余645,394,956bytes。prepare/receipt即使在目录外也按完整cap计入，不拿目录size偷漏。初始static CAS复制被install_static预先硬限32MiB，即使额外再计32MiB external上界也远低于live allowance；此bootstrap后、首个event前，两个external caps计入底层transport的`extra_bytes`，后续每次CAS/journal append前都占账。completion的目录真实物理size减去这笔虚拟预留，reader再加回完整external caps检查。

report硬上限8MiB，classes单份canonical≤2MiB（before/after共4MiB）、host单份≤64KiB、rows≤12×4096bytes、prepared路径≤4096bytes、census摘要≤4096bytes；完整source entries仍在static seal，摘要有数量及全部entries语义SHA。最终仍按整个canonical report硬验8MiB。receipt对每个arm保留≤16KiB的完整27轴/状态汇总及完整ledger的事件数和语义SHA；完整ledger和native IO仍在CAS并在输出摘要前逐事件全重建，不重复展开ledger到receipt。源清单/static/table整体执行32MiB而非每项32MiB；manifest root/capsule项沿冻结finite bounds，固定portion包含wrapper sources≤64KiB。无base64/NPY额外文件，无新增per-event index，不靠压缩率/fixture平均值。GPU/wall是实际保护，不宣称先验fit或equalFLOPs。

## Own HOST saved消费

自己的`.transport/.report/.completion/.consume`，RAM分别`.ram_transport/.ram_report/.ram_consume`且默认Reader拒绝。manifest绑定whole report SHA/bytes、whole CAS/index/journal/static。completion单独CLI外部SHA认证，防止只改clock仍保持manifestSHA的漏洞；记录manifest/report及目录fsync返回后的whole边界，明确不将completion自身write/fsync/shutdown冒认为其可自证时刻，返回前再实际guard。

Reader完整文件census、CAS domain/index、原序journal hashchain、有限W/spine、无环无隐藏对象、每个返回native call与其完整capsule逐项比对及原model/source/precision单例事件。static.json为canonical排序，但原typed CAS保留dict插入顺序；消费者先核二者完整语义相同，再从真实CAS static root按原顺序重建pack/index精确计数，避免把语义相同的不同序列化错认篡改。saved consumer比对整个prepare/12rows、全部27轴，再复用原`validate_rows/validate_tree`重放所有native IO、queue/choose/caps/ridge/exactgate/beam/whole-segmented。此为复用原数学，不称独立数学实现；不加载模型、不运行NN。生产consume重新完整authority build，worker不执行。

失败保留：运行错误按原合同阻挡后续槽；不发布partial modelplan。运输/guard/static seal/模型异常均不能成功commit。运输已有目录调用冻结reserved failure保存全12行/完整错误/原prefix，不删除不退款不retry；模型加载或seal前失败用own before_dispatch_failure保存12个blocked行，不能伪装native结果。I/O同时失败无法保证新receipt，原已写字节保留。

## CLI（MAIN审后，当前不授权生产）

在`/home/gbk/code/LeWorldModel/le-wm-main`使用`-m`，不直接执行文件：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_compact_host_cuda_20261029_v1 proof
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_compact_host_cuda_20261029_v1 prepare --output results/agent_p3_c3_compact_host_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_compact_host_cuda_20261029_v1 verify --prepared PREPARED --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_compact_host_cuda_20261029_v1 run --prepared PREPARED --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_c3_compact_host_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_c3_compact_host_cuda_20261029_v1 consume --prepared PREPARED --prepare-sha ACTUAL_PREPARE_SHA --output ACTUAL_OUTPUT --manifest-sha ACTUAL_MANIFEST_SHA --completion-sha ACTUAL_COMPLETION_SHA --receipt results/agent_p3_c3_compact_host_cuda_20261029_v1_consume_MAIN01.json
```

无未来SHA常量，MAIN实际run输出三SHA。report由manifest认证，无需另作未认证report路径。receipt占事前16MiB配额。science/production_ready flags始终false，不因工程成功宣称C3/P1/原R等价/physical release。

## 测试状态

具体RAM回归结果、版本差异与冻结SHA见交付消息，不冒称旧durable24/25项覆盖本HOST wrapper。尚未执行生产prepare、fullstrict、NN、CUDA或physics；工程实现与科学准入分别判断。
