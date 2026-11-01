# Teacher fidelity v2：仅prepare无损传输修复

v1保持冻结。MAIN报告v1实际build canonical为357081B、compact为345227B，超过原262144B磁盘prepare cap；未产生prepare、未调用神经网络。这是工程序列化失败，不是科学负结果。

v2独立复制v1，唯一实验外变化为新stem/schema和prepare transport；无v1全局猴补。全部inputs/evaluators/bindings保留，未去重、裁字段或改变浮点值。预测输入、12固定分母、预算、先seal后score和指标实现不变。以下沿用合同说明中的CLI均为v2。

## Prepare envelope（新增）

磁盘prepare仍≤262144B，原200MiB floor不变。semantic canonical JSON内存上限独立为1048576B（1MiB），不是磁盘扩容。build返回原完整对象；semantic SHA为原saved.canonical(p)字节SHA256（含原canonical空格/换行规则），不是compact对象hash。新schema及transport contract字段自然形成新的v2 semantic SHA，不声称与v1摘要相等。

encode_prepare：原canonical→gzip level9/mtime0单member→标准base64→compact JSON envelope；字段严格为schema/codec/semantic_bytes/semantic_sha256/payload。外部prepare SHA绑定整个磁盘envelope。verify先read_bytes严格校验外部SHA和磁盘cap，再decode_prepare，最后与当前完整build逐对象相等检查（源闭包重建不变）。

decode_prepare：严格字段/schema/codec/整数字节数/标准base64；zlib wbits31最多输出1MiB+1，要求eof、无unused_data、无unconsumed_tail；拒绝截断、CRC错误、尾随字节和第二gzip member，检查解码长度/SHA，JSON解析后必须重新canonical逐字节等于解码原文，且是本版本prepare。无解压到文件，无无限制gzip.decompress。run/consume只经verify取得原semantic对象，不能把envelope当forward输入。

新增RAM回归使用12行真实结构类人工float32上下文/evaluators与1100条人工bindings，使semantic大于256KiB但小于1MiB，证明所有字段/原canonical字节无损、envelope≤原cap，验证重建变化拒绝；另测两gzip/尾随/CRC/截断/炸弹/错误SHA与长度/非canonical/内存与磁盘cap。人工fixture不冒充科研结果。MAIN负责实际build/prepare和HOSTCUDA；本交付不运行实际prepare、模型或物理。

---

# Moving live teacher-only factual H1 runtime v1

仅本stem source/tests/innerdoc三新文件。遵守冻结prereg
`p1_moving_live_feedback_teacher_fidelity_protocol_20261027_v1.md`，SHA
`43c903c821c161b4c79cc1211e5a3d3d7d6766695511f6294f4b69e1ac037103`。
不修改driver、analysis v1/v2或prereg；MAIN审读、prepare与HOST CUDA执行。
本交付仅RAM/mock，不产生prepare/actual report、不加载权重、不运行CPU/GPU神经或物理。
experiment-code用于固定预算、durable日志和故障测试；不训练/拟合/调参/绘图。
goal API仍被已取消的旧blocked goal占用，不假complete旧任务或整个研究。

## Authority、prepare与严格自身轨迹join

prepare默认固定实际live report
`results/p1_moving_live_feedback_physical_20261027_v1_main01/report.json`，SHA
`990843e8bba15955db5163b9b5d77c1f8eecaa1870d3745b49d46cc22ca895e3`；
原prepare `results/p1_moving_live_feedback_physical_20261027_v1_prepare_main01.json`，SHA
`bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5`。
原analysis v2 actual SHA
`5df7dd90b3da010d1bafbb66198334272ec6470b7b0f7bc62991e4b0dae0b5d8`。
只有build的显式路径参数允许定位同SHA文件；CLI不提供放宽authority或换cohort选项。
源分析helper固定v2 SHA535ea82c...、moving load_task固定246b9194...，不用错误state5版。
原report终结errors空、12full10、model unchanged与paired lineage成立才可prepare。

固定12：两个source_slot×两prefix_arm(GN/D2P)×O/I/R，root-major/layer-major/OIR顺序。
prepare.layers按(slot,arm)join原task_id/selection，不从outcome选根或后缀。
真实warmup_return与control_return按各自episode_id、step和float32 raw_bits复原；
prefix逐位等于原封存prefix，suffix逐位等于本episode dispatch seal/authorized dispatch/实际return。
O核自己的原plan suffix；I/R核本次suffix request/response SHA与实际新ridge seal，不调用compiler。
完整journal SHA/chain、prepare、counts、episode_closed与row闭合。

每个自己的checkpoint5/10、fresh_observation、guard_passed、原PNG file/pixel SHA、
encode_request/response/row.encoding、原CUDA/frozen model及各一次visual calls逐项join。
NPY namespace严格episode_encode5/10/native192.npy；不能拿同像素其它episode借账。
state/goal/final_state均严格7维，不截velocity，模型请求不含state/native标签。
24个旧NPY仅hash封存；prepare只decode12个prefix NPY，不decode任何terminal标签。
原task使用moving.load_task，原goal/Q/model/public float32 normalizer不变、训练provenance仍unknown。

prepare.source bindings包含旧task/model/source闭包、必要PNG/receipt/NPY/journal、
新三文件/prereg，不复制大模型或旧输出。verify完整重建输入并重hash，不调用神经/物理。
小prepare≤256KiB；缺关键输入/身份/hash拒绝，不能借其它root/episode补全。

## Forward域与label域隔离

prepare.inputs是严格白名单forward包：schema、identity、latents3、history2、raw5、mean/std、input_digest。
不含goal、terminal值/路径、hidden state、native结果或witness。
`latents3=[原source z5,原source z10,本episodeactual prefix5]`；
`history2=normalize_float32(concat(本次warmup末5,实际prefix5)).reshape(2,10)`。
raw5为本episode实际suffix。normalize仅原float32 subtract/divide，不/100/tanh/clip/padding。
I也在本诊断用actual prefix，不倒写其原imagined控制权限。

独立evaluators仅保身份、标签/原prefix NPY receipt和原goal，forward回调永远只拿inputs。
所有12个prediction NPY和全资源ledger先durable写入`predictions_seal.json`，
再写`all_predictions_sealed`事件，随后才通过label_loader解码terminal NPY并score。
预测失败则没有prediction seal、没有任何label decode；保全部12行和实耗attempt。
后期label错误保全部已封存预测，不重跑forward，不产生代用label。

## 原模型与实际meter

一条H1：AE输入[1,3,10]→原model.predict([1,3,192],实际AE embedding)→
实际predictor→原pred_proj完整3tokens→最后float32[192]。
原模型只加载一次、eval/requires_grad false、HOSTCUDA-only、inference_mode、
MATH SDPA、TF32off、autocastoff，原tensor SHA前后不变。
保存AE/predictor/pred_proj每个实际module全部输入输出原float32数组，核全3context和输出chronology。
不调用compile_suffix，不做零动作nominal、J/solve/候选选择或额外replay。

Python先求AE参数、后进入predict API，所以实际ledger顺序为
AE→predict_API→predictor→pred_proj→predict_API return；AE失败不能虚报predict API attempt。
模块pre-hook在实际执行前计attempt，post-hook在CUDA同步和durable数组保存后计completion。
model.predict仅在局部guard内临时实例计量，退出恢复原实例属性/类方法解析；不改他人module globals。
AD及torch.linalg solve/SVD等用显式进程排他禁止guard，全部恢复；禁止image encoder/projector入口。
全12恰12 predict_API/AE/predictor/pred_proj，API和module不相加当24forward。
AD/J/SVD/encoder/image_projector/physics均0，无CPU fallback。

## 输出、consume、partial和工程cap

normal run schema为本stem`.report`，status `teacher_factual_recorded_not_admitted`，
fixed_denominator12、independent_tasks2、P1/P4_admission=false、training/selection=false。
每行保identity/status/prediction/label/metrics；先前not_started及失败partial全部保留。
metrics为原float64 mean192：factual_MSE、predicted_Q/actual_Q及差、actual_prefix_terminal_mse。
不把原生0/12或R-I观测改善作为teacher误差结论，不拟合、不做显著性、不声称泛化或新控制收益。

8MiB整输出cap＋200MiB floor，普通写留256KiB final reserve；whole-job300s从run入口含verify/load/score/final。
allocated≤3GiB/reserved≤4GiB；每operation和写入检查。封存首次错误、secondary cleanup/hash/source/report尽力保存。
partial report保真实资源/已发生attempt，输出已存在则拒绝，不retry、不提高cap/缩水分母。
最终写盘失败会显式stderr，不冒称有完整report。

consume为纯saved：外部report SHA、prepare/source完整重建、精确artifact文件集/SHA、
prediction seal/ledger、先seal后score事件、原episode label/goal mean192复算、
全3token AE→predictor→pred_proj原数组桥接及最后token逐位一致；不运行模型。
返回`.consume`诊断rows，不发布可执行控制器commands。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；以下由MAIN在完整审读后执行，本交付未执行prepare/run：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2 prepare --output results/p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2_prepare_main01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2 verify --prepared /ABS/TEACHER_PREPARE --prepare-sha ACTUAL_TEACHER_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2 run --prepared /ABS/TEACHER_PREPARE --prepare-sha ACTUAL_TEACHER_PREPARE_SHA --output results/p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2_cuda_main01
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2 consume --output results/p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2_cuda_main01 --report-sha ACTUAL_TEACHER_REPORT_SHA
```

RAM测试覆盖全12先seal后label、原driver人工环境/人工张量生成的保存bytes接线、
未知未来字段/跨episode拒绝、full3真实hook/API顺序与恢复、额外forward/AD/image encoder禁止、
normalizer尺度、partial不退款、label失败、consume原数组/指标与artifact严格核验。
mock与静态源码验证不代表真实CUDA已成功；无训练/拟合/科学准入。

