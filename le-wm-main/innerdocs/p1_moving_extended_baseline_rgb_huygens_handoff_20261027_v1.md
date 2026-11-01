# Huygens comparison v3：extended RGB saved 接口交接

这是独立新接口文档，不修改已封存RGB三文件或共享文档。供MAIN转交Huygens；
当前没有可用的直接线程发送工具，不宣称已直接发送。仅说明消费合同，不执行CUDA/物理，
不把准备成功当实际RGB完成或P1准入。

## 固定输入与身份

repo cwd：`/home/gbk/code/LeWorldModel/le-wm-main`。
producer module：`experiments.spectral_reachability.p1_moving_extended_baseline_rgb_cuda_20261027_v1`。
其source/tests/doc SHA分别为：

- `da384d180254858238a50c0ecc602313204947060041b45597d43e6195f4ef9a`
- `af7b3ac32af0bb4aae8ed3582dbb280f3b1cfb4aee03d95a52bba1b7dd3885ef`
- `13762495bd3795095b54135ba4bd20abbdaa8b5a2bebdbd2ea9b9d6e011d2c36`

实际saved prepare：
`results/p1_moving_extended_baseline_rgb_cuda_20261027_v1_prepare01.json`，SHA
`f094c5c28dbc6c157e6cb2ef014bd5825826dbe55a6fc614b101482055ed1411`，
136961 bytes、301 bindings。prepare schema为producer stem加`.prepare`。

父physical report：
`results/p1_moving_extended_baseline_physical_20261027_v1_main01/report.json`，SHA
`f05ce7e2d773bb046c6c8ba905c9e86c367c7e19ad09174d4509e1823a297038`；
父prepare SHA `9819cd6857e043492d27a239728fb3eee6d6ed5542546796467bf0ae83425fd7`；
父selection SHA `431bebaa7f95439b40583b1476ad78209023324a5c6f10a963b2101d04aae351`。
父provider源码SHA固定为
`a0f7348ebe8c6e061b5d63a1ba4a8ecb064a26e7f7a1ef4be57fe8b07270d799`。

实际RGB report尚未由本交接文档绑定。消费时必须由MAIN另外提供实际report路径及完整外部SHA，
不能用预期路径、prepare SHA或本文件充当actual report authority。

## 全分母与事件差异（v3必须更新）

arms严格有序：`GRASP_H3_bounded`、`joint_H2_spectral_packet_mapped`。
固定2 roots × 2 arms = 4 branches，各prefix5/terminal10，共8 images。
images/rows按root-major、arm-major、role(prefix5后terminal10)顺序，index为0..7。
join identity为`(root_slot, arm, selection_digest)`；source_slot与root_slot对应同一旧moving根。
不要只按row号跨实验配对，也不要截取成功子集。

完整成功时`events.jsonl`共56行，即每图严格以下7条有序事件：

1. `image_attempt`：`provenance.index`对应该图。
2. `raw_image_saved`：`index`及`receipt`对应真实输入PNG副本。
3. `module_attempt`：`entry.image=i, resource=encoder, completed=false`。
4. `module_return`：同encoder，`completed=true`。
5. `module_attempt`：同图projector，`completed=false`。
6. `module_return`：同projector，`completed=true`。
7. `image_saved`：`receipt`为完成row（含rawpixel/latent/metrics/provenance）。

module entry.index依次为2i、2i+1；attempt/return共享同一module entry.index。
不能套旧RGB“每图6条/总48条”断言；新增raw_image_saved发生在任何神经dispatch之前。
完整路径无额外run_started/run_finished事件；不要假造事件hashchain，events文件由artifact SHA封存。
失败可能有module_error/execution_error/forbidden及不足7条的图；这不是完整56行合同，
必须保留固定8行分母及实际attempted/completed，不能补齐、忽略错误或重试。

## artifacts：18项map，目录另含report本体

完整成功report.artifact_sha256应精确为18个相对平面文件名：

- `binding.json`、`events.jsonl`。
- 对每个i=00..07：`image_ii.rgb.png`和`image_ii.native192.npy`，共16项。

map不含`report.json`（生成map时report尚未写）；正常输出目录合计19文件。
不要误将“18 artifact”解释成含report的18文件目录；也不能只验证NPY而漏掉8份PNG副本。
文件集、每项SHA、report本体外部SHA均需严格验证；拒绝遗漏、额外、重复名、越界路径或symlink。
`binding.json`含prepared绝对路径和prepare_sha256，应与report及MAIN固定prepare一致。

每row.rawpixel含`file/sha256/pixel_sha256/dtype=uint8/shape=[224,224,3]`，
PNG文件逐字等于row.provenance.checkpoint.file_sha256所绑定父physical原图；
解码RGB字节SHA等于checkpoint.pixel_sha256，不以重新压缩PNG替代原字节证明。
每row.latent含`file/sha256/dtype=float32/shape=[192]`；NPY使用allow_pickle=False，
有限native float32[192]，不重新标准化或做projection/Qfit。

## report/metrics 与纯saved入口

report.schema为producer stem加`.report`，完整status：
`moving_extended_baseline_rgb_encoded_not_admitted`。
要求errors为空、first_error为null、source_verified=true、model_unchanged=true，
model_hash_before/after相等且对应原frozen模型、8 rows均encoded，
fixed_branch_denominator=4、fixed_image_denominator=8。
resources.encoder/projector均attempted=completed=8；new_predict/new_AD/new_physics均0。
P1_admission/P2_admission/training均false，new_independent_samples=0。

`row.provenance`提供封存task/root/goal/checkpoint/selection与compiler authority/report身份。
prepare.arguments提供physical_report/physical_sha/physical_prepare_sha/physical_source_sha/arms。
prepare.bindings提供完整输入源码/任务/模型/父物理/ownconsume闭包SHA，不另复制大模型。
provider调用各planner真正ownconsume并重建实际选中commands，非generic自报模型plan。

对NPY z、provenance.root r和goal g，按float64复算：

- actual_Q = mean((z-g)^2)，root_Q = mean((r-g)^2)。
- actual_Q_change = actual_Q-root_Q；actual_vs_root_mse = mean((z-r)^2)。
- rawpixel_mean = uint8 RGB转float64后的全像素均值。
- rawpixel_vs_prefix5_mse = mean((RGB-RGB_prefix5)^2)，保持0..255原像素尺度，不除255。
- actual_Q_change_from_prefix5 = actual_Q-同branch真实prefix5 actual_Q。

prefix5后两项配对差为0；terminal只能同identity真实prefix配对。
predicted_Q/predicted_vs_real_mse/Q_improvement_sign_reversed均null，
prediction_status=`unknown_not_recomputed`；未知不得当0或伪造预测误差。

producer提供`read(path, expected_sha)`、`verify(prepared, prepare_sha)`、`build(**arguments)`、
纯数组`assemble(...)`和`compare(item,z)`；verify/build会重hash闭包并调用saved ownconsume，
不加载神经/物理，但它们是producer自验证，不能冒称独立指标证明。
comparison可直接读取外部SHA绑定的report/prepare/events/PNG/NPY并按上述算术独立复算，
不调用run、不重编码图，不读native结果选择臂。

## 运行边界

16MiB输出cap＋200MiB floor，512KiB最终报告预留；MAIN独占实际CUDA。
wall_seconds=300仅覆盖encode_images入口之后的编码循环检查，
不包含build/verify/model load，也不是整个job的硬300秒超时。
原target-free moving root/task/goal/native192/identity meanQ不变，
这是known-root校准比较，不是P1通过、任务成功或全算力公平认证。
