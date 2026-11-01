# Target-overlay matched factual CUDA fidelity v1

源码：`experiments/spectral_reachability/p1_archive_target_overlay_fidelity_cuda_20261027_v1.py`。
RAM 测试：`tests/test_p1_archive_target_overlay_fidelity_cuda_20261027_v1.py`。
本 agent 只实现与 mock 测试；不调用实际 material/build/prepare，不运行模型、CUDA 或模拟器。
所有实际执行由 MAIN 完整 review 后进行。无新 goal/Q，无 planner 候选、参数选择、训练或 Q fit。

## 固定科学问题与分母

只测 **paired rendering observation effect**：同一原始 archive 动作和同一 native 物理轨迹，
三种观测条件各自的 native192 factual 预测误差如何变化。
固定 episode ordinal `[1,6229]`，不是新的 outcome-selected 根。
每条件都独立 encode 自己的 checkpoint `[0,5,10,15,20]`；
条件为 `archive`、`nativeoff`、`nativeon`。
不由 RGB MAE 或后续 latent 误差选条件、调参、拟合 Q；不称 training lineage 已确认、
模型预测排序已修复、独立确认或 P1 科学准入。

六条 episode×condition 序列共 **30 encoder + 30 projector**。
每条序列固定四个 full-context native forward，按以下顺序：

| kind | 输入 latent 时间 | 三包原 raw action | 对应真实 label |
|---|---|---|---|
| H1_10_to_15 | 实际0/5/10 | 0:5、5:10、10:15 | 同条件实际15 |
| fresh_H2_10_to_15 | 从原实际0/5/10独立重开 | 0:5、5:10、10:15 | 同条件实际15 |
| fresh_H2_15_to_20 | 实际5/10、该 fresh H2 自己的预测15 | 5:10、10:15、15:20 | 同条件实际20 |
| teacher_forced_15_to_20 | 实际5/10/15 | 5:10、10:15、15:20 | 同条件实际20 |

共 **24 predict +24 action_encoder +24 predictor +24 pred_proj，0 AD**。
fresh H2 第一次预测必须与独立 H1 的完整3×192输出逐位一致；不复用缓存来少算，
不 allclose、不重试。teacher-forced 只接同条件真实15，绝不跨条件喂 latent。
历史归一化复用冻结 fidelity.normalized：原 float32 raw[0:20]，time-major，
原 mean/std，无 `/100`、裁剪、tanh 或 planner ball 限制（这是 factual archive，不是优化动作域）。

每次预测输出到同条件实际 successor 的 native192 MSE；
`persistence_last_input_MSE` 是每次调用最后输入 latent 的持久性对照，
`persistence_origin10_MSE` 是从原实际10出发的持久性对照。
对 fresh H2 terminal，前者用预测15、后者用实际10，字段明确分开，不能混作同一 horizon。
另报告两根各五个同期 nativeoff/nativeon latent MSE，纯数组运算、零新神经调用。
所有24输出 durable 后才评分。没有科学 pass 阈值或赢家选择。

## Producer 接口与外部 SHA

已完整读取 producer innerdoc：`innerdocs/p1_archive_target_overlay_probe_20261027_v1.md`。
严格消费其自己的 tag，不重命名旧 alignment 输出：

- report schema `p1_archive_target_overlay_probe_20261027_v1.report`，只允许 `overlay_probe_processed`。
- `overlay_probe_partial_native_stop`、engineering_failed、缺根、缺 checkpoint 一律拒绝编码，不补齐。
- 外部 report 文件 SHA、外部 producer source SHA、外部 prepare 文件 SHA 必须显式传 CLI。
- `execution_binding.json` 精确为 `prepared`（绝对路径）与 `prepare_file_sha256`。
- report.prepare_id 单独核 `producer.digest(prepare)`，不是 prepare 文件 SHA；绝不混用两者。
- 复用 producer.verify 的纯保存输入/选定 H5 slice/旧 off PNG 校验，**不调用 producer.execute**。
  不读取整个 H5、不宣称46GB文件已有完整 SHA，也不重新模拟。

每根15 frame按 step0/5/10/15/20 × overlay_index0/1/2，flag 为 off/on/off；
同时核真实 step、index、flag 与预期文件名，不能只信数组位置。
核每次 pre/post full invariant 相等、triplet内 snapshot 不变、跨时间 goal/renderer 不变、
snapshot 对应 step 的 native_return、40个原 raw command 的 float32 bits、
30个 frame receipts、实际操作分账、完整 journal hashchain及报告 tail、branch begin/end。
两次 off 的 PNG 文件 SHA和实际像素必须精确相同，旧 off 的 file/pixels/state/snapshot 全true。
PNG须原RGB uint8 224×224，无 resize/convert 补救。

**offreplay 不编码**：它是验证渲染可逆性的重复观察，不属于30个预声明编码样本。
`no_dedup=true` 适用于这30个样本，即便 archive 与 off 恰好相等也独立计算；
不适用于把 replay 校验另充编码样本。replay 文件、像素与 snapshot 证据仍完整核验并绑定。
full snapshot 不包括 solver hidden cache；本实现不扩大 producer 对物理隐藏状态的保证。

MAIN 已提供实际 producer pins（此处仅用于命令，本 agent 未执行实际 build）：

```text
source  9845afec0c0ae3cf5636606f3bbd7e2e75734dbd6b7c2648eae25e46fc8f9586
prepare 1bcc1a9e4002cc2887ac1796c62f543a8fb24d6115b2d66de47b8ce6df7a4fb6
report  184809d72b4e4a6194cb9d7591b26f4a2fe09d423a826ab02c1fcb5cf340c430
```

## 原数组、计量与失败证据

模型为原冻结 local native checkpoint；eval、requires_grad false、CUDA-only。
MATH SDP，matmul/cuDNN TF32 false，离线模型加载。复用冻结 zero-AD guard；不启动任何 AD。
后台请求 whitelist 只有身份、kind、latent_history、raw_packets、tokens，无 label、goal、Q或物理 state。

保存所有实际完整 native module tensor leaves：
encoder 输入与结构化输出（包括 ViT 全 patch last_hidden_state、若有 pooler_output）、
projector 输入/输出、action_encoder 输入/输出、model.predict 输入/输出、predictor 输入/输出、
pred_proj 输入/输出，以及原 RGB uint8、原 raw20、所有请求 raw packets/tokens、真实 latent192和完整3×192预测。
不缩为 CLS-only，不丢 patch，不改 float16，不用重建近似代替原数组。
前向 pre-hook 先记 attempt，再保存输入；返回并同步后立即保存输出、记 completed。
所有 arrays 均为原精度 NPY、无损 gzip，禁止 pickle。
三层 receipt：`sha256`=压缩文件字节、`npy_sha256`=解压原NPY字节、
`raw_sha256`=连续数组字节，另保 shape/dtype。三者命名不同，避免重复 keyword。

30图/24请求完整 census在运行前建立。失败后不续跑、不补样本，已写输入/输出和 journal保留；
未完成项保持 failed_partial 或 not_started。模块 ledger 保留实际 attempt/completed。
hook cleanup、AD guard cleanup、final model hash、final source、journal finalize、artifact hash各自留错，
原 execution 首错优先，不把工程失败记作模型负结果。
若写盘本身失败，仅已保存证据与 attempt 下界可信，不冒称完整回执。

report schema `p1_archive_target_overlay_fidelity_cuda_20261027_v1.report`；
完整status `overlay_factual_fidelity_saved_not_admitted`，否则 `partial_or_failed`。
有 images、requests、module_ledger、resources、artifact_sha256、errors/first_error、
model_hash_before/after、source_verified、`limits=limits.receipt()`、AD/physics/training和P1边界。
analysis.json保存24项MSE/persistence和10项paired latent distances。

硬限制：prepare **256KiB**，只存输入 receipt/源绑定，不内嵌图像或 Q；超限明确报错，不能盲调。
run **8MiB**总产物（含partial/journal/report），预留512KiB最终报告，每次写维持 **200MiB floor**，
启动还要200MiB+8MiB。CUDA600s、allocated3GiB/reserved4GiB；实测峰值/elapsed/minimum-free保存在limits。
不复制上游报告/PNG/模型到产物，只保存本实验实际数组与元数据。
压缩并不保证任意神经输出一定小于cap；若真实运行超限，保留工程失败，不删数组/改变精度来通过。

## RAM 验证与 MAIN 命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -s -p no:cacheprovider tests/test_p1_archive_target_overlay_fidelity_cuda_20261027_v1.py
```

29 tests：明确 producer assemble、hashchain、partial/旧tag拒绝、triplet/动作/快照篡改拒绝、
同条件 chronology、fresh H2与teacherforced分路、30/24分母、失败输入数组保留、CUDA-interface拒绝CPU、
完整结构化 native capture、cleanup首错、三层hash回归、完整合成体积。
fixture仅 RAM NumPy fake，无实际native physics、torch计算、研究数据读取或pytest tmp文件。
完整合成30/24 native trace含156条实际接口mock receipt、非零随机ViT全patch，
原NPY共27,897,600 bytes，压缩及报告5,100,916 bytes；这只是存储合同测试，不是实际模型结果或体积保证。

MAIN review冻结三文件后，显式外部SHA创建NEW prepare：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p1_archive_target_overlay_fidelity_cuda_20261027_v1 prepare \
  --source-report /home/gbk/code/LeWorldModel/le-wm-main/results/p1_archive_target_overlay_probe_20261027_v1_main01/report.json \
  --source-sha 184809d72b4e4a6194cb9d7591b26f4a2fe09d423a826ab02c1fcb5cf340c430 \
  --producer-source-sha 9845afec0c0ae3cf5636606f3bbd7e2e75734dbd6b7c2648eae25e46fc8f9586 \
  --producer-prepare-sha 1bcc1a9e4002cc2887ac1796c62f543a8fb24d6115b2d66de47b8ce6df7a4fb6 \
  --output results/p1_archive_target_overlay_fidelity_cuda_20261027_v1_prepare_main01.json

.venv/bin/python -B -m experiments.spectral_reachability.p1_archive_target_overlay_fidelity_cuda_20261027_v1 verify \
  --prepared results/p1_archive_target_overlay_fidelity_cuda_20261027_v1_prepare_main01.json --prepare-sha ACTUAL_NEW_PREPARE_SHA

.venv/bin/python -B -m experiments.spectral_reachability.p1_archive_target_overlay_fidelity_cuda_20261027_v1 run \
  --prepared results/p1_archive_target_overlay_fidelity_cuda_20261027_v1_prepare_main01.json --prepare-sha ACTUAL_NEW_PREPARE_SHA \
  --output results/p1_archive_target_overlay_fidelity_cuda_20261027_v1_cuda_main01
```

新prepare SHA必须由MAIN真实生成后独立计算，不能填写producer prepare SHA来替代。
remaining：MAIN实际build/prepare/CUDA及实际artifact/资源核验；本交付不声称这些步骤已完成。

MAIN接手后实际只读material发现sealed.core不是存在的接口；prepare前改用enc.core.hash_value，
新增三外部SHA在读取前拒绝的回归。该AttributeError是工程问题，不计为模型负结果。
最终源码/测试/文档以MAIN新prepare绑定为准，不沿用该修复前的交付SHA。
