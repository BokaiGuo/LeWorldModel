# Prospective C1 native JVP finite-difference consumer

独占新增 module/tests/doc；prepare/tests 不运行神经或物理，主线程显式 run-cuda。
沿用 prospective compiler 的同 bridge、七个 allowed numerical assets、native frozen loader/model identity、
seed0、float32 predictor、TF32 off、SDPA MATH。不调用 archived-data prepare/verify_manifest/validate_cuda、
read_arrays、row_inputs 或 frame15/target/evaluator reader。不改旧 seals。
使用 experiment-code 技能的有界协议/logging/tests；不执行重跑或画图步骤。

## 固定接口与预算

compiler 输出目录固定为 `results/p5_prospective_c1_cuda_compiler_20261027_v1_cuda_main_v1`。
prepare 先封存 source/test/doc + compiler manifest/bridge/loader/input hashes。
无实际 receipt 时状态只能是 source_prepared_awaiting_actual_JVP_reference。
随后 bind-reference 独占新目录，封存实际 receipt FILE SHA、journal 与所需27数组。

直接复用 compiler 保存的 official_encoder_1_x_norm_patchtokens[:3]（context0/5/10）、
proprio_encoder_1 expand 后的同一 features、normalized_history、normalized_nominal。
patch cache 含第四张独立 goal 的 feature，但 FD 只用前三张，goal patch 不进入预测。
缓存来源是 prospective compiler 的七个 allowed assets；不重新编码。
加载同 frozen CUDA models，并检查 before/after identity；encoder/proprio forward 被禁止。
新增 predictor40，RGB encode0/frames0，proprio0，JVP0，autograd.grad0，physics0。
不新增 nominal forward，saved nominal 不冒充新的独立 F(0) replay。

固定 z0 是 native five-zero packet 经原 float32 dataset projection 后的 normalized outgoing packet，
不是 normalized zero。eps=.01/.005，各 i=0..9，依次 plus/minus，总共40次：

```
F(z) = native_predictor(saved_visual, cat(saved_history2,z), saved_proprio)[0][:,-1]
zplus/minus = z0.float32 + float32(±eps)*ei
FD_i = (F(zplus).float64-F(zminus).float64)/(2*float64(eps))
```

无 tanh/native inverse/额外归一化，无 packet/goal/root 重选，无 eps 调整或接受阈值。
保存40个 rounded normalized inputs 及 effective displacements；分母用预声明 eps，
不事后按 rounded displacement 重标定。
保存40个完整 plus/minus successor `[98304]` float32，40个 input `[10]` float32，
两档完整 FD `[98304,10]` float64（约32MiB；不下载模型/数据）。
saved response_columns 必须与十个 jvp_column 原序完全一致。
数值报告 matrix Frob/error/relative-to-JVP/cosine/maxabs、各列对应指标、两eps敏感性、
saved primal vs saved nominal；零分母记 null，accepted/acceptance_threshold 永远 null。

## 失败与边界

使用原 durable Ledger，prehook先写intent/attempt，posthook记returned；数组独占/fsync/hash，
journal hash-chain。异常保留 partial/nonfinite 返回值与原错误；禁止第41次dispatch。
不足40或完整性检查失败为 FD_failed，CLI非零。
成功 native_JVP_FD_comparison_recorded 仅指诊断完整，不表示JVP正确或物理收益。

若 compiler 失败，仅允许显式 bind-reference --diagnostic-failed-compiler 新诊断prepare。
仍要求十列JVP、nominal、context数组完整finite、model before/after相同；缺列拒绝。
诊断成功独立状态 diagnostic_FD_recorded_from_failed_compiler，保留原失败，不冒充完整compiler。
不调用 frozen_branches，不 reselection、不写 compiler 目录；P5/C2 始终 false。

## 主线程命令

以下 SHA 均为独立保留的文件 SHA，不是 canonical bridge digest；cwd为 le-wm-main。

```bash
.venv/bin/python -m pytest -q tests/test_p5_prospective_c1_native_fd_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p5_prospective_c1_native_fd_20261027_v1 prepare --compiler-manifest results/p5_prospective_c1_cuda_compiler_20261027_v1_ready_main_v1/manifest_frozen.json --compiler-manifest-sha256 COMPILER_MANIFEST_FILE_SHA --output results/p5_prospective_c1_native_fd_20261027_v1_source_prepare_v1
.venv/bin/python -m experiments.spectral_reachability.p5_prospective_c1_native_fd_20261027_v1 bind-reference --manifest results/p5_prospective_c1_native_fd_20261027_v1_source_prepare_v1/manifest_frozen.json --manifest-sha256 SOURCE_PREPARE_FILE_SHA --compiler-receipt-sha256 ACTUAL_COMPILER_RECEIPT_FILE_SHA --output results/p5_prospective_c1_native_fd_20261027_v1_bound_main_v1
.venv/bin/python -m experiments.spectral_reachability.p5_prospective_c1_native_fd_20261027_v1 run-cuda --manifest results/p5_prospective_c1_native_fd_20261027_v1_bound_main_v1/manifest_frozen.json --manifest-sha256 BOUND_MANIFEST_FILE_SHA --output results/p5_prospective_c1_native_fd_20261027_v1_cuda_main_v1
```

主线程审查后在 fresh process 无 physics PYTHONPATH 下执行 CUDA。
非神经 linear/constant tests 不算 CUDA/JVP 实证。
