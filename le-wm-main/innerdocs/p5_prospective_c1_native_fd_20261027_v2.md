# Native prospective C1 FD consumer v2: explicit compiler/receipt binding

新增 v2 wrapper/tests/doc；不修改已封存 v1 consumer、compiler、bridge、executor 或结果。
继承 v1 的原封不动 CUDA core、FD loop、normalization、comparison、durable Ledger；
固定 eps=.01/.005，各10维central FD，40predict、0encode、0autograd、0physics。
保持 SDPA MATH、TF32 off、frozen native loader、同 allowed context/nominal，不读 archived target reader。
无事后数值门、无 eps/selectedpacket 调整，P5/C2 false。
experiment-code 技能仅用于有界源码、tests、日志/封存；未获授权执行神经/物理。

## 新接线合同

v2 没有 hardcoded compiler-output path，也无 v1/v2 run directory 默认值。
三次非神经封存均写 exclusive 新目录：

1. freeze-source：提前封存本 wrapper/tests/doc 以及继承 v1 source/tests/doc、compiler source、seal helper。
   状态 source_prepared_awaiting_explicit_compiler_manifest，不表示 compiler ready 或 JVP完整。
2. prepare：显式新 --compiler-manifest 和该文件 SHA；验证同 compiler policy、ready bridge、7个allowed inputs，
   绑定新 bridge canonical digest、compiler input digest、loader/model/preprocessing/source hashes。
3. bind-reference：显式 --compiler-receipt PATH 和 FILE SHA；
   receipt 的 compiler manifest SHA 必须等于第2阶段显式manifest，bridge/allowed-input/model identities必须相同。
   因此不能把旧 bridge/v1 receipt 接到新 executor-v2 bridge/compiler 上。
   27个缓存/JVP数组必须来自该 receipt 同目录；journal验证FILE SHA、hash-chain、顺序和terminal状态。

main 新目录预期 `results/p5_prospective_c1_cuda_compiler_20261027_v1_cuda_main_v2`，
但 v2 consumer不编码这个路径；主线程按实际 receipt路径显式传入。
source manifest不绑定任何旧运行作为默认/替代；只有新compiler manifest ready后才允许prepare。
没有 receipt时只能等待；不能自动回退旧结果，也不会触发 CUDA。
显式 --diagnostic-failed-compiler 沿用 v1严格诊断合同，仍要求完整十列与context，
不能把 compiler_failed 标成成功 compiler；不会输出动作执行授权。

## 主线程命令

cwd 为 le-wm-main；SHA均为独立保留的文件SHA。以下无占位“通过”结果：

```bash
.venv/bin/python -m pytest -q tests/test_p5_prospective_c1_native_fd_20261027_v1.py tests/test_p5_prospective_c1_native_fd_20261027_v2.py
.venv/bin/python -m experiments.spectral_reachability.p5_prospective_c1_native_fd_20261027_v2 freeze-source --output results/p5_prospective_c1_native_fd_20261027_v2_source_prepare_v1
.venv/bin/python -m experiments.spectral_reachability.p5_prospective_c1_native_fd_20261027_v2 prepare --manifest results/p5_prospective_c1_native_fd_20261027_v2_source_prepare_v1/manifest_frozen.json --manifest-sha256 SOURCE_SHA --compiler-manifest NEW_COMPILER_PREPARE/manifest_frozen.json --compiler-manifest-sha256 NEW_COMPILER_MANIFEST_SHA --output results/p5_prospective_c1_native_fd_20261027_v2_inputs_prepare_v1
.venv/bin/python -m experiments.spectral_reachability.p5_prospective_c1_native_fd_20261027_v2 bind-reference --manifest results/p5_prospective_c1_native_fd_20261027_v2_inputs_prepare_v1/manifest_frozen.json --manifest-sha256 INPUT_PREPARE_SHA --compiler-receipt results/p5_prospective_c1_cuda_compiler_20261027_v1_cuda_main_v2/compiler_receipt.json --compiler-receipt-sha256 NEW_COMPILER_RECEIPT_SHA --output results/p5_prospective_c1_native_fd_20261027_v2_bound_main_v1
.venv/bin/python -m experiments.spectral_reachability.p5_prospective_c1_native_fd_20261027_v2 run-cuda --manifest results/p5_prospective_c1_native_fd_20261027_v2_bound_main_v1/manifest_frozen.json --manifest-sha256 BOUND_SHA --output results/p5_prospective_c1_native_fd_20261027_v2_cuda_main_v1
```

只有最后一行是主线程 CUDA命令；本交付不执行它。
继承 v1完整保存 plus/minus/FD及有效rounded normalized inputs，比较全部十列，不定义事后接受阈值。
新 run receipt另列explicit compiler manifest/receipt路径与SHA，原始错误与partial/final counters不丢失。
非神经tests包括旧/新manifest混接拒绝、源hash冲突、跨目录context拒绝、journal篡改拒绝。
可选现存v1 journal只读test仅验证解析兼容，不将其作为新v2运行默认输入或效果证据。
