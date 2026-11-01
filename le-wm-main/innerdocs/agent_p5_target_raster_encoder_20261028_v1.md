# P5 target raster encoder：10图与16组cross-condition诊断

仅本stem新source/tests/doc，不改renderer v3、v5编码器、模型或已有结果。本模块有真实可执行prepare/verify/HOST CUDA/完整consume，不是泛manifest；但**v3实际严格receipt必须MAIN提供**，无未来SHA/默认报告值，missing/partial均拒绝。worker不运行神经/renderer/物理/网络。

## 输入与严格来源join

prepare六个具名必填参数：static_manifest/path+sha256、static_report/path+sha256、static_receipt/path+sha256。仅接受绑定源码SHA `b2b41d1aee81d37a1569fe4943fa1b31cb8c7dbee30790fa4c07ee7133a97fcd` 的 `agent_p5_renderer_factorial_20261028_v3`；调用其完整verify及consume，canonical精确返回值与实际persisted receipt相等。native report仍是v2 kernel schema；必须外层v3.consume身份和实际launch边通过，不把report.status当验收。完整重跑其五baseline byte gate、全部10条件、每图full-public-state前后精确一致、调用账本与源文件检查。

将v3 native_manifest的每个source_state精确join原v5四词full-public-state、model spec、old physics transition；goal沿v3原goal provenance。每新输入RGB及before/after路径/SHA/bytes均绑定，独占完整5state×visible/white=10图固定顺序：xy visible/white，yx visible/white，x_minus_x visible/white，zero_zero visible/white，goal visible/white。static实际state只允许原合同已声明readonly timestep0.01→0.0的render-view重建，不称dynamic restore或private contact cache复原。所有实际RGB都由v3原生生成，本模块不涂像素、不重新渲染、不读coverage选择或改变词。

old coverage/reward/done四条来自原physics全精度状态及transition，独立并列并标source，不把静态复制history当新测量。goal agent=[64,448]仍是nuisance，不是coverage目标。prep中legacy metadata不会传给encoder；实际模型输入仅10张RGB。没有teacher/readout/adapter/拟合。

## 冻结encoder与固定输出

复用冻结 `agent_p5_terminal_goal_attribution_20261028_v5` SHA `aece72c75efa24f8c5e2df12aff38a954df31307a0476edf28be2295b1320025` 的纯load_runtime/encode/source_closure调用，不修改其TAG、protocol、verify、EXPECTED或其他globals。加载同v5模型spec/loader_manifest/expected_model_receipts，模型before==after==原expected强等；原strict checkpoint load与imported source闭包必须通过。

原RGB uint8 224×224×3→BCHW float32 /255→原native make_transforms224，无flip/resize augmentation，原normalize，math SDPA/inference_mode，官方x_norm_patchtokens；每次输出float32 **256 tokens×384 channels**。不交换轴、不截CLS/register来代替本ABI。无CPU神经fallback，需MAIN明确confirm-host-cuda且CUDA可用/4GiB free scratch；CUDA进程不能带physics overlay/PYTHONPATH或已import pymunk。

每图单独batch1 encode一次，**即使RGB完全相同也不去重**。新增encoder10API/10forward samples/10frames，predictor/proprio0，AD/backward/JVP/VJP/SVD/physics0；forward由真实模块pre/post hook计量，before-dispatch限制每个固定image_key只能一次，失败attempt不退款。checkpoint加载不是额外forward，原context/predict/goal编码不重跑。本次goal visible和white各新encode一次，包含在10内，不拿旧goal token充新条件。API/samples/walltime/旧physics资源不相加，不宣称等FLOPs。

全部10个full token原样NPY fsync后，额外持久all10_tokens_sealed事件，**此后才评分**。固定16行=4word×terminal raster2×goal raster2，每行全256×384 float64 saved MSE；没有mask/region/选词/赢家/排序或新planner评分。另固定5state的visible-white feature MSE供诊断，不调metric。coverage不同单位单列。此根已知构造OOD，只能区分renderer敏感性/feature metric错配线索，不是新决策收益、机制因果识别或跨模型闭环；scientific_admission/C2始终false。

独立consumer重新验证v3实际严格receipt+所有输入闭包，重放10组attempted/returned/artifact事件和31st all-token-seal，核API/sample/frame/image顺序，逐10个token SHA/bytes/shape/dtype/path、目录完整inventory，独立重算16+5行，与原coverage并栏精确比较。少一图/词/condition、token字节/模型/metadata漂移都拒绝。partial保留10image与16cross分母，缺值null，不填0、不重试。hooks清理、model receipts/source及最终saved authority任一失败均不放行。

32MiB output含1MiB尾部reserve、200MiB disk floor、600s包含CUDA load/10encode/最终saved验证（最初saved preflight在timer前），不承诺CUDA C调用立即响应alarm；最后wall仍需<600。4GiB free scratch不是峰值保证。exclusive新目录，不覆盖失败。无神经/physics自动启动或网络下载入口。

## RAM测试与MAIN准确命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。测试使用指定venv，RAM strict边/模块hook/token生命周期不构成真实编码证据；冻结模块globals不改。真实成功v3消费尚未取得时不以mock宣称实际prepare成功。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_target_raster_encoder_20261028_v1.py
```

MAIN用实际v3产物路径/SHA填六项；以下路径按MAIN命名示例，若实际路径不同须显式传实际路径，不能复制未来hash。receipt必须full v3 consume实际完成。三个STATIC_*_SHA是MAIN实测，不是默认值：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_target_raster_encoder_20261028_v1 prepare \
 --manifest results/agent_p5_target_raster_encoder_20261028_v1_prepare_MAIN01.json \
 --static-manifest results/agent_p5_renderer_factorial_20261028_v3_prepare_MAIN01.json \
 --static-manifest-sha256 STATIC_PREPARE_SHA \
 --static-report results/agent_p5_renderer_factorial_20261028_v3_render_MAIN01/report.json \
 --static-report-sha256 STATIC_REPORT_SHA \
 --static-receipt results/agent_p5_renderer_factorial_20261028_v3_consume_MAIN01.json \
 --static-receipt-sha256 STATIC_RECEIPT_SHA
sha256sum results/agent_p5_target_raster_encoder_20261028_v1_prepare_MAIN01.json

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_target_raster_encoder_20261028_v1 verify \
 --manifest results/agent_p5_target_raster_encoder_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA

env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_target_raster_encoder_20261028_v1 cuda \
 --manifest results/agent_p5_target_raster_encoder_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA \
 --confirm-host-cuda --output results/agent_p5_target_raster_encoder_20261028_v1_cuda_MAIN01
sha256sum results/agent_p5_target_raster_encoder_20261028_v1_cuda_MAIN01/report.json

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_target_raster_encoder_20261028_v1 consume \
 --manifest results/agent_p5_target_raster_encoder_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA \
 --report results/agent_p5_target_raster_encoder_20261028_v1_cuda_MAIN01/report.json --report-sha256 ACTUAL_REPORT_SHA \
 --output results/agent_p5_target_raster_encoder_20261028_v1_consume_MAIN01.json
```

待完成：MAIN提供实际v3 strict receipt、审读本三件套、实际prepare/verify、HOST CUDA与strict。worker不执行这些神经/物理动作；输入尚未strict时prepare应拒绝，不发科学准入。
