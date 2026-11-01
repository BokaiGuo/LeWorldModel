# P5：真实重放十图接入冻结编码器（MAIN）

本新版本不修改失败静态v3、旧十图编码器或封存结果。旧编码器只接受静态render-view/dt=0合同，
不能靠重贴schema接受真实native replay。本版本完整接入新replay renderer的prepare/report/receipt，
逐source native状态（包括dt=.01）、原goal、原四词transition和原模型闭包校验，再编码全部十图。

prepare必须显式提供三个真实文件及各SHA，加MAIN已审读的实际renderer源码SHA；没有未来默认值。
完整重跑renderer.verify/consume并与持久receipt精确一致：5state/10cell、实际805 Space.step账本，
不接受partial、旧static消费或仅status。renderer后续若改变源码，需重新审读、prepare，不改已封存字节。
上游真实replay费用805steps单列；本CUDA作业新增physics=0，不把两层成本混淆。

逐词状态与原v5 full-public-state完全相同，所有新before/after都与source状态相同；原coverage/reward
只作为旧测量并栏。图像由原生renderer产生，不涂像素/改shape/关闭接触点，也不选择有利word。
goal agent仍为nuisance，不重新定义任务。prepare与consumer无模型/物理执行。

新CUDA/consume/Meter/Store/评分/预处理/模型加载函数从冻结旧十图实现逐AST保持完全相同，
各自在新模块命名空间解析本版本verify，不做全局monkeypatch。旧源码SHA也绑定。
10图固定顺序（四词+goal，各visible/white），每图batch1一次，所有完整256×384 float32 tokens
先持久封存再算16个cross-condition float64全量MSE；另算5个同state两颜色特征差。
不mask、不选winner、不拟合新Q、不作planner评分。API/sample/frame分栏，predictor/AD/新physics均0。
模型frozen/eval、before/after/expected强等、实际CUDA检查及模块hook；不允许CPU神经回退。
32MiB上限含尾部reserve，200MiB余量，600秒CUDA load/encode/final验证。

MAIN命令在inner根执行。所有ACTUAL_*只能在对应真实产物完成后填写：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_main_p5_replay_raster_encoder_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.main_p5_replay_raster_encoder_20261028_v1 prepare \
 --manifest results/main_p5_replay_raster_encoder_20261028_v1_prepare_MAIN01.json \
 --render-manifest ACTUAL_RENDER_PREPARE --render-manifest-sha256 ACTUAL_PREPARE_SHA \
 --render-report ACTUAL_RENDER_REPORT --render-report-sha256 ACTUAL_REPORT_SHA \
 --render-receipt ACTUAL_RENDER_RECEIPT --render-receipt-sha256 ACTUAL_RECEIPT_SHA \
 --renderer-source-sha256 ACTUAL_REVIEWED_RENDERER_SOURCE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.main_p5_replay_raster_encoder_20261028_v1 verify --manifest ACTUAL_ENCODER_PREPARE --manifest-sha256 ACTUAL_ENCODER_PREPARE_SHA
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.main_p5_replay_raster_encoder_20261028_v1 cuda --manifest ACTUAL_ENCODER_PREPARE --manifest-sha256 ACTUAL_ENCODER_PREPARE_SHA --confirm-host-cuda --output results/main_p5_replay_raster_encoder_20261028_v1_cuda_MAIN01
.venv/bin/python -B -m experiments.spectral_reachability.main_p5_replay_raster_encoder_20261028_v1 consume --manifest ACTUAL_ENCODER_PREPARE --manifest-sha256 ACTUAL_ENCODER_PREPARE_SHA --report ACTUAL_CUDA_REPORT --report-sha256 ACTUAL_CUDA_REPORT_SHA --output results/main_p5_replay_raster_encoder_20261028_v1_consume_MAIN01.json
```

这是一根known constructed OOD的raster敏感性诊断，不是独立确认、模型理论否定、任务收益或P5/C2通过。
实际prepare/运行/consume以MAIN真实产物为准，tests和本文不构成编码完成证据。
