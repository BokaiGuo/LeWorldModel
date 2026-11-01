# MPPI v2 nominal physical → selected RGB adapter

仅新source/tests/doc，不修改任何sealed closure，不新增审计器。MAIN独占prepare与CUDA；本侧只做RAM mock tests，不调用模型/模拟器。

## 固定实际输入

实际物理report `results/p1_constructive_mppi_physical_20261027_v1_main01/report.json`：本侧仅对这一实际结果单文件计算SHA256，值为 `2f5221e8684100ab8ad78b96ce77e6a17a6b5f0ee908efca3a5fa9fba544ffdc`。
其execution_binding声明physical prepare SHA `c7b9bead505e9cd1e84f7b626ae129db3aa764083aae242000dc3914070d7d02`，MAIN prepare/verify时独立重验该文件hash。
封存physical adapter source SHA沿用已交付 `c3bc45a943ffdf56b4dbaea1c1f92fdffbf7ca793939f00644561f3d6a7473df`。
原CUDA report SHA `0db92a35da17b21eca3fa9d11ab90025bc8610aabe7e634af55c18ddaa9c82f9`。

固定两原opaque tasks，单臂kernel_faithful_mppi_nominal，顺序(slot0,prefix5)、(slot0,terminal10)、(slot1,prefix5)、(slot1,terminal10)。不按success或Q选择图像，全部分支full10/initial_exact/error-free才可编码。只消费本次实际物理PNG，不借用旧endpoint。原MPPI最后nominal index34派发及72predict+72action_encoder/root搜索合同保持原样，不能冒充best sample、66predict或官方实现。

## 新schema的RAM接线

复用gradient RGB adapter的模式，调用own physical adapter.build恢复原consume命令权威，核对execution_binding→新adapter prepare→manifest/原实际commands。绝不调用generic physical.verify或adapter.verify/material/load_source，不解析sourcepack/certificate/witness。旧closure文件仅MAIN prepare/run时hash重验。

先要求journal首条prepare与原adapter prepare完整相等；深复制RAM后，仅同步把prepare和journal首条prepare的schema tag规范为generic tag，再交给未改动selected RGB v2 assemble。原磁盘文件绝不重写。保留原两根identity、goal绑定、全部10commands、初末state、资源attempt/return配平与checkpoint/guard/PNG映射核验。

PNG解码、encode_images、compare、Store、Ledger/guard复用selected RGB核心。独立run负责own verifier/receipt：frozen native CUDA encoder+projector，各4次，float32(192,)原始NPY全保留，完全相同图像也不去重。MATH SDPA、禁TF32、实际模块pre/post-hook、同步计量、无CPU fallback；0predict/0action_encoder/0AD/0physics/0training。原始attempt/完成与部分数组错误留存，失败不重试；final model/source/artifact错误独立记录，未加载模型不能声称model_unchanged。

原identity192 meanQ不拟合；输出actual_Q、root_Q、actual_Q_change及actual_vs_root_mse。predicted_Q/预测误差为unknown，不为得到数值盲拼候选history。既有搜索/物理/原编码成本分别保留，不声称全pipeline等算力。

official_MPPI=false、paper_strong_MPPI_requirement_closed=false、P1/P2 admission=false；0success不是理论否定。RAM tests不是实际CUDA结果。本侧没有运行actual build/prepare，也未遍历hash实际closure；MAIN接手时才按合同完整verify。

## 大小与MAIN命令

输出cap2MiB，diskfloor200MiB，final128KiB；4NPY约3.5KiB。不复制PNG/模型。prepare<256KiB且独占新文件，source/tests/doc参与封存，创建后不得原地修改。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_mppi_rgb_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_rgb_cuda_20261027_v1 prepare --output results/p1_constructive_mppi_rgb_cuda_20261027_v1_prepare01.json
# MAIN独立核验prepare SHA并设置PREPARE_SHA。
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_rgb_cuda_20261027_v1 verify --prepared results/p1_constructive_mppi_rgb_cuda_20261027_v1_prepare01.json --prepare-sha "$PREPARE_SHA"
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_mppi_rgb_cuda_20261027_v1 run --prepared results/p1_constructive_mppi_rgb_cuda_20261027_v1_prepare01.json --prepare-sha "$PREPARE_SHA" --output results/p1_constructive_mppi_rgb_cuda_20261027_v1_cuda_main01
```
