# Gradient MPC actual RGB adapter v1

只新增本stem三文件，不改gradient compiler、physical adapter或generic selected RGB closure。
MAIN独占prepare及HOST CUDA。本交付不执行神经/物理、不训练、不新增Q、不分派审计。

## 固定来源与图像顺序

physical report: `results/p1_constructive_gradient_mpc_physical_20261027_v1_main01/report.json`
外部SHA256 `134356eb4524998055e46b83d30cfb44369195356b8890ac1a6aea170208a75a`。
physical adapter prepare SHA256 `cabfc3a3ab6560ed9ca13e7ca4fa2960197ace7ad4b98c308927276b21c05984`。
模型命令只来自已封存gradient own consume，report SHA
`8eadbc6068ee4949f280892442eda45298934ece87dd7ae38a44b4f93d78d728`。
批准两task export/identity沿用原adapter常量，不改任务/goal/normalizer。

固定 `(slot0,gradient_MPC,step5)`, `(slot0,gradient_MPC,step10)`,
`(slot1,gradient_MPC,step5)`, `(slot1,gradient_MPC,step10)`。
全部分支必须完整full10/initialexact/无错误，否则不开始编码；不按native success选择或替换终点。
已保存物理结果的成功标签不是编码选择条件，不在本adapter新增科学准入。

## 新physical prepare schema的接线

不能调用generic physical.verify（它只能重建generic schema），也不调用
physical adapter.verify/material/load_source（避免本RGB程序解析sourcepack/certificate）。
固定外部prepare SHA，经 execution_binding→adapter prepare→manifest→own consume
映射核验。核对新schema、原始journal首项完整prepare相等、source identity、guard、
全部实际10commands、branch次序、资源事件、goalRGB和PNG像素/file SHA。
prepare的原始bindings只做hash复验，不解析其evaluator-only内容。
封存原physical/gradient/encoding/task/source bytes；无模型输入来自state/certificate/witness。

generic v2的assemble仅接受generic tag，故本adapter先验证原journal prepare完全相等，
再在深复制RAM视图同时规范化prepare及journal首项的schema tag。
其他字段及原封存文件不变，随后复用generic完整数组join。不是将旧物理报告冒充新结果。

## 编码和预算

复用generic selected RGB的PNG decoder、encode_images、compare、Store、Ledger/guard；
独立run只负责own verify/receipt/schema及MAIN-only CUDA加载，未patch任何global。
每张图native frozen encoder→existing projector→原始float32(192,) NPY，4encoder/4projector，
即使图像相同也不去重。0predict/0actionencoder/0AD/0physics/0training。
HOST CUDA唯一后端、MATH SDPA、禁TF32、同步实际module返回、无CPU fallback。
durable attempt/return/image_saved/错误；partial不重试，前后模型tensor SHA、来源最终重验。
模型未加载时model_unchanged为null，不能伪称复核成功。
原Q是identity mean192，计算actual_Q/root_Q/actual_Q_change/native MSE；
predicted_Q和预测误差保持unknown，不盲拼optimizer内部history。
所有继承compiler/physics/原编码成本另外已支付，不称全pipeline成本公平。

输出cap2MiB，保留200MiB，final128KiB；4NPY约3.5KiB，不复制PNG/checkpoint。
prepare独占<256KiB元数据，封存本source/test/doc及模型/PNG/order，实际不得更改closure。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_gradient_mpc_rgb_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_gradient_mpc_rgb_cuda_20261027_v1 prepare --output results/p1_constructive_gradient_mpc_rgb_cuda_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_gradient_mpc_rgb_cuda_20261027_v1 verify --prepared results/p1_constructive_gradient_mpc_rgb_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA_FROM_PREVIOUS_COMMAND
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_gradient_mpc_rgb_cuda_20261027_v1 run --prepared results/p1_constructive_gradient_mpc_rgb_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA_FROM_PREVIOUS_COMMAND --output results/p1_constructive_gradient_mpc_rgb_cuda_20261027_v1_cuda_main01
```

RAM tests仅验证接线、失败关闭、四张无去重计量与原始NPY，不是实际CUDA或效能证据。
本侧26 RAM tests通过；另对实际封存输入执行一次纯saved的RAM build，四图顺序正确，
序列化prepare约63KiB，无元数据写入、无模型加载。不是新的编码实验证据。
本交付尚未prepare/launch；实际数值需MAIN运行。P1/P2 admission均false。
