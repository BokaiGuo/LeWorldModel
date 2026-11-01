# Line-search GN actual RGB native192 producer

新独立 producer，非 B8 24帧producer改名/旧latent复用。固定读取 MAIN 实际 physical report：
`results/p1_line_search_gn_physical_20261027_v1_main01/report.json`
SHA `3a41fe83f375e5bb3f390e9f3a0e8e17dd70c2d49a7ebe3a99079f5774de85c1`；
prepare02 SHA `f36f915ccfc2687da056810d1241da433190eff430bb54fa7b7b35a6af3818d5`。
这两个实际输入在 MAIN 提供后才绑定。不得回填success：两根physical均false、物体pose unchanged；
这里不重新计算或替换物理标签，不将latent/Q结果解释成任务成功。

## 输入与执行合同

prepare/verify仅saved读取，无模型或物理；显式run才MAIN HOST CUDA。
调用physical.verify重建其prepare02/selected plan/sourceclosure，并检查实际report和journal SHA/hashchain、
branch_closed完整row、全部6个checkpoint_saved。逐步10commands与selected预测raw完全匹配，
checkpoint0/5/10状态按实际trace匹配，文件路径限定本次physical目录。
PNG逐帧验证文件SHA及224x224 RGB像素SHA，计划/root/task/model/normalizer/goal/原Q均绑定。
终态预测来自本次line-search `root_i.json.gz.final_prediction`，绝不从旧B8终点替代。
预测对照chronology：step0为原task context最后latent，step5/10为本次wholeH2两packet末latent。

顺序固定source-slot root0[0,5,10]、root1[0,5,10]。6帧即使像素完全相同也独立执行保存计费，
无cache/dedup。CUDA frozen native encoder + existing projector，原image transform、batch1/time1，
0predict/actionencoder/AD/physics/training。禁止CPU neural fallback或新readout/Q拟合。
原Q只用于保存后的分析，不作用于raw NPY；每根Q/goal只在prepare存一份，帧通过root身份引用。

各模块分别durable attempt/return/error，预dispatch cap与root/step顺序，encoder/projector输入
CUDA device与shape核验。模型eval/frozen、before/after tensor hash与原planner tensor hash匹配，
sources/PNG/actual报告/prepare/原model文件在run后再次重建核验。
image-only模型路径显式禁止predictor/actionencoder/pred_proj及AD入口。
restore失败追加不覆盖执行首错；未load模型时model_unchanged=None，不冒称模型复核。

每帧保存`frame_00.native192.npy`至`frame_05.native192.npy`，shape192、dtype float32、
完整原始native数值，allow_pickle=False、独占创建、flush/fsync、exact readback与SHA。
严格拒绝float64/不正确shape/nonfinite，不偷偷cast。NPY先保存再进行任何Q代价分析。
`analysis.json`仅包含同root原Q下actual/predicted MSE、native预测误差；无跨Q相减，无科学passing阈值。
独立6encoder+6projector真实入口账；若projector失败，已经return的encoder仍计completed。
失败保留全部已有NPY/attempt/errors，未执行帧在report中blocked_not_attempted，停止不重试。

输出总cap2MiB，含全部NPY/journal/analysis/report，预留128KiB final；保持200MiB floor。
prepare单JSON<512KiB，hash-only closure不复制大source/模型/PNG；两根原Q/goal除外。
输出通常仅几十KiB，但不能声称任意共享disk/fsync故障都能持久化；遇故障只能partial。

## 测试与MAIN命令

mock单元测试不需要actual fixture：6重复像素仍12模块调用及6独立NPY、原Q数值、身份/路径/step/动作
篡改、CPU/grad/shape/cap拒绝、projector/sync/fsync失败分账、restore首错保留。
MAIN提供实际SHA后追加单独saved collect测试，不调用torch/模型/physics。mock不代表CUDA证据。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_line_search_gn_rgb_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_rgb_cuda_20261027_v1 prepare --output results/p1_line_search_gn_rgb_cuda_20261027_v1_prepare01.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_rgb_cuda_20261027_v1 verify --prepared results/p1_line_search_gn_rgb_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
# MAIN审查后独占执行；实现agent不运行：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_rgb_cuda_20261027_v1 run --prepared results/p1_line_search_gn_rgb_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_line_search_gn_rgb_cuda_20261027_v1_cuda_main01
```

本实现不修改任何已sealed source/report，预算与physics本身的20public steps及旧planner156/156分开记账。
