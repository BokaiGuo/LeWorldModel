# Nonlinear suffix library v2：仅补实际 installed class 源闭包

独占源码 `experiments/spectral_reachability/agent_p1_nonlinear_suffix_library_20261028_v2.py`，测试 `tests/test_agent_p1_nonlinear_suffix_library_20261028_v2.py`。完整复制冻结 v1 源码，除新 NAME、installed 源解析/绑定函数及 build metadata 外，不改算法、run、strict consume。v1 SHA `9472538fa3c63ca66626177030b3e9b656737b4444578f7338af76948995e864` 亦列入绑定，旧文件/失败报告不修改。

## 原因与最小修复

MAIN 报告 v1 CUDA load 后在 `presealed loaded model classes` fail closed，report SHA `bb155464bef333add0cb78b0390c78fd3e4495d9a709b61cc54a0833ce233e96`；MAIN 确认没有候选 attempted、没有 predict，模型前后 hash 相同。这是源码清单遗漏，不是模型/科学负结果，失败原样保留。该事实来自 MAIN，本 worker 没有重跑。

MAIN load-only 实测遗漏8个 installed 模块：

- `torch.nn.modules.activation`
- `torch.nn.modules.batchnorm`
- `torch.nn.modules.container`
- `torch.nn.modules.conv`
- `torch.nn.modules.dropout`
- `torch.nn.modules.linear`
- `torch.nn.modules.normalization`
- `transformers.activations`

`installed_class_sources()` 使用 `importlib.machinery.PathFinder.find_spec` 逐层搜索 package metadata，沿 `submodule_search_locations` 解析，不执行/import torch/transformers package，不加载模型。要求全部8项存在、origin 是绝对路径、resolve 后为真实 `.py` 文件、8路径互异；保存每个真实installed path/SHA。任一缺失、无文件或与既有binding冲突立即拒绝，不skip内建类、不放宽运行处的存在检查。build 合并8项 SHA 与冻结 v1源 SHA 到完整bindings，额外输出 `installed_class_sources` 表；verify 原样完整重建source.consume/verify及本build，run前后原样hash全部bindings，原inspect.getfile模型类检查逐项保持。

## 未变协议

全部4root-condition槽/168个固定prefix，已知2根不是4独立根；8步eta1，原identity192 Q/normalizer/动作cap/chart不变；新增3376predict/1344scalar reverse，无新视觉/physics/J/SVD/训练。iterate0仍按原float64 bounded_ridge witness + 原NumPy cast/tanh literal rawbits认证，81/168 postcast norm略超cap仍合法，不投影iterate0、不加epsilon；只有新iterate用原projected inward rounding。全部候选/失败/重复/零方向保留，stable模型Q选择不读outcomes。chronology、模块IO/反向ledger、partial failclosed及P1/freshness/physical_release false均不变。完整算法和费用解释见冻结v1 innerdoc。

测试通过AST逐函数/类比较证明：除build外所有v1函数/类字面AST一致；build仅新增installed merger赋值与返回字段；原module常量除NAME均相同。新测试复用全部40个v1 RAM/saved-only回归，并增加真实8模块metadata读取、fresh subprocess证明无torch/transformers import、每路径篡改导致hash/verify拒绝、每模块缺失拒绝、冲突/无源文件拒绝。测试fixture只改tmp文件，不触碰installed源/模型，不伪造生产回执。RAM测试不是科学证据。

## MAIN 命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`。冻结后才生产prepare；SHA占位只填真实输出。完整upstream verify仍可能耗时，与v1一致，未改成receipt shortcut。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_p1_nonlinear_suffix_library_20261028_v2.py tests/test_agent_p1_nonlinear_suffix_library_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p1_nonlinear_suffix_library_20261028_v2 prepare --output results/agent_p1_nonlinear_suffix_library_20261028_v2_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p1_nonlinear_suffix_library_20261028_v2 verify --prepared results/agent_p1_nonlinear_suffix_library_20261028_v2_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p1_nonlinear_suffix_library_20261028_v2 run --prepared results/agent_p1_nonlinear_suffix_library_20261028_v2_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p1_nonlinear_suffix_library_20261028_v2_cuda_MAIN01 --authorization MAIN_FIXED42_NONLINEAR_SUFFIX_HOST_CUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p1_nonlinear_suffix_library_20261028_v2 consume --output results/agent_p1_nonlinear_suffix_library_20261028_v2_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p1_nonlinear_suffix_library_20261028_v2_consume_MAIN01.json
```

本worker只运行RAM/NumPy及saved metadata测试和CLI help；不运行生产prepare、CUDA、neural CPU、physics或网络。实际loaded-class闭包与执行结果由MAIN审读后确认。如再出现未绑定真实class源仍failclosed，新版本修复，不修改本版封存结果，不称P1通过。
