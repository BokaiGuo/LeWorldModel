# Moving-history factual fidelity：固定两root实验

仅评价source已实际执行的factual successor；不是planner、准入审计器或P1通过证据。新的planner/physical结果不参与选root或标签。主线程独占代码审查和实际CUDA；本实现方不prepare、不执行模型或物理。

## 固定输入与边界

sourcepack：`results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json`，SHA `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`。
原source prepare SHA `c1292f0ebfb1827e5da92b32ab13d2e4a12c025c4741277ab4c4f478d6ee8cb9`。
moving encoder report：`results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json`，SHA `e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`。
task00/01 SHA分别为 `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`、`101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。

使用moving encoder自己的`load_task`，绝不调用旧zero-history loader。核对原prepare、执行journal完整hashchain、实际20个public step的attempt/return、10个warmup latest-action bits、10个witness native return与预执行commit命令、encoder task及所有native192 NPY。replay0的0/5/10 RGB必须逐像素等于moving blind history，20 RGB等于新task goal，15 RGB绑定PNG文件SHA及decoded像素SHA。源文件/模型/出口/本实验三文件均纳入prepare，run前后重验。不读取certificate或hidden state来选择模型输入。

witness raw10仅作为既成factual标签对应的动作条件；不形成planner候选、不执行witness或选择。实际非零raw history保持原顺序；原public mean/std、training provenance unknown，不再除100、不clip/tanh、不拟合。旧task已有root三个latent和goal20，不重编码。

## 最小固定协议

每slot按以下顺序产生四次fresh predict，slot固定0、1，不去重：

|调用|latent context|两历史packet|outgoing|标签|
|---|---|---|---|---|
|root_H1|真实0/5/10|实际warmup10|实际witness前5|真实15|
|observed_prefix_H1|真实5/10/15|warmup后5+witness前5|witness后5|真实20|
|open_H2_first|真实0/5/10|实际warmup10|witness前5|真实15|
|open_H2_terminal|真实5/10/第三次调用预测15|warmup后5+witness前5|witness后5|真实20|

H2重新执行第一跳，不能借用root_H1或真实15。每次模型callback仅接收slot/kind/latent_history/raw_packets/normalized tokens，不接收goal/标签/source metadata。完整float32 prediction、最后native192、输入raw packet/token/latent及AE before/after均NPY durable保存；AE不mask。全部八次完成并落盘后才计算MSE。persistence分别用真实10→15、真实15→20、真实10→15、真实10→20。MSE为float64差平方均值，模型输入输出仍原float32。无科学阈值、无选根、无独立确认声明。

## 固定资源与失败语义

仅新增两张真实step15：2 encoder + 2 existing projector；8 model.predict API、8 predictor.forward、8 action_encoder、8 pred_proj，分别24tokens（每次3）。API与子模块分开计数，不是16次预测。固定分母2slots、8predict calls；attempt在调用前durable，return在同步及cap检查后记账，不retry。异常保留此前输出、failed_partial/not_started行、完整module ledger与first_error；journal.close错误追加，不覆盖first error。

HOST CUDA only，与原encoder GPU名/CUDA/torch版本绑定，冻结原192模型tensor SHA；MATH SDP、TF32 off。0AD/grad/engine/physics/reset/render/selection/training；不实现CPU神经fallback。wall600s，allocated3221225472B、reserved4294967296B。总新run输出上限2097152B（包括PNG、NPY、partial、journal及report），普通写保留262144B供final；每写保留209715200B磁盘。prepare<256KiB且保留200MiB；首次Limits检查即使用实际output.parent disk_usage，非mock常数。源封存仅存SHA/path，不复制大closure。cap失败留存partial，不提高预算重试。report出口SHA列出此前所有实际产物，report自身SHA由主线程外部取得。

## MAIN命令（尚未执行）

在`/home/gbk/code/LeWorldModel/le-wm-main`使用项目解释器：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_moving_history_fidelity_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_fidelity_cuda_20261027_v1 prepare --sourcepack results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json --source-sha c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6 --encoder-report results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json --encoder-sha e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6 --output results/p1_constructive_moving_history_fidelity_cuda_20261027_v1_prepare01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_fidelity_cuda_20261027_v1 verify --prepared results/p1_constructive_moving_history_fidelity_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
timeout 650s env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_fidelity_cuda_20261027_v1 run --prepared results/p1_constructive_moving_history_fidelity_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_constructive_moving_history_fidelity_cuda_20261027_v1_cuda_main01
```

`PREPARE_SHA`必须替换为MAIN prepare实际输出的SHA。MAIN独立审查后决定是否运行；不覆盖已有文件。mock仅验证接线/记账，RAM saved重建仅验证现存输入可接；真实CUDA结论待MAIN的完整report，不能将构造witness成功或低latent MSE归为planner成功/P1通过。

## 实现方验证（2026-10-27）

上述无cache/pyc命令：29 passed。全部fixture在RAM；覆盖fresh H2第三次调用接线、真实非零history平移、原normalizer不clip、标签出口隔离、float32 NPY、完整/partial固定分母、2image/8API硬cap、24token记账、失败不重试、磁盘/显存/时限预调用拒绝、source action bits/hashchain拒绝、首次真实disk probe、journal close不覆盖first error、输出cap/floor及路径拒绝。mock不是实际CUDA结果。

实际saved输入RAM调用`material(...)`成功：2slots、80项source/model/export绑定、canonical prepare为19,867B；`torch`未导入，无新prepare/result写入。三文件冻结后MAIN自行审阅和执行，不接本轮planner/physical输出来筛根。
