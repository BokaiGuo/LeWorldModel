# Moving2 full-H2 gradient open-loop基线

本实现不是审计agent，也不是官方复现、feedback MPC、等总算力比较或P1通过。MAIN独占审查、准入、HOST CUDA/物理执行与独立复算。仅新增本stem source/test/innerdoc，不改旧sealed或共享文件；实现方仅RAM/mock验证，不prepare、不运行神经/物理。两根固定旧moving calibration任务，不独立确认。

## 固定算法、合法输入

直接复用已审读`p1_constructive_gradient_mpc_cuda_20261027_v1`的optimize/proposal/choice/journal_check以及NativeGradient.gradient。32次、η=1、单zero logits初始化不变；每次接受投影更新，不line search、不早停；从zero及全部32iterates稳定最小(cost,iteration)选择，然后nominal/whole/segmented三个fresh full-H2路径独立重放。zero仅指待优化未来logits，不是历史动作。目标为原identity native192 terminal meanQ，无action penalty/readout/训练/阈值拟合。

使用`moving_history_planners_cuda_20261027_v1.load_export`→moving encoder自己`load_task`；不调用原gradient.build、旧base.load_export或zero-only schema。新NativeGradient只强化_start非零raw/token校验，原完整H2状态及动作history可微图、scalar grad和一次tanh chain原样继承。历史原raw10为五次[.1,0]加五次[0,.1]的真实float32，原public normalizer不改、不clip/tanh、不重复缩放。task含目标RGB的模型latent（任务目标合法），但部署输入不包含witness/hidden state/native outcome；with_target=False指无目标marker的输入render合同，不等于无任务goal。

两task路径固定在`results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz`和`task_01.json.gz`，SHA分别a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b、101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74。
encoder report SHA e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6，sourcepack SHA c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6。sourcepack仅计算外部文件SHA作溯源封存，不解析内容、不读取witness/cert/物理report。encoder receipt逐slot核对task/id/blind身份、模型hash与8encoder/8projector已付成本。源闭包与weights/config、三新文件、明确task出口SHA全部prepare绑定，run前后及consumer重验；仅封存path/SHA，不复制大源包。

## 预算与计量

|资源|每root|两root|
|---|---:|---:|
|predict / action_encoder各|136|272|
|其中AD前向predict|64|128|
|非AD前向predict|72|144|
|scalar reverse sweep / grad API|32|64|
|proposal|32|64|
|selection|1|2|

backward meter与grad API是同32次scalar反向，不相加成64；不做逐192维VJP/Jacobian、SVD或linear solve。predict meter源语义为predictor.forward，新增model.predict API attempt/return与全部实际predictor latent/token输入逐一对应，二者不是额外预测。每root136次×3tokens，包括每个iterate、ADforward和winner三路径；总816个latent/action packet token位置。new encoder/projector/physics/reset/render=0；原编码成本单列继承，不称全pipeline免费。不加GRASP等新臂。

HOST CUDA only，原frozen192 tensor hash，GPU名/torch/CUDA版本与encoder receipt绑定；MATH SDP，matmul/cudnn TF32 off，无CPU fallback。wall600s；allocated≤3221225472B，reserved≤4294967296B。prepare≤262144B；run全部文件（含partial/journal/final）≤8388608B，普通写保留262144B给final；每写保留209715200B磁盘。首次资源检查就读真实output.parent free。外部建议timeout650s，不提高内部cap。

attempt调用前durable，CUDA同步后return；失败不retry、不refund，保留固定2rows、failed_partial/not_started、grad/API资源及first_error，close错误追加不覆盖。若源、数值路径、模型、资源或写入异常则commands_released=false。原result保存所有33候选/32梯度/真实路径；另外保存own model_plan及selection SHA。own consumer只接受本schema、完整两根及准确文件census、源SHA/模型/计数/全部136context/梯度/投影/选择/3路径重放闭包；拒绝partial、旧zero report、其他planner的plan，consumer无新神经调用。文件report自身SHA由MAIN外部取得。

## MAIN命令

在`/home/gbk/code/LeWorldModel/le-wm-main`执行；所有prepare/run仍由MAIN决定，本交付不执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_gradient_baseline_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_gradient_baseline_cuda_20261027_v1 prepare --task-exports results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_01.json.gz --task-shas a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b 101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74 --loader-sha 246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63 --encoder-report results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json --encoder-sha e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6 --sourcepack results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json --source-sha c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6 --output results/p1_moving_gradient_baseline_cuda_20261027_v1_prepare01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_gradient_baseline_cuda_20261027_v1 verify --prepared results/p1_moving_gradient_baseline_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
timeout 650s env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_gradient_baseline_cuda_20261027_v1 run --prepared results/p1_moving_gradient_baseline_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_moving_gradient_baseline_cuda_20261027_v1_cuda_main01
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_gradient_baseline_cuda_20261027_v1 consume --output results/p1_moving_gradient_baseline_cuda_20261027_v1_cuda_main01 --report-sha ACTUAL_REPORT_SHA
```

替换MAIN实际prepare及report SHA，不覆盖已有输出。新plan尚不被旧moving physical consumer自动准入：未来物理若需要，应由MAIN另授权新consumer显式调用本own consume。无需本实验结果来运行MAIN其他任务。

## 实现验证与状态

2026-10-27，上述禁pyc/cache的RAM/mock测试命令：30 passed。两root各33候选/32梯度/136输入/3路径重放完整走通，并经own consumer往返；完整RAM journal+results+plans+report为1,873,990B。覆盖实际非零raw/history平移、AD与replay输入篡改拒绝、API因果顺序及cap、同一32反向不重复相加、CPU backend预入口拒绝、外部SHA拒绝、源loader隔离、partial fixed rows/first error保留及资源cap。Mock数值不是模型或物理证据，也不保证实际输出必然不触cap；实际触cap必须partial/fail-closed。

实际明确两task/encoder/source文件的saved RAM build：2slots、121项源/模型/出口绑定，预计prepare 26,163B；未导入torch，未写prepare/result。sourcepack只核对文件SHA，未解析witness/hidden state/native outcome。使用experiment-code技能固定协议、durable记账及错误保留；未引入其通用训练/调参/作图流程。goal完成仅指此有界三文件实现交付，不是P1科学准入完成。
