# P2 actual-midpoint last-latent-only suffix5：实现与 MAIN 运行接口

## 独占交付与事前问题

仅新建同名 source/tests/innerdoc，不改旧 midpoint/gap/计划/封存结果/MAIN 文档。worker 只做 RAM 与 saved metadata/NumPy 检查；没有启动神经（CPU/GPU）、physics、网络、训练或子智能体。本实现不等于科学准入。

固定原四个 known roots、joint/composition 两臂，共八项原 raw10，不挑根、动作、outcome 或指标。问题是：**在原模型输入中只纠正 step5 最后一个 latent，是否改变 suffix5 的终态预测误差和两臂排序，使其与原 actual terminal teacher Q 对齐？** 不预设答案；所有正、零、负和 ties 均保留。不重编译后缀，不生成替代动作，不重拟合 Q/readout/normalizer，不放宽原 success 或 P1。

## 上游真实资产与复用

- 原 midpoint CUDA report SHA `7b0b5c00735a7326326a4619c86539fdb51b1bf99f384fe8794a71a6b4e6321c`。
- 原 strict consume SHA `4e61d9251f86550d0830c33d5607cfe58d4b308b97865f7fc64de0013d25d135`。
- MAIN 独立8数组复算 SHA `cff183a8726af0e9f19a16f6b68534f60651edf289bdbf7331b95f1d83ca66ae`。

以上三个固定文件路径/SHA、旧 midpoint source/tests/doc 的最终 SHA 都在新源码常量中。`collect()` 调用旧 midpoint 的 saved-only verify/validate_run，而不是旧 execute；重核原 prepare、模型/源码文件哈希、八张 actual step5 的原生数组与调用日志、严格回执及 MAIN 独立数值；旧 goal/Q、原动作、root/state/PNG identity 全部继承。原 midpoint 源码保持 SHA `e1f505809313550a110db1029e6c38d1addafdf7846dbd4f26c4ddea92246cfc` 不变。

已实际核对：八个 paid selected mid contexts 全有对应 actual z5，不需要补预测；原 raw10/goal/Q/终态 selected context 也精确闭合。实际 saved 构建得到8个 jobs、222个 consumed bindings，无新模型调用。root0 两臂 mid context 与实际 z5 同 SHA；全八项保留，七个unique step5图，不冒称八个独立状态。检索未发现已完成的对应 P2 teacher-forced suffix producer/result；本任务不重复旧 step5/step10 encoder。

## 精确 ABI：只换一个 latent

原完整 midcontext：`latent_history[3,192]`、`action_history[2,10]`，原 `layout/context_length/latent_dim/action_dim` 不变。新 context 只令 `latent_history[-1] = matching actual_z5`；前两项 latent 和两项 incoming token 字节值不变。不是从 terminal context 切片凑 mid，不把旧 root signal 当 child，不重编码 older history。

原 suffix raw5 是 sealed raw10 的 `[5:10]`，float32 `[5,2]`；原 mean/std float32 `[2]` 按 `(raw5-mean)/std`，time-major reshape 为一个 `[10]` token。将其追加在原两项 history **之后**，得到 outgoing `[3,10]`，不在 append 前先 drop history。设备上 batch=1：

|真实入口|原生输入|原生输出|本轮上限|
|---|---|---|---:|
|action_encoder|`[1,3,10]`|`[1,3,192]`|8|
|predict API|teacher latents 与 AE output，各 `[1,3,192]`|`[1,3,192]`|8|
|predictor（predict 内）|相同两个 `[1,3,192]`|`[1,3,192]`|8|
|pred_proj（predict 内）|predictor output flatten `[3,192]`|`[3,192]`|8|

`native_abi(config)` 检查原 Predictor num_frames=3、输入/输出192、原 Embedder input10/emb192、原 pred_proj 192→192。不是新模型或新的generate API。全3个输出都保存，下一边界只用 API output `[0,-1]`；另外两项是原 API 必需输出，不计作额外 raw packets/child/predict。

下一 context 按原 recurrence：latents=`teacher_latents[1:] + [predicted_z10]`，history=`outgoing[1:]`。这没有执行新物理状态。原 actual z10 来自 gap 已核同 PNG/动作/goal/Q/model 的真实终态编码，直接复用，不重编码，也绝不用新 predicted_z10 冒充 actual terminal。

## 预算、精度与原生持久化

固定8 predict API、8 AE、8 predictor、8 pred_proj；**0 encoder、0原encoder projector、0AD、0physics**。AE/API/predictor 各8个 batch×3 temporal rows；pred_proj 共24个flatten rows，仍只有8个后缀终态，不把24行冒称24条独立路径。计量按 native entry 的实际 input shape 汇总 axis0 samples/temporal rows，不padding预算。

predict API 包含 predictor/pred_proj，wall是 inclusive，不能把这些wall轴相加当独立耗时；另存整次run wall。CUDA_ns/FLOPs未知为null，不填0。原 planner/derivative/search、原physics、8个midpoint encoder/projector和8个终态encoder费用继承另列，不称等全资源比较、物理收益或更省预算。

显式 HOST `cuda:0`，原 compatible loader/weights/config、全模型 frozen eval。新实例的 state hash 必须首先等于已封存 midpoint run 的 model state hash，且本次前后完全相同。源码/权重/config pre/post streaming SHA；不修改旧 globals/runner。

采用同一 MATH-only SDPA guard：math=true，flash/mem-efficient/cuDNN SDPA=false；matmul与cuDNN TF32=false；CUDA autocast disabled。before、每次 native entry、after 都记录并 strict 核对；退出恢复上下文/TF32，恢复失败不能算成功。encoder/encode/projector/rollout与公开AD入口明确拒绝。native参数必须CUDA且float32，拒CPU fallback、kwargs/错误shape、额外/错序调用。

固定嵌套调用树为 AE attempt/enter/return → predict attempt/enter → predictor attempt/enter/return → pred_proj attempt/enter/return → predict return。不是四个互不相关回执。尝试先fsync，再保存所有输入NPY，然后记录native enter才真正调用；returned与saved分开，失败不可退账，不自动retry。第一工程异常不被二次 journal/cleanup 错误覆盖，full8 rows 保留 failed/blocked_not_attempted，不换动作或只报已成功子集。

每项持久化 paid latents、原history、actual_z5、teacher latents、raw suffix5、原mean/std、归一化suffix token、outgoing tokens；每个入口所有参数、完整输出；API full3、末帧、下一完整context。均为原生float32 NPY，无pickle，逐文件dtype/shape/SHA。strict核对所有buffer连接，包括 AE输出→API/实际predictor第二输入、predictor输出flatten→pred_proj、pred_proj输出reshape→API、API末帧→nextcontext。不是只凭一个终态向量。

独占 NEW prepare/source seal、NEW sibling launch claim、NEW run dir和NEW consume JSON；不覆盖已有文件、不修改冻结prepare、不删除失败产物。64MiB输出cap、200MiB floor、900s operation-boundary wall，非抢占watchdog。source seal包括新三文件、旧midpoint及其helpers、原模型/transform/torch attention源码；未将实现签名误作科学证书。

## 事前固定 consumer 输出与解释范围

每项同时列 original open-loop、new teacher-forced suffix、unchanged actual terminal Q；同Q的 latent/projected MSE、teacher-forced-minus-open-loop Q 与MSE变化。四根全部列 C-J signed differences、替换前后 paired Q gap、是否改变排序、是否与 actual terminal teacher Q 的排序（含exact tie）一致。没有事后epsilon、阈值调参、赢家过滤或收益保证。

如果替换后排序仍错，说明仅该末latent替换在这个保留历史的模型调用中不足以恢复原终态Q排序；若对齐，也只支持本次 last-latent intervention 的描述，不证明物理因果或修复完整任务。更早history可能仍有representation问题，suffix dynamics/model error与latent-Q对native任务不对齐仍可能存在。新预测没有改变任何真实 outcome，native success、旧0/4与负结果不重标。P1/P2/scientific_admission全部false；无fresh cohort或泛化结论，不是full C3。

## MAIN 可执行接口

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。freeze后 MAIN 审核再prepare：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_midpoint_teacher_suffix_20261028_v1 prepare --output results/agent_p2_midpoint_teacher_suffix_20261028_v1_prepare_MAIN01
```

MAIN将返回的真实 prepare SHA 放入 `P2_SUFFIX_PREP_SHA`，不能填假未来SHA：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_midpoint_teacher_suffix_20261028_v1 verify --prepared results/agent_p2_midpoint_teacher_suffix_20261028_v1_prepare_MAIN01 --prepare-sha "$P2_SUFFIX_PREP_SHA"
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_midpoint_teacher_suffix_20261028_v1 run --authorize-host-cuda --prepared results/agent_p2_midpoint_teacher_suffix_20261028_v1_prepare_MAIN01 --prepare-sha "$P2_SUFFIX_PREP_SHA" --output results/agent_p2_midpoint_teacher_suffix_20261028_v1_cuda_MAIN01
```

第二条只由 MAIN 在主机执行。完整run返回后，将实际report SHA放入 `P2_SUFFIX_REPORT_SHA`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_midpoint_teacher_suffix_20261028_v1 consume --run-dir results/agent_p2_midpoint_teacher_suffix_20261028_v1_cuda_MAIN01 --report-sha "$P2_SUFFIX_REPORT_SHA" --output results/agent_p2_midpoint_teacher_suffix_20261028_v1_consume_MAIN01.json
```

prepare/verify/consume均saved-only、不加载模型；没有隐式run。运行缺MAIN显式flag会在torch import前拒绝。strict consumer 拒partial/缺项/漂移/未知model hash/错精度或错误数组与journal，不补0。

## 实际测试及待执行项

worker已真实运行saved `collect()` 并核8 jobs/222 bindings。首轮45新增前43项RAM/saved测试通过；最终冻结以最终回复联合回归终态为准。覆盖 actual z5/paid历史精确接线、真实config ABI、missing/null/bool/非float32/错误normalizer拒绝、fixed4x2分母、nested forward ABI/调用树、样本与模块实账、CPU/额外encoder/AD拒绝、完整原生数组/journal变异、第一失败保全8分母及不可退款、public prepare/verify/consume RAM端到端。冷子进程禁止导入torch/gym/physics后实际读saved；这是隔离测试，不是子智能体，不执行CPU神经。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_midpoint_teacher_suffix_20261028_v1.py tests/test_agent_p2_midpoint_teacher_20261028_v1.py tests/test_agent_p2_native_composition_gap_20261028_v1.py
```

未运行本轮真实prepare/CUDA/strict输出，无suffix新预测科学结果。MAIN尚需审核最终三SHA、生成实际prepare并运行8predict/AE/predictor/pred_proj和strict；无缺失midpoint资产需要另补模型。个人交付完成条件为实现/测试/文档/命令/SHA后冻结，而不是研究通过。冻结后此v1不再append，改动只能新版本。
